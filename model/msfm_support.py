"""
Multi-Scale Frequency Module (MSFM) for support feature decoupling.

Adapted from FaRMamba (FFT + CBAM + MSCA). Batched dynamic-size FFT backend.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(in_planes // ratio, 1), 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(in_planes // ratio, 1), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv1(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class AttentionModule(nn.Module):
    """MSCA-style large-kernel depthwise attention (FaRMamba MSCA)."""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * u


def _build_triangular_freq_masks(H, W, num_masks, device, dtype):
    """FaRMamba-style triangular masks on fftshift grid; last mask is full-band."""
    size = max(H, W)
    masks = []
    for i in reversed(range(num_masks - 1)):
        msize = max(size // (2 ** i), 1)
        mask = torch.ones(msize, msize, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=0)
        mask = torch.rot90(mask, k=1, dims=(0, 1))
        pad_h = size - mask.shape[0]
        pad_w = size - mask.shape[1]
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        if mask.shape[-2] != H or mask.shape[-1] != W:
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True)
            mask = mask.squeeze(0).squeeze(0)
        masks.append(mask)
    full = torch.ones(H, W, device=device, dtype=dtype)
    masks.append(full)
    return masks


_CUFFT_CUDA_OK = None


def _cuda_fft_available(device):
    """Probe once: cuFFT on GPU (broken on some new GPUs with old PyTorch/CUDA stacks)."""
    global _CUFFT_CUDA_OK
    if device.type != 'cuda':
        return False
    if _CUFFT_CUDA_OK is not None:
        return _CUFFT_CUDA_OK
    try:
        probe = torch.randn(1, 1, 8, 8, device=device, dtype=torch.float32)
        torch.fft.fft2(probe, norm='ortho')
        _CUFFT_CUDA_OK = True
    except RuntimeError:
        _CUFFT_CUDA_OK = False
    return _CUFFT_CUDA_OK


class DynamicFFTTransform(nn.Module):
    """Batched 2D FFT band split; works for arbitrary H x W."""

    def __init__(self, num_freq_masks=4):
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self._mask_cache = {}

    def _get_masks(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self._mask_cache:
            self._mask_cache[key] = _build_triangular_freq_masks(
                H, W, self.num_freq_masks, device, dtype,
            )
        return self._mask_cache[key]

    def forward(self, x):
        B, C, H, W = x.shape
        orig_device = x.device

        # FFT must run in fp32; fp16 cuFFT also requires power-of-two sizes (e.g. 60x60 fails).
        with torch.cuda.amp.autocast(enabled=False):
            work = x.float().contiguous()
            if work.is_cuda and not _cuda_fft_available(work.device):
                work = work.cpu()

            fft = torch.fft.fft2(work, norm='ortho')
            fft = torch.fft.fftshift(fft, dim=(-2, -1))
            masks = self._get_masks(H, W, work.device, torch.float32)

            lh, hl, hh, ll = [], [], [], []
            for idx, mask in enumerate(masks):
                m = mask.view(1, 1, H, W)
                band = fft * m
                band = torch.fft.ifftshift(band, dim=(-2, -1))
                spatial = torch.fft.ifft2(band, norm='ortho').real
                if idx == 0:
                    lh.append(spatial)
                elif idx == 1:
                    hl.append(spatial)
                elif idx == 2:
                    hh.append(spatial)
                else:
                    ll.append(spatial)

            outs = ll[0], lh[0], hl[0], hh[0]
            if work.device != orig_device:
                outs = tuple(t.to(orig_device) for t in outs)
            return outs


class MSFMCore(nn.Module):
    """FaRMamba CombinedModule (FFT + CBAM + MSCA + 1x1 proj)."""

    def __init__(self, in_planes=256, num_freq_masks=4):
        super().__init__()
        self.in_planes = in_planes
        self.fft = DynamicFFTTransform(num_freq_masks=num_freq_masks)

        self.cbam_lh = CBAM(in_planes=in_planes)
        self.cbam_hl = CBAM(in_planes=in_planes)
        self.cbam_hh = CBAM(in_planes=in_planes)
        self.cbam_ll = CBAM(in_planes=in_planes)

        self.msca_high = AttentionModule(dim=in_planes * 3)
        self.msca_combined = AttentionModule(dim=in_planes * 4)
        self.finalconv = nn.Conv2d(in_planes * 4, in_planes, kernel_size=1)

        nn.init.zeros_(self.finalconv.weight)
        nn.init.zeros_(self.finalconv.bias)

    def forward(self, x):
        ll, lh, hl, hh = self.fft(x)

        lh = self.cbam_lh(lh)
        hl = self.cbam_hl(hl)
        hh = self.cbam_hh(hh)
        ll = self.cbam_ll(ll)

        high_in = torch.cat([lh, hl, hh], dim=1)
        high = self.msca_high(high_in) + high_in

        combined_in = torch.cat([ll, high], dim=1)
        combined = self.msca_combined(combined_in) + combined_in
        return self.finalconv(combined)


class SupportMSFM(nn.Module):
    """Apply MSFM on aggregated support features with residual (init ~= identity)."""

    def __init__(self, in_planes=256, msfm_type='FFT'):
        super().__init__()
        msfm_type = str(msfm_type).upper()
        if msfm_type in {'NONE', 'IDENTITY'}:
            self.msfm = None
        elif msfm_type == 'FFT':
            self.msfm = MSFMCore(in_planes=in_planes)
        else:
            raise ValueError(f"Unsupported msfm_type={msfm_type}. Use FFT or NONE.")

    def forward(self, supp_feat):
        if self.msfm is None:
            return supp_feat
        return supp_feat + self.msfm(supp_feat)
