import json
import os

import cv2
import numpy as np

PASCAL_CLASSES = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor',
}

DEFAULT_MASK_COLOR = (30, 144, 255)  # BGR, dodger blue
DEFAULT_GT_COLOR = (0, 200, 0)
DEFAULT_PRED_COLOR = (0, 0, 255)


def compute_episode_iou(pred, gt, ignore_label=255):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    valid = gt != ignore_label
    pred_fg = (pred == 1) & valid
    gt_fg = (gt == 1) & valid
    inter = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def mask_to_binary(mask, class_id=None):
    if class_id is None:
        return (mask == 1).astype(np.uint8)
    return (mask == class_id).astype(np.uint8)


def overlay_mask(image, mask, color=DEFAULT_MASK_COLOR, alpha=0.45):
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = image.copy()
    fg = mask.astype(bool)
    for c in range(3):
        overlay[:, :, c] = np.where(
            fg,
            alpha * color[c] + (1.0 - alpha) * image[:, :, c],
            image[:, :, c],
        )
    return overlay.astype(np.uint8)


def add_title(image, title, bar_height=28):
    h, w = image.shape[:2]
    canvas = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 255
    canvas[bar_height:, :] = image
    cv2.putText(
        canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return canvas


def resize_keep_aspect(image, target_h):
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def build_episode_figure(
    support_images,
    support_masks,
    query_image,
    gt_mask,
    pred_mask,
    class_name,
    iou,
    panel_h=220,
):
    panels = []
    for idx, (img, msk) in enumerate(zip(support_images, support_masks), start=1):
        vis = overlay_mask(img, msk, color=DEFAULT_MASK_COLOR)
        panels.append(add_title(resize_keep_aspect(vis, panel_h), f'S{idx}'))

    query_vis = resize_keep_aspect(
        query_image.astype(np.uint8) if query_image.dtype != np.uint8 else query_image, panel_h
    )
    if query_vis.ndim == 2:
        query_vis = cv2.cvtColor(query_vis, cv2.COLOR_GRAY2BGR)
    elif query_vis.shape[2] == 3:
        query_vis = cv2.cvtColor(query_vis, cv2.COLOR_RGB2BGR)
    panels.append(add_title(query_vis, 'Query'))

    gt_vis = overlay_mask(query_image, gt_mask, color=DEFAULT_GT_COLOR)
    panels.append(add_title(resize_keep_aspect(gt_vis, panel_h), 'GT'))

    pred_vis = overlay_mask(query_image, pred_mask, color=DEFAULT_PRED_COLOR)
    panels.append(add_title(resize_keep_aspect(pred_vis, panel_h), f'Pred IoU={iou:.3f}'))

    bar_h = panels[0].shape[0] - panel_h
    max_h = max(p.shape[0] for p in panels)
    aligned = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.ones((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8) * 255
            panel = np.vstack([pad, panel])
        aligned.append(panel)

    figure = np.hstack(aligned)
    figure = add_title(figure, class_name, bar_height=32)
    return figure


def select_best_episodes(episode_records, top_k=5, select_mode='topk'):
    if not episode_records:
        return []

    if select_mode == 'per_class':
        best_by_class = {}
        for rec in episode_records:
            cls = rec['class_chosen']
            if cls not in best_by_class or rec['iou'] > best_by_class[cls]['iou']:
                best_by_class[cls] = rec
        selected = sorted(best_by_class.values(), key=lambda x: (-x['iou'], x['episode_idx']))
        return selected[:top_k]

    return sorted(episode_records, key=lambda x: (-x['iou'], x['episode_idx']))[:top_k]


def save_episode_manifest(selected, output_dir, seed, arch, split, shot):
    manifest = {
        'seed': int(seed),
        'arch': arch,
        'split': int(split),
        'shot': int(shot),
        'episodes': [],
    }
    for rank, rec in enumerate(selected, start=1):
        manifest['episodes'].append({
            'rank': rank,
            'episode_idx': int(rec['episode_idx']),
            'iou': float(rec['iou']),
            'class_chosen': int(rec['class_chosen']),
            'class_name': rec['class_name'],
            'query_image_path': rec['query_image_path'],
            'query_label_path': rec['query_label_path'],
            'support_image_paths': rec['support_image_paths'],
            'support_label_paths': rec['support_label_paths'],
        })

    manifest_path = os.path.join(output_dir, 'selected_episodes.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


def load_episode_manifest(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_sub_val_list(split, data_set='pascal', use_split_coco=False):
    if data_set == 'pascal':
        if split == 3:
            return list(range(16, 21))
        if split == 2:
            return list(range(11, 16))
        if split == 1:
            return list(range(6, 11))
        return list(range(1, 6))
    raise NotImplementedError('Only PASCAL manifest replay is implemented.')


def load_episode_from_paths(episode_spec, ann_type='mask'):
    from util.get_weak_anns import transform_anns

    class_chosen = int(episode_spec['class_chosen'])
    query_image = cv2.imread(episode_spec['query_image_path'], cv2.IMREAD_COLOR)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    query_label = cv2.imread(episode_spec['query_label_path'], cv2.IMREAD_GRAYSCALE)

    gt_mask = np.zeros_like(query_label, dtype=np.uint8)
    gt_mask[query_label == class_chosen] = 1
    gt_mask[query_label == 255] = 255

    support_images = []
    support_masks = []
    for img_path, lbl_path in zip(
        episode_spec['support_image_paths'],
        episode_spec['support_label_paths'],
    ):
        support_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        support_label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        support_mask = np.zeros_like(support_label, dtype=np.uint8)
        support_mask[support_label == class_chosen] = 1
        ignore_pix = np.where(support_label == 255)
        support_mask, _ = transform_anns(support_mask, ann_type)
        support_mask[ignore_pix] = 255
        support_images.append(support_image)
        support_masks.append((support_mask == 1).astype(np.uint8))

    return {
        'class_chosen': class_chosen,
        'class_name': episode_spec.get('class_name', PASCAL_CLASSES.get(class_chosen, 'class_{}'.format(class_chosen))),
        'query_image_path': episode_spec['query_image_path'],
        'query_label_path': episode_spec['query_label_path'],
        'support_image_paths': list(episode_spec['support_image_paths']),
        'support_label_paths': list(episode_spec['support_label_paths']),
        'query_image': query_image,
        'gt_mask': (gt_mask == 1).astype(np.uint8),
        'support_images': support_images,
        'support_masks': support_masks,
    }
