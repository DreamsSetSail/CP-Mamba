import hashlib
import os

import cv2
import numpy as np

from util.vis_fss import (
    DEFAULT_GT_COLOR,
    DEFAULT_MASK_COLOR,
    DEFAULT_PRED_COLOR,
    add_title,
    overlay_mask,
    resize_keep_aspect,
)


def make_episode_key(meta):
    """Unique fingerprint for one support+query episode (paths + class)."""
    class_chosen = int(meta['class_chosen'])
    query_image = meta['query_image_path']
    query_label = meta['query_label_path']
    support_images = list(meta['support_image_paths'])
    support_labels = list(meta['support_label_paths'])
    parts = [str(class_chosen), query_image, query_label] + support_images + support_labels
    return hashlib.md5('||'.join(parts).encode('utf-8')).hexdigest()[:16]


def episode_spec_from_meta(meta):
    return {
        'class_chosen': int(meta['class_chosen']),
        'query_image_path': meta['query_image_path'],
        'query_label_path': meta['query_label_path'],
        'support_image_paths': list(meta['support_image_paths']),
        'support_label_paths': list(meta['support_label_paths']),
    }


def assert_same_episode_spec(spec_a, spec_b):
    for key in spec_a:
        if spec_a[key] != spec_b[key]:
            raise ValueError('Episode mismatch on {}: {!r} vs {!r}'.format(key, spec_a[key], spec_b[key]))


def align_row_width(row, target_width):
    h, w = row.shape[:2]
    if w == target_width:
        return row
    if w > target_width:
        return cv2.resize(row, (target_width, h), interpolation=cv2.INTER_AREA)
    pad = np.ones((h, target_width - w, 3), dtype=np.uint8) * 255
    return np.hstack([row, pad])


def build_method_row(
    support_images,
    support_masks,
    query_image,
    gt_mask,
    pred_mask,
    row_title,
    pred_iou,
    panel_h=200,
    show_support=True,
):
    panels = []
    if show_support:
        for idx, (img, msk) in enumerate(zip(support_images, support_masks), start=1):
            vis = overlay_mask(img, msk, color=DEFAULT_MASK_COLOR)
            panels.append(add_title(resize_keep_aspect(vis, panel_h), 'S{}'.format(idx)))

    query_vis = resize_keep_aspect(
        query_image.astype(np.uint8) if query_image.dtype != np.uint8 else query_image,
        panel_h,
    )
    if query_vis.ndim == 2:
        query_vis = cv2.cvtColor(query_vis, cv2.COLOR_GRAY2BGR)
    elif query_vis.shape[2] == 3:
        query_vis = cv2.cvtColor(query_vis, cv2.COLOR_RGB2BGR)
    panels.append(add_title(query_vis, 'Query'))

    gt_vis = overlay_mask(query_image, gt_mask, color=DEFAULT_GT_COLOR)
    panels.append(add_title(resize_keep_aspect(gt_vis, panel_h), 'GT'))

    pred_vis = overlay_mask(query_image, pred_mask, color=DEFAULT_PRED_COLOR)
    panels.append(add_title(resize_keep_aspect(pred_vis, panel_h), 'Pred IoU={:.3f}'.format(pred_iou)))

    bar_h = panels[0].shape[0] - panel_h
    max_h = max(p.shape[0] for p in panels)
    aligned = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.ones((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8) * 255
            panel = np.vstack([pad, panel])
        aligned.append(panel)

    row = np.hstack(aligned)
    return add_title(row, row_title, bar_height=32)


def build_dual_compare_figure(
    episode_data,
    proto_iou,
    gated_iou,
    panel_h=200,
    proto_name='Proto',
    gated_name='Gated',
):
    delta = proto_iou - gated_iou
    class_name = episode_data['class_name']
    query_name = os.path.basename(episode_data.get('query_image_path', ''))
    episode_key = episode_data.get('episode_key', '')
    header = '{} | ep{} | query={} | delta={:+.3f} (Proto {:.3f} vs Gated {:.3f}) | same support+query'.format(
        class_name, episode_data.get('episode_idx', ''), query_name, delta, proto_iou, gated_iou,
    )
    if episode_key:
        header = '{} | key={}'.format(header, episode_key)

    row_proto = build_method_row(
        support_images=episode_data['support_images'],
        support_masks=episode_data['support_masks'],
        query_image=episode_data['query_image'],
        gt_mask=episode_data['gt_mask'],
        pred_mask=episode_data['proto_pred_mask'],
        row_title='{} | IoU={:.3f}'.format(proto_name, proto_iou),
        pred_iou=proto_iou,
        panel_h=panel_h,
    )
    row_gated = build_method_row(
        support_images=episode_data['support_images'],
        support_masks=episode_data['support_masks'],
        query_image=episode_data['query_image'],
        gt_mask=episode_data['gt_mask'],
        pred_mask=episode_data['gated_pred_mask'],
        row_title='{} | IoU={:.3f}'.format(gated_name, gated_iou),
        pred_iou=gated_iou,
        panel_h=panel_h,
    )

    target_w = max(row_proto.shape[1], row_gated.shape[1])
    row_proto = align_row_width(row_proto, target_w)
    row_gated = align_row_width(row_gated, target_w)
    figure = np.vstack([row_proto, row_gated])
    return add_title(figure, header, bar_height=36)


def _query_to_bgr(query_image):
    query_vis = query_image.astype(np.uint8) if query_image.dtype != np.uint8 else query_image.copy()
    if query_vis.ndim == 2:
        return cv2.cvtColor(query_vis, cv2.COLOR_GRAY2BGR)
    if query_vis.shape[2] == 3:
        return cv2.cvtColor(query_vis, cv2.COLOR_RGB2BGR)
    return query_vis


def export_compare_episode_folder(
    episode_data,
    rank,
    output_dir,
    proto_iou,
    gated_iou,
    panel_h=200,
    proto_name='Proto',
    gated_name='Gated',
):
    """Save rank folder with supports, query, gt, pred1, pred2, and combined figure (10 images for 5-shot)."""
    rank_dir = os.path.join(output_dir, 'rank{:02d}'.format(rank))
    os.makedirs(rank_dir, exist_ok=True)

    query_image = episode_data['query_image']
    gt_mask = episode_data['gt_mask']
    saved_files = []

    for idx, (img, msk) in enumerate(
        zip(episode_data['support_images'], episode_data['support_masks']),
        start=1,
    ):
        support_vis = overlay_mask(img, msk, color=DEFAULT_MASK_COLOR)
        support_path = os.path.join(rank_dir, 'support{:02d}.png'.format(idx))
        cv2.imwrite(support_path, support_vis)
        saved_files.append(support_path)

    query_path = os.path.join(rank_dir, 'query.png')
    cv2.imwrite(query_path, _query_to_bgr(query_image))
    saved_files.append(query_path)

    gt_path = os.path.join(rank_dir, 'gt.png')
    cv2.imwrite(gt_path, overlay_mask(query_image, gt_mask, color=DEFAULT_GT_COLOR))
    saved_files.append(gt_path)

    pred1_path = os.path.join(rank_dir, 'pred1.png')
    pred1_vis = overlay_mask(query_image, episode_data['proto_pred_mask'], color=DEFAULT_PRED_COLOR)
    pred1_vis = add_title(pred1_vis, '{} IoU={:.3f}'.format(proto_name, proto_iou))
    cv2.imwrite(pred1_path, pred1_vis)
    saved_files.append(pred1_path)

    pred2_path = os.path.join(rank_dir, 'pred2.png')
    pred2_vis = overlay_mask(query_image, episode_data['gated_pred_mask'], color=DEFAULT_PRED_COLOR)
    pred2_vis = add_title(pred2_vis, '{} IoU={:.3f}'.format(gated_name, gated_iou))
    cv2.imwrite(pred2_path, pred2_vis)
    saved_files.append(pred2_path)

    figure = build_dual_compare_figure(
        episode_data=episode_data,
        proto_iou=proto_iou,
        gated_iou=gated_iou,
        panel_h=panel_h,
        proto_name=proto_name,
        gated_name=gated_name,
    )
    combined_name = 'rank{:02d}_ep{:04d}_{}_delta{:+.3f}.png'.format(
        rank,
        episode_data['episode_idx'],
        episode_data['class_name'],
        proto_iou - gated_iou,
    )
    combined_path = os.path.join(rank_dir, combined_name)
    cv2.imwrite(combined_path, figure)
    saved_files.append(combined_path)

    return rank_dir, saved_files


def select_max_delta_episodes(episode_records, top_k=5, select_mode='max_delta', min_delta=0.0):
    filtered = [r for r in episode_records if r['delta_iou'] >= min_delta]
    if not filtered:
        return []

    if select_mode == 'per_class':
        best_by_class = {}
        for rec in filtered:
            cls = rec['class_chosen']
            if cls not in best_by_class or rec['delta_iou'] > best_by_class[cls]['delta_iou']:
                best_by_class[cls] = rec
        selected = sorted(
            best_by_class.values(),
            key=lambda x: (-x['delta_iou'], -x['proto_iou'], x['episode_idx']),
        )
        return _dedup_by_episode_key(selected[:top_k])

    ranked = sorted(
        filtered,
        key=lambda x: (-x['delta_iou'], -x['proto_iou'], x['episode_idx']),
    )
    return _dedup_by_episode_key(ranked[:top_k])


def _dedup_by_episode_key(records):
    seen = set()
    unique = []
    for rec in records:
        key = rec.get('episode_key')
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(rec)
    return unique
