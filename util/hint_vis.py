import os

import cv2
import numpy as np
import torch


def prepare_hint_dirs(save_root):
    hint_root = os.path.join(save_root, "hints")
    overlay_dir = os.path.join(hint_root, "overlay")
    mask_dir = os.path.join(hint_root, "mask")
    color_dir = os.path.join(hint_root, "color")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    return overlay_dir, mask_dir, color_dir


def save_hint_visualizations(hint_dirs, names, opt_img, hint_color, hint_mask):
    overlay_dir, mask_dir, color_dir = hint_dirs

    opt_img_cpu = opt_img.detach().cpu().to(torch.uint8)
    hint_color_cpu = hint_color.detach().cpu().to(torch.uint8)
    hint_mask_cpu = hint_mask.detach().cpu()

    for idx, name in enumerate(names):
        opt_np = opt_img_cpu[idx].permute(1, 2, 0).numpy()
        hint_color_np = hint_color_cpu[idx].permute(1, 2, 0).numpy()
        mask_np = hint_mask_cpu[idx, 0].numpy()
        mask_bool = mask_np > 0.5

        overlay_np = opt_np.copy()
        overlay_np[mask_bool] = hint_color_np[mask_bool]

        mask_vis = (mask_bool.astype(np.uint8) * 255)
        mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)

        cv2.imwrite(os.path.join(overlay_dir, name), overlay_np[:, :, ::-1])
        cv2.imwrite(os.path.join(mask_dir, name), mask_vis)
        cv2.imwrite(os.path.join(color_dir, name), hint_color_np[:, :, ::-1])