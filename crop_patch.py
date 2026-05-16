import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


def split_into_patches(img: Image.Image, patch_size: int = 8):
    """Split image into non-overlapping patches of size (patch_size, patch_size).
    Crops the image to a multiple of patch_size."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    arr = arr[:h2, :w2]

    patches = []
    coords = []  # (row, col) in patch grid
    for y in range(0, h2, patch_size):
        for x in range(0, w2, patch_size):
            p = arr[y:y + patch_size, x:x + patch_size]
            patches.append(p)
            coords.append((y // patch_size, x // patch_size))
    return patches, coords, (h2, w2)


def save_random_patches(
    image_path: str,
    out_dir: str = "random_patches",
    patch_size: int = 8,
    num_patches: int = 16,
    seed: int = 0,
):
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    patches, coords, cropped_hw = split_into_patches(img, patch_size=patch_size)

    if len(patches) == 0:
        raise ValueError("No patches generated. Is the image smaller than patch_size?")

    random.seed(seed)
    k = min(num_patches, len(patches))
    chosen_idx = random.sample(range(len(patches)), k=k)

    for i, idx in enumerate(chosen_idx):
        patch_arr = patches[idx]
        r, c = coords[idx]
        patch_img = Image.fromarray(patch_arr)
        patch_img.save(out_dir / f"{image_path.stem}_ps{patch_size}_r{r}_c{c}_{i:03d}.png")

    print(f"Saved {k} random patches to: {out_dir.resolve()}")
    print(f"Original size: {img.size}, Cropped to: {cropped_hw[::-1]} (W,H), Total patches: {len(patches)}")


if __name__ == "__main__":
    # 用法示例：把这里改成你的图片路径
    save_random_patches(
        image_path="/data/yjy_data/SAM2/hint_outputs_test/color_hint_by_dots/12_2160_2160.jpg",
        out_dir="/NAS_data/yjy/linshi_image/color_hint_by_dots",
        patch_size=64,
        num_patches=100,  # 随机导出 20 个 patch
        seed=42,
    )
