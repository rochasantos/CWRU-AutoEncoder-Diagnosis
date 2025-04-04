import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image

def extract_patches_rgb(spectrogram_rgb, patch_width=224, patch_stride=112):
    """
    Divide um espectrograma RGB horizontalmente em janelas (C, H, W)
    """
    c, h, w = spectrogram_rgb.shape
    patches = []

    for start in range(0, w - patch_width + 1, patch_stride):
        patch = spectrogram_rgb[:, :, start:start + patch_width]  # (3, 801, patch_width)
        patches.append(patch)

    return patches


roots_dir = [
    # "data/spectrogram/cwru/014/N",
    # "data/spectrogram/cwru/014/I",
    # "data/spectrogram/cwru/014/O",
    # "data/spectrogram/cwru/014/B",
    # "data/spectrogram/cwru/021/N",
    # "data/spectrogram/cwru/021/I",
    # "data/spectrogram/cwru/021/O",
    "data/spectrogram/cwru/021/B",
]

def generate_chunk():
    for root_dir in roots_dir:
        severity_load = root_dir[-5:-2]
        output_dir = root_dir.replace(severity_load, severity_load[1:])
        for filename in os.listdir(root_dir):
            img = Image.open(root_dir+'/'+filename)
            spectrogram = np.array(img) # (H, W, 3)
            spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).permute(2, 0, 1) # (C, H, W)
            patches = extract_patches_rgb(spectrogram_tensor, patch_width=801, patch_stride=512)
            for i, patch in enumerate(patches):
                output_path = os.path.join(output_dir, f"{filename[:-4]}_{i}.png")
                patch_np = patch.permute(1, 2, 0).numpy().astype(np.uint8)  # (H, W, 3) para imshow
                plt.imshow(patch_np)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                # plt.show()