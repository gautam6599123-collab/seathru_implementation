# %%

import matplotlib.pyplot as plt
import numpy as np

from seathru.reference.seathru import linear_to_srgb

# %%


def show_backscatter_results(image, backscatter, corrected, index=0):
    """
    Inputs are GPU or CPU tensors with shape [B, 3, H, W].
    """
    original_np = image[index].detach().cpu().permute(1, 2, 0).numpy()
    backscatter_np = backscatter[index].detach().cpu().permute(1, 2, 0).numpy()
    corrected_np = corrected[index].detach().cpu().permute(1, 2, 0).numpy()

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    images = [
        (original_np, "Original"),
        (backscatter_np, "Estimated backscatter"),
        (corrected_np, "Backscatter removed"),
    ]

    for ax, (array, title) in zip(axes, images):
        ax.imshow(linear_to_srgb(np.clip(array, 0, 1)))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# %%


def show_reconstruction_results(image, reconstructed, index=0):
    """
    Inputs are GPU or CPU tensors with shape [B, 3, H, W].
    """
    original_np = image[index].detach().cpu().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed[index].detach().cpu().permute(1, 2, 0).numpy()

    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    images = [
        (original_np, "Original"),
        (reconstructed_np, "Reconstructed"),
    ]

    for ax, (array, title) in zip(axes, images):
        ax.imshow(linear_to_srgb(np.clip(array, 0, 1)))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
