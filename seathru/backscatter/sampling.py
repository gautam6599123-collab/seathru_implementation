# %%

import numpy as np
import torch

from .fitting import fit_B_channel_batch

# %%


def dark_samples(
    I: np.ndarray,
    z: np.ndarray,
    bins: int = 10,
    percentile: float = 1.0,
    max_samples: int = 4096,
    min_pixels_per_bin: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dark pixels from depth bins for batched backscatter fitting.

    Args:
        I: RGB image, shape [H, W, 3], values preferably in [0, 1].
        z: Depth map, shape [H, W]. Invalid values may be NaN or zero.
        bins: Number of equally spaced depth bins.
        percentile: Keep pixels at or below this darkness percentile per bin.
        max_samples: Maximum number of selected samples. Output is padded
            to this size.
        min_pixels_per_bin: Skip bins with fewer valid pixels.
        seed: Seed used when a bin has more selected samples than its quota.

    Returns:
        z_samples: [max_samples], float32
        rgb_samples: [3, max_samples], float32, channel-first
        valid: [max_samples], bool. False entries are padding.

    Notes:
        The original depth-bin and darkest-percentile selection is preserved.
        To keep depth bins represented when limiting the sample count, each
        bin receives an approximately equal sample quota.
    """
    if I.ndim != 3 or I.shape[2] != 3:
        raise ValueError(f"I must have shape [H, W, 3], got {I.shape}")
    if z.shape != I.shape[:2]:
        raise ValueError(f"z must have shape {I.shape[:2]}, got {z.shape}")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")

    I = np.asarray(I, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)

    # Filter invalid pixels and require strictly positive depth.
    ok = np.isfinite(z) & np.all(np.isfinite(I), axis=2) & (z > 0)

    zz = z[ok]
    rgb = I[ok]

    # Fixed-size, padded outputs.
    z_samples = np.zeros(max_samples, dtype=np.float32)
    rgb_samples = np.zeros((3, max_samples), dtype=np.float32)
    valid = np.zeros(max_samples, dtype=bool)

    # No valid depth values.
    if zz.size == 0:
        return z_samples, rgb_samples, valid

    z_min = zz.min()
    z_max = zz.max()

    # A constant depth map belongs to one effective bin.
    if z_min == z_max:
        edges = np.linspace(z_min, z_max + 1e-6, bins + 1)
    else:
        edges = np.linspace(z_min, z_max, bins + 1)

    darkness = np.linalg.norm(rgb, axis=1)
    rng = np.random.default_rng(seed)

    # Distribute the maximum sample budget across depth bins.
    base_quota, remainder = divmod(max_samples, bins)

    selected_z = []
    selected_rgb = []

    for k in range(bins):
        if k == bins - 1:
            m = (zz >= edges[k]) & (zz <= edges[k + 1])
        else:
            m = (zz >= edges[k]) & (zz < edges[k + 1])

        idx = np.flatnonzero(m)

        if idx.size < min_pixels_per_bin:
            continue

        cut = np.percentile(darkness[idx], percentile)
        dark_idx = idx[darkness[idx] <= cut]

        if dark_idx.size == 0:
            continue

        quota = base_quota + (1 if k < remainder else 0)

        # If a bin contributes too many samples, randomly subsample it.
        if dark_idx.size > quota:
            dark_idx = rng.choice(dark_idx, size=quota, replace=False)

        selected_z.append(zz[dark_idx])
        selected_rgb.append(rgb[dark_idx])

    if not selected_z:
        return z_samples, rgb_samples, valid

    zs = np.concatenate(selected_z)
    rgbs = np.concatenate(selected_rgb, axis=0)

    n = min(zs.size, max_samples)

    z_samples[:n] = zs[:n]
    rgb_samples[:, :n] = rgbs[:n].T
    valid[:n] = True

    return z_samples, rgb_samples, valid


# %%


def estimate_backscatter_for_batch(image, depth, max_samples=4096):
    B, _, _, _ = image.shape

    z_samples_list = []
    rgb_samples_list = []
    valid_samples_list = []

    for i in range(B):
        I_np = image[i].detach().permute(1, 2, 0).cpu().numpy()

        # depth[i]: [1, H, W] -> [H, W]
        z_np = depth[i].detach().squeeze(0).cpu().numpy()

        z_s, rgb_s, valid_s = dark_samples(
            I_np,
            z_np,
            bins=10,
            percentile=1.0,
            max_samples=max_samples,
        )

        z_samples_list.append(z_s)
        rgb_samples_list.append(rgb_s)
        valid_samples_list.append(valid_s)

    z_samples = torch.from_numpy(np.stack(z_samples_list)).to(
        device=image.device, dtype=image.dtype
    )
    rgb_samples = torch.from_numpy(np.stack(rgb_samples_list)).to(
        device=image.device, dtype=image.dtype
    )
    valid_samples = torch.from_numpy(np.stack(valid_samples_list)).to(
        device=image.device
    )

    params, losses = fit_B_channel_batch(
        z=z_samples,
        y=rgb_samples,
        valid=valid_samples,
        restarts=10,
        steps=2000,
        lr=0.03,
    )

    return params, losses
