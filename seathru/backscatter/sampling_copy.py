# %%

import torch

from .fitting import fit_B_channel_batch

# %%


def dark_samples_gpu(
    image: torch.Tensor,
    depth: torch.Tensor,
    bins: int = 10,
    percentile: float = 1.0,
    max_samples: int = 4096,
    min_pixels_per_bin: int = 10,
    seed: int = 0,
):
    """
    GPU equivalent of dark_samples.

    Args:
        image: [B, 3, H, W], floating point RGB
        depth: [B, 1, H, W], floating point depth
        bins: Number of equally spaced depth bins
        percentile: Keep pixels at or below this darkness percentile
        max_samples: Fixed output sample count per image
        min_pixels_per_bin: Skip bins with fewer valid pixels
        seed: Random seed for subsampling

    Returns:
        z_samples:   [B, max_samples]
        rgb_samples: [B, 3, max_samples]
        valid:       [B, max_samples]
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B, 3, H, W], got {image.shape}")

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B, 1, H, W], got {depth.shape}")

    if image.shape[0] != depth.shape[0] or image.shape[2:] != depth.shape[2:]:
        raise ValueError("image and depth dimensions do not match")

    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    if max_samples < bins:
        raise ValueError("max_samples should be >= bins")

    B, _, H, W = image.shape
    device = image.device
    dtype = image.dtype

    # Flatten spatial dimensions.
    rgb = image.reshape(B, 3, H * W)
    z = depth.reshape(B, H * W)

    # Valid pixels: finite RGB, finite positive depth.
    valid_pixels = torch.isfinite(z) & torch.isfinite(rgb).all(dim=1) & (z > 0)

    # Fixed-size outputs, allocated on GPU.
    z_samples = torch.zeros((B, max_samples), device=device, dtype=dtype)
    rgb_samples = torch.zeros((B, 3, max_samples), device=device, dtype=dtype)
    sample_valid = torch.zeros((B, max_samples), device=device, dtype=torch.bool)

    # Reproducible random generator on the same device.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    base_quota, remainder = divmod(max_samples, bins)

    for b in range(B):
        mask = valid_pixels[b]

        if not mask.any():
            continue

        zb = z[b, mask]
        rgbb = rgb[b, :, mask]  # [3, N]

        z_min = zb.min()
        z_max = zb.max()

        # Handle constant-depth images.
        if z_min == z_max:
            edges = torch.linspace(
                z_min,
                z_max + 1e-6,
                bins + 1,
                device=device,
                dtype=dtype,
            )
        else:
            edges = torch.linspace(
                z_min,
                z_max,
                bins + 1,
                device=device,
                dtype=dtype,
            )

        darkness = torch.linalg.vector_norm(rgbb, dim=0)

        selected_z = []
        selected_rgb = []

        for k in range(bins):
            if k == bins - 1:
                in_bin = (zb >= edges[k]) & (zb <= edges[k + 1])
            else:
                in_bin = (zb >= edges[k]) & (zb < edges[k + 1])

            idx = torch.where(in_bin)[0]

            if idx.numel() < min_pixels_per_bin:
                continue

            bin_darkness = darkness[idx]

            # Match np.percentile(..., percentile).
            threshold = torch.quantile(
                bin_darkness,
                percentile / 100.0,
            )

            dark_idx = idx[bin_darkness <= threshold]

            if dark_idx.numel() == 0:
                continue

            quota = base_quota + (1 if k < remainder else 0)

            if dark_idx.numel() > quota:
                perm = torch.randperm(
                    dark_idx.numel(),
                    device=device,
                    generator=generator,
                )
                dark_idx = dark_idx[perm[:quota]]

            selected_z.append(zb[dark_idx])
            selected_rgb.append(rgbb[:, dark_idx])

        if not selected_z:
            continue

        zs = torch.cat(selected_z)
        rgbs = torch.cat(selected_rgb, dim=1)

        n = min(zs.numel(), max_samples)

        z_samples[b, :n] = zs[:n]
        rgb_samples[b, :, :n] = rgbs[:, :n]
        sample_valid[b, :n] = True

    return z_samples, rgb_samples, sample_valid


# %%


def estimate_backscatter_for_batch(image, depth, max_samples=4096):
    z_samples, rgb_samples, valid_samples = dark_samples_gpu(
        image=image,
        depth=depth,
        bins=10,
        percentile=1.0,
        max_samples=max_samples,
        min_pixels_per_bin=10,
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
