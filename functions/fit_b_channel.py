# %%

import numpy as np
import torch

# %%


def backscatter_model(z, params):
    """
    z:      [B, 1, 1, S]
    params: [B, C, R, 4]

    Returns:
        predicted backscatter: [B, C, R, S]
    """
    B_inf = params[..., 0].unsqueeze(-1)
    beta_B = params[..., 1].unsqueeze(-1)
    J_p = params[..., 2].unsqueeze(-1)
    beta_Dp = params[..., 3].unsqueeze(-1)

    return B_inf * (1.0 - torch.exp(-beta_B * z)) + J_p * torch.exp(-beta_Dp * z)


def _inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def fit_B_channel_batch(
    z,
    y,
    valid=None,
    restarts=10,
    steps=2000,
    lr=0.03,
    seed=0,
):
    """
    Batched equivalent of fit_B_channel.

    Args:
        z: [B, S] sampled depths
        y: [B, C, S] sampled RGB values
        valid: optional [B, S] boolean mask
        restarts: number of random initializations per image/channel
        steps: Adam optimization steps
        lr: Adam learning rate
        seed: random seed

    Returns:
        best_params: [B, C, 4]
        best_loss:   [B, C]
    """
    if z.ndim != 2:
        raise ValueError(f"z must have shape [B, S], got {z.shape}")
    if y.ndim != 3:
        raise ValueError(f"y must have shape [B, C, S], got {y.shape}")
    if z.shape[0] != y.shape[0] or z.shape[1] != y.shape[2]:
        raise ValueError("z and y dimensions are inconsistent")

    device = z.device
    dtype = y.dtype
    B, C, S = y.shape
    R = restarts

    z = z.to(device=device, dtype=dtype)

    if valid is None:
        valid = torch.ones((B, S), dtype=torch.bool, device=device)
    else:
        valid = valid.to(device=device, dtype=torch.bool)

    # Remove invalid or non-finite samples before evaluating the model.
    valid = valid & torch.isfinite(z) & torch.isfinite(y).all(dim=1)

    # Replace invalid entries with finite values. The mask excludes them
    # from the objective, but this also prevents NaNs in intermediate ops.
    z = torch.where(valid, z, torch.zeros_like(z))
    y = torch.where(valid[:, None, :], y, torch.zeros_like(y))

    counts = valid.sum(dim=1)  # [B]
    if torch.any(counts == 0):
        raise ValueError("At least one image has no valid samples")

    # Bounds in the order:
    # B_inf, beta_B, J_p, beta_Dp
    lower = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    upper = torch.tensor([1.0, 5.0, 1.0, 5.0], device=device, dtype=dtype)

    # Random initialization within the parameter bounds.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    initial = torch.rand(
        (B, C, R, 4),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    initial = lower + initial * (upper - lower)

    # Optimize unconstrained variables, mapping them into the bounds
    # with a sigmoid.
    unit_initial = (initial - lower) / (upper - lower)
    raw = torch.nn.Parameter(_inverse_sigmoid(unit_initial))

    optimizer = torch.optim.Adam([raw], lr=lr)

    # Broadcast dimensions:
    # z -> [B, 1, 1, S]
    # y -> [B, C, 1, S]
    # mask -> [B, 1, 1, S]
    z_fit = z[:, None, None, :]
    y_fit = y[:, :, None, :]
    mask = valid[:, None, None, :].to(dtype)

    for _ in range(steps):
        optimizer.zero_grad()

        params = lower + (upper - lower) * torch.sigmoid(raw)
        prediction = backscatter_model(z_fit, params)

        squared_error = (prediction - y_fit).square()
        loss = (squared_error * mask).sum(dim=-1) / mask.sum(dim=-1)

        # Each [image, channel, restart] fit has its own loss.
        # Summing lets autograd update all fits in one backward pass.
        loss.sum().backward()
        optimizer.step()

    # Evaluate the final parameters and select the best restart.
    with torch.no_grad():
        params = lower + (upper - lower) * torch.sigmoid(raw)
        prediction = backscatter_model(z_fit, params)

        squared_error = (prediction - y_fit).square()
        final_loss = (squared_error * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B, C, R]

        best_restart = final_loss.argmin(dim=-1)  # [B, C]

        gather_idx = best_restart[..., None, None].expand(B, C, 1, 4)
        best_params = params.gather(dim=2, index=gather_idx).squeeze(2)

        best_loss = final_loss.gather(dim=2, index=best_restart[..., None]).squeeze(2)

    return best_params, best_loss


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
    B, C, H, W = image.shape

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
