# %%

import torch

# %%


def beta_model_gpu(z, params):
    """
    Two-term exponential attenuation model.

    z:      [B, 1, 1, S]
    params: [B, C, R, 4] with order [a, b, c, d]

    Returns:
        beta: [B, C, R, S]
    """
    a = params[..., 0].unsqueeze(-1)
    b = params[..., 1].unsqueeze(-1)
    c = params[..., 2].unsqueeze(-1)
    d = params[..., 3].unsqueeze(-1)

    return a * torch.exp(b * z) + c * torch.exp(d * z)


# %%


def _logit(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


# %%


def coarse_beta_gpu(
    E: torch.Tensor,
    z: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the coarse attenuation estimate.

    Args:
        E: [B, 3, H, W], backscatter-removed RGB image.
        z: [B, 1, H, W] or [B, H, W], depth map.
        eps: Minimum value used to avoid log(0) and division by zero.

    Returns:
        beta0: [B, 3, H, W]
    """
    if E.ndim != 4 or E.shape[1] != 3:
        raise ValueError(f"E must have shape [B, 3, H, W], got {E.shape}")

    if z.ndim == 3:
        z = z.unsqueeze(1)

    if z.ndim != 4 or z.shape[1] != 1:
        raise ValueError(f"z must have shape [B, 1, H, W], got {z.shape}")

    if E.shape[0] != z.shape[0] or E.shape[2:] != z.shape[2:]:
        raise ValueError("E and z dimensions do not match")

    E_safe = E.clamp(min=eps, max=1.0)
    z_safe = z.clamp_min(eps)

    beta0 = -torch.log(E_safe) / z_safe

    return beta0


# %%


def refine_beta_batch(
    z,
    E,
    beta0=None,
    max_samples=30000,
    restarts=10,
    steps=2000,
    lr=0.03,
    f_scale=0.1,
    seed=0,
):
    """
    Batched PyTorch version of refine_beta.

    Args:
        z: [B, 1, H, W] or [B, H, W]
        E: [B, 3, H, W], backscatter-removed image
        beta0: [B, 3, H, W], initial coarse beta estimate
        max_samples: maximum sampled pixels per image
        restarts: number of parameter initializations per image/channel
        steps: Adam steps
        lr: Adam learning rate
        f_scale: soft_l1 scale, matching the SciPy implementation
        seed: random seed

    Returns:
        best_params: [B, 3, 4], parameters [a, b, c, d]
        best_loss: [B, 3], robust objective per image/channel
    """
    if z.ndim == 4:
        if z.shape[1] != 1:
            raise ValueError("4D z must have shape [B, 1, H, W]")
        z = z[:, 0]

    if z.ndim != 3:
        raise ValueError(f"z must have shape [B,H,W] or [B,1,H,W], got {z.shape}")

    if E.ndim != 4 or E.shape[1] != 3:
        raise ValueError(f"E must have shape [B,3,H,W], got {E.shape}")

    if z.shape[0] != E.shape[0] or z.shape[1:] != E.shape[2:]:
        raise ValueError("z and E dimensions do not match")

    if beta0 is None:
        beta0 = coarse_beta_gpu(E, z[:, None])

    if beta0.shape != E.shape:
        raise ValueError("beta0 must have the same shape as E")

    B, C, _, _ = E.shape
    device, dtype = E.device, E.dtype
    R = restarts

    # Flatten spatial dimensions.
    z_flat = z.reshape(B, -1)
    E_flat = E.reshape(B, C, -1)
    beta0_flat = beta0.reshape(B, C, -1)

    # Match the original validity requirements:
    # valid positive depth, E in (1e-4, 1), and beta0 in (0, 10).
    valid = (
        torch.isfinite(z_flat)
        & (z_flat > 0)
        & torch.isfinite(E_flat).all(dim=1)
        & (E_flat > 1e-4).all(dim=1)
        & (E_flat < 1.0).all(dim=1)
        & torch.isfinite(beta0_flat).all(dim=1)
        & (beta0_flat > 0).all(dim=1)
        & (beta0_flat < 10).all(dim=1)
    )

    # Gather a fixed number of samples per image.
    z_samples = torch.zeros((B, max_samples), device=device, dtype=dtype)
    E_samples = torch.ones((B, C, max_samples), device=device, dtype=dtype)
    sample_valid = torch.zeros((B, max_samples), device=device, dtype=torch.bool)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for i in range(B):
        idx = torch.where(valid[i])[0]

        if idx.numel() == 0:
            raise ValueError(f"Image {i} has no valid pixels for beta fitting")

        if idx.numel() > max_samples:
            perm = torch.randperm(idx.numel(), device=device, generator=generator)
            idx = idx[perm[:max_samples]]

        n = idx.numel()
        z_samples[i, :n] = z_flat[i, idx]
        E_samples[i, :, :n] = E_flat[i, :, idx]
        sample_valid[i, :n] = True

    # Depth range per image controls the exponent bounds.
    z_min = torch.where(valid, z_flat, torch.full_like(z_flat, float("inf"))).amin(
        dim=1
    )

    z_max = torch.where(valid, z_flat, torch.full_like(z_flat, float("-inf"))).amax(
        dim=1
    )

    z_range = (z_max - z_min).clamp_min(1e-6)

    # Original bounds:
    # a,c in [0,5], b,d in [-10/z_range, 0].
    lower = torch.zeros((B, 1, 1, 4), device=device, dtype=dtype)
    upper = torch.empty_like(lower)

    upper[..., 0] = 5.0
    upper[..., 1] = 0.0
    upper[..., 2] = 5.0
    upper[..., 3] = 0.0

    lower[..., 1] = -10.0 / z_range[:, None, None]
    lower[..., 3] = -10.0 / z_range[:, None, None]

    # Initialize a,c using the median coarse beta, split between terms.
    # The decay rates are initialized in the original negative range.
    beta0_valid = torch.where(
        valid[:, None, :],
        beta0_flat,
        torch.full_like(beta0_flat, float("nan")),
    )

    median_beta = torch.nanmedian(beta0_valid, dim=-1).values
    median_beta = torch.nan_to_num(
        median_beta, nan=0.1, posinf=0.1, neginf=0.1
    ).clamp_min(1e-4)

    initial = torch.empty((B, C, R, 4), device=device, dtype=dtype)

    # Reproducible random initialization.
    rand = torch.rand(
        (B, C, R, 4),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    # a and c initialized around 20%-80% of median beta.
    initial[..., 0] = (0.2 + 0.6 * rand[..., 0]) * median_beta[:, :, None]

    initial[..., 2] = (0.2 + 0.6 * rand[..., 2]) * median_beta[:, :, None]

    # b and d initialized between -2/z_range and -0.1/z_range.
    initial[..., 1] = -(0.1 + 1.9 * rand[..., 1]) / z_range[:, None, None]

    initial[..., 3] = -(0.1 + 1.9 * rand[..., 3]) / z_range[:, None, None]

    # Clamp initialization strictly inside bounds before inverse mapping.
    unit_initial = (initial - lower) / (upper - lower)
    raw = torch.nn.Parameter(_logit(unit_initial))

    optimizer = torch.optim.Adam([raw], lr=lr)

    z_fit = z_samples[:, None, None, :]  # [B,1,1,S]
    E_fit = E_samples[:, :, None, :]  # [B,C,1,S]
    mask = sample_valid[:, None, None, :].to(dtype)

    log_E = -torch.log(E_fit.clamp_min(1e-8))

    for _ in range(steps):
        optimizer.zero_grad()

        params = lower + (upper - lower) * torch.sigmoid(raw)

        beta = beta_model_gpu(z_fit, params).clamp_min(1e-6)

        # Original residual: z_hat - z, where z_hat = -log(E)/beta(z).
        z_hat = log_E / beta
        residual = z_hat - z_fit

        # SciPy soft_l1 rho(r^2):
        # 2*f^2 * (sqrt(1 + (r/f)^2) - 1)
        scaled = residual / f_scale
        robust = 2.0 * (f_scale**2) * (torch.sqrt(1.0 + scaled.square()) - 1.0)

        loss = (robust * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)

        loss.sum().backward()
        optimizer.step()

    # Select the best restart for every image and channel.
    with torch.no_grad():
        params = lower + (upper - lower) * torch.sigmoid(raw)

        beta = beta_model_gpu(z_fit, params).clamp_min(1e-6)
        z_hat = log_E / beta
        residual = z_hat - z_fit

        scaled = residual / f_scale
        robust = 2.0 * (f_scale**2) * (torch.sqrt(1.0 + scaled.square()) - 1.0)

        final_loss = (robust * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(
            1
        )  # [B,C,R]

        best_idx = final_loss.argmin(dim=-1)

        best_params = params.gather(
            dim=2,
            index=best_idx[..., None, None].expand(B, C, 1, 4),
        ).squeeze(2)

        best_loss = final_loss.gather(
            dim=2,
            index=best_idx[..., None],
        ).squeeze(2)

    return best_params, best_loss
