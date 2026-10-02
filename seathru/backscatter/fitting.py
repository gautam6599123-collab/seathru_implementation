# %%

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


# %%


def _inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


# %%


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
