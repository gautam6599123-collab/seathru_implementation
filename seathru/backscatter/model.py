# %%

import torch

# %%


def B_map_gpu(z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    z:      [B, 1, H, W]
    params: [B, 3, 4]
    returns [B, 3, H, W]
    """
    if z.ndim != 4 or z.shape[1] != 1:
        raise ValueError(f"Expected z [B, 1, H, W], got {z.shape}")

    if params.ndim != 3 or params.shape[1:] != (3, 4):
        raise ValueError(f"Expected params [B, 3, 4], got {params.shape}")

    if z.shape[0] != params.shape[0]:
        raise ValueError("Batch dimensions of z and params do not match")

    Binf = params[:, :, 0, None, None]
    betaB = params[:, :, 1, None, None]
    Jp = params[:, :, 2, None, None]
    betaDp = params[:, :, 3, None, None]

    z_rgb = z.expand(-1, 3, -1, -1)

    return Binf * (1.0 - torch.exp(-betaB * z_rgb)) + Jp * torch.exp(-betaDp * z_rgb)


# %%


def remove_backscatter_gpu(I, z, params):
    """
    I:      [B, 3, H, W]
    z:      [B, 1, H, W]
    params: [B, 3, 4]
    """
    if I.ndim != 4 or I.shape[1] != 3:
        raise ValueError(f"Expected I [B, 3, H, W], got {I.shape}")

    if I.shape[0] != z.shape[0] or I.shape[-2:] != z.shape[-2:]:
        raise ValueError("Image and depth dimensions do not match")

    B = B_map_gpu(z, params)
    D = torch.clamp(I - B, min=1e-6, max=1.0)

    return B, D


# %%


def coarse_beta_gpu(E, z):

    E_safe = E.clamp(1e-5, 1.0)
    z_safe = z.clamp_min(1e-6)

    return -torch.log(E_safe) / z_safe


# %%


def reconstruct_scene_gpu(D, z, beta_map, max_exponent=10.0):

    exponent = (beta_map * z).clamp(min=0.0, max=max_exponent)

    return D * torch.exp(exponent)
