# %%

import torch

# %%


# %%


# %%


def B_map_gpu(z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    z:      (B, 1, H, W)
    params: (B, 3, 4)

    Returns:
        B: (B, 3, H, W)
    """

    Binf = params[:, :, 0, None, None]
    betaB = params[:, :, 1, None, None]
    Jp = params[:, :, 2, None, None]
    betaDp = params[:, :, 3, None, None]

    z = z.expand(-1, 3, -1, -1)

    exp_B = torch.exp(-betaB * z)
    exp_D = torch.exp(-betaDp * z)

    B = Binf * (1.0 - exp_B) + Jp * exp_D

    return B


# %%


def remove_backscatter_gpu(I, z, params):

    B = B_map_gpu(z, params)

    D = torch.clamp(I - B, min=1e-6, max=1.0)

    return B, D


# %%


def coarse_beta_gpu(E, z):

    E_safe = E.clamp(1e-5, 1.0)
    z_safe = z.clamp_min(1e-6)

    return -torch.log(E_safe) / z_safe


def reconstruct_scene_gpu(D, z, beta_map, max_exponent=10.0):

    exponent = (beta_map * z).clamp(min=0.0, max=max_exponent)

    return D * torch.exp(exponent)
