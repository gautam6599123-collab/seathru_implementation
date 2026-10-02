# %%

import torch

# %%


def reconstruct_scene_gpu(D, z, beta_map, max_exponent=10.0):
    """
    Reconstruct scene radiance from the backscatter-corrected image.

    Args:
        D:         [B, 3, H, W] corrected image
        z:         [B, 1, H, W] depth map
        beta_map:  [B, 3, H, W] estimated attenuation coefficients
        max_exponent: upper limit for beta*z to avoid exponential overflow

    Returns:
        J: [B, 3, H, W] reconstructed scene image
    """
    if D.ndim != 4 or D.shape[1] != 3:
        raise ValueError(f"D must have shape [B, 3, H, W], got {D.shape}")

    if z.ndim == 3:
        z = z.unsqueeze(1)

    if z.ndim != 4 or z.shape[1] != 1:
        raise ValueError(f"z must have shape [B, 1, H, W], got {z.shape}")

    if beta_map.ndim != 4 or beta_map.shape[1] != 3:
        raise ValueError(f"beta_map must have shape [B, 3, H, W], got {beta_map.shape}")

    if D.shape[0] != z.shape[0] or D.shape[0] != beta_map.shape[0]:
        raise ValueError("Batch dimensions of D, z, and beta_map must match")

    if D.shape[-2:] != z.shape[-2:] or D.shape[-2:] != beta_map.shape[-2:]:
        raise ValueError("Spatial dimensions of D, z, and beta_map must match")

    # Ensure all tensors are on the same device
    if D.device != z.device or D.device != beta_map.device:
        raise ValueError("D, z, and beta_map must be on the same device")

    # beta_map and z broadcast to [B, 3, H, W]
    exponent = (beta_map * z).clamp(min=0.0, max=max_exponent)

    J = D * torch.exp(exponent)

    return J
