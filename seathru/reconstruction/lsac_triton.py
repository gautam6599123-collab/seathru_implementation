# %%

import torch
import triton
import triton.language as tl

# %%


@triton.jit
def _lsac_step(
    D,
    Z,
    A,
    OUT,
    DIFF,
    H: tl.constexpr,
    W: tl.constexpr,
    EPS: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = idx < H * W

    y = idx // W
    x = idx % W

    # Centre pixel
    zc = tl.load(Z + idx, mask=inside, other=0.0)

    d0 = tl.load(D + 3 * idx + 0, mask=inside, other=0.0)
    d1 = tl.load(D + 3 * idx + 1, mask=inside, other=0.0)
    d2 = tl.load(D + 3 * idx + 2, mask=inside, other=0.0)

    center_valid = (
        inside
        & tl.is_finite(zc)
        & (zc > 0.0)
        & tl.is_finite(d0)
        & tl.is_finite(d1)
        & tl.is_finite(d2)
    )

    # Previous iteration's centre value
    a0 = tl.load(A + 3 * idx + 0, mask=inside, other=0.0)
    a1 = tl.load(A + 3 * idx + 1, mask=inside, other=0.0)
    a2 = tl.load(A + 3 * idx + 2, mask=inside, other=0.0)

    # Neighbour indices and boundary masks
    idx_up = idx - W
    idx_down = idx + W
    idx_left = idx - 1
    idx_right = idx + 1

    mask_up = inside & (y > 0)
    mask_down = inside & (y < H - 1)
    mask_left = inside & (x > 0)
    mask_right = inside & (x < W - 1)

    # Neighbour depth and D values
    zu = tl.load(Z + idx_up, mask=mask_up, other=0.0)
    zd = tl.load(Z + idx_down, mask=mask_down, other=0.0)
    zl = tl.load(Z + idx_left, mask=mask_left, other=0.0)
    zr = tl.load(Z + idx_right, mask=mask_right, other=0.0)

    def neighbour_valid(nmask, zn, ni):
        nd0 = tl.load(D + 3 * ni + 0, mask=nmask, other=0.0)
        nd1 = tl.load(D + 3 * ni + 1, mask=nmask, other=0.0)
        nd2 = tl.load(D + 3 * ni + 2, mask=nmask, other=0.0)

        return (
            nmask
            & tl.is_finite(zn)
            & (zn > 0.0)
            & tl.is_finite(nd0)
            & tl.is_finite(nd1)
            & tl.is_finite(nd2)
            & (tl.abs(zc - zn) <= EPS)
        )

    good_up = center_valid & neighbour_valid(mask_up, zu, idx_up)
    good_down = center_valid & neighbour_valid(mask_down, zd, idx_down)
    good_left = center_valid & neighbour_valid(mask_left, zl, idx_left)
    good_right = center_valid & neighbour_valid(mask_right, zr, idx_right)

    # Load previous-iteration illuminant values from good neighbours
    au0 = tl.load(A + 3 * idx_up + 0, mask=good_up, other=0.0)
    au1 = tl.load(A + 3 * idx_up + 1, mask=good_up, other=0.0)
    au2 = tl.load(A + 3 * idx_up + 2, mask=good_up, other=0.0)

    ad0 = tl.load(A + 3 * idx_down + 0, mask=good_down, other=0.0)
    ad1 = tl.load(A + 3 * idx_down + 1, mask=good_down, other=0.0)
    ad2 = tl.load(A + 3 * idx_down + 2, mask=good_down, other=0.0)

    al0 = tl.load(A + 3 * idx_left + 0, mask=good_left, other=0.0)
    al1 = tl.load(A + 3 * idx_left + 1, mask=good_left, other=0.0)
    al2 = tl.load(A + 3 * idx_left + 2, mask=good_left, other=0.0)

    ar0 = tl.load(A + 3 * idx_right + 0, mask=good_right, other=0.0)
    ar1 = tl.load(A + 3 * idx_right + 1, mask=good_right, other=0.0)
    ar2 = tl.load(A + 3 * idx_right + 2, mask=good_right, other=0.0)

    # Number of valid neighbours
    count = (
        good_up.to(tl.float32)
        + good_down.to(tl.float32)
        + good_left.to(tl.float32)
        + good_right.to(tl.float32)
    )

    denom = tl.maximum(count, 1.0)

    mean0 = (au0 + ad0 + al0 + ar0) / denom
    mean1 = (au1 + ad1 + al1 + ar1) / denom
    mean2 = (au2 + ad2 + al2 + ar2) / denom

    # Jacobi update:
    # a_new = p * D + (1-p) * neighbour_mean
    update = center_valid & (count > 0.0)

    new0 = tl.where(update, P * d0 + (1.0 - P) * mean0, a0)
    new1 = tl.where(update, P * d1 + (1.0 - P) * mean1, a1)
    new2 = tl.where(update, P * d2 + (1.0 - P) * mean2, a2)

    # Invalid pixels retain their previous values.
    tl.store(OUT + 3 * idx + 0, new0, mask=inside)
    tl.store(OUT + 3 * idx + 1, new1, mask=inside)
    tl.store(OUT + 3 * idx + 2, new2, mask=inside)

    # Per-pixel maximum change, used for convergence testing
    diff0 = tl.abs(new0 - a0)
    diff1 = tl.abs(new1 - a1)
    diff2 = tl.abs(new2 - a2)

    pixel_diff = tl.maximum(diff0, tl.maximum(diff1, diff2))
    tl.store(DIFF + idx, pixel_diff, mask=inside)


# %%


def lsac_triton(
    D: torch.Tensor,
    z: torch.Tensor,
    p: float = 0.01,
    epsilon_fraction: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-5,
    f: float = 2.0,
    check_every: int = 1,
):
    """
    Triton implementation of the LSAC Jacobi iteration.

    Inputs:
        D: (H, W, 3), CUDA float32, contiguous
        z: (H, W), CUDA float32, contiguous

    Returns:
        E, a, eps, n_iterations, history

    check_every=1 checks convergence after every iteration.
    Larger values reduce synchronization overhead, but may perform
    extra iterations after convergence.
    """

    if not D.is_cuda or not z.is_cuda:
        raise ValueError("D and z must be CUDA tensors.")

    if D.dtype != torch.float32 or z.dtype != torch.float32:
        raise ValueError("D and z must be float32.")

    if D.ndim != 3 or D.shape[-1] != 3:
        raise ValueError("D must have shape (H, W, 3).")

    if z.ndim != 2 or D.shape[:2] != z.shape:
        raise ValueError("z must have shape (H, W), matching D.")

    D = D.contiguous()
    z = z.contiguous()

    H, W, _ = D.shape
    device = D.device

    # Match the reference validity definition.
    valid = torch.isfinite(z) & (z > 0) & torch.isfinite(D).all(dim=-1)

    valid_z = z[valid]

    if valid_z.numel() == 0:
        raise ValueError("No valid pixels found.")

    z_min = valid_z.min().item()
    z_max = valid_z.max().item()

    z_range = z_max - z_min
    eps = epsilon_fraction * max(z_range, 1e-6)

    # Ping-pong buffers. a starts at zero, as in the NumPy version.
    a0 = torch.zeros_like(D)
    a1 = torch.empty_like(D)

    diff = torch.empty((H * W,), device=device, dtype=torch.float32)

    BLOCK = 256
    grid = (triton.cdiv(H * W, BLOCK),)

    a_current = a0
    a_next = a1

    n_iterations = 0
    history = []

    for iteration in range(max_iterations):
        _lsac_step[grid](
            D,
            z,
            a_current,
            a_next,
            diff,
            H,
            W,
            eps,
            p,
            BLOCK,
        )

        a_current, a_next = a_next, a_current
        n_iterations = iteration + 1

        # Check convergence periodically.
        if n_iterations % check_every == 0:
            max_difference = diff.max().item()

            if max_difference < tolerance:
                break

    a = a_current
    E = torch.clamp(f * a, 0.0, 1.0)

    return E, a, eps, n_iterations, history
