# %%
# import relevant library functions
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.optimize import least_squares

# %%


def load_img(path: Path):
    """
    Loads the image into memory. Requires at least a 3-channel RGB image.
    Grayscale images will raise an error. RGBA images will be converted to RGB.
    """
    x = iio.imread(path)

    # Raise an error if the image is 2D (grayscale) or has fewer than 3 channels
    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError(
            f"Expected at least a 3-channel RGB image. Got shape: {x.shape}"
        )

    # Keep only the first 3 channels (discard Alpha/transparency)
    if x.shape[-1] > 3:
        x = x[..., :3]

    # Convert to float32
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = np.asarray(x, dtype=np.float32)

    return np.clip(x, 0, 1)


# %%


def load_depth(path: Path):
    """
    Loads a depth map from the Sea-Thru dataset.
    Converts 0 values to NaNs and preserves the scale in meters.
    """
    # Read the depth map
    depth = iio.imread(path)

    # Convert to float32 to allow for NaN values
    depth = depth.astype(np.float32)

    # Convert invalid depth values (0) to NaN
    depth[depth == 0.0] = np.nan

    return depth


# %%


def B_model(
    z: np.ndarray, Binf: float, betaB: float, Jp: float, betaDp: float
) -> np.ndarray:
    """
    Evaluates the theoretical backscatter model for a given depth.

    Args:
        z: Depth array (meters).
        Binf: (B^infinity) The veiling light/asymptotic background color.
        betaB: (beta^B) The backscatter attenuation coefficient.
        Jp: (J') Ambient illumination/forward-scattered light at the camera.
        betaDp: (beta^D') The attenuation coefficient for the direct signal.
    """
    return Binf * (1 - np.exp(-betaB * z)) + Jp * np.exp(-betaDp * z)


def dark_samples(
    I: np.ndarray, z: np.ndarray, bins: int = 10, percentile: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bins the depth map and extracts the darkest 1% pixels in each bin to isolate
    the water column/backscatter signal from scene objects.
    """
    # 1. Filter out invalid depth/pixel values and ensure depth is strictly positive
    ok = np.isfinite(z) & np.all(np.isfinite(I), axis=2) & (z > 0)
    zz = z[ok]
    rgb = I[ok]

    # Create depth bin edges from the valid depth range
    edges = np.linspace(zz.min(), zz.max(), bins + 1)

    # Calculate pixel darkness using the L2 norm across RGB channels
    darkness = np.linalg.norm(rgb, axis=1)

    Z_samples, RGB_samples = [], []

    for k in range(bins):
        # 2. Isolate pixels within the current depth bin
        if k == bins - 1:
            m = (zz >= edges[k]) & (zz <= edges[k + 1])
        else:
            m = (zz >= edges[k]) & (zz < edges[k + 1])

        idx = np.where(m)[0]

        # Skip bins with insufficient data to prevent statistical anomalies
        if len(idx) < 10:
            continue

        # 3. Find the intensity threshold for the darkest 'percentile' (e.g., 1%)
        cut = np.percentile(darkness[idx], percentile)
        d = idx[darkness[idx] <= cut]

        Z_samples.append(zz[d])
        RGB_samples.append(rgb[d])

    # Return concatenated arrays of the selected dark pixels and their depths
    return np.concatenate(Z_samples), np.concatenate(RGB_samples)


def fit_B_channel(
    zs: np.ndarray, ys: np.ndarray, restarts: int = 10, seed: int = 0
) -> np.ndarray:
    """
    Fits the non-linear backscatter model to the sampled dark pixels using
    multistart least squares optimization to avoid local minima.
    """
    rng = np.random.default_rng(seed)

    # Parameter bounds defined by the physical model limits:
    # [Binf, betaB, Jp, betaDp]
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([1.0, 5.0, 1.0, 5.0])

    best = None

    def residual(p: np.ndarray) -> np.ndarray:
        return B_model(zs, *p) - ys

    # Multistart optimization: The backscatter model is non-convex,
    # so we run the solver from multiple random initializations.
    for _ in range(restarts):
        # Generate initial guesses within physically plausible sub-ranges
        x0 = np.array(
            [
                rng.uniform(0.05, 0.8),  # Binf
                rng.uniform(0.01, 2.0),  # betaB
                rng.uniform(0.0, 0.3),  # Jp
                rng.uniform(0.01, 2.0),  # betaDp
            ]
        )

        try:
            # Trust Region Reflective (trf) is the default for bounded least_squares
            r = least_squares(
                residual, x0, bounds=(lo, hi), max_nfev=5000, loss="linear"
            )

            # Evaluate the Mean Squared Error (MSE) of the fit
            mse = np.mean(r.fun**2)

            # Keep the parameters that yield the lowest MSE
            if best is None or mse < best[0]:
                best = (mse, r.x)

        except ValueError:
            # Catch specific solver failures (e.g., NaNs in Jacobian)
            # rather than silently passing all Exceptions
            pass

    if best is None:
        raise RuntimeError(
            "Backscatter fit failed across all restarts. Check input data normalization."
        )

    return best[1]


def estimate_backscatter(
    I: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates backscatter parameters for all three color channels independently.
    """
    # Extract the dark samples representing the water column
    zs, rgb = dark_samples(I, z)

    # Fit the 4 model parameters for R, G, and B separately
    params = np.vstack([fit_B_channel(zs, rgb[:, c], seed=100 + c) for c in range(3)])

    return params, zs, rgb


def B_map(z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Constructs the full-resolution backscatter map for the image using the
    estimated per-channel parameters.

    Args:
        z: 2D array of depth values (Height, Width).
        params: 2D array of shape (3, 4) containing [Binf, betaB, Jp, betaDp]
                for the R, G, and B channels respectively.

    Returns:
        B: 3D array (Height, Width, 3) representing the estimated backscatter
           contribution at each pixel.
    """
    # Initialize an empty array with the same spatial dimensions as z, plus 3 color channels
    B = np.zeros((*z.shape, 3), dtype=np.float32)

    # Evaluate the backscatter model for Red (0), Green (1), and Blue (2)
    for c in range(3):
        # Unpack the 4 parameters for the current channel into the B_model
        B[..., c] = B_model(z, *params[c])

    return B


def remove_backscatter(
    I: np.ndarray, z: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolates the direct signal (D) by subtracting the estimated backscatter (B)
    from the original image (I).

    Args:
        I: The original normalized RGB image (Height, Width, 3).
        z: The depth map (Height, Width).
        params: The backscatter parameters (3, 4) returned by estimate_backscatter.

    Returns:
        B: The generated backscatter map (Height, Width, 3).
        D: The isolated direct signal map (Height, Width, 3), clamped to positive values.
    """
    # Generate the backscatter map using the physical model parameters
    B = B_map(z, params)

    # Subtract backscatter from the observed image to get the direct signal (D).
    #
    # Important: We clip the lower bound to 1e-6 rather than 0.0.
    # The next step in the Sea-thru pipeline requires transforming D into log-space
    # (e.g., np.log(D)) to estimate the wide-band attenuation coefficients.
    # Clipping at a small epsilon prevents log(0) -inf errors or domain errors from negative values.
    D = np.clip(I - B, 1e-6, 1.0)

    return B, D


# %%


def display_rgb(x: np.ndarray, gamma: float = 1 / 2.2) -> np.ndarray:
    """
    Applies simple gamma correction to linear RGB data for monitor display.

    Args:
        x: Linear RGB image array.
        gamma: Gamma correction factor (default ~0.45).

    Returns:
        Gamma-corrected RGB array clamped to [0.0, 1.0].
    """
    # A single clip is mathematically sufficient and slightly faster.
    # Any value in [0.0, 1.0] raised to a positive power remains in [0.0, 1.0].
    return np.clip(x, 0.0, 1.0) ** gamma


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """
    Applies accurate, piece-wise sRGB color space conversion.
    """
    x_clipped = np.clip(x, 0.0, 1.0)

    # The exact sRGB standard uses a linear transformation near zero to
    # prevent an infinite slope, transitioning to a gamma curve for brighter pixels.
    return np.where(
        x_clipped <= 0.0031308,
        12.92 * x_clipped,
        1.055 * (x_clipped ** (1.0 / 2.4)) - 0.055,
    )


# %%
def shift(a: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """
    Shifts a 2D or 3D NumPy array spatially by (dy, dx) pixels.

    Args:
        a: Input array of shape (H, W) or (H, W, C).
        dy: Vertical shift distance (positive shifts down, negative shifts up).
        dx: Horizontal shift distance (positive shifts right, negative shifts left).
        fill: Value used to pad the empty spaces created by the shift.

    Returns:
        A shifted copy of the array.
    """
    out = np.full_like(a, fill)
    H, W = a.shape[:2]

    # Calculate target slicing (where data goes in the output array)
    ys = slice(max(0, dy), min(H, H + dy))
    xs = slice(max(0, dx), min(W, W + dx))

    # Calculate source slicing (where data comes from in the input array)
    y0 = slice(max(0, -dy), min(H, H - dy))
    x0 = slice(max(0, -dx), min(W, W - dx))

    out[ys, xs] = a[y0, x0]
    return out


def lsac(
    D: np.ndarray,
    z: np.ndarray,
    p: float = 0.01,
    epsilon_fraction: float = 0.01,
    max_iterations: int = 2000,
    tolerance: float = 1e-5,
    f: float = 2.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, list[list[float]]]:
    """
    Sea-thru Local Space Average Color (LSAC) illuminant estimation.
    Estimates the scene illuminant by iteratively averaging colors within
    local spatial neighborhoods that share similar depths.

    Args:
        D: Direct signal array, shape (H, W, 3).
        z: Range/depth map in meters, shape (H, W).
        p: Weighting parameter for the direct signal in the update step.
        epsilon_fraction: Fraction of the total depth range used to define
                          the neighborhood depth threshold (epsilon).
        max_iterations: Maximum number of solver iterations.
        tolerance: Convergence threshold based on the maximum absolute change.
        f: Geometry scaling factor. (f=2 for perpendicular orientation).
        verbose: If True, prints convergence progress.

    Returns:
        E: Estimated local illuminant map, shape (H, W, 3).
        a: Final LSAC local space average, shape (H, W, 3).
        eps: The computed depth neighborhood threshold.
        n_iterations: Total iterations performed before convergence.
        history: Tracking of the median RGB values during the solver run.
    """
    H, W, _ = D.shape

    logger = logging.getLogger("SeaThruBenchmark")

    # 1. Isolate Valid Pixels
    # Ensure depths and signal values are finite and physically valid (z > 0)
    valid = np.isfinite(z) & np.all(np.isfinite(D), axis=2) & (z > 0)

    if not np.any(valid):
        raise ValueError("No valid pixels found in the depth map.")

    # 2. Compute Depth Threshold (Epsilon)
    # Epsilon determines if a neighboring pixel is close enough in 3D space
    # to be considered part of the same physical object/plane.
    z_valid = z[valid]
    z_range = np.nanmax(z_valid) - np.nanmin(z_valid)
    eps = epsilon_fraction * max(z_range, 1e-6)

    # 3. Initialization
    # Initialize the local average color 'a' as zeros.
    a = np.zeros_like(D, dtype=np.float32)
    history = []

    # 4. Iterative Optimization loop
    for iteration in range(max_iterations):
        # Accumulators for the neighboring pixels
        total = np.zeros_like(D, dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

        # Iterate over the 4-connected spatial neighborhood (Up, Down, Left, Right)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Shift the depth and 'a' maps to align neighbors to the center pixel
            zn = shift(z, dy, dx, fill=np.nan)
            an = shift(a, dy, dx, fill=0.0)

            # A neighbor is valid if it exists, is finite, and its depth
            # falls within the epsilon threshold of the center pixel's depth.
            good = valid & np.isfinite(zn) & (np.abs(z - zn) <= eps)

            # Accumulate the valid neighbors' colors
            total += an * good[..., None]
            count += good

        # 5. Compute Neighborhood Average (a'_c)
        # Prevent division by zero by setting empty neighborhood counts to 1
        mean = total / np.maximum(count[..., None], 1.0)

        # 6. LSAC Update Step
        # Formula: a_c(x,y) = p * D_c(x,y) + (1 - p) * a'_c(x,y)
        update = valid & (count > 0)
        anew = a.copy()

        anew[update] = p * D[update] + (1 - p) * mean[update]

        # 7. Evaluate Convergence
        # Measure the maximum absolute change across all valid pixels
        difference = np.nanmax(np.abs(anew[valid] - a[valid]))
        a = anew

        # Record the median RGB values for tracking stability
        median_rgb = [float(np.nanmedian(a[..., c][valid])) for c in range(3)]
        history.append(median_rgb)

        # 8. Logging
        if verbose and (iteration % 50 == 0):
            logger.info(
                f"iteration = {iteration + 1:4d}, "
                f"change = {difference:.8f}, "
                f"median = [{median_rgb[0]:.6f}, {median_rgb[1]:.6f}, {median_rgb[2]:.6f}]"
            )

        # 9. Early Stopping
        if difference < tolerance:
            if verbose:
                logger.info(f"\nConverged after {iteration + 1} iterations.")
            break

    n_iterations = iteration + 1

    # 10. Estimate Illuminant (E)
    # Scale the converged average color by the geometry factor (f)
    E = f * a
    E = np.clip(E, 0.0, 1.0)

    return E, a, eps, n_iterations, history


# %%
def coarse_beta(E: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Computes a coarse, noisy estimate of the direct signal attenuation
    coefficient (beta_D) directly from the illuminant map.

    Formula: beta_c^D(z) = -log(E_c) / z
    """
    # Clip E to avoid log(0) errors and limit z to avoid division by zero.
    E_safe = np.clip(E, 1e-5, 1.0)
    z_safe = np.maximum(z[..., None], 1e-6)

    return -np.log(E_safe) / z_safe


def beta_model(z: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Two-term exponential model for wideband attenuation.
    Because camera channels capture a broad spectrum (wideband), a single
    exponential does not perfectly model attenuation. The two-term model
    accounts for different wavelengths decaying at different rates.
    """
    return a * np.exp(b * z) + c * np.exp(d * z)


def refine_beta(
    z: np.ndarray, E: np.ndarray, beta0: np.ndarray, restarts: int = 10, seed: int = 0
) -> np.ndarray:
    """
    Fits a smooth two-term exponential model per color channel to the noisy coarse beta estimates.
    Returns the evaluated full-resolution beta map (H, W, 3).
    """
    rng = np.random.default_rng(seed)

    # 1. Isolate valid data points
    m = np.isfinite(z) & np.all(np.isfinite(E) & (E > 1e-4) & (E < 1.0), axis=2)
    zz = z[m].astype(float)
    ee = E[m].astype(float)
    bb = beta0[m].astype(float)

    # 2. Filter out extreme coarse beta values
    m2 = np.all(np.isfinite(bb) & (bb > 0.0) & (bb < 10.0), axis=1)
    zz, ee, bb = zz[m2], ee[m2], bb[m2]

    # 3. Subsample data for optimization speed
    if len(zz) > 30000:
        ids = rng.choice(len(zz), 30000, replace=False)
        zz, ee, bb = zz[ids], ee[ids], bb[ids]

    zr = max(zz.max() - zz.min(), 1e-3)
    lo = np.array([0.0, -10.0 / zr, 0.0, -10.0 / zr])
    hi = np.array([5.0, 0.0, 5.0, 0.0])

    # Initialize the final spatial beta map
    beta_map = np.zeros_like(E)

    # 4. Optimize R, G, and B independently
    for c in range(3):
        best = None
        ee_c = ee[:, c]
        bb_c = bb[:, c]

        for _ in range(restarts):
            s = np.clip(np.median(bb_c), 0.02, 2.0)
            x0 = np.array(
                [
                    rng.uniform(0.2, 0.8) * s,
                    -rng.uniform(0.1, 2.0) / zr,
                    rng.uniform(0.2, 0.8) * s,
                    -rng.uniform(0.1, 2.0) / zr,
                ]
            )
            x0 = np.clip(x0, lo + 1e-8, hi - 1e-8)

            def residual(p: np.ndarray) -> np.ndarray:
                # Predict beta using the two-term model for a single channel
                beta = beta_model(zz, *p)
                zhat = -np.log(ee_c) / np.maximum(beta, 1e-6)
                return zhat - zz

            try:
                r = least_squares(
                    residual,
                    x0,
                    bounds=(lo, hi),
                    loss="soft_l1",
                    f_scale=0.1,
                    max_nfev=3000,
                )
                mse = np.mean(r.fun**2)

                if best is None or mse < best["mse"]:
                    best = {"params": r.x, "mse": mse}
            except ValueError:
                pass

        if best is None:
            raise RuntimeError(f"beta_D refinement failed to converge for channel {c}.")

        # 5. Evaluate the smooth model for the entire depth map for this channel
        beta_map[..., c] = beta_model(z, *best["params"])

    return beta_map


def reconstruct_scene(
    D: np.ndarray, z: np.ndarray, beta_map: np.ndarray, max_exponent: float = 10.0
) -> np.ndarray:
    """
    Reconstructs the original scene colors (J) by reversing the wideband
    direct signal attenuation.

    Formula: J_c = D_c * exp(beta_c^D(z) * z)

    Args:
        D: The isolated direct signal, shape (H, W, 3).
        z: The depth map, shape (H, W).
        beta_map: The spatially varying beta_D coefficients evaluated at each pixel, shape (H, W, 3).
        max_exponent: A safety threshold to prevent math overflows at extreme depths.
    """
    # Calculate the exponent term for the reconstruction
    exponent = beta_map * z[..., None]

    # Clip the exponent to prevent np.exp() from overflowing to infinity.
    # (exp(10) is ~22026, which is already a massive physical multiplier).
    exponent = np.clip(exponent, 0.0, max_exponent)

    # Reverse the attenuation
    J = D * np.exp(exponent)

    return J


def gray_world(J: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Gray World white balancing to the reconstructed scene.
    Assumes that the average color of a diverse scene should be neutral gray
    (i.e., mean(R) == mean(G) == mean(B)).

    Args:
        J: The reconstructed linear image, shape (H, W, 3).

    Returns:
        Js: The white-balanced image clamped to [0, 1].
        mean: The original color channel means.
        scale: The scaling factors applied to each channel.
    """
    # Ensure physically valid pixels (greater than 0, no NaNs)
    x = np.clip(J, 0.0, None)
    valid = np.all(np.isfinite(x), axis=2)

    # 1. Calculate the mean intensity of each color channel across valid pixels
    mean = np.array([x[..., c][valid].mean() for c in range(3)])

    # 2. Determine the target gray point (the average of the channel means)
    target = mean.mean()

    # 3. Calculate the scalar needed to shift each channel to the target
    scale = target / np.maximum(mean, 1e-6)

    # 4. Apply the scaling and clamp to valid image range
    Js = np.clip(x * scale[None, None, :], 0.0, 1.0)

    return Js, mean, scale
