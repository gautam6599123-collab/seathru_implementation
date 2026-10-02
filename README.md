# Sea-Thru: A Method for Removing Water from Underwater Images
An implementation of the method described in: `Akkaynak, D. and Treibitz, T. "Sea-Thru: A Method for Removing Water
from Underwater Images." CVPR, 2019.`

This repository contains a physics-based implementation of Sea-Thru for
estimating and removing the veiling-light/backscatter component in
underwater images, together with GPU-oriented components for processing.
Overview
Underwater images are affected by wavelength-dependent attenuation and
backscatter. Sea-Thru uses scene depth and an image formation model to
estimate the backscatter contribution and recover a representation of
the direct signal.
The implementation includes:
- RGB image and depth-map loading.
- Extraction of dark pixels across depth intervals.
- Nonlinear fitting of the backscatter model independently for each
  color channel.
- Construction of a full-resolution backscatter map and subtraction
  from the observed image.
- GPU-oriented implementations for batched processing, depth-aware
  refinement, and reconstruction.
The repository is an implementation/research project, not an official
release by the authors of the Sea-Thru paper.
Method
Let (I) denote the observed RGB image, (z) the scene depth, (B) the
estimated backscatter, and (D) the direct signal. The implementation
estimates the backscatter using the model
z}`\right`{=tex}) + J' e^{-`\beta`{=tex}^{D'}z}, \]

where the four parameters are fitted separately for each color channel.

The estimated backscatter is subtracted from the observed image to
obtain the direct component:

\[ D = `\max`{=tex}(I-B,,0). \]

The code also contains depth-aware processing and reconstruction
components. See the source modules for implementation details and
current interfaces.

## Repository contents

The main implementation files include:

| File                | Description                                                                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `seathru.py`        | NumPy/SciPy reference implementation of image/depth loading, dark-sample extraction, backscatter fitting, and backscatter estimation/removal. |
| `dataset.py`        | Dataset utilities for pairing and loading image/depth data.                                                                                   |
| `sampling.py`       | Backscatter model functions                                                                                                                   |
| `fitting.py`        | Sampling utilities for selecting observations used in parameter estimation.                                                                   |
| `fit_b_channel.py`  | Backscatter channel-fitting implementation                                                                                                    |
| `beta_map.py`       | Coarse estimation and refinement of attenuation-related parameters                                                                            |
| `lsac_triton.py`    | Triton implementation of the LSAC refinement step.                                                                                            |
| `reconstruction.py` | Scene reconstruction from the direct signal, depth, and estimated parameters.                                                                 |
| `prefetcher.py`     | CUDA batch prefetching utilities                                                                                                              |
| `functions.py`      | Supporting functions                                                                                                                          |
| `example_gpu.py`    | GPU example/entry point                                                                                                                       |

The repository is under active development; function signatures and the
recommended execution path may change.

## Requirements

The implementation uses Python and scientific-computing/deep-learning
libraries. Depending on which path you run, you may need:

-   Python
-   NumPy
-   SciPy
-   ImageIO
-   tifffile
-   PyTorch
-   TorchCodec
-   Triton
-   A compatible NVIDIA GPU and CUDA-enabled PyTorch installation for
    the GPU/Triton components

Install the versions appropriate for your environment. GPU support
depends on compatibility between the NVIDIA driver, CUDA, PyTorch, and
Triton.

## Data

The implementation expects paired RGB images and depth maps. The dataset
loader accepts pairs of image and depth paths. The GPU dataset
implementation reads RGB images using TorchCodec and depth maps using
`tifffile`.

Important data assumptions in the current code:

-   Images must contain at least three color channels; the reference
    loader discards channels beyond RGB.
-   Integer images are normalized to the range (\[0,1\]).
-   Depth maps are converted to `float32`.
-   A depth value of zero is treated as invalid and converted to `NaN`.
-   Depth values are otherwise kept on their original scale. Ensure that
    the depth units and scale are consistent with the model and dataset.

You will need to prepare the image/depth path pairs for your dataset.
The repository does not prescribe a universal directory structure.

## Usage

### NumPy/SciPy reference implementation

The `seathru.py` module provides the reference-style processing
functions, including:

-   `load_img(path)` and `load_depth(path)`
-   `dark_samples(I, z)`
-   `fit_B_channel(zs, ys, ...)`
-   `estimate_backscatter(I, z)`
-   `B_map(z, params)`
-   `remove_backscatter(I, z, params)`

A typical processing sequence is:

1.  Load an RGB image and its corresponding depth map.
2.  Estimate backscatter parameters from dark samples.
3.  Generate the backscatter map.
4.  Subtract the estimated backscatter from the observed image.
5.  Inspect and save the resulting direct-signal image.

Example:

``` python
from pathlib import Path
from seathru import load_img, load_depth, estimate_backscatter, remove_backscatter

image = load_img(Path("path/to/image.png"))
depth = load_depth(Path("path/to/depth.tif"))

params, depths, samples = estimate_backscatter(image, depth)
backscatter, direct = remove_backscatter(image, depth, params)
```

The exact import path may need to be adjusted depending on how the
repository is installed or invoked.

### GPU implementation

GPU-oriented components are provided in separate modules. They use
PyTorch tensors and, for LSAC refinement, Triton. Refer to
`example_gpu.py` and the function definitions in the relevant modules
for the current invocation details.

For GPU execution:

1.  Install a CUDA-enabled PyTorch build compatible with your system.
2.  Install a compatible Triton version.
3.  Prepare paired image/depth paths.
4.  Construct the dataset and data loader using the repository's current interfaces.
5.  Run the GPU processing pipeline and inspect intermediate and final outputs.

No performance figures are claimed here; runtime depends on image
resolution, GPU, software versions, and algorithm settings.

## Notes and limitations

-   This repository implements a research method and should be treated as experimental software.
-   Results depend on the quality and scale of the depth maps, image normalization, and the validity of the model assumptions.
-   Invalid depth values are represented as `NaN`; downstream operations must handle them appropriately.
-   The NumPy/SciPy and GPU implementations may differ in numerical behavior and should be compared when validating changes.
-   Evaluate results visually and, where suitable reference images are available, with quantitative metrics. A visually plausible output alone does not establish physical accuracy.

## Citation

If you use this repository, please cite it as:

``` bibtex
@misc{seathru_implementation,
  author       = {Gautam Singh},
  title        = {A Sea-Thru Implementation},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/gautam6599-collab/seathru_implementation}}
}
```

Please also cite the original Sea-Thru paper when referring to the
method:

``` bibtex
@inproceedings{akkaynak2019sea,
  title     = {Sea-Thru: A Method for Removing Water From Underwater Images},
  author    = {Akkaynak, Derya and Treibitz, Tali},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

## Acknowledgements

This project is based on the Sea-Thru method proposed by Derya Akkaynak
and Tali Treibitz.
