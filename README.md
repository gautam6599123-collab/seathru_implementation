# Sea-Thru: Underwater Image Restoration

Implementation of the Sea-Thru method by Akkaynak and Treibitz (CVPR 2019).

Sea-Thru uses scene depth and an underwater image formation model to estimate backscatter and recover the direct signal.

## Features

- Load RGB images and depth maps.
- Sample dark pixels across depth intervals.
- Fit backscatter parameters for each color channel.
- Estimate and remove backscatter.
- Refine depth-aware parameters and reconstruct the scene.
- GPU processing with PyTorch and Triton.

This is an independent implementation, not an official release by the paper's authors.

## Method

The observed image `I` contains direct signal and backscatter. The implementation estimates the backscatter map `B` from depth `z`, then computes:

`D = max(I - B, 0)`

See the source code for the model and reconstruction details.

## Main files

| File                | Purpose                           |
| ------------------- | --------------------------------- |
| `seathru.py`        | NumPy/SciPy implementation        |
| `dataset.py`        | Image and depth loading           |
| `sampling.py`       | Sampling observations for fitting |
| `fitting.py`        | Parameter fitting                 |
| `fit_b_channel.py`  | Per-channel backscatter fitting   |
| `beta_map.py`       | Attenuation parameter estimation  |
| `lsac_triton.py`    | Triton LSAC refinement            |
| `reconstruction.py` | Scene reconstruction              |
| `prefetcher.py`     | CUDA batch prefetching            |
| `example_gpu.py`    | GPU example                       |

The code is under development, so interfaces may change.

## Requirements

Depending on the implementation path, you may need:

- Python, NumPy, SciPy
- ImageIO and `tifffile`
- PyTorch and TorchCodec
- Triton
- NVIDIA GPU with a compatible CUDA/PyTorch setup for GPU processing

## Data

Provide paired RGB images and depth maps. The current loader:

- Keeps the first three image channels.
- Normalizes integer images to `[0, 1]`.
- Converts depth maps to `float32`.
- Treats zero depth values as invalid (`NaN`).

Depth units and scale must be consistent with the model.

## Usage

### NumPy/SciPy

`seathru.py` provides functions for loading data, sampling dark pixels, fitting backscatter, and removing it.

```python
from pathlib import Path
from seathru import load_img, load_depth, estimate_backscatter, remove_backscatter

image = load_img(Path("path/to/image.png"))
depth = load_depth(Path("path/to/depth.tif"))

params, depths, samples = estimate_backscatter(image, depth)
backscatter, direct = remove_backscatter(image, depth, params)
```

Adjust the import path to match your setup.

### GPU

The GPU implementation uses PyTorch, with Triton for LSAC refinement. See `example_gpu.py` and the relevant module functions for the current pipeline.

## Notes

- Results depend on depth quality, image scaling, and model assumptions.
- The NumPy/SciPy and GPU paths may differ numerically.
- Compare outputs visually and use quantitative metrics when reference images are available.

## Citation

Please cite this repository:

```bibtex
@misc{seathru_implementation,
  author       = {Gautam Singh},
  title        = {A Sea-Thru Implementation},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/gautam6599-collab/seathru_implementation}}
}
```

Also cite the original paper:

```bibtex
@inproceedings{akkaynak2019sea,
  title     = {Sea-Thru: A Method for Removing Water From Underwater Images},
  author    = {Akkaynak, Derya and Treibitz, Tali},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

## Acknowledgements

Based on the Sea-Thru method by Derya Akkaynak and Tali Treibitz.
