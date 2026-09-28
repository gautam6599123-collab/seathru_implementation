import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from benchmark import benchmark_seathru
from seathru import (
    coarse_beta,
    estimate_backscatter,
    gray_world,
    linear_to_srgb,
    load_depth,
    load_img,
    lsac,
    reconstruct_scene,
    refine_beta,
    remove_backscatter,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sea-Thru processor or benchmark.")
    parser.add_argument(
        "--benchmark", type=str, help="Path to dataset directory for benchmarking"
    )
    parser.add_argument(
        "--image", type=str, default="data/example.png", help="Path to single image"
    )
    parser.add_argument(
        "--depth", type=str, default="data/example.tif", help="Path to single depth map"
    )
    args = parser.parse_args()

    if args.benchmark:
        benchmark_seathru(args.benchmark, sample_size=10)
    else:
        image_path = Path(args.image)
        depth_path = Path(args.depth)

        if image_path.exists() and depth_path.exists():
            print(f"Processing single image: {image_path.name}")
            I = load_img(image_path)
            z = load_depth(depth_path)

            params, _, _ = estimate_backscatter(I, z)
            B, D = remove_backscatter(I, z, params)
            E, a, eps, iters, hist = lsac(D, z, verbose=False)
            beta0 = coarse_beta(E, z)
            beta_map = refine_beta(z, E, beta0, seed=0)
            J = reconstruct_scene(D, z, beta_map)
            Js, _, _ = gray_world(J)
            Js_srgb = linear_to_srgb(Js)
            Js_uint8 = (np.clip(Js_srgb, 0.0, 1.0) * 255).astype(np.uint8)

            out_dir = image_path.parent
            output_name = f"{image_path.stem}_restored.png"
            output_path = out_dir / output_name
            iio.imwrite(output_path, Js_uint8)

        else:
            print(f"File(s) not found:\n- {image_path}\n- {depth_path}")
            print(
                "To benchmark an entire folder, use: python main.py --benchmark path/to/dataset"
            )
