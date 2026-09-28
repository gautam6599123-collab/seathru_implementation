import json
import random
import time
from pathlib import Path

from logger import setup_logger
from seathru import (
    coarse_beta,
    estimate_backscatter,
    gray_world,
    load_depth,
    load_img,
    lsac,
    reconstruct_scene,
    refine_beta,
    remove_backscatter,
)


def benchmark_seathru(dataset_dir: str | Path, sample_size: int = 10, seed: int = 42):
    logger = setup_logger("benchmark_run.log")
    dataset_dir = Path(dataset_dir)
    image_paths = sorted(list(dataset_dir.rglob("*.png")))

    rng = random.Random(seed)
    sampled_images = rng.sample(image_paths, min(sample_size, len(image_paths)))

    results = []
    total_run_time = 0.0

    logger.info(f"Starting benchmark on {len(sampled_images)} images...")
    logger.info(
        "Hardware: 11th Gen Intel i5-1135G7 (8) @ 4.200GHz & Intel TigerLake-LP GT2 [Iris Xe Graphics]"
    )

    for img_path in sampled_images:
        depth_path = img_path.with_suffix(".tif")
        if not depth_path.exists():
            logger.warning(f"Depth map missing for {img_path.name}, skipping.")
            continue

        try:
            t0 = time.perf_counter()
            I = load_img(img_path)
            z = load_depth(depth_path)
            t_io = time.perf_counter() - t0

            t1 = time.perf_counter()
            params, _, _ = estimate_backscatter(I, z)
            B, D = remove_backscatter(I, z, params)
            t_b = time.perf_counter() - t1

            t2 = time.perf_counter()
            # Set verbose=True if you want LSAC steps logged
            E, a, eps, iters, hist = lsac(D, z, verbose=False)
            t_l = time.perf_counter() - t2

            t3 = time.perf_counter()
            beta0 = coarse_beta(E, z)
            beta_map = refine_beta(z, E, beta0, seed=0)
            t_beta = time.perf_counter() - t3

            t4 = time.perf_counter()
            J = reconstruct_scene(D, z, beta_map)
            Js, _, _ = gray_world(J)
            t_reconstruct = time.perf_counter() - t4

            total = t_io + t_b + t_l + t_beta + t_reconstruct
            total_run_time += total

            metrics = {
                "image": img_path.name,
                "resolution": I.shape[:2],
                "io_time_sec": t_io,
                "backscatter_time_sec": t_b,
                "lsac_time_sec": t_l,
                "beta_map_sec": t_beta,
                "t_reconstruct": t_reconstruct,
                "total_time_sec": total,
                "lsac_iterations": iters,
            }
            results.append(metrics)

            logger.info(
                f"[{img_path.name}] Res: {I.shape[:2]} | Total: {total:.2f}s | "
                f"IO: {t_io:.2f}s | B_est: {t_b:.2f}s | LSAC: {t_l:.2f}s ({iters} iters)"
            )

        except Exception as e:
            logger.error(f"Failed processing {img_path.name}: {str(e)}")

    logger.info(f"Benchmarking complete. Total time: {total_run_time:.2f}s")

    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Structured metrics saved to {output_file}")

    return results, total_run_time
