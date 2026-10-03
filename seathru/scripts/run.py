# %%

import time

import torch
from torch.utils.data import DataLoader

from seathru.backscatter.model import remove_backscatter_gpu
from seathru.backscatter.sampling import estimate_backscatter_for_batch
from seathru.data.dataset import SeaThruDataset
from seathru.data.prefetcher import CUDAPrefetcher
from seathru.reconstruction.beta_map import (
    beta_model_gpu,
    coarse_beta_gpu,
    refine_beta_batch,
)
from seathru.reconstruction.lsac_triton import lsac_triton
from seathru.reconstruction.reconstruction import reconstruct_scene_gpu
from seathru.visualisation import show_backscatter_results, show_reconstruction_results

# %%


def main():

    # -----------------------------
    # Configuration
    # -----------------------------

    pairs = [
        ("data/T_S02958.png", "data/depthT_S02958.tif"),
        # Add more image/depth pairs here
    ]

    batch_size = 2
    num_workers = 8

    # -----------------------------
    # Dataset and DataLoader
    # -----------------------------

    dataset = SeaThruDataset(pairs)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    # -----------------------------
    # Device and prefetcher
    # -----------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda:0")
    prefetcher = CUDAPrefetcher(loader, device)

    # -----------------------------
    # Processing
    # -----------------------------

    while True:
        batch = prefetcher.next()

        if batch is None:
            break

        image = batch["image"]  # [B, 3, H, W]
        depth = batch["depth"]  # [B, 1, H, W]

        print("Image:", image.shape, image.device)
        print("Depth:", depth.shape, depth.device)

        # ---------------------------------
        # 1. Estimate backscatter parameters
        # ---------------------------------

        params, losses = estimate_backscatter_for_batch(image, depth)

        # ---------------------------------
        # 2. Remove backscatter
        # ---------------------------------

        with torch.no_grad():
            backscatter, corrected = remove_backscatter_gpu(
                I=image,
                z=depth,
                params=params,
            )

        print("Parameters [B, C, 4]:")
        print(params)

        print("Fit losses [B, C]:")
        print(losses)

        show_backscatter_results(image, backscatter, corrected)

        # ---------------------------------
        # 3. LSAC refinement, one image at a time
        # ---------------------------------

        E_batch = []

        with torch.no_grad():
            for i in range(corrected.shape[0]):
                # NCHW -> HWC, as required by lsac_triton
                D_i = corrected[i].permute(1, 2, 0).contiguous()

                # [1, H, W] -> [H, W]
                z_i = depth[i, 0].contiguous()

                E_i, _, _, _, _ = lsac_triton(D_i, z_i)

                # Convert HWC -> CHW so the batch is [B, 3, H, W]
                E_i = E_i.permute(2, 0, 1).contiguous()

                E_batch.append(E_i)

        E = torch.stack(E_batch, dim=0)

        print("LSAC output:", E.shape, E.device)

        # ---------------------------------
        # 4. Estimate and refine beta
        # ---------------------------------

        beta0 = coarse_beta_gpu(E, depth)
        print("E:", E.shape)
        print("depth:", depth.shape)
        # print("beta_params:", beta_params.shape)
        # print("beta_map:", beta_map.shape)
        print("corrected:", corrected.shape)
        beta_params, _ = refine_beta_batch(depth, E, beta0)

        beta_map = beta_model_gpu(depth, beta_params)

        # ---------------------------------
        # 5. Reconstruct scene
        # ---------------------------------

        J = reconstruct_scene_gpu(corrected, depth, beta_map)

        show_reconstruction_results(image, J)


# %%

t0 = time.perf_counter()
main()
t1 = time.perf_counter() - t0

print(f"{t1}")
