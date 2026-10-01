# %%

import torch

from dataset.dataset import CUDAPrefetcher, DataLoader, SeaThruDataset
from functions.fit_b_channel import estimate_backscatter_for_batch
from functions.functions import B_map_gpu, remove_backscatter_gpu

# %%

pairs = [("data/example.png", "data/example.tif")]

dataset = SeaThruDataset(pairs)
batch_size = 1

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
)

# %%

device = torch.device("cuda:0")
assert torch.cuda.is_available()

prefetcher = CUDAPrefetcher(loader, device)

while True:
    batch = prefetcher.next()

    if batch is None:
        break

    image = batch["image"]
    depth = batch["depth"]

    print("Image:", image.shape, image.device)
    print("Depth:", depth.shape, depth.device)

    params, losses = estimate_backscatter_for_batch(image, depth)
    depth_4d = depth.unsqueeze(1)

    with torch.no_grad():
        backscatter, corrected = remove_backscatter_gpu(
            I=image,
            z=depth_4d,
            params=params,
        )

    print("Parameters [B, C, 4]:")
    print(params)
    print("Fit losses [B, C]:")
    print(losses)

    # Display first image (batch size is 1).
    original_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
    backscatter_np = backscatter[0].detach().cpu().permute(1, 2, 0).numpy()
    corrected_np = corrected[0].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.clip(original_np, 0, 1))
    axes[0].set_title("Original")

    axes[1].imshow(np.clip(backscatter_np, 0, 1))
    axes[1].set_title("Estimated backscatter")

    axes[2].imshow(np.clip(corrected_np, 0, 1))
    axes[2].set_title("Backscatter removed")

    for ax in axes:
        x.axis("off")

    plt.tight_layout()
    plt.show()
