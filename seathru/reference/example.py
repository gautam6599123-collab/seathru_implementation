# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt

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

# %%


t1 = time.perf_counter()

image_path = "../../data/example.png"
depth_path = "../../data/example.tif"

I = load_img(Path(image_path))
z = load_depth(Path(depth_path))

params, _, _ = estimate_backscatter(I, z)
B, D = remove_backscatter(I, z, params)
plt.imshow(linear_to_srgb(I))

plt.imshow(linear_to_srgb(D))


E, a, eps, iters, hist = lsac(D, z, verbose=False)


beta0 = coarse_beta(E, z)


beta_map = refine_beta(z, E, beta0, seed=0)


J = reconstruct_scene(D, z, beta_map)


Js, _, _ = gray_world(J)


plt.imshow(linear_to_srgb(Js))

t2 = t1 - time.perf_counter()

print(f"{t2}")
