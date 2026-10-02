# %%

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image

# %%


class SeaThruDataset(Dataset):
    def __init__(self, pairs):
        """
        pairs: list of (image_path, depth_path)
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, depth_path = self.pairs[index]
        # Load RGB PNG using TorchCodec
        image = decode_image(
            str(image_path),
            mode="RGB",
            output_dtype="auto",
        )

        # CHW, integer -> CHW, float32 in [0, 1]
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0

        elif image.dtype == torch.uint16:
            image = image.to(torch.float32) / 65535.0

        else:
            raise TypeError(f"Unexpected image dtype: {image.dtype} for {image_path}")

        # Load depth TIFF
        depth = tifffile.imread(depth_path)
        depth = np.asarray(depth, dtype=np.float32)
        # Zero means invalid depth
        depth[depth == 0.0] = np.nan
        # H,W -> 1,H,W
        depth = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0)
        # Return CPU tensors
        return {
            "image": image.contiguous(),
            "depth": depth.contiguous(),
            "image_path": str(image_path),
            "depth_path": str(depth_path),
        }
