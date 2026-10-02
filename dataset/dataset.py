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


# %%
class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader_iter = iter(loader)
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)

        self.next_batch = None
        self.next_event = None

        # Keep CPU batches alive until their asynchronous copies finish.
        self.host_batches_in_flight = []

        self.preload()

    def preload(self):
        # Release host batches whose DMA transfers have completed.
        self.host_batches_in_flight = [
            (event, batch)
            for event, batch in self.host_batches_in_flight
            if not event.query()
        ]

        try:
            host_batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            self.next_event = None
            return

        with torch.cuda.stream(self.copy_stream):
            image = host_batch["image"].to(
                self.device,
                non_blocking=True,
            )

            depth = host_batch["depth"].to(
                self.device,
                non_blocking=True,
            )

            # Record an event after both transfers have been enqueued.
            event = torch.cuda.Event()
            event.record(self.copy_stream)

        self.next_batch = {
            "image": image,
            "depth": depth,
            "image_path": host_batch["image_path"],
            "depth_path": host_batch["depth_path"],
        }
        self.next_event = event

        # Retain the pinned CPU tensors until the transfer has completed.
        self.host_batches_in_flight.append((event, host_batch))

    def next(self):
        if self.next_batch is None:
            return None

        current_stream = torch.cuda.current_stream(self.device)

        # Ensure this batch's transfer finishes before processing it.
        current_stream.wait_event(self.next_event)

        batch = self.next_batch

        # Tell PyTorch that these tensors are now used by current_stream.
        batch["image"].record_stream(current_stream)
        batch["depth"].record_stream(current_stream)

        # Enqueue the next batch's transfer before returning this batch.
        # The caller can then process this batch while that transfer runs.
        self.preload()

        return batch
