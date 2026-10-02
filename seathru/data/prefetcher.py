# %%

import torch

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
