import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from nanogpt.models.utils import get_device

if TYPE_CHECKING:
    from nanogpt.data.base import Dataset


@dataclass
class TrainConfig:
    num_epochs: int
    num_batches: int
    batch_size: int


@dataclass
class LRConfig:
    max_lr: float = 6e-4  # maximum learning rate
    min_lr: float = 6e-5  # minimum learning rate
    warmup_steps: int = 64  # linear warmup to maximum LR
    decay_steps: int = 512  # cosine decay to minimum LR


def get_lr(step: int, lr_config: LRConfig) -> float:
    # Linear warmup
    if step < lr_config.warmup_steps:
        return lr_config.max_lr * (step + 1) / lr_config.warmup_steps
    # Minimum after decaying
    if step > lr_config.decay_steps:
        return lr_config.min_lr
    # Cosine decay to minimum LR
    decay_ratio = (step - lr_config.warmup_steps) / (lr_config.decay_steps - lr_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))  # starts at 1 and goes to 0
    return lr_config.min_lr + coeff * (lr_config.max_lr - lr_config.min_lr)


def train(
    model: nn.Module,
    train_data: "Dataset",
    train_config: TrainConfig,
    lr_config: LRConfig | None = None,
    compile: bool = True,
    seed: int | None = None,
) -> nn.Module:
    # Get device
    device = get_device()

    # Maybe make reproducible
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    # Reduce precision to speed up training
    torch.set_float32_matmul_precision("high")  # TF32

    # Send model to device and maybe compile
    model.to(device)  # in-place
    if compile:  # pragma: no cover
        model = torch.compile(model)  # type: ignore[assignment]

    # Get optimizer
    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    # Handle micro batches
    num_micro_batches = train_config.batch_size // (train_data.batch_size * train_data.sequence_length)

    # Optimization loop
    print("Optimizing")
    for epoch in range(train_config.num_epochs):
        train_data.reset()
        for batch in range(train_config.num_batches):
            # TODO: remove time
            # Start time
            t0 = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Handle micro batches
            batch_loss = torch.zeros(1, dtype=torch.bfloat16).to(device)
            for _ in range(num_micro_batches):
                # Get data
                inputs, targets = next(train_data)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Calculate loss
                with torch.autocast(device_type=device, dtype=torch.bfloat16):  # reduce precision to speed up training
                    logits, loss = model(inputs, targets)
                # Scale the loss to account for gradient accumulation, because the gradients just add on each successive
                # backward(). Addition of gradients corresponds to a SUM in the objective, but instead of a SUM we want
                # MEAN. Scale the loss here so it comes out right.
                loss = loss / num_micro_batches
                batch_loss += loss.detach()  # detach from graph to only track value
                loss.backward()  # accumulates gradients (does a +=)

            # Clip gradients
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradient

            # Get and apply learning rate for this iteration
            if lr_config is not None:
                step = epoch * train_config.num_batches + batch
                lr = get_lr(step, lr_config)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Step and update parameters
            optimizer.step()

            # End time
            if device == "cuda":
                torch.cuda.synchronize()  # wait for GPU to finish
            t1 = time.time()
            dt = (t1 - t0) * 1000  # s -> ms

            # Print
            print(f"epoch {epoch}, batch {batch}, loss: {batch_loss.item():.3f}, norm: {norm:.3f}, dt: {dt:.2f}ms")

    return model
