import os

import torch

from nanogpt.data.base import DATA_DIR, Dataset
from nanogpt.data.tokenizer import Tokenizer

# Specify paths
# Downloaded from: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
FILE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")


def _load_data() -> str:
    with open(FILE_PATH) as f:
        return f.read()


class Shakespeare(Dataset):
    def __init__(
        self,
        B: int,
        T: int,
        encoding_type: str = "gpt2",
        character_limit: int | None = None,
        infinite_iter: bool = False,
    ):
        # Initialize
        super().__init__()
        self.B = B
        self.T = T
        self.position = 0
        self.infinite_iter = infinite_iter

        # Load data and tokenize
        # Note: all data is in memory
        self.tokenizer = Tokenizer(encoding_type)
        self.raw = _load_data()
        if character_limit:
            self.raw = self.raw[:character_limit]
        self.tokens = torch.tensor(self.tokenizer.encode(self.raw), dtype=torch.long)

    @property
    def batches_per_epoch(self) -> int:
        return len(self.tokens) // (self.B * self.T)

    @property
    def batch_size(self) -> int:
        return self.B

    @property
    def sequence_length(self) -> int:
        return self.T

    def reset(self):
        self.position = 0

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check position
        if self.position > len(self.tokens):  # pragma: no cover
            if self.infinite_iter:
                self.reset()
            else:
                raise StopIteration

        # Get buffer
        B = self.B
        T = self.T
        buffer = self.tokens[self.position : self.position + B * T + 1]  # + 1 is for last target token

        # Get views
        inputs = buffer[:-1].view(B, T)
        targets = buffer[1:].view(B, T)

        # Advance position
        self.position += B * T

        return inputs, targets
