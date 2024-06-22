import pytest
import tiktoken
import torch
from nanogpt.models.gpt2 import GPT
from nanogpt.models.utils import get_device
from torch.nn import functional as F


@pytest.fixture(scope="session")
def gpt2() -> GPT:
    return GPT.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def device() -> str:
    return get_device()


def test_from_pretrained(gpt2: GPT):
    assert gpt2


def test_inference(gpt2: GPT, device: str):
    # Specify settings
    num_return_sequences = 5  # B
    max_length = 30

    # Prepare for inference
    gpt2.eval()
    gpt2.to(device)

    # Prepare prefix tokens (our input)
    enc = tiktoken.get_encoding("gpt2")
    tokens_py = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens_py, dtype=torch.long)  # (T,); 8 in this particular case
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B, T)
    x = tokens.to(device)

    # Generate; `x` starts as (B, T)
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():  # tells PyTorch we won't be calling backward(); saves memory and time
            logits, _ = gpt2(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline defaults)
            # topk_probs becomes (B, 50), top_k_indices is the same size
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, dim=-1, index=ix)  # (B, 1)
            # append to the sequence
            x = torch.cat([x, xcol], dim=-1)

    # Convert back to text
    out = []
    for i in range(num_return_sequences):
        tokens_py = x[i, :max_length].tolist()
        decoded = enc.decode(tokens_py)
        out.append(decoded)

    # Validate
    exp_out = [
        "Hello, I'm a language model, not a program.\n\nSo this morning I started studying for the interview in the lab"
        ". This was not",
        "Hello, I'm a language model, and one of the main things that bothers me when they create languages is how easy"
        " it becomes to create something that",
        "Hello, I'm a language model, and I wrote it off on the grounds that a language model would make me more fluent"
        ". But I'm not",
        "Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way"
        " we talk about languages",
        "Hello, I'm a language model, a language model I'm using for data modelling. All I did was test the results and"
        " then I wrote some",
    ]
    assert out == exp_out
