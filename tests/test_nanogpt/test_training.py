from nanogpt.data.shakespeare import Shakespeare
from nanogpt.models.gpt2 import GPT, GPTConfig
from nanogpt.training import LRConfig, train


def test_train():
    # Get data and model
    train_data = Shakespeare(B=6, T=32, encoding_type="gpt2")
    model = GPT(GPTConfig(vocab_size=50304))  # make divisible by more powers of 2 for tiny speedup

    # Configure learning rate
    lr_config = LRConfig(warmup_steps=2, decay_steps=6)

    # Train; don't compile for testing
    train(model, train_data, num_epochs=1, num_batches=8, lr_config=lr_config, compile=False, seed=0)
