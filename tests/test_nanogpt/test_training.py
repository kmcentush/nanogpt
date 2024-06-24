from nanogpt.data.shakespeare import Shakespeare
from nanogpt.models.gpt2 import GPT, GPTConfig
from nanogpt.training import LRConfig, TrainConfig, train


def test_train():
    # Define sizes
    batch_size = 32  # total number of tokens per batch
    micro_batch_size = 4  # batches of tokens per dataset iteration
    sequence_length = 8  # sequence length of tokens per dataset iteration
    assert batch_size % (micro_batch_size * sequence_length) == 0  # ensure divisible or else data will get cropped

    # Get data and model
    train_data = Shakespeare(B=micro_batch_size, T=sequence_length, encoding_type="gpt2")
    model = GPT(GPTConfig(vocab_size=50304))  # make divisible by more powers of 2 for tiny speedup

    # Configure training and learning rate
    train_config = TrainConfig(num_epochs=1, num_batches=8, batch_size=batch_size)
    lr_config = LRConfig(warmup_steps=2, decay_steps=6)

    # Train; don't compile for testing
    train(model, train_data, train_config, lr_config=lr_config, compile=False, seed=0)
