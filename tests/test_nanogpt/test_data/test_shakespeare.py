from nanogpt.data.shakespeare import Shakespeare


def test_next_batch():
    # Initialize
    B = 4
    T = 6
    data = Shakespeare(B=B, T=T, encoding_type="gpt2", character_limit=1000)

    # Get one batch
    inputs, targets = next(data)

    # Validate
    assert data.batches_per_epoch == 11
    assert inputs.shape == (B, T)
    assert targets.shape == (B, T)
    assert (inputs.view(-1)[1:] == targets.view(-1)[:-1]).all()
    assert inputs[0, 0] == 5962
    assert targets[-1, -1] == 198
