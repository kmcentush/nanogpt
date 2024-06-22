from nanogpt.data.tokenizer import Tokenizer


def test_encode_decode():
    # Initialize
    tokenizer = Tokenizer("gpt2")
    text = "Hello world!"

    # Encode
    encoded = tokenizer.encode(text)
    assert encoded == [15496, 995, 0]

    # Decode
    decoded = tokenizer.decode(encoded)
    assert decoded == text
