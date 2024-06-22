import tiktoken


class Tokenizer:
    def __init__(self, encoding_type: str):
        self.encoding = tiktoken.get_encoding(encoding_type)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)
