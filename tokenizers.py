class CharacterTokenizer:
    vocab: list[str]

    # TODO: support passing in a corpus
    def __init__(self, text: str):
        self.vocab = sorted(list(set(text)))

        # String to integer mapping
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        return self.vocab[item]

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab[i] for i in tokens)
