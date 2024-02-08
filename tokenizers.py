from abc import ABC, abstractmethod


# Consider using https://pytorch.org/text/stable/vocab.html
class Tokenizer(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    vocab: list[str]
    unk_token: str = "<UNK>"

    def __init__(self, corpus: str | list[str]):
        if isinstance(corpus, str):
            self.vocab = sorted(set(corpus))
        else:
            vocab_set = set()
            for document in corpus:
                vocab_set.update(set(document))
            self.vocab = sorted(vocab_set)

        # Add unk token for out-of-vocabulary characters
        self.vocab.append(self.unk_token)
        self.unk_idx = len(self.vocab) - 1

        # String to integer mapping
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def __getitem__(self, item: int) -> str:
        return self.vocab[item]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_idx) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocab[i] for i in tokens)
