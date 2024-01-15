import torch
from torch.utils.data import Dataset, DataLoader


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


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        assert tokens.dtype == torch.long
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) // (self.block_size + 1)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]

        # TODO: figure out why the default collate function does weird batching when we don't return tensors
        return x, y


def load_text(input_file: str) -> str:
    with open(input_file) as f:
        return f.read()


def main(
    text_file: str = "data/shakespeare.txt",
    train_pct: float = 0.9,
    block_size=32,
    batch_size: int = 16,
    num_workers: int = 0,
):
    text = load_text(text_file)

    # Train the tokenizer on our text
    tokenizer = CharacterTokenizer(text)
    print("Vocab:", tokenizer.vocab)

    # Encode our text and convert to a tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print("Tokens:", data[:100])

    # Create a validation set
    n = int(train_pct * len(data))
    train = data[:n]
    val = data[n:]

    train_dataset = TokenDataset(train, block_size=block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    for batch, labels in train_loader:
        print("Batch from train loader:", batch[0])
        print("Labels from train loader:", labels[0])
        break


if __name__ == "__main__":
    main()
