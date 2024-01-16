import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import nn

from lm import LanguageModel


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


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    eval_interval: int,
):
    size = len(dataloader.dataset)
    model.train()
    for batch_i, (batch_x, batch_y) in enumerate(dataloader):
        logits, loss = model(batch_x, batch_y)

        if batch_i % eval_interval == 0:
            print(f"Batch {batch_i * len(batch_x)}/{size} Loss: {loss.item()}")

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def main(
    text_file: str = "data/shakespeare.txt",
    train_pct: float = 0.9,
    block_size=32,
    batch_size: int = 16,
    num_workers: int = 0,
    n_embd: int = 64,
    eval_interval: int = 100,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
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
    # TODO: create validation loader
    val = data[n:]

    train_dataset = TokenDataset(train, block_size=block_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    # Print a sample instance
    sample_train_batch = next(iter(train_loader))
    print(
        f"""First instance from train loader: 
        x = {sample_train_batch[0][0]}
        y = {sample_train_batch[1][0]}"""
    )

    # TODO: verify that this works on GPU
    with torch.device(device):
        model = LanguageModel(
            vocab_size=len(tokenizer), block_size=block_size, n_embd=n_embd
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        train_loop(train_loader, model, optimizer, eval_interval=eval_interval)


if __name__ == "__main__":
    main()
