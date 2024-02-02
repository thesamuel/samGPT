import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn

from datasets import TokenDataset
from lm import LanguageModel
from tokenizers import CharacterTokenizer


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
    # TODO: for big data, we shouldn't load it into memory
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
            vocab_size=len(tokenizer), block_size=block_size, n_embd=n_embd, n_head=4
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        train_loop(train_loader, model, optimizer, eval_interval=eval_interval)

    empty_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(empty_context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
