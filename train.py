import random
from collections.abc import Sized
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import TokenizerDataset, DolmaDataset
from lm import LanguageModel
from tokenizers import CharacterTokenizer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_text(input_file: str) -> str:
    with open(input_file) as f:
        return f.read()


def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    eval_interval: int,
    device: str,
):
    size = len(dataloader.dataset) if isinstance(dataloader.dataset, Sized) else None

    model.train()
    for batch_i, (batch_x, batch_y) in enumerate(dataloader):
        logits, loss = model(batch_x.to(device), batch_y.to(device))

        if batch_i % eval_interval == 0:
            print(
                f"Batch {batch_i * len(batch_x)}/{size or 'unknown'} Loss: {loss.item()}"
            )

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
    device: str = "cpu",
):
    # Set seeds for reproducibility
    set_seed(42)

    # Initialize Dolma Datasets
    dolma_path = Path("data/dolma")
    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = DolmaDataset(dolma_path / split, shuffle=True)

    # TODO: increase number of tokenizer training examples, implement BPE tokenizer
    # Train the tokenizer on our text
    tokenizer_train = list(islice(iter(datasets["train"]), 10_000))
    tokenizer = CharacterTokenizer(tokenizer_train)
    print("Vocab:", tokenizer.vocab[:100])

    # Create a dataset that streams to tokens
    train_dataset = TokenizerDataset(
        text_dataset=datasets["train"],
        tokenizer=tokenizer,
        block_size=block_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,  # NOTE: this is not supported on MPS (Mac accelerated training)
        num_workers=num_workers,
        # NOTE: we don't specify a shuffle parameter since we use a streaming dataset with imperfect shuffling above
    )

    # Print a sample instance
    sample_train_batch = next(iter(train_loader))
    print(
        f"""First instance from train loader: 
        x = {sample_train_batch[0][0]}
        y = {sample_train_batch[1][0]}"""
    )

    model = LanguageModel(
        vocab_size=len(tokenizer),
        block_size=block_size,
        n_embd=n_embd,
        n_head=4,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        train_loop(
            train_loader,
            model,
            optimizer,
            eval_interval=eval_interval,
            device=device,
        )

    empty_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(
        tokenizer.decode(model.generate(empty_context, max_new_tokens=500)[0].tolist())
    )


if __name__ == "__main__":
    main()
