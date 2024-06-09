import random
import timeit
from itertools import islice
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dolma
from lm import LanguageModel
from tokenizers import CharacterTokenizer, Tokenizer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def show_generations(model: LanguageModel, tokenizer: Tokenizer, device: str):
    with torch.no_grad():
        empty_context = torch.zeros((1, 1), dtype=torch.long, device=device)
        start_time = timeit.default_timer()
        generation = tokenizer.decode(
            model.generate(empty_context, max_new_tokens=500)[0].tolist()
        )
        elapsed = timeit.default_timer() - start_time
        print(f"Generation (took {elapsed:3f} seconds):", generation)


def train_loop(
    dataloader: DataLoader,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    eval_interval: int,
    device: str,
    tokenizer: Optional[Tokenizer] = None,
    generate_at_eval: bool = True,
):
    model.train()
    batch_x: torch.Tensor
    batch_y: torch.Tensor
    for batch_i, (batch_x, batch_y) in enumerate(dataloader):
        logits, loss = model(batch_x.to(device), batch_y.to(device))

        if batch_i % eval_interval == 0:
            print(f"Batch {batch_i * len(batch_x)}, Loss: {loss.item()}")
            if generate_at_eval and tokenizer:
                model.eval()
                show_generations(model, tokenizer, device)
                model.train()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def main(
    block_size: int = 32,
    batch_size: int = 16,
    n_embd: int = 64,
    num_workers: int = 4,
    eval_interval: int = 100,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: str = "cpu",
):
    # Set seeds for reproducibility
    set_seed(42)

    # Initialize Dolma Datasets
    dolma_datapipe = load_dolma("data/dolma/dolma-v1_6-8B-sample")

    # TODO: increase number of tokenizer training examples, implement BPE tokenizer
    # Train the tokenizer on our text
    tokenizer_train = list(islice(iter(dolma_datapipe), 10_000))
    tokenizer = CharacterTokenizer(tokenizer_train)
    print("Vocab:", tokenizer.vocab[:100])

    # Create a dataset that streams to tokens
    tokenized_datapipe = dolma_datapipe.tokenize(
        tokenizer=tokenizer, block_size=block_size
    )
    train_loader = DataLoader(
        tokenized_datapipe,
        batch_size=batch_size,
        pin_memory=True,  # NOTE: this is not supported on MPS (Mac accelerated training)
        num_workers=num_workers,
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
            tokenizer=tokenizer,
            device=device,
        )

    empty_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(
        tokenizer.decode(model.generate(empty_context, max_new_tokens=500)[0].tolist())
    )


if __name__ == "__main__":
    main()
