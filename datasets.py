import random
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from tokenizers import Tokenizer


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, block_size: int):
        # assert tokens.dtype == torch.long
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) // (self.block_size + 1)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]

        # TODO: figure out why the default collate function does weird batching when we don't return tensors
        return x, y


class TokenizerDataset(IterableDataset):
    def __init__(
        self,
        text_dataset: IterableDataset,
        tokenizer: Tokenizer,
        block_size: int,
    ):
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        for document in iter(self.text_dataset):
            tokens = self.tokenizer.encode(document)
            # TODO: append BOS/EOS
            x = tokens[: self.block_size]
            y = tokens[1 : self.block_size + 1]

            # TODO: append pad tokens to avoid throwing away data
            if len(y) != self.block_size:
                continue

            # TODO: do we need these tensors?
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class DolmaDataset(IterableDataset):
    def __init__(self, dolma_dir: Path, shuffle: bool):
        """
        Creates a DolmaDataset

        :param dolma_dir: directory containing dolma json.gz files
        :param shuffle: Enables approximate shuffling which will shuffle the shards and the examples within each shard
        """

        # TODO: support multiple workers by dividing up shards
        self.shard_paths = sorted(dolma_dir.glob("*.json.gz"))
        self.shuffle = shuffle

    def __iter__(self) -> Iterable[str]:
        shard_paths = (
            random.sample(self.shard_paths, k=len(self.shard_paths))
            if self.shuffle
            else self.shard_paths
        )
        for shard_path in shard_paths:
            print(f"Loading shard from {shard_path}...")
            shard = pd.read_json(shard_path, lines=True)
            if self.shuffle:
                print(f"Shuffling shard {shard_path}...")
                shard = shard.sample(frac=1)
            yield from shard["text"]
