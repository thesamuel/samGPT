import pandas as pd
import torch
import torchdata.datapipes as dp
from torch.utils.data import Dataset
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

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


@functional_datapipe("tokenize")
class TokenizerDataset(IterDataPipe):
    def __init__(self, dp: IterDataPipe, tokenizer: Tokenizer, block_size: int) -> None:
        self.dp = dp
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        for document in self.dp:
            tokens = self.tokenizer.encode(document)
            # TODO: append BOS/EOS
            x = tokens[: self.block_size]
            y = tokens[1 : self.block_size + 1]

            # TODO: append pad tokens to avoid throwing away data
            if len(y) != self.block_size:
                continue

            # TODO: do we need these tensors?
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


@functional_datapipe("read_jsonlines")
class JsonLinesIterDataPipe(IterDataPipe):
    def __init__(self, dp: IterDataPipe) -> None:
        self.dp = dp

    def __iter__(self):
        for file in self.dp:
            # TODO: have this return records instead of the specific column we want
            yield pd.read_json(file, lines=True)["text"].tolist()


def load_dolma(root: str, file_shuffle_buffer: int = 1, prefetch_buffer: int = 2):
    return (
        dp.iter.FileLister(root, recursive=True, masks="*.json.gz")
        .shuffle(buffer_size=file_shuffle_buffer)
        .sharding_filter()  # Shard after shuffling
        .read_jsonlines()
        .prefetch(prefetch_buffer)  # Prefetch N files to avoid GPU starvation
        .unbatch()
    )


def test_functional_dataset():
    first_row = next(iter(load_dolma("data/dolma/v1_6-sample")))
    print(first_row)
    assert isinstance(first_row, str)
    assert first_row.startswith("LABOR'S MARTYRS")
