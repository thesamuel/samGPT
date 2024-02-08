# samGPT

A reimplementation of [nanoGPT](https://github.com/karpathy/nanoGPT). Current prototype trains on the AI2
[Dolma](https://huggingface.co/datasets/allenai/dolma) dataset.

Right now everything is very simple:
- Character tokenizer
- Multi-head attention in vanilla PyTorch
- Single worker dataloader

Some planned improvements:
- Add multiple workers
- Investigate tweaks to Transformer (eg. flash attention)
- BPE tokenizer
