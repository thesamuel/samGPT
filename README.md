# samGPT

A reimplementation of [nanoGPT](https://github.com/karpathy/nanoGPT). Current prototype trains on the AI2
[Dolma](https://huggingface.co/datasets/allenai/dolma) dataset.

Features:
- [x] Character tokenizer
- [ ] BPE tokenizer
- [x] Multi-head attention in vanilla PyTorch
- [x] Single worker dataloader
- [x] Multi-worker dataloader
- [ ] Flash attention
- [ ] WIP - KV Cache
