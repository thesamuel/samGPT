#!/usr/bin/env bash

DOLMA_VERSION="v1_6-sample"
DATA_DIR="dolma/$DOLMA_VERSION"
PARALLEL_DOWNLOADS=8

git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"

parallel --progress -j "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR" :::: "dolma/urls/${DOLMA_VERSION}.txt"
