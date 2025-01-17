#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
conda init
conda activate codeRLEval