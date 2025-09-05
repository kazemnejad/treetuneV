#! /bin/bash

set -e

models=(
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
  'Qwen/Qwen3-1.7B'
)

datasets=(
  'zwhe99/DeepMath-103K'
  'agentica-org/DeepScaleR-Preview-Dataset'
  'HuggingFaceH4/aime_2024'
  'MathArena/aime_2025'
  'realtreetune/konkur'
  'realtreetune/konkur-filtered'
)

for model in "${models[@]}"; do
  echo "Downloading $model"
  hf download "$model" --max-workers 8
done

for dataset in "${datasets[@]}"; do
  echo "Downloading $dataset"
  hf download --repo-type dataset "$dataset" --max-workers 8
  python -c "from datasets import load_dataset; load_dataset('$dataset', num_proc=8)"
done