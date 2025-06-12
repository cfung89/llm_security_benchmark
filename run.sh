#! /bin/bash

source .venv/bin/activate

inspect eval inspect_evals/gdm_intercode_ctf --model ollama/mistral-large
inspect eval inspect_evals/cybench --model ollama/mistral-large

inspect eval inspect_evals/gdm_intercode_ctf --model ollama/gemma2
inspect eval inspect_evals/cybench --model ollama/gemma2

if ! [[ -d "./output" ]]; then
    mkdir output
fi
inspect log convert --to json --output-dir ./output ./logs
