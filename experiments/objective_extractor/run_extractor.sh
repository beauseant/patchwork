#!/bin/bash

MODELS=("falcon3:10b-instruct-fp16" "mixtral:8x22b" "qwen3:8b" "qwen3:32b" "llama3.1:8b" "gemma2:9b" "deepseek-r1:8b" "gemma3:4b" "llama4:16x17b" "mistral:7b" "llama3.3:70b" "gpt-5-mini")
#MODELS=("llama3.1:8b")

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <source_directory> <destination_directory> <ollama_host>"
  exit 1
fi

source_directory="$1"
destination_directory="$2"
ollama_host="$3"

if [ ! -d "$source_directory" ]; then
  echo "Error: The source directory '$source_directory' does not exist."
  exit 1
fi
if [ ! -d "$destination_directory" ]; then
  echo "Error: The destination directory '$destination_directory' does not exist."
  exit 1
fi

REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT/backend/np-tools:${PYTHONPATH}"

for model in "${MODELS[@]}"; do
  echo "Available model: $model"
  sanitized_model="${model//:/-}"

  for file in "$source_directory"/*; do
    if [ -f "$file" ]; then
      file_name="$(basename "$file")"
      destination_file="$destination_directory/${file_name}_${sanitized_model}.parquet"

      if [ -e "$destination_file" ]; then
        echo "The file '$file_name' already exists in the destination directory, skipping."
      else
        echo "The file '$file_name' did not exist, processing it."

        python3 -m src.core.objective_extractor.extract \
          --config "$REPO_ROOT/backend/np-tools/src/core/objective_extractor/config/config.yaml" \
          --ollama_host "$ollama_host" \
          --path_to_parquet "$file" \
          --path_save "$destination_directory" \
          --llm_model_type_ex "$model" \
          --llm_model_type_gen "$model" \
          --mode_extractive_generative both
      fi
    fi
  done
done
