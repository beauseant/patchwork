#!/bin/bash
set -euo pipefail

#MODELS=("llama3.1:8b" "mixtral:8x22b" "falcon3:10b-instruct-fp16" "qwen3:8b" "qwen3:32b"  "gemma2:9b" "deepseek-r1:8b" "gemma3:4b" "llama4:16x17b" "mistral:7b" "llama3.3:70b" "gpt-5-mini")

MODELS=("llama3.3:70b" )

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <source_directory> <destination_directory> <ollama_host>"
  exit 1
fi

src_dir="$1"
dest_dir="$2"
ollama_host="$3"

# Logs and dirs
RUN_LOG="$dest_dir/runs_detail.log"        # timeline log (start/end + details)
DONE_LOG="$dest_dir/processed_models.log"  # idempotency ledger
CHECKPOINT_DIR="$dest_dir/checkpoints"
FINAL_DIR="$dest_dir/final"
LOG_DIR="$dest_dir/logs"

mkdir -p "$dest_dir" "$CHECKPOINT_DIR" "$FINAL_DIR" "$LOG_DIR"
touch "$RUN_LOG" "$DONE_LOG"

# Repo env
REPO_ROOT="$(pwd)"
# Safe expansion: append only if PYTHONPATH is set
export PYTHONPATH="$REPO_ROOT/backend/np-tools${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# Count parquet rows (via pandas)
count_rows() {
  python3 - "$1" <<'PY'
import sys, pandas as pd
p=sys.argv[1]
try:
    print(len(pd.read_parquet(p)))
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(2)
PY
}

for model in "${MODELS[@]}"; do
  sanitized_model="${model//:/-}"

  for infile in "$src_dir"/*; do
    [ -f "$infile" ] || continue

    fname="$(basename "$infile")"
    checkpoint_file="$CHECKPOINT_DIR/${fname}_${sanitized_model}.checkpoint.parquet"
    final_file="$FINAL_DIR/${fname}_${sanitized_model}.parquet"
    run_log="$LOG_DIR/${fname}_${sanitized_model}.log"
    entry_key="${fname}|${model}"

    # Idempotent skip if already recorded as done
    if grep -Fxq "$entry_key" "$DONE_LOG"; then
      echo "[INFO] $(timestamp) already done: $entry_key"
      continue
    fi

    # If a final file exists and matches input rows, record and skip
    if [ -f "$final_file" ]; then
      in_rows="$(count_rows "$infile" || true)"
      out_rows="$(count_rows "$final_file" || true)"
      if [[ "$in_rows" == "$out_rows" && -n "$in_rows" ]]; then
        echo "$entry_key" >> "$DONE_LOG"
        echo "[INFO] $(timestamp) found complete final: $entry_key rows=$out_rows"
        continue
      fi
    fi

    echo "[START] $(timestamp) model=${model} file=${fname}" | tee -a "$RUN_LOG"

    in_rows="$(count_rows "$infile" || true)"
    if [[ "$in_rows" == ERROR:* || -z "$in_rows" ]]; then
      echo "[FAIL ] $(timestamp) model=${model} file=${fname} reason='cannot read input parquet'" | tee -a "$RUN_LOG"
      continue
    fi

    # Clean any stale checkpoint for a fresh attempt
    rm -f "$checkpoint_file"

    # Run: write only to checkpoint, capture full Python logs
    # FAULTHANDLER + unbuffered so tracebacks flush immediately
    if PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
      python3 -m src.core.objective_extractor.extract \
        --config "$REPO_ROOT/backend/np-tools/src/core/objective_extractor/config/config_o.yaml" \
        --ollama_host "$ollama_host" \
        --path_to_parquet "$infile" \
        --path_save "$checkpoint_file" \
        --llm_model_type_ex "$model" \
        --llm_model_type_gen "$model" \
        --mode_extractive_generative both \ 
        #--mode_extractive_generative generative \ 
        --enable_checkpoints >"$run_log" 2>&1; then

      if [ ! -f "$checkpoint_file" ]; then
        echo "[FAIL ] $(timestamp) model=${model} file=${fname} reason='missing checkpoint after success'" | tee -a "$RUN_LOG"
        echo "---- Tail of $run_log ----"
        tail -n 50 "$run_log" || true
        echo "--------------------------"
        continue
      fi

      out_rows="$(count_rows "$checkpoint_file" || true)"
      if [[ "$out_rows" != "$in_rows" || -z "$out_rows" ]]; then
        echo "[WARN ] $(timestamp) model=${model} file=${fname} rows_in=${in_rows} rows_out=${out_rows} (mismatch) — keeping checkpoint, not publishing" | tee -a "$RUN_LOG"
        echo "---- Tail of $run_log ----"
        tail -n 50 "$run_log" || true
        echo "--------------------------"
        continue
      fi

      # Atomic publish to final
      mv -f "$checkpoint_file" "$final_file"
      echo "$entry_key" >> "$DONE_LOG"
      echo "[END  ] $(timestamp) model=${model} file=${fname} status=SUCCESS rows=${out_rows} saved=$(basename "$final_file")" | tee -a "$RUN_LOG"
    else
      echo "[FAIL ] $(timestamp) model=${model} file=${fname} status=ERROR (python exit)" | tee -a "$RUN_LOG"
      echo "---- Tail of $run_log ----"
      tail -n 50 "$run_log" || true
      echo "--------------------------"
    fi
  done
done