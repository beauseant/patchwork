#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
EMBEDDER_PY="${EMBEDDER_PY:-${SCRIPT_DIR}/embedder.py}"

MODEL_NAME="${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
DEVICE="${DEVICE:-cuda}"
NORMALIZE_EMBEDDINGS="${NORMALIZE_EMBEDDINGS:-true}"

BATCH_SIZE="${BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"
ID_COL="${ID_COL:-place_id}"
TEXT_COL="${TEXT_COL:-generative_objective}"

SAVE_CHUNK_EMBEDDINGS="${SAVE_CHUNK_EMBEDDINGS:-false}"

PROCESSED_LOG="${PROCESSED_LOG:-${SCRIPT_DIR}/embed_processed.log}"

is_true() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

log_line() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" "$3" "${4:-}" >> "$PROCESSED_LOG"
}

already_ok_in_log() {
  local path="$1"
  [[ -f "$PROCESSED_LOG" ]] || return 1
  awk -F'\t' -v p="$path" '$2=="OK" && $3==p {found=1} END{exit(found?0:1)}' "$PROCESSED_LOG"
}

[[ -f "$EMBEDDER_PY" ]] || { echo "embedder.py no encontrado en: $EMBEDDER_PY" >&2; exit 1; }
touch "$PROCESSED_LOG"

COMMON_ARGS=(--model-name "$MODEL_NAME" --device "$DEVICE")
[[ -n "$TOKENIZER_MODEL" ]] && COMMON_ARGS+=(--tokenizer-model "$TOKENIZER_MODEL")
is_true "$NORMALIZE_EMBEDDINGS" || COMMON_ARGS+=(--no-normalize-embeddings)

CORPORA=(
  "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_outsiders_2024/final"
  "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_00/final"
  "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_01/final"
  "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_all_insiders_2024/part_02/final"
)

job_idx=0
for path_corpus in "${CORPORA[@]}"; do
  ((++job_idx))

  parent_dir="$(dirname "$path_corpus")"
  out_doc="${parent_dir}/embeddings.parquet" # save in parent dir

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $job_idx -> $path_corpus" >&2
  echo "  out-doc-parquet: $out_doc" >&2

  # Si ya existe la salida, lo consideramos procesado y saltamos
  if [[ -f "$out_doc" ]]; then
    echo "  SKIP (ya existe): $out_doc" >&2
    log_line "SKIP" "$path_corpus" "$out_doc" "output_exists"
    continue
  fi

  if already_ok_in_log "$path_corpus"; then
    echo "  SKIP (marcado OK en log): $path_corpus" >&2
    log_line "SKIP" "$path_corpus" "$out_doc" "already_ok_in_log"
    continue
  fi

  cmd=("$PYTHON_BIN" "$EMBEDDER_PY" "${COMMON_ARGS[@]}"
    --path-corpus "$path_corpus"
    --batch-size "$BATCH_SIZE"
    --chunk-size "$CHUNK_SIZE"
    --chunk-overlap "$CHUNK_OVERLAP"
    --id-col "$ID_COL"
    --text-col "$TEXT_COL"
    --out-doc-parquet "$out_doc"
  )

  is_true "$SAVE_CHUNK_EMBEDDINGS" && cmd+=(--save-chunk-embeddings)

  # Ejecuta y loguea estado
  if "${cmd[@]}"; then
    log_line "OK" "$path_corpus" "$out_doc" ""
  else
    log_line "FAIL" "$path_corpus" "$out_doc" "embedder_error"
    exit 1
  fi
done
