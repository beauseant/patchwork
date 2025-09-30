#!/bin/bash
set -euo pipefail  # fail fast, catch unset vars, propagate pipe errors
# set -x           # <- uncomment if you also want bash to echo each command

PATHS=(
    /Users/lbartolome/salvando_la_tesis/data/LicsExtracted_Aug_6/lics_for_topicmodeling.parquet
)

DESTINATION_DIR="/Users/lbartolome/salvando_la_tesis/data/LicsExtracted_Aug_6/preprocessed"

echo "Starting batch preprocessing..."
mkdir -p "$DESTINATION_DIR"

for SOURCE_PATH in "${PATHS[@]}"; do
    FILE_NAME="$(basename "$SOURCE_PATH" .parquet)_preproc.parquet"
    DEST_PATH="$DESTINATION_DIR/$FILE_NAME"
    LOG_FILE="${DEST_PATH}.log"

    echo "-------------------------------------------"
    echo "Processing: $SOURCE_PATH"
    echo "Saving to:  $DEST_PATH"
    echo "Log file:   $LOG_FILE"
    echo "Running preprocessing pipeline..."
    echo

    # Show the exact command (for copy/paste)
    echo ">> python3 -u nlpipe.py --source_path \"$SOURCE_PATH\" --source_type parquet --source pliegos --destination_path \"$DEST_PATH\" --lang es --spacy_model es_core_news_lg --config_file /Users/lbartolome/salvando_la_tesis/NLPipe/config.json"
    echo

    # Run Python unbuffered so logs appear immediately, mirror to terminal and log file
    PYTHONUNBUFFERED=1 python3 -u nlpipe.py \
        --source_path "$SOURCE_PATH" \
        --source_type parquet \
        --source pliegos \
        --destination_path "$DEST_PATH" \
        --lang es \
        --spacy_model en_core_web_lg \
        --config_file /Users/lbartolome/salvando_la_tesis/NLPipe/config.json \
        --do_embeddings \
        2>&1 | tee "$LOG_FILE"

    status=${PIPESTATUS[0]}  # exit code of python in the pipeline

    if [ $status -eq 0 ]; then
        echo "Successfully processed: $FILE_NAME"
    else
        echo "Failed to process: $FILE_NAME (exit code $status). See $LOG_FILE"
    fi
done

echo "-------------------------------------------"
echo "Batch processing complete!"
