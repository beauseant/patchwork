#!/bin/bash

PATHS=(
    /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/df_cpv5_to_lemmatize.parquet
)

DESTINATION_DIR="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/"

echo "Starting batch preprocessing..."

for SOURCE_PATH in "${PATHS[@]}"; do
    FILE_NAME=$(basename "$SOURCE_PATH" .parquet)_preproc.parquet
    DEST_PATH="$DESTINATION_DIR/$FILE_NAME"

    echo "-------------------------------------------"
    echo "Processing: $SOURCE_PATH"
    echo "Saving to:  $DEST_PATH"
    echo "Running preprocessing pipeline..."

    python /export/usuarios_ml4ds/lbartolome/Repos/my_repos/DomAIn-Analyzer/src/NLPipe/nlpipe.py \
    --source_path "$SOURCE_PATH" \
    --source_type parquet \
    --source place \
    --destination_path "$DEST_PATH" \
    --lang es \
    --spacy_model es_core_news_lg \
    --config_file /export/usuarios_ml4ds/lbartolome/Repos/patchwork/data_preparation/config_prepoc.json \
    --do_embeddings

    if [ $? -eq 0 ]; then
        echo "Successfully processed: $FILE_NAME"
    else
        echo "Failed to process: $FILE_NAME"
    fi
done

echo "-------------------------------------------"
echo "Batch processing complete!"