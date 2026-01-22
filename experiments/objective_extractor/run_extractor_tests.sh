REPO_ROOT="$(pwd)"
# Safe expansion: append only if PYTHONPATH is set
export PYTHONPATH="$REPO_ROOT/backend/np-tools${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"

PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 python3 -m src.core.objective_extractor.extract \
--config "/export/usuarios_ml4ds/lbartolome/Repos/patchwork/backend/np-tools/src/core/objective_extractor/config/config_o.yaml" \
--ollama_host "http://kumo01.tsc.uc3m.es:11434" \
--path_to_parquet "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/data//insiders_outsiders_500_500.parquet" \
--path_save "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results_no_limit/checkpoints/insiders_outsiders_500_500.parquet_qwen3-32b.checkpoint.parquet" \
--llm_model_type_ex "qwen3:32b" \
--llm_model_type_gen "qwen3:32b" \
--mode_extractive_generative generative \
--enable_checkpoints