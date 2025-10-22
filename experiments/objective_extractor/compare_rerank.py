
import argparse
import json
import pathlib
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
import os

repo_root = os.getcwd()
np_tools_path = os.path.join(repo_root, "backend", "np-tools")

if np_tools_path not in sys.path:
    print(np_tools_path)
    sys.path.insert(0, np_tools_path)

from src.core.objective_extractor.file_utils import init_logger, load_yaml_config_file
from src.core.objective_extractor.extract import ObjectiveExtractor

def paired_bootstrap(a, b, n=5000, seed=123):
    """Paired bootstrap CI for mean(a - b). a, b = 1D arrays (same length)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]; b = b[mask]
    m = len(a)
    if m == 0:
        return None
    idx = rng.integers(0, m, size=(n, m))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return np.percentile(diffs, [2.5, 50, 97.5])


def win_rate(a, b):
    """Fraction of rows with a > b (ignoring NaNs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() == 0:
        return math.nan
    return float((a[mask] > b[mask]).mean())


def evaluate_row_bm25_matrix(extractor, text, gold):
    """
    For a single row, run:
      - BM25 ON + (rerank ON/OFF)
      - BM25 OFF + (rerank ON/OFF)
    using extractor._prepare_both_contexts(text) in each BM25 setting,
    then run extract() (extractive & generative) on both contexts.

    Returns a dict with metrics + objectives.
    """
    out = {}
    bm25_orig = getattr(extractor, "enable_bm25", False)

    def run_for_bm25(flag, prefix):
        extractor.enable_bm25 = bool(flag)

        # Build contexts for rerank ON/OFF under this BM25 setting
        (ctx_on, meta_on), (ctx_off, meta_off) = extractor._prepare_both_contexts(text)

        # ---- Extractive
        t0 = time.time()
        obj_e_on, _, bP_e_on, bR_e_on, bF_e_on, tP_e_on, tR_e_on, tF_e_on, _ = extractor.extract(
            text, gold_objective=gold, option="extractive",
            precomputed_context=(ctx_on, meta_on)
        )
        ext_time_on = time.time() - t0

        t0 = time.time()
        obj_e_off, _, bP_e_off, bR_e_off, bF_e_off, tP_e_off, tR_e_off, tF_e_off, _ = extractor.extract(
            text, gold_objective=gold, option="extractive",
            precomputed_context=(ctx_off, meta_off)
        )
        ext_time_off = time.time() - t0

        # ---- Generative
        t0 = time.time()
        obj_g_on, _, bP_g_on, bR_g_on, bF_g_on, tP_g_on, tR_g_on, tF_g_on, _ = extractor.extract(
            text, gold_objective=gold, option="generative",
            precomputed_context=(ctx_on, meta_on)
        )
        gen_time_on = time.time() - t0

        t0 = time.time()
        obj_g_off, _, bP_g_off, bR_g_off, bF_g_off, tP_g_off, tR_g_off, tF_g_off, _ = extractor.extract(
            text, gold_objective=gold, option="generative",
            precomputed_context=(ctx_off, meta_off)
        )
        gen_time_off = time.time() - t0

        # Scores & times
        out.update({
            # Extractive (rerank ON)
            f"{prefix}extractive_objective_rerank_on": obj_e_on,
            f"{prefix}extractive_bert_precision_rerank_on": bP_e_on,
            f"{prefix}extractive_bert_recall_rerank_on": bR_e_on,
            f"{prefix}extractive_bert_f1_rerank_on": bF_e_on,
            f"{prefix}extractive_token_precision_rerank_on": tP_e_on,
            f"{prefix}extractive_token_recall_rerank_on": tR_e_on,
            f"{prefix}extractive_token_f1_rerank_on": tF_e_on,
            f"{prefix}extractive_time_seconds_rerank_on": ext_time_on,
            # Extractive (rerank OFF)
            f"{prefix}extractive_objective_rerank_off": obj_e_off,
            f"{prefix}extractive_bert_precision_rerank_off": bP_e_off,
            f"{prefix}extractive_bert_recall_rerank_off": bR_e_off,
            f"{prefix}extractive_bert_f1_rerank_off": bF_e_off,
            f"{prefix}extractive_token_precision_rerank_off": tP_e_off,
            f"{prefix}extractive_token_recall_rerank_off": tR_e_off,
            f"{prefix}extractive_token_f1_rerank_off": tF_e_off,
            f"{prefix}extractive_time_seconds_rerank_off": ext_time_off,

            # Generative (rerank ON)
            f"{prefix}generative_objective_rerank_on": obj_g_on,
            f"{prefix}generative_bert_precision_rerank_on": bP_g_on,
            f"{prefix}generative_bert_recall_rerank_on": bR_g_on,
            f"{prefix}generative_bert_f1_rerank_on": bF_g_on,
            f"{prefix}generative_token_precision_rerank_on": tP_g_on,
            f"{prefix}generative_token_recall_rerank_on": tR_g_on,
            f"{prefix}generative_token_f1_rerank_on": tF_g_on,
            f"{prefix}generative_time_seconds_rerank_on": gen_time_on,
            # Generative (rerank OFF)
            f"{prefix}generative_objective_rerank_off": obj_g_off,
            f"{prefix}generative_bert_precision_rerank_off": bP_g_off,
            f"{prefix}generative_bert_recall_rerank_off": bR_g_off,
            f"{prefix}generative_bert_f1_rerank_off": bF_g_off,
            f"{prefix}generative_token_precision_rerank_off": tP_g_off,
            f"{prefix}generative_token_recall_rerank_off": tR_g_off,
            f"{prefix}generative_token_f1_rerank_off": tF_g_off,
            f"{prefix}generative_time_seconds_rerank_off": gen_time_off,
        })

        # Minimal meta of interest (namespaced)
        def _get(d, k, default=None):
            try:
                return d.get(k, default) if isinstance(d, dict) else default
            except Exception:
                return default

        meta_keys = [
            "bm25_used", "bm25_reasons", "regex_anchor_found", "top_dense_score",
            "total_nodes", "retriever_anchor_found", "query_strategy"
        ]
        for k in meta_keys:
            out[f"{prefix}meta_on_{k}"] = _get(meta_on, k, None)
            out[f"{prefix}meta_off_{k}"] = _get(meta_off, k, None)

        out[f"{prefix}rerank_changed_top1_on"]  = _get(meta_on,  "rerank_changed_top1", False)
        out[f"{prefix}rerank_changed_top1_off"] = _get(meta_off, "rerank_changed_top1", False)

    # Run both settings
    run_for_bm25(True,  "bm25_on_")
    run_for_bm25(False, "bm25_off_")

    extractor.enable_bm25 = bm25_orig
    return out


def main():
    ap = argparse.ArgumentParser(description="BM25 ON/OFF and Rerank ON/OFF eval with JSON dumps (includes BERT & token metrics)")
    ap.add_argument("--config", type=str, default="backend/np-tools/src/core/objective_extractor/config/config_o.yaml")
    ap.add_argument("--path_to_parquet", type=str, default="/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/data/insiders_outsiders_500_500.parquet")
    ap.add_argument("--calculate_on", type=str, default="texto_tecnico")
    ap.add_argument("--evaluate_on", type=str, default="title")
    ap.add_argument("--ollama_host", type=str, default="http://kumo01.tsc.uc3m.es:11434")
    ap.add_argument("--llm_model_type", type=str, default="llama3.1:8b")
    ap.add_argument("--llm_model_type_gen", type=str, default="llama3.1:8b")
    ap.add_argument("--llm_model_type_ex", type=str, default="llama3.1:8b")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of rows to evaluate")
    ap.add_argument("--enable_bm25", action="store_true", help="Initial BM25 flag; per-row we run both ON/OFF anyway")
    # Output paths
    ap.add_argument("--out_parquet", type=str, default="bm25_matrix_results_v2.parquet")
    ap.add_argument("--out_gen_json", type=str, default="generative_results_v2.json")
    ap.add_argument("--out_ext_json", type=str, default="extractive_results_v2.json")
    args = ap.parse_args()

    logger = init_logger(pathlib.Path(args.config), "bm25_matrix_eval_json")

    # Resolve LLM models
    llm_model_type_gen = args.llm_model_type_gen or args.llm_model_type
    llm_model_type_ex  = args.llm_model_type_ex  or args.llm_model_type

    extractor = ObjectiveExtractor(
        config_path=pathlib.Path(args.config),
        ollama_host=args.ollama_host,
        calculate_on=args.calculate_on,
        evaluate_on=args.evaluate_on,
        llm_model_type_ex=llm_model_type_ex,
        llm_model_type_gen=llm_model_type_gen,
        enable_bm25=args.enable_bm25,
    )

    # Load and filter
    df = pd.read_parquet(args.path_to_parquet)
    if args.calculate_on == "texto_administrativo":
        df = df[df.resultado_administrativo == "Descargado correctamente"]
        df = df[~df['texto_administrativo'].str.startswith('[ERROR:')]
    elif args.calculate_on == "texto_tecnico":
        df = df[df.resultado_tecnico == "Descargado correctamente"]
        df = df[~df['texto_tecnico'].str.startswith('[ERROR:')]

    if args.max_rows:
        df = df.head(args.max_rows)

    # Will include metadata_title if present
    has_meta_title = "metadata_title" in df.columns

    logger.info("Evaluating %d rows (BM25 ON/OFF and Rerank ON/OFF)...", len(df))

    rows = []
    gen_json_records = []
    ext_json_records = []

    tqdm.pandas()
    for ridx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        text = row[extractor.calculate_on]
        gold = row[extractor.evaluate_on] if extractor.evaluate_on in row else None
        metadata_title = (row["metadata_title"] if has_meta_title else None)

        try:
            res = evaluate_row_bm25_matrix(extractor, text, gold)
            rows.append(res)

            safe_gold = None if gold is None or (isinstance(gold, float) and math.isnan(gold)) else str(gold)
            safe_meta_title = None if metadata_title is None or (isinstance(metadata_title, float) and math.isnan(metadata_title)) else str(metadata_title)

            def pack(block_prefix, kind):  # kind in {"generative","extractive"}
                return {
                    "rerank_on": {
                        "objective": res.get(f"{block_prefix}{kind}_objective_rerank_on"),
                        "bert_precision": res.get(f"{block_prefix}{kind}_bert_precision_rerank_on"),
                        "bert_recall":    res.get(f"{block_prefix}{kind}_bert_recall_rerank_on"),
                        "bert_f1":        res.get(f"{block_prefix}{kind}_bert_f1_rerank_on"),
                        "token_precision":res.get(f"{block_prefix}{kind}_token_precision_rerank_on"),
                        "token_recall":   res.get(f"{block_prefix}{kind}_token_recall_rerank_on"),
                        "token_f1":       res.get(f"{block_prefix}{kind}_token_f1_rerank_on"),
                        "time_s":         res.get(f"{block_prefix}{kind}_time_seconds_rerank_on"),
                    },
                    "rerank_off": {
                        "objective": res.get(f"{block_prefix}{kind}_objective_rerank_off"),
                        "bert_precision": res.get(f"{block_prefix}{kind}_bert_precision_rerank_off"),
                        "bert_recall":    res.get(f"{block_prefix}{kind}_bert_recall_rerank_off"),
                        "bert_f1":        res.get(f"{block_prefix}{kind}_bert_f1_rerank_off"),
                        "token_precision":res.get(f"{block_prefix}{kind}_token_precision_rerank_off"),
                        "token_recall":   res.get(f"{block_prefix}{kind}_token_recall_rerank_off"),
                        "token_f1":       res.get(f"{block_prefix}{kind}_token_f1_rerank_off"),
                        "time_s":         res.get(f"{block_prefix}{kind}_time_seconds_rerank_off"),
                    },
                }

            gen_entry = {
                "row_index": ridx,
                "gold": safe_gold,
                "metadata_title": safe_meta_title,
                "bm25_on":  pack("bm25_on_",  "generative"),
                "bm25_off": pack("bm25_off_", "generative"),
            }
            ext_entry = {
                "row_index": ridx,
                "gold": safe_gold,
                "metadata_title": safe_meta_title,
                "bm25_on":  pack("bm25_on_",  "extractive"),
                "bm25_off": pack("bm25_off_", "extractive"),
            }

            gen_json_records.append(gen_entry)
            ext_json_records.append(ext_entry)

        except Exception as e:
            logger.exception("Row failed, skipping: %s", e)
            rows.append({})
    out = pd.DataFrame(rows)

    def pr(label, a, b):
        if a not in out.columns or b not in out.columns:
            print(f"\n{label}\n  Missing columns: {a} or {b}")
            return
        ci = paired_bootstrap(out[a].values, out[b].values)
        wr = win_rate(out[a].values, out[b].values)
        mean_a = np.nanmean(out[a].values)
        mean_b = np.nanmean(out[b].values)
        print(f"\n{label}")
        print(f"  mean {a}: {mean_a:.6f}")
        print(f"  mean {b}: {mean_b:.6f}")
        if ci is not None:
            print(f"  Δ ({a} - {b}) CI95%: [{ci[0]:.6f}, {ci[1]:.6f}, {ci[2]:.6f}]")
        else:
            print("  Δ CI95%: N/A")
        print(f"  win-rate ({a} > {b}): {wr:.3f}")

    # BM25 ON vs OFF comparisons at fixed rerank
    pr("Generative BERT-F1 (rerank ON): BM25 ON vs OFF",
       "bm25_on_generative_bert_f1_rerank_on", "bm25_off_generative_bert_f1_rerank_on")
    pr("Generative time (s) (rerank ON): BM25 ON vs OFF",
       "bm25_on_generative_time_seconds_rerank_on", "bm25_off_generative_time_seconds_rerank_on")

    pr("Generative BERT-F1 (rerank OFF): BM25 ON vs OFF",
       "bm25_on_generative_bert_f1_rerank_off", "bm25_off_generative_bert_f1_rerank_off")
    pr("Generative time (s) (rerank OFF): BM25 ON vs OFF",
       "bm25_on_generative_time_seconds_rerank_off", "bm25_off_generative_time_seconds_rerank_off")

    pr("Extractive BERT-F1 (rerank ON): BM25 ON vs OFF",
       "bm25_on_extractive_bert_f1_rerank_on", "bm25_off_extractive_bert_f1_rerank_on")
    pr("Extractive time (s) (rerank ON): BM25 ON vs OFF",
       "bm25_on_extractive_time_seconds_rerank_on", "bm25_off_extractive_time_seconds_rerank_on")

    pr("Extractive BERT-F1 (rerank OFF): BM25 ON vs OFF",
       "bm25_on_extractive_bert_f1_rerank_off", "bm25_off_extractive_bert_f1_rerank_off")
    pr("Extractive time (s) (rerank OFF): BM25 ON vs OFF",
       "bm25_on_extractive_time_seconds_rerank_off", "bm25_off_extractive_time_seconds_rerank_off")

    # save
    out_path = pathlib.Path(args.out_parquet)
    out.to_parquet(out_path, index=False)
    print(f"\nSaved per-row matrix DataFrame to: {out_path.resolve()}")

    gen_json_path = pathlib.Path(args.out_gen_json)
    with gen_json_path.open("w", encoding="utf-8") as f:
        json.dump(gen_json_records, f, ensure_ascii=False, indent=2)
    print(f"Saved generative objectives & scores to: {gen_json_path.resolve()}")

    ext_json_path = pathlib.Path(args.out_ext_json)
    with ext_json_path.open("w", encoding="utf-8") as f:
        json.dump(ext_json_records, f, ensure_ascii=False, indent=2)
    print(f"Saved extractive objectives & scores to: {ext_json_path.resolve()}")


if __name__ == "__main__":
    main()
