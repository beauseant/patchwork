import json, re, csv
from collections import Counter, defaultdict
from scipy.stats import friedmanchisquare, spearmanr, kendalltau, binomtest
from pathlib import Path
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay


def load_json(path):
    """Loads JSON or JSONL file into a list of records."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict): return [data]
        if isinstance(data, list): return data
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return []

def get_selected_label(record):
    """Extract label from selected_model field."""
    sel = record.get("label_annotations", {}).get("selected_model", {})
    if isinstance(sel, dict) and sel:
        return next(iter(sel.keys()))
    elif isinstance(sel, str):
        return sel
    return None


def summarize(json_path):
    """Print total, labeled, and label distribution."""
    records = load_json(json_path)
    total = len(records)
    counter = Counter(get_selected_label(r) for r in records if get_selected_label(r))
    labeled = sum(counter.values())

    print("\n" + "=" * 60)
    print("Summary of human annotations")
    print("=" * 60)
    print(f"Total instances          : {total}")
    print(f"Total labeled instances  : {labeled}\n")

    for lab in ["A es mejor", "B es mejor", "Ambos son malos"]:
        c = counter.get(lab, 0)
        p = c / labeled * 100 if labeled else 0
        print(f"{lab:15s}: {c:5d} ({p:5.1f}%)")
    print("=" * 60 + "\n")
    return records


def score_models(csv_path, records):
    """Join annotations with CSV metadata and compute model scores."""
    label_by_id = {}
    for r in records:
        rid = r.get("id") or r.get("place_id")
        lab = get_selected_label(r)
        if rid and lab:
            label_by_id[str(rid)] = lab

    points = defaultdict(Counter)
    decisions = defaultdict(Counter)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            if rid not in label_by_id:
                continue
            task = (row.get("task_type") or "").strip().lower()
            a, b = (row.get("item_a_model") or "").strip(), (row.get("item_b_model") or "").strip()
            if not a or not b:
                continue
            lab = label_by_id[rid]
            if lab == "A es mejor":
                points[task][a] += 3
            elif lab == "B es mejor":
                points[task][b] += 3
            elif lab == "Ambos son malos":
                points[task][a] += 1
                points[task][b] += 1
            decisions[task][lab] += 1

    print("#" * 70)
    print("Model scores by task_type")
    print("#" * 70)
    for task, counter in points.items():
        print(f"\nTask type: {task or '(unknown)'}")
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))  # tie-break by name
        for model, score in ranked:
            print(f"  {model:<20s}: {score}")
        if ranked:
            top = ranked[0][1]
            winners = [m for m, s in ranked if s == top]
            print(f"  ➤ Winner(s): {', '.join(winners)} with {top} points")
        mix = ", ".join(f"{k}={v}" for k, v in decisions[task].items())
        print(f"  Decisions: {mix}")
    print()
    return points


def build_wide(df, which_key, metric_pattern):
    """
    Rows = place_id, cols = models. Keeps only rows where all selected models have values (paired).
    Example metric_pattern: 'rougeL_f1', 'bert_f1', 'len_words'
    """
    pat = re.compile(rf"^(?P<model>.+)_{which_key}_{metric_pattern}$")
    rows = []
    for col in df.columns:
        m = pat.match(col)
        if not m:
            continue
        model = m.group("model")
        sub = df[["place_id", col]].dropna().rename(columns={col: "value"})
        sub["model"] = model
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    wide = long_df.pivot(index="place_id", columns="model", values="value")
    return wide.dropna(how="any")


def run_tests(df_wide, human_scores, task, metric_name):
    """
    Friedman (paired) + Kendall tau-b (and Spearman) between human order and metric medians.
    Positive tau / rho => metric agrees with human ranking.
    """
    if df_wide.empty or task not in human_scores:
        print(f"\n[{metric_name}] No data for {task}.")
        return

    ranked = sorted(human_scores[task].items(), key=lambda kv: (-kv[1], kv[0]))  # best→worst
    human_order = [m for m, _ in ranked]

    cols = [m for m in human_order if m in df_wide.columns]
    if len(cols) < 3:
        print(f"\n[{metric_name}] Not enough models ({len(cols)}).")
        return

    W = df_wide[cols].dropna(how="any")
    if W.empty:
        print(f"\n[{metric_name}] No paired rows.")
        return

    # Friedman
    stat, p = friedmanchisquare(*[W[c].values for c in cols])

    # Medians per model
    med = W.median()

    # Make higher rank number = better human rank so positive tau/rho => agreement
    rank_idx = list(range(len(cols), 0, -1))
    x = rank_idx
    y = med[cols].values

    # Kendall tau-b (preferred for ranks with possible ties)
    tau, p_tau = kendalltau(x, y, alternative="two-sided", method="auto")
    # Spearman (for reference)
    rho, p_rho = spearmanr(x, y)

    print(f"\n=== {metric_name} ({task}) ===")
    print(f"Models (human best→worst): {', '.join(cols)}")
    print(f"Paired instances (rows): {W.shape[0]}")
    print(f"Friedman χ²={stat:.3f}, p={p:.3g}  (H0: all models equal)")
    print(f"Kendall tau={tau:.3f}, p={p_tau:.3g}  (tau>0 ⇒ metric agrees with humans)")
    print(f"Spearman rho={rho:.3f}, p={p_rho:.3g} (rho>0 ⇒ metric agrees with humans)")
    for m in cols:
        print(f"  {m:20s}: median={med[m]:.4f}")


def human_pairwise_significance(csv_path, records, min_trials=10):
    """
    For each model pair, count A/B wins (skip 'Ambos son malos') and run
    two-sided binomial test against p=0.5. Holm-correct p-values.
    """
    label_by_id = {}
    for r in records:
        rid = r.get("id") or r.get("place_id")
        lab = get_selected_label(r)
        if rid and lab in {"A es mejor", "B es mejor"}:
            label_by_id[str(rid)] = lab

    # Count wins per pair
    counts = defaultdict(lambda: [0, 0])  # key=(m1,m2) sorted; value=[wins_m1, wins_m2]
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            lab = label_by_id.get(rid)
            if lab is None:
                continue
            a, b = (row.get("item_a_model") or "").strip(), (row.get("item_b_model") or "").strip()
            if not a or not b:
                continue
            m1, m2 = sorted([a, b])
            # who won relative to (m1, m2)?
            if lab == "A es mejor":
                winner = a
            else:  # "B es mejor"
                winner = b
            if winner == m1:
                counts[(m1, m2)][0] += 1
            else:
                counts[(m1, m2)][1] += 1

    # Binomial tests
    results = []
    for (m1, m2), (w1, w2) in counts.items():
        n = w1 + w2
        if n < min_trials:
            continue
        res = binomtest(w1, n=n, p=0.5, alternative="two-sided")
        p = res.pvalue
        rate = w1 / n if n else float("nan")
        results.append({"model_i": m1, "model_j": m2, "wins_i": w1, "wins_j": w2,
                        "n": n, "p_raw": p, "winrate_i": rate})

    if not results:
        print("\n[Human pairwise] Not enough comparisons.")
        return

    df = pd.DataFrame(results).sort_values("p_raw").reset_index(drop=True)

    # Holm correction
    m = len(df)
    df["p_holm"] = [min((m - i) * p, 1.0) for i, p in enumerate(df["p_raw"].values)]
    # Ensure monotone non-decreasing adjusted p-values in sorted order
    for i in range(1, m):
        if df.at[i, "p_holm"] < df.at[i - 1, "p_holm"]:
            df.at[i, "p_holm"] = df.at[i - 1, "p_holm"]

    print("\n" + "-" * 70)
    print("Human pairwise significance (binomial tests, Holm-corrected)")
    print(f"(Showing pairs with n >= {min_trials})")
    sig = df[df["p_holm"] < 0.05]
    if sig.empty:
        print("No significant pairwise differences after correction.")
    else:
        for _, r in sig.iterrows():
            better = r["model_i"] if r["wins_i"] > r["wins_j"] else r["model_j"]
            rate = r["winrate_i"] if better == r["model_i"] else 1 - r["winrate_i"]
            print(f"{better} beats the other (n={int(r['n'])}, win rate={rate:.2f}) "
                  f"p_raw={r['p_raw']:.3g}, p_holm={r['p_holm']:.3g}")
    return df

def analyze_all(json_path, csv_path
                #parquet_path
    ):
    records = summarize(json_path)
    points = score_models(csv_path, records)
    #df = pd.read_parquet(parquet_path)
    

    path_dfs = Path("/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results/final/")

    # --- ROUGE setup ---
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    rouge_metrics = ["precision", "recall", "f1"]
    keys = ["generative", "extractive"]  # the two objectives to compare against title

    def add_rouge_and_lengths(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute:
        • ROUGE(1/2/L) P/R/F1 between title and each *_objective
        • Lengths for each objective: *_len_chars, *_len_words
        """
        # Ensure objective columns exist and are strings
        for k in keys:
            col = f"{k}_objective"
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        # --- Lengths (vectorized) ---
        for k in keys:
            obj = f"{k}_objective"
            df[f"{k}_len_chars"] = df[obj].str.len()
            # .str.split(None) splits on arbitrary whitespace; .str.len() gives word count
            df[f"{k}_len_words"] = df[obj].str.split().str.len().astype("Int64")

        # --- ROUGE (row-wise) ---
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        # Prepare columns
        for k in keys:
            for rt in rouge_types:
                for m in rouge_metrics:
                    df[f"{k}_{rt}_{m}"] = np.nan

        # Compute per row
        for idx, row in df.iterrows():
            title = row.get("title", "")
            if not isinstance(title, str):
                title = "" if pd.isna(title) else str(title)

            for k in keys:
                candidate = row[f"{k}_objective"]
                try:
                    scores = scorer.score(title, candidate)
                except Exception:
                    scores = {rt: rouge_scorer.Score(precision=0.0, recall=0.0, fmeasure=0.0) for rt in rouge_types}

                for rt in rouge_types:
                    df.at[idx, f"{k}_{rt}_precision"] = scores[rt].precision
                    df.at[idx, f"{k}_{rt}_recall"]    = scores[rt].recall
                    df.at[idx, f"{k}_{rt}_f1"]        = scores[rt].fmeasure

        return df

    # --- Columns to keep/rename/merge ---
    cols_keep = []
    for key in ["generative", "extractive"]:
        for type in ["bert", "token"]:
            cols_keep.extend([f'{key}_{type}_precision', f'{key}_{type}_recall', f'{key}_{type}_f1'])
        cols_keep.extend([f"{key}_time_seconds"])

    # Add ROUGE columns
    for key in keys:
        for rt in rouge_types:
            cols_keep.extend([f"{key}_{rt}_precision", f"{key}_{rt}_recall", f"{key}_{rt}_f1"])

    # --- NEW: add length columns ---
    for key in keys:
        cols_keep.extend([f"{key}_len_chars", f"{key}_len_words"])

    df_final = None

    for path_df in path_dfs.iterdir():
        if path_df.suffix == '.parquet':
            df = pd.read_parquet(path_df)

            # Compute ROUGE + length columns
            df = add_rouge_and_lengths(df)

            model_name = path_df.stem.split("parquet_")[-1].split(".parquet")[0]

            # Select and rename columns
            base_cols = ["place_id", "title"]
            existing_cols_keep = [c for c in cols_keep if c in df.columns]  # guard in case some metrics are absent
            df_selected = df[base_cols + existing_cols_keep].copy()

            # Prefix metrics with model name
            rename_dict = {col: f"{model_name}_{col}" for col in existing_cols_keep}
            df_selected.rename(columns=rename_dict, inplace=True)

            # Merge
            if df_final is None:
                df_final = df_selected
            else:
                df_final = df_final.merge(df_selected, on=["place_id", "title"], how="outer")

    # --- Summary stats (all numeric columns except keys) ---
    numeric_cols = [col for col in df_final.columns if col not in ["place_id", "title"]]
    df_stats = df_final[numeric_cols].describe().T
    df = df_final.copy()

    metrics = {
        "ROUGE-L F1": r"rougeL_f1",
        "BERT F1": r"bert_f1",
        "Length (words)": r"len_words",
    }

    for metric_name, pattern in metrics.items():
        for task in ["generative", "extractive"]:
            wide = build_wide(df, task, pattern)
            run_tests(wide, points, task, metric_name)

    human_pairwise_significance(csv_path, records, min_trials=10)


LABELS = ["A es mejor", "B es mejor", "Ambos son malos"]

def records_to_label_map(records):
    """id/place_id -> label"""
    out = {}
    for r in records:
        rid = r.get("id") or r.get("place_id")
        lab = get_selected_label(r)
        if rid and lab in LABELS:
            out[str(rid)] = lab
    return out

def build_paired_annotations(json_path_1, json_path_2, csv_path):
    rec1 = load_json(json_path_1)
    rec2 = load_json(json_path_2)

    m1 = records_to_label_map(rec1)
    m2 = records_to_label_map(rec2)

    common_ids = sorted(set(m1).intersection(m2))

    # Bring task_type + model names from the CSV (optional but useful for slicing)
    meta = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            meta[rid] = {
                "task_type": (row.get("task_type") or "").strip().lower(),
                "item_a_model": (row.get("item_a_model") or "").strip(),
                "item_b_model": (row.get("item_b_model") or "").strip(),
            }

    rows = []
    for rid in common_ids:
        row = {"id": rid, "label_1": m1[rid], "label_2": m2[rid]}
        row.update(meta.get(rid, {"task_type": None, "item_a_model": None, "item_b_model": None}))
        rows.append(row)

    return pd.DataFrame(rows)

def agreement_report(df_pairs, labels=LABELS):
    y1 = df_pairs["label_1"].tolist()
    y2 = df_pairs["label_2"].tolist()

    acc = np.mean(np.array(y1) == np.array(y2))
    kappa = cohen_kappa_score(y1, y2, labels=labels)

    print("\n" + "="*70)
    print("Inter-annotator agreement")
    print("="*70)
    print(f"Common labeled instances: {len(df_pairs)}")
    print(f"Percent agreement       : {acc:.3f}")
    print(f"Cohen's kappa (nominal) : {kappa:.3f}")
    print("="*70)

    # By task_type (if present)
    if "task_type" in df_pairs.columns:
        for task, g in df_pairs.groupby("task_type", dropna=False):
            y1t, y2t = g["label_1"].tolist(), g["label_2"].tolist()
            if len(g) < 5:
                continue
            acct = np.mean(np.array(y1t) == np.array(y2t))
            kappat = cohen_kappa_score(y1t, y2t, labels=labels)
            print(f"\nTask: {task}  (n={len(g)})")
            print(f"  agreement: {acct:.3f} | kappa: {kappat:.3f}")


def plot_label_distribution(df_pairs, name1="jarenas", name2="carlosdi", labels=LABELS, savepath=None):
    c1 = df_pairs["label_1"].value_counts().reindex(labels, fill_value=0)
    c2 = df_pairs["label_2"].value_counts().reindex(labels, fill_value=0)

    x = np.arange(len(labels))
    width = 0.38

    plt.figure()
    plt.bar(x - width/2, c1.values, width, label=name1)
    plt.bar(x + width/2, c2.values, width, label=name2)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution (common items)")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


def plot_confusion(df_pairs, name1="jarenas", name2="carlosdi", labels=LABELS, savepath=None):
    cm = confusion_matrix(df_pairs["label_1"], df_pairs["label_2"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion matrix: {name1} (rows) vs {name2} (cols)")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()
    
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import csv
from collections import defaultdict

LABELS = ["A es mejor", "B es mejor", "Ambos son malos"]

def records_to_label_map(records):
    out = {}
    for r in records:
        rid = r.get("id") or r.get("place_id")
        lab = get_selected_label(r)
        if rid and lab in LABELS:
            out[str(rid)] = lab
    return out

def build_pairs_df(json_path_1, json_path_2, csv_path):
    m1 = records_to_label_map(load_json(json_path_1))
    m2 = records_to_label_map(load_json(json_path_2))
    common = sorted(set(m1).intersection(m2))

    meta = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            meta[rid] = {
                "task_type": (row.get("task_type") or "").strip().lower(),
                "a_model": (row.get("item_a_model") or "").strip(),
                "b_model": (row.get("item_b_model") or "").strip(),
            }

    rows = []
    for rid in common:
        row = {"id": rid, "ann1": m1[rid], "ann2": m2[rid]}
        row.update(meta.get(rid, {"task_type": None, "a_model": None, "b_model": None}))
        # clave del par (ordenado o no ordenado, tú eliges)
        row["pair_ordered"] = f"{row['a_model']}  vs  {row['b_model']}"
        row["pair_unordered"] = " vs ".join(sorted([row["a_model"], row["b_model"]]))
        rows.append(row)

    return pd.DataFrame(rows)

def agreement_stats(y1, y2, labels=LABELS):
    y1 = np.array(y1); y2 = np.array(y2)
    acc = (y1 == y2).mean()
    kappa = cohen_kappa_score(y1, y2, labels=labels)
    return acc, kappa, len(y1)

def report_agreement(df, by=None, min_n=10, labels=LABELS):
    acc, kappa, n = agreement_stats(df["ann1"], df["ann2"], labels=labels)
    print("\n" + "="*70)
    print("AGREEMENT (overall)")
    print("="*70)
    print(f"n={n} | percent_agreement={acc:.3f} | cohen_kappa={kappa:.3f}")

    if by is None:
        return

    print("\n" + "-"*70)
    print(f"AGREEMENT by {by} (showing groups with n >= {min_n})")
    print("-"*70)

    rows = []
    for key, g in df.groupby(by, dropna=False):
        if len(g) < min_n:
            continue
        a, k, n = agreement_stats(g["ann1"], g["ann2"], labels=labels)
        rows.append((key, n, a, k))

    rows.sort(key=lambda x: (-x[1], -x[2]))  # por tamaño y luego acuerdo
    for key, n, a, k in rows[:50]:
        print(f"{str(key)[:80]:80s} | n={n:4d} | agree={a:.3f} | kappa={k:.3f}")

    if len(rows) > 50:
        print(f"... ({len(rows)-50} more groups)")


if __name__ == "__main__":
    analyze_all(
    json_path="experiments/objective_extractor/potato/tasks/phase1/annotation_output/jarenas@ing.uc3m.es/annotated_instances.jsonl",
    csv_path="experiments/objective_extractor/potato/tasks/phase1/data_files/Selected_Comparisons.csv",
    )

    analyze_all(
    json_path="experiments/objective_extractor/potato/tasks/phase1/annotation_output/carlosdi@pa.uc3m.es/annotated_instances.jsonl",
    csv_path="experiments/objective_extractor/potato/tasks/phase1/data_files/Selected_Comparisons.csv",
    )

    csv_path = "experiments/objective_extractor/potato/tasks/phase1/data_files/Selected_Comparisons.csv"

    j_path = "experiments/objective_extractor/potato/tasks/phase1/annotation_output/jarenas@ing.uc3m.es/annotated_instances.jsonl"
    c_path = "experiments/objective_extractor/potato/tasks/phase1/annotation_output/carlosdi@pa.uc3m.es/annotated_instances.jsonl"

    df_pairs = build_paired_annotations(j_path, c_path, csv_path)

    agreement_report(df_pairs)
    plot_label_distribution(df_pairs, name1="jarenas", name2="carlosdi",
                        savepath="label_distribution.png")
    plot_confusion(df_pairs, name1="jarenas", name2="carlosdi",
                savepath="confusion_matrix.png")


    df_pairs = build_pairs_df(j_path, c_path, csv_path)

    # acuerdo global
    report_agreement(df_pairs)

    # acuerdo por task_type
    report_agreement(df_pairs, by="task_type", min_n=20)

    # acuerdo por par de modelos (ordenado, A vs B)
    report_agreement(df_pairs, by="pair_ordered", min_n=20)

    # o si quieres tratar (A,B) igual que (B,A)
    report_agreement(df_pairs, by="pair_unordered", min_n=20)