import json, re, csv
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import friedmanchisquare, spearmanr, kendalltau, binomtest


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
    import pdb; pdb.set_trace()

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

def analyze_all(json_path, csv_path, parquet_path):
    records = summarize(json_path)
    points = score_models(csv_path, records)
    df = pd.read_parquet(parquet_path)

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

if __name__ == "__main__":
    analyze_all(
        json_path="jarenas@ing.uc3m.es/annotated_instances.jsonl",
        csv_path="data_files/Selected_Comparisons.csv",
        parquet_path="results_comparison_final.parquet"
    )
