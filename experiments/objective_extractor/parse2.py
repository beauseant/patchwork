import json, re, csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import friedmanchisquare, spearmanr, kendalltau, binomtest

from rouge_score import rouge_scorer



# =============================================================================
# Este script hace 3 cosas (sin quitar nada de lo existente):
#   (1) Ranking humano por tarea (generative/extractive) usando TU scoring tal cual.
#   (2) Agreement entre anotadores:
#         - Agreement nominal de 3 clases (A/B/ambos malos): percent agreement + Cohen's kappa
#         - Por task_type y por par de modelos
#         - Figuras: distribución y matriz de confusión
#         - NUEVO: agreement "orientado a elección de modelo" (lo que te importa para escoger modelos):
#             • Top-1 winner agreement por tarea
#             • Correlación entre rankings (Kendall τ) por tarea
#             • Pairwise winner agreement (solo en casos A vs B, ignorando "Ambos son malos")
#   (3) Relación humanos ↔ métricas automáticas:
#         - Tests Friedman + Kendall/Spearman entre ranking humano y medianas de métricas
#         - Figuras automáticas por métrica y tarea
#         - (Opcional) análisis por instancia delta(A-B) vs etiqueta humana + figura
#
# Nota de interpretación:
#   - Cohen’s kappa aquí SÍ aplica: 3 clases nominales (no ordinales).
#   - Pero para "explicar que el agreement es suficiente para elegir el mejor modelo",
#     lo más convincente es mostrar que los anotadores coinciden en el ranking/ganador
#     aunque discrepen en algunas etiquetas individuales.
# =============================================================================


# ----------------------------
# 0) Helpers IO + labels
# ----------------------------
LABELS = ["A es mejor", "B es mejor", "Ambos son malos"]
MODEL_ALIAS1 = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "falcon": "falcon3-10b-instruct-fp16",
}

MODEL_ALIAS2 = {
    "mistral-7b": "Mistral-7B",
    "mixtral-8x22b": "Mixtral-8×22B",
    "gemma2-9b": "Gemma-2-9B",
    "gemma3-4b": "Gemma-3-4B",
    "llama3.1-8b": "LLaMA-3.1-8B",
    "llama4-16x17b": "LLaMA-4-16×17B",
    "qwen3-8b": "Qwen-3-8B",
    "qwen3-32b": "Qwen-3-32B",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "falcon": "Falcon-10B",
    "falcon3-10b-instruct-fp16": "Falcon-10B",
    "deepseek-r1-8b": "DeepSeek-R1-8B",
}

def slugify(value):
    """Lowercase-safe slug for filenames."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "plot"


def finalize_plot(fig, save_path=None, show=True):
    """Save figure if path provided and optionally display it."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Plot saved] {save_path}")
    if show:
        plt.show()
    plt.close(fig)

def load_json_or_jsonl(path):
    """Loads JSON or JSONL file into list of records."""
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
    """Extract label from label_annotations.selected_model (dict or str)."""
    sel = record.get("label_annotations", {}).get("selected_model", {})
    if isinstance(sel, dict) and sel:
        return next(iter(sel.keys()))
    if isinstance(sel, str):
        return sel
    return None

def records_to_label_map(records):
    """id/place_id -> label (only LABELS)."""
    out = {}
    for r in records:
        rid = r.get("id") or r.get("place_id")
        lab = get_selected_label(r)
        if rid and lab in LABELS:
            out[str(rid)] = lab
    return out


# ----------------------------
# 1) Ranking humano por tarea (TU scoring tal cual)
# ----------------------------
def human_ranking_points(csv_path, jsonl_path):
    """
    Devuelve:
      - df_points: tabla con puntos por modelo y tarea (long format)
      - points: dict task -> Counter(model -> points)
      - decisions: dict task -> Counter(label -> count)
    """
    records = load_json_or_jsonl(jsonl_path)
    label_by_id = records_to_label_map(records)

    points = defaultdict(Counter)
    decisions = defaultdict(Counter)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            lab = label_by_id.get(rid)
            if lab is None:
                continue

            task = (row.get("task_type") or "").strip().lower()
            a = (row.get("item_a_model") or "").strip()
            b = (row.get("item_b_model") or "").strip()
            
            # normalize
            a = MODEL_ALIAS1.get(a, a)
            b = MODEL_ALIAS1.get(b, b)
            
            #a = MODEL_ALIAS2.get(a, a)
            #b = MODEL_ALIAS2.get(b, b)
            if not a or not b:
                continue

            # TU regla (no modificar)
            if lab == "A es mejor":
                points[task][a] += 3
            elif lab == "B es mejor":
                points[task][b] += 3
            elif lab == "Ambos son malos":
                points[task][a] += 1
                points[task][b] += 1

            decisions[task][lab] += 1

    # tabla “larga”
    rows = []
    for task, c in points.items():
        for model, pts in c.items():
            rows.append({"task_type": task, "model": model, "points": pts})
    df_points = pd.DataFrame(rows).sort_values(["task_type","points"], ascending=[True, False])

    return df_points, points, decisions

def plot_human_ranking(df_points, output_dir=None, show=True):
    """Barplot por task_type, opcionalmente guardando resultados."""
    for task, g in df_points.groupby("task_type"):
        g = g.sort_values("points", ascending=False)
        x = np.arange(len(g))
        fig, ax = plt.subplots()
        ax.bar(x, g["points"])
        ax.set_title(f"Human ranking by points — {task}")
        ax.set_ylabel("Points")
        ax.set_xticks(x)
        ax.set_xticklabels(g["model"].tolist(), rotation=30, ha="right")
        fig.tight_layout()
        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"human_ranking_{slugify(task)}.png"
        finalize_plot(fig, save_path=save_path, show=show)


# ----------------------------
# 2) Agreement entre dos anotadores
# ----------------------------
def build_pairs_df(json_path_1, json_path_2, csv_path):
    """Devuelve df con ann1/ann2 + meta (task + modelos) para slicing."""
    m1 = records_to_label_map(load_json_or_jsonl(json_path_1))
    m2 = records_to_label_map(load_json_or_jsonl(json_path_2))
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
                "place_id": row.get("place_id", None),
            }

    rows = []
    for rid in common:
        row = {"id": rid, "ann1": m1[rid], "ann2": m2[rid]}
        row.update(meta.get(rid, {"task_type": None, "a_model": None, "b_model": None, "place_id": None}))
        row["pair_ordered"] = f"{row['a_model']} vs {row['b_model']}"
        row["pair_unordered"] = " vs ".join(sorted([row["a_model"], row["b_model"]]))
        rows.append(row)

    return pd.DataFrame(rows)

def agreement_stats(y1, y2, labels=LABELS):
    """
    Agreement nominal de 3 clases:
      - percent agreement: proporción de etiquetas idénticas
      - Cohen's kappa: agreement corrigiendo por azar
    """
    y1 = np.array(y1); y2 = np.array(y2)
    acc = (y1 == y2).mean()
    kappa = cohen_kappa_score(y1, y2, labels=labels)
    return acc, kappa, len(y1)

def report_agreement(df_pairs, by=None, min_n=10, labels=LABELS):
    """Imprime overall y (opcional) por grupos."""
    acc, kappa, n = agreement_stats(df_pairs["ann1"], df_pairs["ann2"], labels=labels)
    print("\n" + "="*70)
    print("INTER-ANNOTATOR AGREEMENT (3 labels: A/B/ambos malos)")
    print("="*70)
    print(f"n={n} | percent_agreement={acc:.3f} | cohen_kappa={kappa:.3f}")

    if by is None:
        return

    print("\n" + "-"*70)
    print(f"AGREEMENT by {by} (showing groups with n >= {min_n})")
    print("-"*70)

    rows = []
    for key, g in df_pairs.groupby(by, dropna=False):
        if len(g) < min_n:
            continue
        a, k, n = agreement_stats(g["ann1"], g["ann2"], labels=labels)
        rows.append((key, n, a, k))

    rows.sort(key=lambda x: (-x[1], -x[2]))
    for key, n, a, k in rows:
        print(f"{str(key)[:80]:80s} | n={n:4d} | agree={a:.3f} | kappa={k:.3f}")

def plot_confusion(df_pairs, labels=LABELS, title="Confusion matrix", output_dir=None, show=True):
    """Figura: matriz de confusión 3×3 (ann1 vs ann2)."""
    cm = confusion_matrix(df_pairs["ann1"], df_pairs["ann2"], labels=labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"confusion_matrix_{slugify(title)}.png"
    finalize_plot(fig, save_path=save_path, show=show)

def plot_label_distribution(df_pairs, labels=LABELS, name1="ann1", name2="ann2",
                            output_dir=None, show=True):
    """Figura: distribución de etiquetas de ambos anotadores sobre items comunes."""
    c1 = df_pairs["ann1"].value_counts().reindex(labels, fill_value=0)
    c2 = df_pairs["ann2"].value_counts().reindex(labels, fill_value=0)

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, c1.values, width, label=name1)
    ax.bar(x + width / 2, c2.values, width, label=name2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Label distribution (common items)")
    ax.legend()
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"label_distribution_{slugify(name1)}_vs_{slugify(name2)}.png"
    finalize_plot(fig, save_path=save_path, show=show)


# ----------------------------
# 2B) NUEVO: Agreement “orientado a elección de modelo”
#      (Esto es lo que te sirve para argumentar que, aunque κ sea medio,
#       los anotadores coinciden en qué modelo es mejor.)
# ----------------------------
def points_to_rank(points_by_task):
    """
    Convierte dict task->Counter(model->points) en ranking:
      rank[task] = [model1, model2, ...] (best→worst, tie-break por nombre)
    """
    rank = {}
    for task, counter in points_by_task.items():
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        rank[task] = [m for m,_ in ranked]
    return rank

def top1_winner(points_by_task):
    """Devuelve ganador top-1 por tarea según puntos."""
    winners = {}
    for task, counter in points_by_task.items():
        if not counter:
            winners[task] = None
            continue
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        winners[task] = ranked[0][0]
    return winners

def report_choice_agreement(points1, points2):
    """
    Reporta:
      - winner agreement por task
      - Kendall τ entre rankings por task (solo modelos comunes)
    """
    r1 = points_to_rank(points1)
    r2 = points_to_rank(points2)
    w1 = top1_winner(points1)
    w2 = top1_winner(points2)

    print("\n" + "="*70)
    print("MODEL-SELECTION AGREEMENT (ranking/winner)")
    print("="*70)

    # Winner agreement
    tasks = sorted(set(r1.keys()) | set(r2.keys()))
    for task in tasks:
        print(f"Task={task:10s} | ann1_winner={w1.get(task)} | ann2_winner={w2.get(task)}"
              f" | same={w1.get(task) == w2.get(task)}")

    # Ranking correlation (Kendall τ)
    print("\nRanking correlation (Kendall τ) per task (common models only):")
    for task in tasks:
        if task not in r1 or task not in r2:
            continue
        common = [m for m in r1[task] if m in r2[task]]
        if len(common) < 3:
            print(f"  {task:10s}: not enough common models (n={len(common)})")
            continue
        # ranks: lower index = better
        a = [r1[task].index(m) for m in common]
        b = [r2[task].index(m) for m in common]
        tau, p = kendalltau(a, b, method="auto")
        print(f"  {task:10s}: tau={tau:.3f}, p={p:.3g} (tau>0 ⇒ similar ranking)")

def pairwise_winner_agreement(df_pairs, labels=LABELS, min_n=20):
    """
    Agreement sobre la *elección de ganador* en comparaciones A vs B:
      - Ignora "Ambos son malos" (porque no elige ganador)
      - En cada item, un anotador elige A o B; medimos % de coincidencia
      - Reporta global y por task_type
    """
    df = df_pairs.copy()
    df = df[df["ann1"].isin(["A es mejor", "B es mejor"]) & df["ann2"].isin(["A es mejor", "B es mejor"])].copy()

    if df.empty:
        print("\n[Pairwise winner agreement] No items with A/B decisions for both annotators.")
        return

    # Mapeo a ganador concreto (nombre de modelo)
    def winner(row, lab_col):
        if row[lab_col] == "A es mejor":
            return row["a_model"]
        return row["b_model"]

    df["winner1"] = df.apply(lambda r: winner(r, "ann1"), axis=1)
    df["winner2"] = df.apply(lambda r: winner(r, "ann2"), axis=1)
    acc = (df["winner1"] == df["winner2"]).mean()

    print("\n" + "="*70)
    print("PAIRWISE WINNER AGREEMENT (A vs B only; ignoring 'Ambos son malos')")
    print("="*70)
    print(f"n={len(df)} | percent_same_winner={acc:.3f}")

    if "task_type" in df.columns:
        for task, g in df.groupby("task_type", dropna=False):
            if len(g) < min_n:
                continue
            acct = (g["winner1"] == g["winner2"]).mean()
            print(f"  task={task:10s} | n={len(g):4d} | same_winner={acct:.3f}")

def plot_winner_overlap(points1, points2, title="Top-1 winner agreement",
                        output_dir=None, show=True):
    """
    Figura simple: barras indicando si el winner coincide por tarea.
    """
    w1 = top1_winner(points1)
    w2 = top1_winner(points2)
    tasks = sorted(set(w1.keys()) | set(w2.keys()))
    same = [int(w1.get(t) == w2.get(t)) for t in tasks]

    x = np.arange(len(tasks))
    fig, ax = plt.subplots()
    ax.bar(x, same)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["different", "same"])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Winner match")
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"winner_overlap_{slugify(title)}.png"
    finalize_plot(fig, save_path=save_path, show=show)


# ----------------------------
# 3) Métricas automáticas (ROUGE + longitudes) y merge multi-modelo
# ----------------------------
def add_rouge_and_lengths(df, keys=("generative", "extractive")):
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    rouge_metrics = ["precision", "recall", "f1"]

    # asegurar strings
    for k in keys:
        obj_col = f"{k}_objective"
        if obj_col not in df.columns:
            df[obj_col] = ""
        df[obj_col] = df[obj_col].fillna("").astype(str)

    # longitudes (proxy de "verbosity"/coste)
    for k in keys:
        obj_col = f"{k}_objective"
        df[f"{k}_len_chars"] = df[obj_col].str.len()
        df[f"{k}_len_words"] = df[obj_col].str.split().str.len().astype("Int64")

    # ROUGE (row-wise) comparando title vs objective (como en tu pipeline)
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    for k in keys:
        for rt in rouge_types:
            for m in rouge_metrics:
                df[f"{k}_{rt}_{m}"] = np.nan

    for idx, row in df.iterrows():
        title = row.get("title", "")
        if not isinstance(title, str):
            title = "" if pd.isna(title) else str(title)

        for k in keys:
            candidate = row.get(f"{k}_objective", "")
            try:
                scores = scorer.score(title, candidate)
            except Exception:
                scores = {rt: rouge_scorer.Score(precision=0.0, recall=0.0, fmeasure=0.0) for rt in rouge_types}

            for rt in rouge_types:
                df.at[idx, f"{k}_{rt}_precision"] = scores[rt].precision
                df.at[idx, f"{k}_{rt}_recall"]    = scores[rt].recall
                df.at[idx, f"{k}_{rt}_f1"]        = scores[rt].fmeasure

    return df

def load_and_merge_parquets(results_dir, keys=("generative","extractive")):
    """
    Lee todos los parquet de results_dir (uno por modelo),
    calcula ROUGE+len, y devuelve df_final con columnas prefixadas por modelo.

    IMPORTANT:
      - Si los parquets están indexados por place_id, se hace reset_index().
      - El nombre de modelo se deriva del filename (stem); debe coincidir con item_a_model/item_b_model en el CSV.
    """
    results_dir = Path(results_dir)
    df_final = None

    # columnas que intentamos conservar si existen (BERT/Token + tiempo)
    cols_keep = []
    for key in keys:
        for typ in ["bert", "token"]:
            cols_keep.extend([f"{key}_{typ}_precision", f"{key}_{typ}_recall", f"{key}_{typ}_f1"])
        cols_keep.append(f"{key}_time_seconds")

    # ROUGE cols + lengths
    rouge_types = ["rouge1","rouge2","rougeL"]
    for key in keys:
        for rt in rouge_types:
            cols_keep.extend([f"{key}_{rt}_precision", f"{key}_{rt}_recall", f"{key}_{rt}_f1"])
        cols_keep.extend([f"{key}_len_chars", f"{key}_len_words"])

    for p in results_dir.iterdir():
        if p.suffix != ".parquet":
            continue
            
        if p.stem.endswith("_filtrado"):
            continue

        df = pd.read_parquet(p)

        # asegurar place_id como columna (si está indexado)
        if "place_id" not in df.columns:
            df = df.reset_index()

        # asegurar title
        if "title" not in df.columns:
            raise ValueError(f"{p.name} no tiene columna 'title' (necesaria para ROUGE).")

        df = add_rouge_and_lengths(df, keys=keys)

        model_name = p.stem.split("parquet_")[-1]  # ajusta si tu naming difiere
        base_cols = ["place_id", "title"]

        existing = [c for c in cols_keep if c in df.columns]
        df_sel = df[base_cols + existing].copy()

        # prefix
        df_sel.rename(columns={c: f"{model_name}_{c}" for c in existing}, inplace=True)

        # merge
        if df_final is None:
            df_final = df_sel
        else:
            df_final = df_final.merge(df_sel, on=["place_id","title"], how="outer")

    if df_final is None:
        raise ValueError("No se encontraron .parquet en results_dir")

    return df_final


# ----------------------------
# 4) Comparación humanos ↔ métricas: Friedman + Kendall/Spearman (como tu script)
# ----------------------------
def build_wide(df, which_key, metric_pattern):
    """
    Rows=place_id, Cols=modelos.
    metric_pattern ejemplo: 'rougeL_f1' o 'bert_f1' o 'len_words'
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

def run_tests(df_wide, human_points_by_task, task, metric_name):
    """
    Friedman (paired) + Kendall tau-b / Spearman entre orden humano y medianas de la métrica.

    Interpreta:
      - Friedman p<0.05: la métrica ve diferencias entre modelos.
      - tau/rho > 0: la métrica ordena modelos parecido a humanos.
      - tau/rho < 0: la métrica ordena al revés que humanos.
    """
    if df_wide.empty or task not in human_points_by_task:
        print(f"\n[{metric_name}] No data for {task}.")
        return

    ranked = sorted(human_points_by_task[task].items(), key=lambda kv: (-kv[1], kv[0]))
    human_order = [m for m, _ in ranked]

    cols = [m for m in human_order if m in df_wide.columns]
    if len(cols) < 3:
        print(f"\n[{metric_name}] Not enough models ({len(cols)}).")
        return

    W = df_wide[cols].dropna(how="any")
    if W.empty:
        print(f"\n[{metric_name}] No paired rows.")
        return

    stat, p = friedmanchisquare(*[W[c].values for c in cols])
    med = W.median()

    # rank humano: mayor=mejor (para que tau>0 sea “acuerdo”)
    x = list(range(len(cols), 0, -1))
    y = med[cols].values

    tau, p_tau = kendalltau(x, y, alternative="two-sided", method="auto")
    rho, p_rho = spearmanr(x, y)

    print(f"\n=== {metric_name} ({task}) ===")
    print(f"Models (human best→worst): {', '.join(cols)}")
    print(f"Paired instances (rows): {W.shape[0]}")
    print(f"Friedman χ²={stat:.3f}, p={p:.3g}")
    print(f"Kendall tau={tau:.3f}, p={p_tau:.3g} (tau>0 ⇒ métrica alinea con humanos)")
    print(f"Spearman rho={rho:.3f}, p={p_rho:.3g} (rho>0 ⇒ métrica alinea con humanos)")
    for m in cols:
        print(f"  {m:20s}: median={med[m]:.4f}")

def plot_metric_medians(df_final, human_points_by_task, task, metric_pattern, title=None,
                        output_dir=None, show=True):
    """
    Figura: barras de la mediana de una métrica por modelo, ordenadas por ranking humano.
    """
    wide = build_wide(df_final, task, metric_pattern)
    if wide.empty:
        print("No wide table for plotting.")
        return

    if task not in human_points_by_task:
        print(f"No human ranking available for task '{task}'.")
        return

    ranked = sorted(human_points_by_task[task].items(), key=lambda kv: (-kv[1], kv[0]))
    models = [m for m,_ in ranked if m in wide.columns]
    if not models:
        print(f"No overlapping models for plotting medians ({task}).")
        return
    med = wide[models].median()

    x = np.arange(len(models))
    fig, ax = plt.subplots()
    ax.bar(x, med.values)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(f"Median {metric_pattern}")
    ax.set_title(title or f"Median {metric_pattern} by model — {task} (ordered by humans)")
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"metric_medians_{slugify(task)}_{slugify(metric_pattern)}.png"
    finalize_plot(fig, save_path=save_path, show=show)

def plot_rank_vs_metric(df_final, human_points_by_task, task, metric_pattern, title=None,
                        output_dir=None, show=True):
    """
    Figura: scatter de ranking humano vs ranking por métrica (mediana).
      - X: rank humano (1=mejor)
      - Y: rank por métrica (1=mejor)
    Ideal para argumentar "la métrica no captura preferencias humanas" (nube/no diagonal).
    """
    wide = build_wide(df_final, task, metric_pattern)
    if wide.empty:
        print("No wide table for rank-vs-metric plot.")
        return

    if task not in human_points_by_task:
        print(f"No human ranking available for task '{task}'.")
        return

    # ranking humano
    ranked = sorted(human_points_by_task[task].items(), key=lambda kv: (-kv[1], kv[0]))
    models = [m for m,_ in ranked if m in wide.columns]
    if len(models) < 3:
        print(f"Not enough models for rank-vs-metric plot ({task}).")
        return

    med = wide[models].median()
    # ranks (1=best) para humanos y métrica
    human_rank = {m: i+1 for i,m in enumerate(models)}
    metric_rank = {m: i+1 for i,m in enumerate(med.sort_values(ascending=False).index.tolist())}

    xs = [human_rank[m] for m in models]
    ys = [metric_rank[m] for m in models]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    for m, x_val, y_val in zip(models, xs, ys):
        ax.text(x_val, y_val, m, fontsize=8)

    ax.set_xlabel("Human rank (1=best)")
    ax.set_ylabel(f"Metric rank by median {metric_pattern} (1=best)")
    ax.set_title(title or f"Human rank vs metric rank — {task} / {metric_pattern}")
    ax.invert_xaxis()  # opcional: mejor a la izquierda
    ax.invert_yaxis()  # opcional: mejor arriba
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"rank_vs_metric_{slugify(task)}_{slugify(metric_pattern)}.png"
    finalize_plot(fig, save_path=save_path, show=show)


# ----------------------------
# 5) (Opcional) Coincidencia a nivel instancia: delta(A-B) vs etiqueta humana
# ----------------------------
LABEL_MAP = {"A es mejor": 1, "B es mejor": -1, "Ambos son malos": 0}

def per_instance_alignment(df_final, csv_path, jsonl_path, task="generative", metric="rougeL_f1"):
    """
    Crea df con delta_métrica = metric(A) - metric(B) y label humano (+1/-1/0).
    Requiere que el CSV tenga place_id (porque df_final va por place_id).

    Esto sirve para medir si la métrica predice decisiones humanas a nivel item:
      - pred = sign(delta)
      - correct si pred == label y label != 0
    """
    records = load_json_or_jsonl(jsonl_path)
    label_by_id = records_to_label_map(records)

    # index df_final por place_id (string)
    dfi = df_final.copy()
    dfi["place_id"] = dfi["place_id"].astype(str)
    dfi = dfi.set_index("place_id", drop=False)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id"))
            lab = label_by_id.get(rid)
            if lab not in LABEL_MAP:
                continue

            if (row.get("task_type") or "").strip().lower() != task:
                continue

            pid = row.get("place_id")
            if pid is None:
                continue
            pid = str(pid)

            a = (row.get("item_a_model") or "").strip()
            b = (row.get("item_b_model") or "").strip()

            colA = f"{a}_{task}_{metric}"
            colB = f"{b}_{task}_{metric}"
            if colA not in dfi.columns or colB not in dfi.columns:
                continue
            if pid not in dfi.index:
                continue

            valA = dfi.at[pid, colA]
            valB = dfi.at[pid, colB]
            if pd.isna(valA) or pd.isna(valB):
                continue

            delta = float(valA) - float(valB)
            rows.append({"id": rid, "place_id": pid, "task": task, "a": a, "b": b,
                         "label": LABEL_MAP[lab], "delta": delta})

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    d["pred"] = np.sign(d["delta"]).astype(int)  # +1 si A mejor según métrica
    d["correct"] = (d["pred"] == d["label"]) & (d["label"] != 0)
    return d

def plot_delta_hist(d, title="Delta metric (A - B)", output_dir=None, show=True):
    """Figura: histograma de delta(A-B)."""
    fig, ax = plt.subplots()
    ax.hist(d["delta"], bins=30)
    ax.set_title(title)
    ax.set_xlabel("delta")
    ax.set_ylabel("count")
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"delta_hist_{slugify(title)}.png"
    finalize_plot(fig, save_path=save_path, show=show)
    
# ----------------------------
# 4B) NUEVO: Comparativa ANOT1 vs ANOT2 vs AUTOMÁTICA (resumen + figura)
# ----------------------------
def agreement_for_subset(df_pairs, subset_mask=None, labels=LABELS):
    """
    Devuelve (acc, kappa, n) para un subconjunto de df_pairs.
    Si subset_mask es None -> usa todo df_pairs.
    """
    if subset_mask is None:
        df = df_pairs
    else:
        df = df_pairs[subset_mask].copy()

    if df.empty:
        return np.nan, np.nan, 0

    return agreement_stats(df["ann1"], df["ann2"], labels=labels)


def rank_from_scores(scores_dict, higher_is_better=True):
    """
    Convierte dict model->score a dict model->rank (1=best).
    Rompe empates por nombre.
    """
    items = list(scores_dict.items())
    if higher_is_better:
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    else:
        items = sorted(items, key=lambda kv: (kv[1], kv[0]))
    return {m: i + 1 for i, (m, _) in enumerate(items)}


def build_comparison_table(df_final, points1, points2, df_pairs, task, metric_pattern, metric_name):
    """
    Construye una tabla con:
      - points_ann1 / points_ann2
      - rank_ann1 / rank_ann2
      - median_metric
      - rank_metric (por mediana; 1=best)
      - agreement (acc/kappa) para ese task (3 labels)

    Ojo: usa build_wide() y medianas sobre filas "paired" de place_id.
    """
    wide = build_wide(df_final, task, metric_pattern)
    if wide.empty:
        return pd.DataFrame(), (np.nan, np.nan, 0)

    # Medianas por modelo (métrica automática)
    med_metric = wide.median().to_dict()

    # Puntos humanos por modelo (si faltan modelos, rellena 0 para poder comparar)
    p1 = dict(points1.get(task, Counter()))
    p2 = dict(points2.get(task, Counter()))

    # Conjunto de modelos comparables (presentes en humanos o en métrica)
    models = sorted(set(med_metric.keys()) | set(p1.keys()) | set(p2.keys()))

    # Completar faltantes
    for m in models:
        p1.setdefault(m, 0)
        p2.setdefault(m, 0)
        med_metric.setdefault(m, np.nan)

    # Rankings
    r1 = rank_from_scores(p1, higher_is_better=True)
    r2 = rank_from_scores(p2, higher_is_better=True)
    # En métricas asumimos "más alto = mejor"
    rm = rank_from_scores({m: med_metric[m] for m in models if not pd.isna(med_metric[m])},
                          higher_is_better=True)

    # Agreement (3 labels) para subset del task
    mask_task = (df_pairs["task_type"] == task) if ("task_type" in df_pairs.columns) else None
    acc, kappa, n = agreement_for_subset(df_pairs, subset_mask=mask_task)

    rows = []
    for m in models:
        rows.append({
            "task": task,
            "metric": metric_name,
            "model": m,
            "points_ann1": p1[m],
            "points_ann2": p2[m],
            "rank_ann1": r1.get(m, np.nan),
            "rank_ann2": r2.get(m, np.nan),
            "median_metric": med_metric[m],
            "rank_metric": rm.get(m, np.nan),
        })

    df_cmp = pd.DataFrame(rows)
    # Orden por ranking promedio humano (para que el plot sea legible)
    df_cmp["rank_human_avg"] = df_cmp[["rank_ann1", "rank_ann2"]].mean(axis=1)
    df_cmp = df_cmp.sort_values(["rank_human_avg", "model"], ascending=[True, True]).reset_index(drop=True)

    return df_cmp, (acc, kappa, n)


def plot_annotators_vs_metric_ranks(df_cmp, agreement_tuple, output_dir=None, show=True):
    """
    Figura PRINCIPAL: compara RANKS (1=best) de ann1, ann2 y métrica automática.
    - X: modelos ordenados por rank humano promedio
    - Y: rank (invertido para que 1=best esté arriba)
    """
    if df_cmp.empty:
        print("Empty comparison table; skipping rank plot.")
        return

    task = df_cmp["task"].iloc[0]
    metric = df_cmp["metric"].iloc[0]
    acc, kappa, n = agreement_tuple

    models = df_cmp["model"].tolist()
    x = np.arange(len(models))

    y1 = df_cmp["rank_ann1"].astype(float).values
    y2 = df_cmp["rank_ann2"].astype(float).values
    ym = df_cmp["rank_metric"].astype(float).values

    fig, ax = plt.subplots()
    ax.plot(x, y1, marker="o", label="ann1 (rank by points)")
    ax.plot(x, y2, marker="o", label="ann2 (rank by points)")
    ax.plot(x, ym, marker="o", label=f"auto (rank by median {metric})")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Rank (1=best)")
    ax.set_title(f"Ranking comparison — {task} — {metric}\n"
                 f"Inter-annotator agreement (3 labels): n={n}, agree={acc:.3f}, kappa={kappa:.3f}")
    ax.invert_yaxis()  # 1 arriba = mejor
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"compare_ranks_{slugify(task)}_{slugify(metric)}.png"
    finalize_plot(fig, save_path=save_path, show=show)


def normalize_series(s):
    """Min-max normalize (robusto a constantes). Devuelve np.array."""
    s = np.asarray(s, dtype=float)
    finite = np.isfinite(s)
    if not finite.any():
        return s
    mn, mx = np.nanmin(s[finite]), np.nanmax(s[finite])
    if mx == mn:
        return np.where(finite, 0.5, np.nan)
    out = (s - mn) / (mx - mn)
    return out


def plot_annotators_vs_metric_bars(df_cmp, agreement_tuple, output_dir=None, show=True):
    """
    Figura ALTERNATIVA: barras normalizadas para comparar "magnitudes":
      - points ann1 (normalizado)
      - points ann2 (normalizado)
      - median_metric (normalizado)
    """
    if df_cmp.empty:
        print("Empty comparison table; skipping bar plot.")
        return

    task = df_cmp["task"].iloc[0]
    metric = df_cmp["metric"].iloc[0]
    acc, kappa, n = agreement_tuple

    models = df_cmp["model"].tolist()
    x = np.arange(len(models))

    p1 = normalize_series(df_cmp["points_ann1"].values)
    p2 = normalize_series(df_cmp["points_ann2"].values)
    mm = normalize_series(df_cmp["median_metric"].values)

    width = 0.28
    fig, ax = plt.subplots()
    ax.bar(x - width, p1, width, label="ann1 points (norm)")
    ax.bar(x,         p2, width, label="ann2 points (norm)")
    ax.bar(x + width, mm, width, label=f"auto median {metric} (norm)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Normalized score (0–1)")
    ax.set_title(f"Score comparison (normalized) — {task} — {metric}\n"
                 f"Inter-annotator agreement (3 labels): n={n}, agree={acc:.3f}, kappa={kappa:.3f}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    save_path = None
    if output_dir is not None:
        save_path = Path(output_dir) / f"compare_scores_{slugify(task)}_{slugify(metric)}.png"
    finalize_plot(fig, save_path=save_path, show=show)


###### comaparative graphmw
def _human_points_by_place(csv_path, ann_jsonl_path, task):
    """
    Construye puntos humanos POR PLACE_ID usando TU regla (3/3/1+1).
    Devuelve: dict place_id -> Counter(model->points)
    """
    records = load_json_or_jsonl(ann_jsonl_path)
    label_by_id = records_to_label_map(records)

    points_by_place = {}  # pid -> Counter
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("task_type") or "").strip().lower() != task:
                continue

            rid = str(row.get("id"))
            lab = label_by_id.get(rid)
            if lab not in LABEL_MAP:
                continue

            pid = row.get("place_id")
            if pid is None:
                continue
            pid = str(pid)

            a = (row.get("item_a_model") or "").strip()
            b = (row.get("item_b_model") or "").strip()
            if not a or not b:
                continue

            if pid not in points_by_place:
                from collections import Counter
                points_by_place[pid] = Counter()

            # TU regla (idéntica a tu scoring global, pero por place_id)
            if lab == "A es mejor":
                points_by_place[pid][a] += 3
            elif lab == "B es mejor":
                points_by_place[pid][b] += 3
            elif lab == "Ambos son malos":
                points_by_place[pid][a] += 1
                points_by_place[pid][b] += 1

    return points_by_place


def _ranks_from_scores(scores_dict, higher_is_better=True):
    """
    scores_dict: model -> score (numérico)
    Devuelve: model -> rank (1=best)
    Empates: se rompen por nombre.
    """
    items = list(scores_dict.items())
    items = [(m, v) for m, v in items if pd.notna(v)]
    if not items:
        return {}

    if higher_is_better:
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
    else:
        items = sorted(items, key=lambda kv: (kv[1], kv[0]))

    return {m: i + 1 for i, (m, _) in enumerate(items)}


def _human_rank_distributions(csv_path, ann_jsonl_path, task, models=None):
    """
    Devuelve:
      - models_all: lista de modelos
      - dist: dict model -> list[ranks] (uno por place_id donde el modelo aparece)
    """
    points_by_place = _human_points_by_place(csv_path, ann_jsonl_path, task)

    # Model universe
    if models is None:
        models_all = sorted({m for pid in points_by_place for m in points_by_place[pid].keys()})
    else:
        models_all = list(models)

    dist = {m: [] for m in models_all}

    for pid, counter in points_by_place.items():
        # Ranking en este place_id (solo modelos presentes)
        ranks = _ranks_from_scores(dict(counter), higher_is_better=True)
        for m, r in ranks.items():
            if m in dist:
                dist[m].append(r)

    return models_all, dist


def _auto_rank_distributions(df_final, task, metric_pattern, models=None):
    """
    Ranking por place_id inducido por una métrica automática.
    Para cada place_id:
      - obtiene los valores de la métrica por modelo
      - rankea modelos (más alto = mejor)
    Devuelve: dist dict model -> list[ranks]
    """
    dfi = df_final.copy()
    dfi["place_id"] = dfi["place_id"].astype(str)

    # Seleccionar columnas tipo: {model}_{task}_{metric_pattern}
    suffix = f"_{task}_{metric_pattern}"
    cols = [c for c in dfi.columns if c.endswith(suffix)]

    # modelos presentes en df_final para esa métrica
    auto_models = [c[: -len(suffix)] for c in cols]  # quita el sufijo
    if models is None:
        models_all = sorted(set(auto_models))
    else:
        models_all = list(models)

    # map modelo -> columna (solo si existe)
    col_by_model = {m: f"{m}{suffix}" for m in models_all if f"{m}{suffix}" in dfi.columns}

    dist = {m: [] for m in models_all}

    for _, row in dfi.iterrows():
        # scores por place_id
        scores = {}
        for m, col in col_by_model.items():
            v = row.get(col, np.nan)
            if pd.notna(v):
                scores[m] = float(v)

        ranks = _ranks_from_scores(scores, higher_is_better=True)
        for m, r in ranks.items():
            dist[m].append(r)

    return models_all, dist


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_metric_boxes_by_model(
    df_final,
    csv_path,
    ann1_jsonl,
    ann2_jsonl,
    task,
    metrics,  # dict: {"ROUGE-L F1":"rougeL_f1", "BERT F1":"bert_f1", ...}
    ann1_name="ann1",
    ann2_name="ann2",
    output_dir=None,
    show=True,
    dpi=600,
    base_fontsize=14,
    lw=2.2,
    annotate_metric_ranks=True,   # <- NUEVO
    badge_fontsize=None,          # <- NUEVO (si None usa base+1)
    badge_pad=0.28,               # <- NUEVO (tamaño del círculo)
):
    """
    FIGURA “PAPER-SAFE”:
      - Panel 1: HUMAN (dos boxplots por modelo: ann1 y ann2)
      - Paneles siguientes: una métrica automática por panel
      - Bordes negros, color solo en el interior
      - Un solo xlabel común
      - Leyenda fuera (global)
      - (Opcional) Anotación con ranking (número en círculo) en cada box de cada métrica automática
    """

    if badge_fontsize is None:
        badge_fontsize = base_fontsize + 1

    # -----------------------------
    # 0) Estilo global (paper)
    # -----------------------------
    mpl.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.linewidth": lw,
        "lines.linewidth": lw,
        "legend.fontsize": base_fontsize - 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # -----------------------------
    # 1) Human distributions
    # -----------------------------
    models1, dist_h1 = _human_rank_distributions(csv_path, ann1_jsonl, task, models=None)
    models2, dist_h2 = _human_rank_distributions(csv_path, ann2_jsonl, task, models=None)
    models_all = sorted(set(models1) | set(models2))

    # -----------------------------
    # 2) Auto distributions
    # -----------------------------
    dist_auto_by_metric = {}
    for metric_name, metric_pattern in metrics.items():
        _, distm = _auto_rank_distributions(df_final, task, metric_pattern, models=models_all)
        dist_auto_by_metric[metric_name] = distm

    # -----------------------------
    # 3) Orden de modelos (rank humano medio)
    # -----------------------------
    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    human_mean_rank = {}
    for m in models_all:
        vals = []
        if len(dist_h1.get(m, [])): vals.append(mean_or_nan(dist_h1[m]))
        if len(dist_h2.get(m, [])): vals.append(mean_or_nan(dist_h2[m]))
        human_mean_rank[m] = float(np.mean(vals)) if vals else np.nan

    models_ordered = sorted(
        models_all,
        key=lambda m: (pd.isna(human_mean_rank[m]), human_mean_rank[m], m)
    )

    # -----------------------------
    # 4) Colores métricas (interior)
    # -----------------------------
    #palette = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    #metric_colors = {m: palette[i % len(palette)] for i, m in enumerate(metrics.keys())}
    
    # set colors by hex
    predefined_colors = ["#d39f9c", "#b5d39c", "#d39cb5", "#d3ba9c", "#9f9cd3"]
    metric_colors = {m: predefined_colors[i % len(predefined_colors)] for i, m in enumerate(metrics.keys())}
    # -----------------------------
    # Helpers
    # -----------------------------
    def set_box_style(bp, facecolor):
        for b in bp["boxes"]:
            b.set(facecolor=facecolor, edgecolor="black", linewidth=lw)
        for key in ["medians", "whiskers", "caps"]:
            for line in bp[key]:
                line.set(color="black", linewidth=lw)

    def add_badge(ax, x, y, txt):
        ax.text(
            x, y, txt,
            ha="center", va="center",
            fontsize=badge_fontsize,
            fontweight="bold",
            zorder=10,
            bbox=dict(
                boxstyle=f"circle,pad={badge_pad}",
                fc="white",
                ec="black",
                lw=lw,
            )
        )

    # -----------------------------
    # 5) Plot
    # -----------------------------
    panel_names = ["Human"] + list(metrics.keys())
    n_panels = len(panel_names)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.9 * n_panels, max(7.2, 0.34 * len(models_ordered) + 2.4)),
        sharey=True,
        gridspec_kw={"wspace": 0.28},
    )
    if n_panels == 1:
        axes = [axes]

    y_positions = np.arange(1, len(models_ordered) + 1)

    # ---------- Panel HUMAN ----------
    ax = axes[0]

    data_ann1 = [dist_h1.get(m, []) for m in models_ordered]
    data_ann2 = [dist_h2.get(m, []) for m in models_ordered]

    off = 0.20
    pos1 = y_positions - off
    pos2 = y_positions + off

    bp1 = ax.boxplot(
        data_ann1,
        vert=False,
        positions=pos1,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    bp2 = ax.boxplot(
        data_ann2,
        vert=False,
        positions=pos2,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    # Estilo Human
    for b in bp1["boxes"]:
        b.set(facecolor="#9cd1d3", edgecolor="black", linewidth=lw)
    for b in bp2["boxes"]:
        b.set(facecolor="#9cd1d3", edgecolor="black", linewidth=lw, hatch="//",)

    for key in ["medians", "whiskers", "caps"]:
        for line in bp1[key] + bp2[key]:
            line.set(color="black", linewidth=lw)

    ax.set_title("Human")
    ax.invert_xaxis()
    ax.set_xlabel("")

    # ---------- Paneles AUTOMÁTICOS ----------
    for j, metric_name in enumerate(metrics.keys(), start=1):
        ax = axes[j]
        distm = dist_auto_by_metric[metric_name]

        # Datos por modelo (en orden)
        data = [distm.get(m, []) for m in models_ordered]

        bp = ax.boxplot(
            data,
            vert=False,
            positions=y_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )

        set_box_style(bp, facecolor=metric_colors[metric_name])

        ax.set_title(metric_name)
        ax.invert_xaxis()
        ax.set_xlabel("")

        # ---- Anotación ranking (número en círculo) ----
        if annotate_metric_ranks:
            # Score = media del rank por modelo (menor=mejor)
            means = []
            for m in models_ordered:
                vals = distm.get(m, [])
                means.append(mean_or_nan(vals))

            # Convertir a ranking (ignorando NaNs: los manda al final)
            order_idx = sorted(
                range(len(models_ordered)),
                key=lambda i: (pd.isna(means[i]), means[i], models_ordered[i])
            )
            rank_of_idx = {idx: r + 1 for r, idx in enumerate(order_idx)}

            # Posición del badge: sobre la mediana del box (o media si está vacío)
            for i, vals in enumerate(data):
                if not vals:
                    continue
                x = float(np.median(vals))
                y = float(y_positions[i])
                add_badge(ax, x, y, str(rank_of_idx[i]))

    # ---------- Y labels ----------
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(models_ordered)
    axes[0].invert_yaxis()

    # ---------- Spines + ticks ----------
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        ax.tick_params(width=lw, length=6)

    # ---------- Título global + xlabel común ----------
    #fig.suptitle(f"Per-instance ranking distributions — task={task}", y=0.98)
    fig.supxlabel("Rank (1=best)", y=0.02)

    # ---------- Leyenda global fuera ----------
    legend_handles = [
        Patch(facecolor="#9cd1d3", edgecolor="black",  linewidth=lw, label="Annotator 1"),
        Patch(facecolor="#9cd1d3", edgecolor="black", linewidth=lw, hatch="//",label="Annotator 2") #hatch="\\\\", linewidth=lw, label="Annotator 2"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.189, 0.005),
    )

    # Reservar espacio
    fig.tight_layout(rect=[0.03, 0.06, 0.995, 0.94])

    # -----------------------------
    # 6) Guardar
    # -----------------------------
    save_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"boxes_by_model_{slugify(task)}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # Paths (ajusta si lo ejecutas desde otro working directory)
    # ---------------------------------------------------------------------
    csv_path="experiments/objective_extractor/potato/tasks/phase1/data_files/Selected_Comparisons.csv"
    ann1_jsonl="experiments/objective_extractor/potato/tasks/phase1/annotation_output/jarenas@ing.uc3m.es/annotated_instances.jsonl"
    ann2_jsonl="experiments/objective_extractor/potato/tasks/phase1/annotation_output/carlosdi@pa.uc3m.es/annotated_instances.jsonl"
    ann1_name = "jarenas"
    ann2_name = "carlosdi"
    plots_dir = Path("experiments/objective_extractor/plots")
    show_plots = True  # cambia a False si solo quieres guardar sin mostrar

    # ---------------------------------------------------------------------
    # (1) Ranking por anotador + figuras automáticas
    # ---------------------------------------------------------------------
    df_points1, points_dict1, decisions1 = human_ranking_points(csv_path, ann1_jsonl)
    plot_human_ranking(df_points1, output_dir=plots_dir / f"ranking_{ann1_name}", show=show_plots)

    df_points2, points_dict2, decisions2 = human_ranking_points(csv_path, ann2_jsonl)
    plot_human_ranking(df_points2, output_dir=plots_dir / f"ranking_{ann2_name}", show=show_plots)

    # ---------------------------------------------------------------------
    # (2) Agreement nominal de 3 clases (A/B/ambos malos) + figuras automáticas
    # ---------------------------------------------------------------------
    df_pairs = build_pairs_df(ann1_jsonl, ann2_jsonl, csv_path)

    report_agreement(df_pairs)  # global (3 clases)
    report_agreement(df_pairs, by="task_type", min_n=20)      # por tarea
    report_agreement(df_pairs, by="pair_unordered", min_n=20) # por par de modelos (sin orden)

    agreement_dir = plots_dir / "agreement"
    plot_label_distribution(
        df_pairs,
        name1=ann1_name,
        name2=ann2_name,
        output_dir=agreement_dir,
        show=show_plots,
    )
    plot_confusion(
        df_pairs,
        title=f"Confusion matrix: {ann1_name} (rows) vs {ann2_name} (cols)",
        output_dir=agreement_dir,
        show=show_plots,
    )

    # ---------------------------------------------------------------------
    # (2B) NUEVO: Agreement para "elección de modelo"
    #     Esto soporta el argumento: "aunque κ sea medio, el ranking/ganador coincide".
    # ---------------------------------------------------------------------
    report_choice_agreement(points_dict1, points_dict2)
    plot_winner_overlap(
        points_dict1,
        points_dict2,
        title="Top-1 winner agreement by task",
        output_dir=plots_dir / "model_selection",
        show=show_plots,
    )

    # Agreement sobre ganador A vs B (ignorando 'Ambos son malos')
    pairwise_winner_agreement(df_pairs, min_n=20)

    # ---------------------------------------------------------------------
    # (3) Humanos vs métricas automáticas + figuras automáticas
    # ---------------------------------------------------------------------

    results_dir = "/export/data_ml4ds/NextProcurement/pruebas_oct_2025/objective_extractor/results/final/"

    
    df_final = load_and_merge_parquets(results_dir)
    metrics = {
        "ROUGE-L F1": "rougeL_f1",
        "BERT F1": "bert_f1",
        "Length (words)": "len_words",
        "Token F1": "token_f1",
    }

    for metric_name, pat in metrics.items():
        metric_dir = plots_dir / "metrics" / slugify(metric_name)
        for task in ["generative", "extractive"]:
            wide = build_wide(df_final, task, pat)
            run_tests(wide, points_dict1, task, metric_name)

            # Figura 1: medianas por modelo (orden humano)
            plot_metric_medians(
                df_final, points_dict1, task, pat,
                title=f"{metric_name} medians — {task}",
                output_dir=metric_dir,
                show=show_plots,
            )

            # Figura 2: ranking humano vs ranking por métrica
            plot_rank_vs_metric(
                df_final, points_dict1, task, pat,
                title=f"Human rank vs {metric_name} rank — {task}",
                output_dir=metric_dir,
                show=show_plots,
            )

    # ---------------------------------------------------------------------
    # (4) Opcional: Alineamiento por instancia (A vs B) usando deltas + figura
    #     NOTA: aquí se usan paths "demo" del ejemplo original; ajusta a tus rutas.
    # ---------------------------------------------------------------------
    d = per_instance_alignment(
        df_final,
        csv_path,        # usa tu csv_path real
        ann1_jsonl,      # usa tu anotador 1 (o cambia al 2)
        task="generative",
        metric="rougeL_f1"
    )
    if not d.empty:
        print(d[["label","pred","correct"]].value_counts(dropna=False))
        plot_delta_hist(
            d,
            title="generative — rougeL_f1 delta (A-B)",
            output_dir=plots_dir / "per_instance",
            show=show_plots,
        )

    # ---------------------------------------------------------------------
    # (3B) NUEVO: Figura comparativa ann1 vs ann2 vs automática (con agreement en el título)
    # ---------------------------------------------------------------------
    compare_dir = plots_dir / "compare_ann_vs_auto"

    for metric_name, pat in metrics.items():
        for task in ["generative", "extractive"]:
            df_cmp, agr = build_comparison_table(
                df_final=df_final,
                points1=points_dict1,
                points2=points_dict2,
                df_pairs=df_pairs,
                task=task,
                metric_pattern=pat,
                metric_name=metric_name,
            )

            # (Opcional) imprime una tabla resumen (top filas) para debugging/reporting
            if not df_cmp.empty:
                print("\n" + "="*70)
                print(f"SUMMARY TABLE — task={task} — metric={metric_name}")
                print("="*70)
                cols_show = ["model", "points_ann1", "points_ann2", "rank_ann1", "rank_ann2", "median_metric", "rank_metric"]
                print(df_cmp[cols_show].head(30).to_string(index=False))

            # Figura 1: Ranks comparados (la más útil para tu argumento)
            plot_annotators_vs_metric_ranks(
                df_cmp, agr,
                output_dir=compare_dir / slugify(metric_name),
                show=show_plots,
            )

            # Figura 2: Barras normalizadas (útil como apoyo visual)
            plot_annotators_vs_metric_bars(
                df_cmp, agr,
                output_dir=compare_dir / slugify(metric_name),
                show=show_plots,
            )
    """
    metrics = {
        "ROUGE-L F1": "rougeL_f1",
        "BERT F1": "bert_f1",
        "Length (words)": "len_words",
        "Token F1": "token_f1",
    }

    plot_metric_boxes_by_model(
        df_final=df_final,
        csv_path=csv_path,
        ann1_jsonl=ann1_jsonl,
        ann2_jsonl=ann2_jsonl,
        task="generative",
        metrics=metrics,
        ann1_name="jarenas",
        ann2_name="carlosdi",
        output_dir=plots_dir / "paper_boxes",
        show=show_plots,
    )

    plot_metric_boxes_by_model(
        df_final=df_final,
        csv_path=csv_path,
        ann1_jsonl=ann1_jsonl,
        ann2_jsonl=ann2_jsonl,
        task="extractive",
        metrics=metrics,
        ann1_name="jarenas",
        ann2_name="carlosdi",
        output_dir=plots_dir / "paper_boxes",
        show=show_plots,
    )
    """
