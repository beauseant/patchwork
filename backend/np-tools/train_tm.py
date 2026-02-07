"""
Train Mallet LDA topic models per CPV.

Expected folder structure for --data_path:
    data_path/
    ├── cpv_45000000/
    │   ├── stops.txt              # one stopword per line
    │   ├── equivalences.txt       # format  original:replacement
    │   └── corpus.parquet         # must contain columns: place_id, lemmas
    ├── cpv_72000000/
    │   ├── stops.txt
    │   ├── equivalences.txt
    │   └── corpus.parquet
    └── ...
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora


from src.TopicModeling import MalletLDAModel


def load_stopwords(path: Union[str, Path]) -> set:
    """
    Load stopwords from a single .txt file (one word per line).

    Parameters
    ----------
    path : str or Path
        Path to the stopwords file.

    Returns
    -------
    set
        Set of stopwords.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stopwords file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        words = {line.strip() for line in f if line.strip()}
    # Add lowercase variants
    words.update({w.lower() for w in words})
    return words


def load_equivalences(path: Union[str, Path]) -> Dict[str, str]:
    """
    Load equivalences from a single .txt file.
    Format per line: ``original:replacement``

    Parameters
    ----------
    path : str or Path
        Path to the equivalences file.

    Returns
    -------
    dict
        Mapping original → replacement.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Equivalences file not found: {path}")
    equivalences = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parts = line.split(":")
            if len(parts) == 2:
                equivalences[parts[0]] = parts[1]
    return equivalences


def train_test_split(df: pd.DataFrame, frac: float = 0.2):
    """Split a DataFrame into train and test."""
    if frac <= 0:
        return df, df.iloc[0:0]  # empty test set
    test = df.sample(frac=frac, axis=0, random_state=42)
    train = df.drop(index=test.index)
    return train, test


def set_logger(
    name: str = "tm-logger",
    log_level: str = "INFO",
    log_dir: Union[str, Path] = "logs",
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Create a logger with console and/or file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fh = logging.FileHandler(
            log_dir / f"{name}_{ts}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

# ── Text preprocessing ────────────────────────────────────────────


def tkz_clean_str(
    rawtext: str, stopwords: set, equivalents: dict
) -> str:
    """Lowercase, remove stopwords, apply equivalences."""
    if not rawtext:
        return ""
    tokens = rawtext.lower().split()
    tokens = [equivalents.get(w, w) for w in tokens if w not in stopwords]
    # Second pass in case equivalences introduced new stopwords
    tokens = [w for w in tokens if w not in stopwords]
    return " ".join(tokens)


def build_vocabulary(
    data_col: pd.Series,
    min_lemas: int = 3,
    no_below: int = 5,  #10
    no_above: float = 0.90, #0.6
    keep_n: int = 100_000,
) -> set:
    """Build a filtered vocabulary using Gensim Dictionary."""
    data_col = data_col[data_col.apply(lambda x: len(x.split())) >= min_lemas]
    tokens = [doc.split() for doc in data_col.values.tolist()]
    gensim_dict = corpora.Dictionary(tokens)
    gensim_dict.filter_extremes(
        no_below=no_below, no_above=no_above, keep_n=keep_n)
    return {gensim_dict[idx] for idx in range(len(gensim_dict))}


def main():
    parser = argparse.ArgumentParser(description="Train Mallet LDA per CPV")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Root folder with one subfolder per CPV (each with stops.txt, equivalences.txt, corpus.parquet)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where trained models will be saved",
    )
    parser.add_argument(
        "--mallet_path",
        default="/export/usuarios_ml4ds/lbartolome/mallet-2.0.8/bin/mallet",
        help="Path to Mallet binary",
    )
    parser.add_argument(
        "--num_topics",
        default="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,40,50",
        help="Comma-separated list of topic numbers to try",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.0,
        help="Fraction for test split (0 = no test set)",
    )
    parser.add_argument(
        "--min_lemmas",
        type=int,
        default=3,
        help="Minimum number of lemmas per document",
    )
    parser.add_argument(
        "--word_min_len",
        type=int,
        default=2,
        help="Minimum word length",
    )
    parser.add_argument(
        "--log_dir",
        default="/export/usuarios_ml4ds/lbartolome/Repos/patchwork/data/logs",
        help="Directory where logs will be written",
    )
    parser.add_argument(
        "--cpvs",
        type=str,
        default="79,45",
        help="Comma-separated list of CPV codes to process (e.g. '79,45') or 'all' for all subfolders",
    )
    args = parser.parse_args()

    logger = set_logger(name="train_tm", log_dir=args.log_dir)
    num_topics = [int(k) for k in args.num_topics.split(",")]
    data_root = Path(args.data_path)
    output_root = Path(args.output_dir)

    # Iterate over CPV subfolders
    cpv_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not cpv_dirs:
        logger.error(f"No CPV subdirectories found in {data_root}")
        sys.exit(1)

    for cpv_dir in cpv_dirs:
        cpv_name = cpv_dir.name
        if args.cpvs.lower() != "all":
            cpv_code = cpv_name.split("_")[-1]
            if cpv_code not in args.cpvs.split(","):
                logger.info(f"Skipping CPV {cpv_name} (not in selected list)")
                continue
            
        logger.info(f"{'=' * 40}")
        logger.info(f"Processing CPV: {cpv_name}")
        logger.info(f"{'=' * 40}")

        # ── Validate required files ──
        stops_file = cpv_dir / "stops.txt"
        eq_file = cpv_dir / "equivalences.txt"
        parquet_files = list(cpv_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No .parquet files in {cpv_dir}, skipping.")
            continue

        # ── Load stops / equivalences ──
        t0 = time.time()
        if stops_file.is_file():
            stopwords = load_stopwords(stops_file)
            logger.info(f"  {len(stopwords)} stopwords loaded ({time.time() - t0:.2f}s)")
        else:
            stopwords = set()
            logger.warning(f"  No stops.txt found in {cpv_dir}, using empty stopwords.")

        t0 = time.time()
        if eq_file.is_file():
            equivalents = load_equivalences(eq_file)
            logger.info(f"  {len(equivalents)} equivalences loaded ({time.time() - t0:.2f}s)")
        else:
            equivalents = {}
            logger.info(f"  No equivalences.txt found in {cpv_dir}, using empty equivalences.")

        # ── Load corpus ──
        df = pd.read_parquet(parquet_files[0])
        df = df.drop_duplicates(subset=["place_id"])
        logger.info(f"  Corpus: {len(df)} docs (after dedup)")

        # ── Preprocess ──
        df["lemmas"] = df["lemmas"].apply(
            lambda x: tkz_clean_str(x, stopwords, equivalents)
        )
        vocabulary = build_vocabulary(df["lemmas"], min_lemas=args.min_lemmas)
        df["lemmas"] = df["lemmas"].apply(
            lambda x: " ".join(w for w in x.split() if w in vocabulary)
        )
        df = df[df["lemmas"].apply(lambda x: len(x.split())) >= args.min_lemmas]
        logger.info(
            f"  After filtering: {len(df)} docs, "
            f"avg {df['lemmas'].apply(lambda x: len(x.split())).mean():.1f} lemmas/doc"
        )

        if df.empty:
            logger.warning(f"  No documents left after filtering for {cpv_name}, skipping.")
            continue

        # ── Train models for each k ──
        coherences = []
        for k in num_topics:
            logger.info(f"  Training with {k} topics...")
            texts_train, texts_test = train_test_split(df, args.test_split)

            model_path = output_root / cpv_name / f"{k}_topics"
            model = MalletLDAModel(
                model_dir=model_path,
                stop_words=list(stopwords),
                word_min_len=args.word_min_len,
                mallet_path=args.mallet_path,
                logger=logger,
            )

            t0 = time.time()
            train_kwargs = dict(
                num_topics=k,
                texts_test=texts_test["lemmas"].tolist() if len(texts_test) > 0 else None,
                ids_test=texts_test["place_id"].tolist() if len(texts_test) > 0 else None,
            )
            model.train(
                texts_train["lemmas"].tolist(),
                texts_train["place_id"].tolist(),
                **train_kwargs,
            )
            model.save_model(model_path / "model_data" / "model.pickle")
            logger.info(f"  Trained in {time.time() - t0:.1f}s → {model_path}")

            # Read coherence from TMmodel if available
            coh_path = model_path / "model_data" / "TMmodel" / "topic_coherence.npy"
            if coh_path.exists():
                cohr = float(np.load(coh_path).mean())
                coherences.append({"num_topics": k, "coherence": cohr})
                logger.info(f"  Coherence (k={k}): {cohr:.4f}")

        # ── Plot coherence and pick best models ──
        if coherences:
            df_c = pd.DataFrame(coherences)
            plt.figure(figsize=(10, 6))
            plt.plot(df_c["num_topics"], df_c["coherence"], marker="o")
            plt.xlabel("Number of Topics")
            plt.ylabel("Coherence Score")
            plt.title(f"Coherence – {cpv_name}")
            plt.grid(True)
            plt.savefig(output_root / cpv_name / "coherence_plot.png")
            plt.close()

            # Best overall
            best = df_c.loc[df_c["coherence"].idxmax()]
            logger.info(
                f"  Best coherence: {best['coherence']:.4f} "
                f"with {int(best['num_topics'])} topics"
            )

    logger.info("All done.")


if __name__ == "__main__":
    main()

# python3 train_tm.py --data_path /export/usuarios_ml4ds/lbartolome/Repos/patchwork/data/models_paper/tr_data --output_dir /export/usuarios_ml4ds/lbartolome/Repos/patchwork/data/models_paper/models