#!/usr/bin/env python3
import argparse
import logging
import sys
import time
import pathlib
from typing import List

import numpy as np
import scipy.sparse as sparse
from sparse_dot_topn import awesome_cossim_topn


def setup_logger(verbosity: int = 1) -> logging.Logger:
    logger = logging.getLogger("tm_sims")
    # Avoid duplicate handlers if run multiple times (e.g., in notebooks)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
    return logger


def calculate_sims(
    logger: logging.Logger,
    tm_model_dir: str,
    topn: int = 300,
    lb: float = 0.0,
) -> pathlib.Path:
    """
    Given the path to a TMmodel, calculate cosine similarities between documents
    (via theta-theta^T) and save them as a sparse matrix distances.npz.

    Parameters
    ----------
    tm_model_dir : str
        Path to TMmodel (expects 'thetas.npz' inside).
    topn : int
        Number of top similar documents to keep per row.
    lb : float
        Lower bound for similarity values (values below are dropped).

    Returns
    -------
    pathlib.Path
        Path to the saved distances.npz file.
    """
    t_start = time.perf_counter()
    tm_folder = pathlib.Path(tm_model_dir)
    thetas_path = tm_folder / "thetas.npz"
    if not thetas_path.exists():
        raise FileNotFoundError(f"Could not find {thetas_path}")

    thetas = sparse.load_npz(thetas_path)
    logger.info(f"Shape of thetas: {thetas.shape}")
    # Use sqrt trick so (sqrt(theta)) * (sqrt(theta))^T yields theta * theta^T row-wise cosine
    thetas_sqrt = thetas.sqrt() if hasattr(thetas, "sqrt") else sparse.csr_matrix(np.sqrt(thetas.A))
    thetas_col = thetas_sqrt.T

    logger.info(f"Computing top-{topn} similarities with lower bound {lb}...")
    sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)
    out_path = tm_folder / "distances.npz"
    sparse.save_npz(out_path, sims)

    t_total = (time.perf_counter() - t_start) / 60.0
    logger.info(f"Saved similarities to {out_path}")
    logger.info(f"Total computation time: {t_total:.2f} min")
    return out_path


def get_doc_by_doc_sims(W: sparse.csr_matrix, ids_corpus: List[str]) -> List[str]:
    """
    Build a string representation of top similarities for each document.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        Upper-triangular or full square similarity matrix (sparse).
    ids_corpus : List[str]
        Document IDs, aligned with W's rows/cols.

    Returns
    -------
    List[str]
        For each row i, a string like "docJ|sim docK|sim ..." (excluding self and
        skipping the first neighbor in original code’s logic).
    """
    # Take upper triangle to avoid duplicates (i<j)
    non_zero_indices = sparse.triu(W, k=1).nonzero()

    # For each row, gather "id|value" for its non-zero upper-triangular neighbors
    sim_str = [
        " ".join(
            [
                f"{ids_corpus[col]}|{W[row, col]}"
                for col in non_zero_indices[1][non_zero_indices[0] == row]
            ][1:]  # keep original "[1:]" behavior
        )
        for row in range(W.shape[0])
    ]
    return sim_str


def export_sim_strings(logger: logging.Logger, tm_model_dir: str) -> pathlib.Path:
    """
    Load distances.npz and corpus.txt, generate a line-based representation,
    and write distances.txt next to distances.npz.
    """
    tm_folder = pathlib.Path(tm_model_dir)
    sims_path = tm_folder / "distances.npz"
    if not sims_path.exists():
        raise FileNotFoundError(
            f"Could not find {sims_path}. Run with --stage compute or both first."
        )

    sims = sparse.load_npz(sims_path)
    logger.info("Loaded similarities matrix.")

    def process_line(line: str) -> str:
        # Keep original behavior: split by ' 0 ', strip quotes
        id_ = line.rsplit(" 0 ", 1)[0].strip()
        id_ = id_.strip('"')
        return id_

    corpus_path = tm_folder.parent.parent / "train_data/corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Could not find {corpus_path}")

    with open(corpus_path, encoding="utf-8") as f:
        ids_corpus = [process_line(line) for line in f]
    logger.info("Loaded corpus ids.")

    logger.info("Building similarities string representation...")
    t0 = time.perf_counter()
    sim_rpr = get_doc_by_doc_sims(sims, ids_corpus)
    t1 = time.perf_counter()
    logger.info(f"Representation finished in {t1 - t0:0.4f} seconds.")

    out_txt = tm_folder / "distances.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for item in sim_rpr:
            f.write(f"{item}\n")
    logger.info(f"Wrote similarities representation to {out_txt}")
    return out_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and/or export document-document similarities from a TMmodel."
    )
    parser.add_argument(
        "--path_tmmodel",
        type=str,
        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/case_models/root_model_30_tpcs_20231028/TMmodel",
        help="Path to TMmodel (expects thetas.npz; writes distances.npz and distances.txt).",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=300,
        help="Top-N similar documents to keep per row when computing.",
    )
    parser.add_argument(
        "--lb",
        type=float,
        default=0.0,
        help="Lower bound for similarity when computing.",
    )
    parser.add_argument(
        "--stage",
        choices=["compute", "export", "both"],
        default="both",
        help="Which stage to run: compute similarities, export text, or both.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=1, help="Increase verbosity."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)

    if args.stage in ("compute", "both"):
        calculate_sims(logger, args.path_tmmodel, args.topn, args.lb)

    if args.stage in ("export", "both"):
        export_sim_strings(logger, args.path_tmmodel)


if __name__ == "__main__":
    main()
