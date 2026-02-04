import argparse
import logging
import pathlib
import sys
import uuid
from datetime import datetime
from typing import Dict, Optional

import yaml
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd
import numpy as np


def _chunk_batch_fn(
    batch: dict,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    chunk_overlap: int,
    text_col: str,
    id_col: str,
) -> dict:
    """
    Chunk a batch of documents into smaller overlapping chunks using the provided tokenizer.
    """
    all_chunk_ids = []
    all_doc_ids = []
    all_texts = []
    all_token_ids = []

    tokenizer.model_max_length = int(1e9)

    stride = chunk_size - chunk_overlap

    if stride <= 0:
        stride = max(1, chunk_size // 2)

    for text, doc_id in zip(batch[text_col], batch[id_col]):
        if not text:
            continue

        encodings = tokenizer(text, truncation=False, add_special_tokens=False)
        input_ids = encodings["input_ids"]

        for i in range(0, len(input_ids), stride):
            start_idx = i
            end_idx = i + chunk_size

            this_chunk_ids = input_ids[start_idx:end_idx]

            if not this_chunk_ids:  # avoid empty chunks
                continue

            this_chunk_text = tokenizer.decode(this_chunk_ids)

            all_chunk_ids.append(str(uuid.uuid4()))
            all_doc_ids.append(doc_id)
            all_texts.append(this_chunk_text)
            all_token_ids.append(this_chunk_ids)

            if end_idx >= len(input_ids):
                break

    return {
        "chunk_id": all_chunk_ids,
        "doc_id": all_doc_ids,
        "text": all_texts,
        "token_ids": all_token_ids
    }


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Helper function to log or print messages.
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
    else:
        print(message)


def load_yaml_config_file(
    config_file: str,
    section: str,
    logger: Optional[logging.Logger] = None,  # CHANGED: allow None
) -> Dict:
    """
    Load a YAML configuration file and return the specified section.
    """
    if not pathlib.Path(config_file).exists():
        log_or_print(
            f"Config file not found: {config_file}", level="error", logger=logger)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    section_dict = config.get(section, {})

    # CHANGED: bugfix (was `if section == {}` which never triggers)
    if not section_dict:
        log_or_print(
            f"Section {section} not found in config file.", level="error", logger=logger)
        raise ValueError(f"Section {section} not found in config file.")

    log_or_print(
        f"Loaded config file {config_file} and section {section}.", logger=logger)

    return section_dict


def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.
    """

    logger_config = load_yaml_config_file(config_file, "logger", logger=None)
    name = name if name else logger_config.get("logger_name", "default_logger")
    log_level = logger_config.get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(logger_config.get("dir_logger", "logs"))
    N_log_keep = int(logger_config.get("N_log_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create path_logs dir if it does not exist
    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    # Generate log file name based on the data
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    # Remove old log files if they exceed the limit
    log_files = sorted(
        dir_logger.glob("*.log"),
        key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    # Create handlers based on config
    if logger_config.get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if logger_config.get("console_log", True):
        console_handler = FlushingStreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


class Embedder(object):
    def __init__(
        self,
        model_name: str,
        tokenizer_model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        device: str = "cuda",  
        normalize_embeddings: bool = True,  
    ):

        # init logger
        self._logger = logger if logger else init_logger(
            config_file="data_preparation/config.yaml",
            name="EMBEDDER",
        )

        self._device = device
        self._normalize_embeddings = normalize_embeddings

        self._init_st_tok_models(
            embeddings_model=model_name,
            tokenizer_model=tokenizer_model,
        )

    def _init_st_tok_models(
        self,
        embeddings_model: str,
        tokenizer_model: Optional[str] = None
    ) -> None:
        """
        Initialize the SentenceTransformer and Tokenizer models.
        """

        self._st = None
        self._logger.info(
            "Initializing SentenceTransformer(%r)", embeddings_model)
        try:
            self._st = SentenceTransformer(embeddings_model, device=self._device)
        except Exception as e:
            self._logger.error(
                "Error initializing SentenceTransformer: %s", str(e))
            raise e

        if tokenizer_model is None:
            self._logger.info("Using embeddings model tokenizer.")
            try:
                self._tok = AutoTokenizer.from_pretrained(
                    self._st.tokenizer.name_or_path)
            except Exception as e:
                self._logger.error("Error initializing tokenizer: %s", str(e))
                raise e
        else:
            self._tok = AutoTokenizer.from_pretrained(tokenizer_model)

    def chunk_corpus(
        self,
        corpus: Dataset,
        chunk_size: int,
        chunk_overlap: int,
        text_col: str = "text",
        id_col: str = "id",
        batch_size: int = 1000,
        num_proc: int = 4,
        adjust_chunk_size: bool = True
    ) -> Dataset:
        """
        Chunk the corpus into smaller overlapping chunks.
        """

        real_chunk_size = chunk_size
        real_overlap = chunk_overlap

        model_limit = self._st.max_seq_length if hasattr(
            self._st, "max_seq_length") else 512

        if adjust_chunk_size and (chunk_size > model_limit):
            self._logger.warning(
                f"Chunk size {chunk_size} exceeds model limit {model_limit}. Adjusting chunk size.")
            real_chunk_size = model_limit

            # Adjust overlap if necessary
            if real_overlap >= real_chunk_size:
                real_overlap = int(real_chunk_size * 0.25)
                self._logger.warning(
                    f"Chunk overlap adjusted to {real_overlap}.")

        self._logger.info(
            f"Chunking corpus with chunk_size={real_chunk_size}, chunk_overlap={real_overlap}...")

        chunked_dataset = corpus.map(
            _chunk_batch_fn,
            fn_kwargs={
                "tokenizer": self._tok,
                "chunk_size": real_chunk_size,
                "chunk_overlap": real_overlap,
                "text_col": text_col,
                "id_col": id_col,
            },
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=corpus.column_names
        )

        return chunked_dataset

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / (n + eps)).astype(np.float32)

    def _embed_chunks(
        self,
        dataset: Dataset,
        text_col: str = "text",
        batch_size: int = 64
    ) -> Dataset:
        """
        Embed the chunks in the dataset using the SentenceTransformer model.
        """
        if self._st is None:
            raise ValueError("SentenceTransformer model is not initialized.")

        self._logger.info("Embedding dataset...")

        def embed_batch(batch):
            embeddings = self._st.encode(
                batch[text_col],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,  
            ).astype(np.float32)

            return {"embeddings": embeddings.tolist()} # return list of lists

        embedded_dataset = dataset.map(
            embed_batch,
            batched=True,
            batch_size=batch_size,
            writer_batch_size=batch_size,
            desc="Embedding chunks",
        )
        return embedded_dataset

    def _aggregate_doc_embeddings_from_chunks_df(
        self,
        df_chunks: pd.DataFrame,
        doc_id_col: str = "doc_id",
        chunk_emb_col: str = "embeddings",
    ) -> pd.DataFrame:
        """
        Aggregate chunk embeddings -> document embeddings (mean pooling + L2 renorm).
        """
        self._logger.info("Aggregating chunk embeddings to document level (mean + L2 norm)...")

        def agg_one_doc(group: pd.DataFrame) -> pd.Series:
            M = np.asarray(group[chunk_emb_col].tolist(), dtype=np.float32)  # (n_chunks, dim)
            doc = M.mean(axis=0)
            # renormalize 
            doc = self._l2_normalize(doc)
            return pd.Series({"doc_embedding": doc.tolist(), "n_chunks": int(len(group))})

        df_docs = (
            df_chunks.groupby(doc_id_col, sort=False)
            .apply(agg_one_doc)
            .reset_index()
        )
        return df_docs

    def embed_corpus(
        self,
        path_corpus: str,
        batch_size: int = 64,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        id_col: str = "place_id",
        text_col: str = "generative_objective",
        out_doc_parquet: Optional[str] = None,
        out_chunk_parquet: Optional[str] = None,
        save_chunk_embeddings: bool = False,
    ) -> pd.DataFrame:
        """
        Loads a parquet corpus, chunks it, embeds chunks on GPU, aggregates to doc embeddings,
        and saves as parquet (doc_id + doc_embedding). Optionally saves chunk embeddings.
        """

        self._logger.info(f"Loading corpus from {path_corpus}...")
        corpus = pd.read_parquet(path_corpus)
        
        # keep only id and text columns
        corpus = corpus[[id_col, text_col]]

        # convert to datasets.Dataset
        dataset = Dataset.from_pandas(corpus)

        # chunk the corpus
        chunked_dataset = self.chunk_corpus(
            corpus=dataset,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            text_col=text_col,
            id_col=id_col,
            batch_size=1000,
            num_proc=4,
            adjust_chunk_size=True,
        )

        # embed the chunks
        embedded_dataset = self._embed_chunks(
            dataset=chunked_dataset,
            text_col="text",
            batch_size=batch_size,
        )


        df_chunks = embedded_dataset.to_pandas()

        # Optionally save chunk embeddings (ids + embeddings)
        if save_chunk_embeddings:
            if out_chunk_parquet is None:
                out_chunk_parquet = str(pathlib.Path(path_corpus).with_suffix("")) + "_chunk_embeddings.parquet"
            self._logger.info(f"Saving chunk embeddings to {out_chunk_parquet} ...")
            df_chunks[["chunk_id", "doc_id", "embeddings"]].to_parquet(out_chunk_parquet, index=False)

        # Aggregate to doc embeddings
        df_docs = self._aggregate_doc_embeddings_from_chunks_df(
            df_chunks=df_chunks,
            doc_id_col="doc_id",
            chunk_emb_col="embeddings",
        )

        # Save doc embeddings (doc_id + doc_embedding)
        if out_doc_parquet is None:
            out_doc_parquet = str(pathlib.Path(path_corpus).with_suffix("")) + "_doc_embeddings.parquet"
        self._logger.info(f"Saving doc embeddings to {out_doc_parquet} ...")
        df_docs[["doc_id", "doc_embedding"]].to_parquet(out_doc_parquet, index=False)

        return df_docs

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a parquet corpus and generate document embeddings."
    )
    parser.add_argument(
        "--path-corpus",
        required=True,
        help="Path to the parquet file containing the corpus.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to use while encoding embeddings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Maximum number of tokens per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Number of overlapping tokens between adjacent chunks.",
    )
    parser.add_argument(
        "--id-col",
        default="place_id",
        help="Column name that contains the unique document identifier.",
    )
    parser.add_argument(
        "--text-col",
        default="generative_objective",
        help="Column name that contains the text to embed.",
    )
    parser.add_argument(
        "--out-doc-parquet",
        default=None,
        help="Output parquet path for document-level embeddings.",
    )
    parser.add_argument(
        "--out-chunk-parquet",
        default=None,
        help="Output parquet path for chunk-level embeddings.",
    )
    parser.add_argument(
        "--save-chunk-embeddings",
        action="store_true",
        help="Persist chunk-level embeddings to parquet as well.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Optional tokenizer model name/path; defaults to the embedding model tokenizer.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device identifier to run the embeddings on (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        default=True,
        help="Apply L2 normalization to chunk embeddings before aggregation.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable L2 normalization on chunk embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    embedder = Embedder(
        model_name=args.model_name,
        tokenizer_model=args.tokenizer_model or None,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
    )

    embedder.embed_corpus(
        path_corpus=args.path_corpus,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        id_col=args.id_col,
        text_col=args.text_col,
        out_doc_parquet=args.out_doc_parquet or None,
        out_chunk_parquet=args.out_chunk_parquet or None,
        save_chunk_embeddings=args.save_chunk_embeddings,
    )


if __name__ == "__main__":
    main()
