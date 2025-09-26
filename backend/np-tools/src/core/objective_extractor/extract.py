import heapq
import logging
import pathlib
from langdetect import detect  # type: ignore
from langdetect.lang_detect_exception import LangDetectException  # type: ignore
import ctranslate2  # type: ignore
import pyonmttok  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore

from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore
import re

from llama_index.core import VectorStoreIndex, Document  # type: ignore
from llama_index.core.node_parser import SentenceSplitter  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.core.schema import TextNode  # type: ignore
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore

from src.core.objective_extractor.prompter import Prompter
from src.core.objective_extractor.file_utils import load_yaml_config_file, init_logger


def safe_detect(text):
    try:
        if text.strip():
            return detect(text)
    except LangDetectException:
        pass
    return None

class CleanedBM25Retriever(BM25Retriever):
    def __init__(self, nodes, **kwargs):
        cleaned_nodes = [
            self._clean_node(n) for n in nodes
            if self._clean_node(n).text.strip()
        ]

        if not cleaned_nodes:
            raise ValueError(
                "CleanedBM25Retriever received no valid nodes after cleaning.")

        super().__init__(nodes=cleaned_nodes, **kwargs)

    def _clean_node(self, node: TextNode) -> TextNode:
        cleaned_text = re.sub(r"[:.,\"()\n\-]", " ", node.text.lower())
        return TextNode(text=cleaned_text, id_=node.node_id, metadata=node.metadata)

class ObjectiveExtractor(object):
    def __init__(
        self,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(
            "/np-tools/src/core/objective_extractor/config/config.yaml"),
        ollama_host: str = "http://kumo01.tsc.uc3m.es:11434",
        **kwargs
    ):
        """
        Initialize the Objective Extractor with configuration and models.
        """
        self._logger = logger if logger else init_logger(config_path, __name__)
        config = load_yaml_config_file(config_path, "extractor", self._logger)

        # Merge config with any additional keyword arguments
        config = {**config, **kwargs}

        self.embed_model = HuggingFaceEmbedding(
            model_name=config.get("embedding_model"))

        model_dir = snapshot_download(repo_id=config.get(
            "translation_model"), revision="main")
        self.ct2_tokenizer = pyonmttok.Tokenizer(
            mode="none", sp_model_path=f"{model_dir}/spm.model")
        self.ct2_translator = ctranslate2.Translator(model_dir)

        self.node_parser = SentenceSplitter(
            chunk_size=config.get("chunk_size"),
            chunk_overlap=config.get("chunk_overlap")
        )

        self._logger.info(
            f"Initializing prompter EXTRACTIVE with model type: {config.get('llm_model_type_ex')}")
        self.prompter_ex = Prompter(model_type=config.get(
            "llm_model_type_ex"), ollama_host=ollama_host)

        self._logger.info(
            f"Initializing prompter GENERATIVE with model type: {config.get('llm_model_type_gen')}")
        self.prompter_gen = Prompter(model_type=config.get(
            "llm_model_type_gen"), ollama_host=ollama_host)

        self.calculate_on = config.get("calculate_on")
        self.top_k = int(config.get("top_k", 20))
        self.max_k = int(config.get("max_k", 10))
        self.min_k = int(config.get("min_k", 3))
        self.fusion_alpha = config.get("fusion_alpha", 0.5)

        with open(config.get("templates", {}).get("generative", "")) as f:
            self.generative_prompt = f.read()

        with open(config.get("templates", {}).get("extractive", "")) as f:
            self.extractive_prompt = f.read()

        self._logger.info(
            "ObjectiveExtractor initialized with config: %s", config_path)

    def extract_extractive(self, text: str) -> str:
        """Return only the extractive objective as text."""
        context = self._prepare_context(text)
        return self.extract(text, option="extractive", precomputed_context=context)

    def extract_generative(self, text: str) -> str:
        """Return only the generative objective as text."""
        context = self._prepare_context(text)
        return self.extract(text, option="generative", precomputed_context=context)

    def extract_both(self, text: str) -> dict:
        """
        Return both objectives in a dict, using a single precomputed context
        (faster and convenient for APIs).
        """
        context = self._prepare_context(text)
        return {
            "extracted_objective": self.extract(text, option="extractive", precomputed_context=context),
            "generated_objective": self.extract(text, option="generative", precomputed_context=context),
        }

    def translate_ca_to_es(self, text: str) -> str:
        tokenized = self.ct2_tokenizer.tokenize(text)
        translated = self.ct2_translator.translate_batch([tokenized[0]])
        return self.ct2_tokenizer.detokenize(translated[0][0]['tokens'])

    def get_adaptive_top_k_from_combined(self, combined_nodes, max_k=10, min_k=3):

        if not combined_nodes:
            return []

        # Sort by score descending
        sorted_nodes = sorted(
            combined_nodes, key=lambda n: n.score or 0, reverse=True)

        scores = [n.score for n in sorted_nodes]
        self._logger.debug(f"Retrieved scores: {scores}")

        # Heuristic: stop adding if big drop in score (confidence decay)
        drop_threshold = 0.75  # relative drop
        top_nodes = [sorted_nodes[0]]

        for i in range(1, min(len(sorted_nodes), max_k)):
            prev_score = sorted_nodes[i - 1].score or 0
            curr_score = sorted_nodes[i].score or 0

            if prev_score == 0 or (curr_score / prev_score) < drop_threshold:
                break
            top_nodes.append(sorted_nodes[i])

        # Enforce bounds
        if len(top_nodes) < min_k:
            top_nodes = sorted_nodes[:min(min_k, len(sorted_nodes))]

        return top_nodes

    def _prepare_context(self, text):
        clean_text = re.sub(r'[\uf0b7]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        doc = Document(text=clean_text)
        nodes = self.node_parser.get_nodes_from_documents([doc])

        top_k = min(self.top_k, len(nodes)) if nodes else 0
        if top_k == 0:
            return clean_text

        vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        try:
            bm25_retriever = CleanedBM25Retriever(
                nodes=nodes, similarity_top_k=top_k)
        except ValueError as e:
            self._logger.warning(
                f"BM25Retriever could not be initialized: {e}")
            bm25_retriever = None

        query = "objeto del contrato, objeto de la contratación, tiene por objeto, objetivos del contrato, objeto del pliego, objectivo"
        combined_nodes = self._combine_retrievers(
            [bm25_retriever, vector_retriever], query, top_k=top_k, fusion_alpha=self.fusion_alpha)
        retrieved_nodes = self.get_adaptive_top_k_from_combined(
            combined_nodes, max_k=self.max_k, min_k=self.min_k)

        context = [n.get_content() for n in retrieved_nodes] or [clean_text]
        detected_languages = [lang for fragment in context if (
            lang := safe_detect(fragment)) is not None]

        catalan_count = detected_languages.count('ca')
        if context and detected_languages and (catalan_count / len(detected_languages) >= 0.75):
            self._logger.info(
                f"Detected {catalan_count} Catalan fragments out of {len(context)}. Translating to Spanish.")
            context = [
                self.translate_ca_to_es(fragment) if lang == 'ca' else fragment
                for fragment, lang in zip(context, detected_languages)
            ]

        return "\n\n".join(context)

    def extract(self, text, option="generative", precomputed_context=None):
        try:
            context_joint = precomputed_context or self._prepare_context(text)
            if option == "generative":
                prompt = self.generative_prompt.format(context=context_joint)
                prompter = self.prompter_gen
            elif option == "extractive":
                prompt = self.extractive_prompt.format(context=context_joint)
                prompter = self.prompter_ex
            else:
                raise ValueError(
                    "Invalid option. Use 'generative' or 'extractive'.")
            result, _ = prompter.prompt(question=prompt, use_context=False)
            return result.strip()
        except Exception as e:
            print(f"EXCEPTION in extract(): {e}")
            return f"ERROR: {e}"

    def _combine_retrievers(self, retrievers, query, top_k=4, fusion_alpha=0.5):
        all_nodes = []
        bm25_nodes = []
        other_nodes = []

        for retriever in retrievers:
            if retriever is None:
                self._logger.warning("Skipping None retriever.")
                continue

            name = type(retriever).__name__.lower()
            query_input = query.replace(",", " ") if "bm25" in name else query
            try:
                results = retriever.retrieve(query_input)
            except Exception as e:
                self._logger.warning(
                    f"Retriever {name} failed with error: {e}")
                continue

            if "bm25" in name:
                bm25_nodes.extend(results)
            else:
                other_nodes.extend(results)

            all_nodes.extend(results)

        # Normalize scores
        def normalize(nodes):
            if not nodes:
                return
            scores = [n.score or 0 for n in nodes]
            min_s, max_s = min(scores), max(scores)
            for n in nodes:
                n.score = (n.score - min_s) / \
                    (max_s - min_s + 1e-5) if max_s > min_s else 0

        normalize(bm25_nodes)
        normalize(other_nodes)

        print(f"BM25 scores: {[n.score for n in bm25_nodes]}")
        print(f"Other scores: {[n.score for n in other_nodes]}")

        # Combine by node ID
        combined = {}
        for n in bm25_nodes + other_nodes:
            nid = n.node.node_id
            if nid not in combined:
                combined[nid] = n
            else:
                existing = combined[nid]
                if fusion_alpha is not None:
                    combined_score = fusion_alpha * \
                        (existing.score or 0) + \
                        (1 - fusion_alpha) * (n.score or 0)
                    existing.score = combined_score
                else:
                    if (n.score or 0) > (existing.score or 0):
                        combined[nid] = n

        final_nodes = list(combined.values())
        final_nodes = heapq.nlargest(
            top_k, final_nodes, key=lambda x: x.score or 0)

        self._logger.info(f"Combined scores: {[n.score for n in final_nodes]}")
        return final_nodes

    def apply_to_dataframe(self, df, mode="both"):
        tqdm.pandas()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")

        if self.calculate_on not in df.columns:
            raise ValueError(
                f"Column '{self.calculate_on}' not found in DataFrame.")

        valid_modes = {"extractive", "generative", "both"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        if mode == "extractive":
            time_start = pd.Timestamp.now()
            self._logger.info(
                f"Applying extractive objective extraction to column '{self.calculate_on}'")
            df["extracted_objective"] = df[self.calculate_on].progress_apply(
                lambda text: self.extract(text, option="extractive")
            )
            time_end = pd.Timestamp.now()
            self._logger.info("Extractive objective extraction completed in %.2f seconds",
                              (time_end - time_start).total_seconds())

        elif mode == "generative":
            time_start = pd.Timestamp.now()
            self._logger.info(
                f"Applying generative objective extraction to column '{self.calculate_on}'")
            df["generated_objective"] = df[self.calculate_on].progress_apply(
                lambda text: self.extract(text, option="generative")
            )
            time_end = pd.Timestamp.now()
            self._logger.info("Generative objective extraction completed in %.2f seconds",
                              (time_end - time_start).total_seconds())

        elif mode == "both":
            def process_both(text):
                context = self._prepare_context(text)
                return pd.Series({
                    "extracted_objective": self.extract(text, option="extractive", precomputed_context=context),
                    "generated_objective": self.extract(text, option="generative", precomputed_context=context),
                })

            self._logger.info(
                f"Applying both extractive and generative extraction to column '{self.calculate_on}'")
            time_start = pd.Timestamp.now()
            results = df[self.calculate_on].progress_apply(process_both)
            df = pd.concat([df, results], axis=1)
            time_end = pd.Timestamp.now()
            self._logger.info("Both extractive and generative extraction completed in %.2f seconds",
                              (time_end - time_start).total_seconds())
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from 'extractive', 'generative', or 'both'.")

        return df
