import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Dict, Optional
import argparse
import heapq
import logging
import math
import pathlib
import re

import pandas as pd  # type: ignore
from bert_score import score  # type: ignore
from src.core.objective_extractor.file_utils import init_logger, load_yaml_config_file
from huggingface_hub import snapshot_download  # type: ignore
from langdetect import detect  # type: ignore
from langdetect.lang_detect_exception import LangDetectException  # type: ignore
from llama_index.core import Document, VectorStoreIndex  # type: ignore
from llama_index.core.node_parser import SentenceSplitter  # type: ignore
from llama_index.core.schema import TextNode  # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore
from src.core.objective_extractor.prompter import Prompter
from tqdm import tqdm  # type: ignore

################################################################################
# Regexes
################################################################################
# Headings / labels
ES_HEADS = r"""
    objeto\s+de\s+la\s+contrataci[oó]n
  | objeto\s+del\s+contrato
  | objeto\s+dei\s+contrato
  | objeto\s+del\s+procedimiento\s+de\s+contrataci[oó]n
  | informaci[oó]n\s+sobre\s+el\s+procedimiento\s+de\s+contrataci[oó]n
  | objetivos?\s+del\s+contrato
  | objeto\s+del\s+pliego
"""

CA_HEADS = r"""
    objecte\s+de\s+la\s+contractaci[oó]
  | objecte\s+del\s+contracte
  | objecte\s+del\s+procediment\s+de\s+contractaci[oó]n?
  | informaci[oó]?\s+sobre\s+el\s+procediment\s+de\s+contractaci[oó]n?
  | objectius?\s+del\s+contracte
  | objecte\s+del\s+plec
"""

GL_HEADS = r"""
    obxecto\s+da\s+contrataci[oó]n
  | obxecto\s+do\s+contrato
  | obxecto\s+do\s+prego
"""

EU_HEADS = r"""
    kontratuaren\s+xedea
  | kontratazioaren\s+xedea
  | kontratuaren\s+helburua
  | kontratazioaren\s+helburua
"""

# Introductory or defining phrases (verbs, typical expressions)
ES_VERBS = r"""
    el\s+presente\s+(?:pliego|proyecto|contrato)\s+(?:tiene|tendr[aá])\s+por\s+objeto
  | tiene\s+por\s+objeto
  | definir\s+las\s+obras\s+de
  | suministro\s+de
  | el\s+objeto\s+es
  | el\s+objetivo\s+es
"""

CA_VERBS = r"""
    el\s+present\s+(?:plec|projecte|contracte)\s+(?:t[eé]|tindr[aà])\s+per\s+objecte
  | t[eé]\s+per\s+objecte
  | definir\s+les\s+obres\s+de
  | subministrament\s+de
  | l'?objecte\s+é?s
  | l'?objectiu\s+é?s
"""

GL_VERBS = r"""
    ten\s+por\s+obxecto
  | definir\s+as\s+obras\s+de
  | subministraci[oó]n\s+de
  | o\s+obxecto\s+é?s
  | o\s+obxectivo\s+é?s
"""

EU_VERBS = r"""
    xedea\s+da
  | helburua\s+da
  | lanak\s+definitzea
  | hornidura\s+de
"""

# -------------------------
# 1) RE_ANCHOR – detection of any relevant anchor phrase or section title
# -------------------------
RE_ANCHOR = re.compile(
    rf"""
    \b(
        # --- Español ---
        {ES_HEADS}
      | {ES_VERBS}

        # --- Català ---
      | {CA_HEADS}
      | {CA_VERBS}

        # --- Galego ---
      | {GL_HEADS}
      | {GL_VERBS}

        # --- Euskera ---
      | {EU_HEADS}
      | {EU_VERBS}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# -------------------------
# 2) PATTERNS_ANCHOR – literal extraction patterns (non-greedy capture)
# -------------------------
END = r"(?=[\.\n;]|$)"  # stop at ., newline, ; or end of text

def after(pattern: str) -> str:
    """Helper to build a consistent non-greedy extractor pattern."""
    return rf"(?:{pattern})[ \t]*[:\-]?[ \t]*([^\n\.;]+)"

PATTERNS_ANCHOR = [
    # --- Español ---
    after(r"el\s+presente\s+(?:pliego|proyecto|contrato)\s+(?:tiene|tendr[aá])\s+por\s+objeto"),
    after(r"tiene\s+por\s+objeto"),
    after(r"el\s+objeto\s+es"),
    after(r"el\s+objetivo\s+es"),
    after(r"definir\s+las\s+obras\s+de"),
    after(r"suministro\s+de"),

    # --- Català ---
    after(r"el\s+present\s+(?:plec|projecte|contracte)\s+(?:t[eé]|tindr[aà])\s+per\s+objecte"),
    after(r"t[eé]\s+per\s+objecte"),
    after(r"l'?objecte\s+é?s"),
    after(r"l'?objectiu\s+é?s"),
    after(r"definir\s+les\s+obres\s+de"),
    after(r"subministrament\s+de"),

    # --- Galego ---
    after(r"ten\s+por\s+obxecto"),
    after(r"o\s+obxecto\s+é?s"),
    after(r"o\s+obxectivo\s+é?s"),
    after(r"definir\s+as\s+obras\s+de"),
    after(r"subministraci[oó]n\s+de"),

    # --- Euskera ---
    after(r"xedea\s+da"),
    after(r"helburua\s+da"),
    after(r"lanak\s+definitzea"),
    after(r"hornidura\s+de"),
]

# patterns that lead to noise (normative/administrative text)
RE_BAD = re.compile(
    r'\b(art[ií]culo|cap[ií]tulo|normativa|legislaci[oó]n|obligaciones?|protecci[oó]n\s+de\s+datos|'
    r'control\s+de\s+calidad|prevenci[oó]n|seguridad\s+y\s+salud|revisi[oó]n\s+de\s+precios|'
    r'valoraci[oó]n|abono\s+de\s+las\s+obras|maquinaria|medios\s+personales|director[a]?\s+de\s+obra|D\.?O\.?)\b',
    re.IGNORECASE,
)

# some cleaning
RE_DOT_LEADER = re.compile(r'\.{5,}')  # "........"

# 1.2 Objeto del contrato.......................15
# 3.4.1 Condiciones técnicas....................28
# Anexo I Documentación........................45
RE_TOC_LINE = re.compile(
    r'^\s*(?:\d+(?:[\.\s]\d+){0,3})\s*[-–.]?\s*[A-ZÁÉÍÓÚÑa-záéíóúñ][^.\n]{2,}\.{3,}\s*\d+\s*$'
)

# 1. OBJETO DEL CONTRATO
# 2.3 Condiciones generales
# 4.1.2 Especificaciones técnicas
RE_NUMBERED_HEADER = re.compile(
    r'^\s*(?:\d+(?:[\.\-\s]\d+){0,4})\s*[-–\.]*\s*[A-ZÁÉÍÓÚÑa-záéíóúñ].{0,120}$'
)

# EXCAVADORA HIDRÁULICA
# 1.- CAMIÓN GRÚA AUTOPROPULSADA
# HORMIGONERA AUTOPROPULSADA DE 350 L
# MÁQUINA FRESADORA DE PAVIMENTO
RE_MACHINERY_LINE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*\.-?\s*)?[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,}){0,6}\.?(\s*\.)*\s*$'
)

# lines that contain only page numbers,
RE_PAGE_ONLY = re.compile(r'^\s*\d+\s*(?:/\s*\d+)?\s*$')
################################################################################

# aux functions to get node id and position
def _nid(n):
    """ Extract node ID robustly from different node types. """
    if hasattr(n, "node") and hasattr(n.node, "node_id"):
        return n.node.node_id
    return getattr(n, "node_id", id(n))


def _pos(n):
    """ Extract node position robustly from different node types. """
    try:
        return n.node.metadata.get("idx", None)
    except Exception:
        return (getattr(n, "metadata", {}) or {}).get("idx", None)


class CleanedBM25Retriever(BM25Retriever):
    """BM25 with light cleaning to be more robust against OCR and weird punctuation."""

    def __init__(self, nodes, **kwargs):
        cleaned_nodes = []
        for n in nodes:
            cn = self._clean_node(n)
            if cn.text.strip():
                cleaned_nodes.append(cn)
        if not cleaned_nodes:
            raise ValueError(
                "CleanedBM25Retriever received no valid nodes after cleaning.")
        super().__init__(nodes=cleaned_nodes, **kwargs)

    def _clean_node(self, node: TextNode) -> TextNode:
        cleaned_text = re.sub(r"[:.,\"()\n\-]", " ", node.text.lower())
        return TextNode(text=cleaned_text, id_=node.node_id, metadata=node.metadata)


class MultiToSpanishTranslator:
    """
    OPUS Marian (Helsinki-NLP) translator: ca/gl/eu -> es using Hugging Face pipelines.
    Load once; route by detected source language.
    """

    def __init__(
        self,
        ca_repo: Optional[str] = "Helsinki-NLP/opus-mt-ca-es",
        gl_repo: Optional[str] = "Helsinki-NLP/opus-mt-gl-es",
        eu_repo: Optional[str] = "Helsinki-NLP/opus-mt-eu-es",
        max_length: int = 512,
    ):
        self._pipes: Dict[str, any] = {}

        def _load(lang: str, repo: Optional[str]):
            if not repo:
                return
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(repo)
            max_pos = getattr(
                mdl.config, "max_position_embeddings", max_length)
            
            def get_free_gpu():
                if not torch.cuda.is_available():
                    return -1
                num_gpus = torch.cuda.device_count()
                mem = [torch.cuda.memory_allocated(i) for i in range(num_gpus)]
                return mem.index(min(mem))

            self._pipes[lang] = pipeline(
                task="translation",
                model=mdl,
                tokenizer=tok,
                device=get_free_gpu(),
                model_kwargs={"max_length": min(max_length, max_pos)}
            )

        _load("ca", ca_repo)
        _load("gl", gl_repo)
        _load("eu", eu_repo)

    def available(self, lang: str) -> bool:
        return lang in self._pipes

    def translate(self, text: str, lang: str) -> str:
        """
        Translate from 'ca' | 'gl' | 'eu' -> 'es'.
        If the lang isn't loaded, returns the input text unchanged.
        """
        pipe = self._pipes.get(lang)
        if not pipe:
            return text
        return pipe(text)[0]["translation_text"]


class ObjectiveExtractor:
    def __init__(
        self,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path(".config/config.yaml"),
        ollama_host: str = "http://kumo01.tsc.uc3m.es:11434",
        **kwargs
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        config = load_yaml_config_file(config_path, "extractor", self._logger)
        config = {**config, **kwargs}

        # embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.get("embedding_model"))
        self.embedding_model_type = config.get("embedding_model")

        # sentence splitter
        self.node_parser = SentenceSplitter(
            chunk_size=config.get("chunk_size"),
            chunk_overlap=config.get("chunk_overlap")
        )

        self._logger.info(
            f"Initializing prompter EXTRACTIVE with model type: {config.get('llm_model_type_ex')}")
        self.prompter_ex = Prompter(config_path=config_path, model_type=config.get(
            "llm_model_type_ex"), ollama_host=ollama_host)

        self._logger.info(
            f"Initializing prompter GENERATIVE with model type: {config.get('llm_model_type_gen')}")
        self.prompter_gen = Prompter(config_path=config_path, model_type=config.get(
            "llm_model_type_gen"), ollama_host=ollama_host)

        self.calculate_on = config.get("calculate_on")
        self.evaluate_on = config.get("evaluate_on")
        self.max_k = config.get("max_k")
        self.min_k = config.get("min_k")
        self.budget_on_top_k = config.get("budget_on_top_k")
        self.enable_rerank = config.get("enable_rerank")

        self.window_anchor_start = config.get("window_anchor_start")
        self.window_anchor_end = config.get("window_anchor_end")

        self.enable_bm25 = config.get("enable_bm25")
        self.bm25_on_anchor = config.get("bm25_on_anchor")
        self.dense_confidence_thr = config.get("dense_confidence_thr")
        self.long_doc_chunk_thr = config.get("long_doc_chunk_thr")
        self.rrf_k = config.get("rrf_k")

        # Noise score parameters
        self.noise_min_denominator = config.get("noise_min_denominator")
        self.noise_words_per_bad_hit = config.get(
            "noise_words_per_bad_hit")

        # Objective snippet length validation parameters
        self.objective_min_length = config.get("objective_min_length")
        self.objective_max_length = config.get("objective_max_length")

        # Context selection parameters
        self.budget_chars = config.get("budget_chars", 3500)
        self.max_per_chunk = config.get("max_per_chunk", 1200)
        self.diversity_pos_gap = config.get("diversity_pos_gap", 1)
        self.rel_drop = config.get("rel_drop", 0.6)
        self.budget_soft = config.get("budget_soft", 0.85)

        # BM25 penalty parameters
        self.bm25_noise_penalty = config.get("bm25_noise_penalty", 0.6)

        # Reranking weight parameters
        self.rerank_base_weight = config.get("rerank_base_weight", 0.45)
        self.rerank_purpose_weight = config.get("rerank_purpose_weight", 0.35)
        self.rerank_position_weight = config.get(
            "rerank_position_weight", 0.10)
        self.rerank_noise_weight = config.get("rerank_noise_weight", 0.10)

        # Text processing parameters
        self.sentence_boundary_ratio = config.get(
            "sentence_boundary_ratio", 0.6)

        # Anchor window parameters
        self.anchor_window_radius_wide = config.get(
            "anchor_window_radius_wide", 2)
        self.anchor_window_radius_narrow = config.get(
            "anchor_window_radius_narrow", 1)

        # Node scoring parameters
        self.window_nodes_base_score = config.get(
            "window_nodes_base_score", 0.25)
        self.pool_nodes_base_score = config.get("pool_nodes_base_score", 0.3)

        # Language detection and translation
        self.translation_threshold = config.get("translation_threshold", 0.75)
        self.translation_max_length = config.get(
            "translation_max_length", 1024)

        # translation models
        tm_ca = config.get("translation_model_ca_es")
        tm_gl = config.get("translation_model_gl_es")
        tm_eu = config.get("translation_model_eu_es")

        self.multi_translator = MultiToSpanishTranslator(
            ca_repo=tm_ca,
            gl_repo=tm_gl,
            eu_repo=tm_eu,
            max_length=self.translation_max_length,
        )

        # Text cleaning thresholds
        self.dot_leader_line_max_length = config.get(
            "dot_leader_line_max_length", 200)
        self.numbered_header_max_length = config.get(
            "numbered_header_max_length", 80)

        # Adaptive threshold calculation
        self.std_dev_factor = config.get("std_dev_factor", 0.25)

        # Separator for context joining
        self.separator = config.get("context_separator", "\n\n")

        with open(config.get("templates", {}).get("generative", "")) as f:
            self.generative_prompt = f.read()
        with open(config.get("templates", {}).get("extractive", "")) as f:
            self.extractive_prompt = f.read()

        self._logger.info(
            "ObjectiveExtractor initialized with config: %s", config_path)

    def _find_object_snippet(
        self,
        text: str,
        min_length: int = 15,
        max_length: int = 350
    ) -> str | None:
        """
        Extracts an explicit 'objective reference phrase (e.g., "el presente pliego tiene por objeto", etc.) from text using regex patterns.

        Parameters:
        -----------
        text: str
            The text to search for objective statements
        min_length: int
            Minimum length for valid objective snippets 
        max_length: int
            Maximum length for valid objective snippets 

        Returns:
        --------
            str | None: The extracted objective snippet if found and valid, None otherwise
        """

        t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
        t = re.sub(r'[ \t\f\v]+', ' ', t)   # collapse spaces/tabs/etc, but NOT '\n'
        t = re.sub(r'\n+', '\n', t).strip()

        for pat in PATTERNS_ANCHOR:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                cand = re.sub(r'\s+', ' ', m.group(1).strip())
                if min_length <= len(cand) <= max_length:
                    return cand
        return None

    def _noise_score(
        self,
        context_candidate: str,
        min_denominator: int = 8,
        words_per_bad_hit: int = 30
    ) -> float:
        """Calculates a noise score of a candidate for context chunk based on the presence of 'bad' patterns.

        The score is calculated as: bad_hits / max(min_denominator, words / words_per_bad_hit)

        Parameters
        ----------
        context_candidate: str
            The text to analyze for noise patterns
        min_denominator: int
            Minimum denominator threshold to ensure that for very short texts, the noise score doesn't become artificially inflated.
        words_per_bad_hit: int
            Scaling factor for text length. For every 'words_per_bad_hit' words, one 'bad' pattern is tolerated before the noise score increases. 

        Returns
        -------
            float: Noise score between 0.0 and 1.0, where higher values indicate more noise
        """

        if not context_candidate:
            return 1.0

        bad_hits = len(RE_BAD.findall(context_candidate))
        words = max(1, len(re.findall(r'\w+', context_candidate)))
        ratio = min(1.0, bad_hits / max(min_denominator,
                    words / words_per_bad_hit))

        return ratio

    def _build_window_around_node(
        self,
        nodes,
        center_idx: int,
        radius: int = 1
    ):
        """Builds a context window around a central node.

        Parameters
        -----------
        nodes: list
            List of nodes with metadata containing 'idx' for position
        center_idx: int
            The index of the central node
        radius: int, optional
            The radius of the context window, defaults to 1

        Returns
        -------
            list: List of nodes within the context window
        """
        bypos = {(getattr(n, "node", None) or n).metadata.get(
            "idx", 0): n for n in nodes}
        positions = sorted(bypos.keys())
        anchor_pos = min(positions, key=lambda p: abs(
            p - center_idx)) if positions else center_idx
        window = []
        for p in range(anchor_pos - radius, anchor_pos + radius + 1):
            if p in bypos:
                window.append(bypos[p])
        return window

    def _clip_to_sentence_boundaries(
        self,
        text: str,
        limit: int
    ) -> str:
        """
        Cuts text to a maximum length without cutting off mid-sentence if possible.

        Parameters
        ----------
        text : str
            The text to be clipped.
        limit : int
            The maximum length of the clipped text.

        Returns
        -------
        str
            The clipped text, ideally ending at a sentence boundary.
        """
        if len(text) <= limit:
            return text
        cut = text[:limit]
        m = re.search(r'(?s)(.*?)([\.!?]\s+|\n\n)(?!.*[\.!?]\s+|\n\n)', cut)
        if m and len(m.group(0)) > limit * self.sentence_boundary_ratio:
            return m.group(0).strip()
        return cut.strip()

    def _wrap_nodes_with_score(
        self,
        nodes,
        base: float = 0.0
    ):
        """
        Wraps nodes to ensure each item has .score and .get_content(). This is needed because nodes come through different paths:
        - Retriever nodes: from BM25 or vector searchers, are wrapped in retriever result objects with .score and .get_content() methods by default.
        - Window nodes: raw TextNodes with only .text attribute and metadata

        Parameters
        ----------
        nodes : list
            List of nodes that may or may not have .score and .get_content() methods.
            Can include TextNode objects, retriever result nodes, or window nodes.
        base : float, optional
            Base score to assign to nodes that don't already have a score, defaults to 0.0.
            Typically used to give window/anchor nodes a baseline score (e.g., 0.25).

        Returns
        -------
        list
            List of nodes where all items are guaranteed to have .score and .get_content() methods.
        """
        wrapped = []
        for n in nodes:
            if hasattr(n, "score") and callable(getattr(n, "get_content", None)):  #  direct node
                wrapped.append(n)
                continue
            node_obj = getattr(n, "node", None) or n

            class _DummyNodeWithScore:
                def __init__(self, node, score):
                    self.node = node
                    self.score = float(score)

                def get_content(self):
                    if hasattr(self.node, "get_content") and callable(getattr(self.node, "get_content", None)):
                        return self.node.get_content()
                    return getattr(self.node, "text", "") or ""

            wrapped.append(_DummyNodeWithScore(
                node_obj, base))  #  wrapped node
        return wrapped

    def _strip_toc_and_equipment(self, text: str) -> str:
        """
        Removes:
        - Lines with dot leaders and page numbers (TOC).
        - Numbered header lines typical of indexes.
        - Dot leader strings.
        - Uppercase machinery listings.
        - Lines that are just page numbers.
        while whitelisting lines that contain anchors (RE_ANCHOR) and are not TOC or page-only lines.

        Parameters
        ----------
        text : str
            The text to be cleaned.

        Returns
        -------
        str
            The cleaned text.
        """

        cleaned_lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if RE_ANCHOR.search(line):
                # preserve the anchor unless it's also a TOC or page-only line
                if RE_TOC_LINE.match(line) or RE_PAGE_ONLY.match(line):
                    continue
                cleaned_lines.append(raw)
                continue
            if RE_PAGE_ONLY.match(line):
                continue
            if RE_TOC_LINE.match(line):
                continue
            if RE_DOT_LEADER.search(line) and len(line) < self.dot_leader_line_max_length:
                continue
            if RE_MACHINERY_LINE.match(line):
                continue
            if RE_NUMBERED_HEADER.match(line) and len(line) <= self.numbered_header_max_length:
                continue
            cleaned_lines.append(raw)

        out = "\n".join(cleaned_lines)
        out = re.sub(r'\.{4,}', '…', out)

        return out

    def _purpose_signal(self, text: str) -> float:
        """Calculates the purpose signal (i.e., if the text contains anchor signals the score is boosted; noise patterns reduce the score). 

        Parameters
        ----------
        text : str
            The text to analyze for purpose signals.

        Returns
        -------
        float: Purpose signal score between 0.0 and 1.0.
        """
        score = 0.0
        if RE_ANCHOR.search(text or ''):
            score += 1.0
        if RE_BAD.search(text or ''):
            score -= 0.5
        return max(0.0, min(1.0, score))

    def rerank_nodes(
        self,
        nodes,
        total_count: int = None
    ):
        """
        Recalculates a node score by re-ranking it based on a weighted combination of a series of factors specific to the objective extraction task:

        new_score = base_weight * base_score + purpose_weight * purpose_signal - noise_weight * noise_score + position_weight * position_bonus

        where:
        - base_score: Original retriever score
        - purpose_signal: Objective-related patterns (+1.0 for anchors, -0.5 for noise)
        - noise_score: Subtracts points for administrative/legal jargon
        - position_bonus: Higher scores for chunks appearing earlier in document


        Parameters
        ----------
        nodes : list
            List of nodes with initial scores from retrievers
        total_count : int, optional
            Total number of chunks for position bonus calculation, defaults to len(nodes)

        Returns
        -------
        list
            Nodes sorted by new scores in descending order
        """

        N = total_count or len(nodes) or 1
        for n in nodes:

            # base_score
            base = float(getattr(n, "score", 0.0) or 0.0)
            if hasattr(n, "get_content") and callable(getattr(n, "get_content", None)):
                txt = n.get_content()
            elif hasattr(n, "node") and hasattr(n.node, "get_content"):
                txt = n.node.get_content()
            else:
                txt = ""

            # purpose_signal
            p = self._purpose_signal(txt)
            try:
                idx = n.node.metadata.get('idx', 0)
            except Exception:
                idx = (getattr(n, "metadata", {}) or {}).get('idx', 0)

            # position_bonus
            pos_bonus = max(0.0, 1.0 - (idx / max(1, N - 1)))

            # noise_score
            noise = self._noise_score(txt)

            # final score
            n.score = (self.rerank_base_weight * base +
                       self.rerank_purpose_weight * p +
                       self.rerank_position_weight * pos_bonus -
                       self.rerank_noise_weight * noise)

        nodes.sort(key=lambda x: float(
            getattr(x, "score", 0.0) or 0.0), reverse=True)
        return nodes

    def _select_context_chunks(
        self,
        nodes,
        required_ids=None,
        min_k=4,
        max_k=9,
        budget_chars=3500,
        max_per_chunk=1200,
        diversity_pos_gap=1,
        rel_drop=0.6,
        budget_soft=0.85,
    ):
        """
        Selects context chunks from a list of nodes (text chunks from document) based on scores, required nodes, budget (character limit), diversity (positional spread), and relative drop-off (score threshold).

        Parameters
        ----------
        nodes : list
            List of nodes with .score and .get_content() methods.
        required_ids : list, optional
            List of node IDs that must be included in the selection.
        min_k : int, optional
            Minimum number of chunks to be selected as context.
        max_k : int, optional
            Maximum number of chunks to be selected as context.
        budget_chars : int, optional
            Maximum total character count for selected chunks.
        max_per_chunk : int, optional
            Maximum character length per individual chunk.
        diversity_pos_gap : int, optional
            Minimum positional gap between selected chunks to ensure diversity.
        rel_drop : float, optional
            Relative score drop threshold for stopping selection.
        budget_soft : float, optional
            Soft budget factor to allow some flexibility in character limit.

        Returns
        -------
        list
            List of selected text chunks as strings.    
        """

        required_ids = set(required_ids or [])

        # Identify positions of required nodes for diversity protection
        required_positions = set()
        for n in nodes:
            nid = _nid(n)
            if nid in required_ids:
                p = _pos(n)
                if p is not None:
                    for d in range(-diversity_pos_gap, diversity_pos_gap + 1):
                        required_positions.add(p + d)

        # Normalize scores to [0, 1]
        scores = [float(getattr(n, "score", 0.0) or 0.0) for n in nodes]
        if scores:
            smin, smax = min(scores), max(scores)
            if smax > smin:
                for n in nodes:
                    n.score = (float(getattr(n, "score", 0.0)
                               or 0.0) - smin) / (smax - smin)
            else:
                for n in nodes:
                    n.score = 0.5

        # ensure we iterate by descending score so early breaks are valid
        nodes = sorted(nodes, key=lambda n: float(
            getattr(n, "score", 0.0) or 0.0), reverse=True)

        # Calculate adaptive base threshold (60th percentile or mu - 0.25*sd)
        vals = [float(getattr(n, "score", 0.0) or 0.0) for n in nodes]
        if vals:
            P = 0.60
            p60_idx = int(math.ceil((1.0 - P) * (len(vals) - 1)))
            p60 = vals[p60_idx]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) /
                           max(1, len(vals) - 1)) if len(vals) > 1 else 0.0
        else:
            p60, mu, sd = 0.0, 0.0, 0.0

        base_thr = max(p60, mu - self.std_dev_factor * sd)

        selected = []
        used_chars = 0
        selected_positions = []
        picked_ids = set()  # avoid duplicates across phases

        # include all required nodes first
        for n in nodes:
            nid = _nid(n)
            if nid not in required_ids:
                continue
            if nid in picked_ids:  # skip if already picked
                continue
            if len(selected) >= max_k:  # enforce max_k
                break
            txt = (n.get_content() or "").strip()
            if not txt:
                continue
            chunk = self._clip_to_sentence_boundaries(txt, max_per_chunk)
            if not chunk:
                continue

            # include requireds regardless of budget (but still count chars)
            extra = len(chunk) + (2 if selected else 0)
            selected.append(chunk)
            used_chars += extra
            pos = _pos(n)
            if pos is not None:
                selected_positions.append(pos)
            picked_ids.add(nid)

        # select remaining best nodes with constraints based on thresholds, diversity, and budget
        prev_score = None
        for n in nodes:
            nid = _nid(n)
            if nid in picked_ids:  # skip duplicates
                continue
            if nid in required_ids:  # keep requireds isolated to the first phase
                continue

            sc = float(getattr(n, "score", 0.0) or 0.0)
            pos = _pos(n)

            # adaptive base threshold
            near_required = (
                pos in required_positions) if pos is not None else False
            if len(selected) >= min_k and sc < base_thr and not near_required:
                break

            # positional diversity
            if pos is not None and selected_positions and diversity_pos_gap > 0:
                too_close = any(
                    abs(pos - p) <= diversity_pos_gap for p in selected_positions)
                if too_close and not near_required:
                    if len(selected) >= min_k:
                        continue

            # marginal utility (relative drop) and soft budget
            if prev_score is not None and len(selected) >= min_k:
                rel = sc / max(prev_score, 1e-6)
                if rel < rel_drop and used_chars > budget_soft * budget_chars and not near_required:
                    break

            txt = (n.get_content() or "").strip()
            if not txt:
                continue
            chunk = self._clip_to_sentence_boundaries(txt, max_per_chunk)
            if not chunk:
                continue
            extra = len(chunk) + (len(self.separator) if selected else 0)

            if len(selected) >= max_k:
                break
            if used_chars + extra > budget_chars and len(selected) >= min_k:
                break

            selected.append(chunk)
            used_chars += extra
            prev_score = sc
            if pos is not None:
                selected_positions.append(pos)
            picked_ids.add(nid)
            if len(selected) >= max_k:
                break

        # guarantee min_k
        i = 0
        while len(selected) < min_k and i < len(nodes):
            n = nodes[i]
            i += 1
            nid = _nid(n)
            if nid in picked_ids or nid in required_ids:
                continue
            if len(selected) >= max_k:
                break
            txt = (n.get_content() or "").strip()
            if not txt:
                continue
            # try to keep diversity unless truly needed for min_k
            pos = _pos(n)
            if pos is not None and selected_positions and diversity_pos_gap > 0:
                too_close = any(
                    abs(pos - p) <= diversity_pos_gap for p in selected_positions)
                if too_close and (len(selected) + 1) < min_k:
                    continue

            chunk = self._clip_to_sentence_boundaries(txt, max_per_chunk)
            if not chunk:
                continue
            extra = len(chunk) + (2 if selected else 0)
            if used_chars + extra <= budget_chars:
                selected.append(chunk)
                used_chars += extra
                if pos is not None:
                    selected_positions.append(pos)
                picked_ids.add(nid)

        return selected

    def _prepare_context(self, text):
        """
        Prepares the context for objective extraction:

        1. Text cleaning: Removes bullets, TOC entries, machinery lists, numbered headers, and page references.
        2. Regex-based Anchor Detection: Searches for explicit objective phrases using pre-defined patterns
        3. Document Chunking: Splits text into overlapping chunks with positional metadata
        4. Multi-Retrieval: Uses vector search (dense) + optional BM25 (sparse) retrieval. BM25 is triggered only when no anchor is found by regex.
        5. Anchor Window Creation: Builds context windows around detected objective anchors (±2 chunks)
        6. RRF Fusion: Combines BM25 and vector results using RRF and noise penalties on BM25 scores with reranking
        7. Adaptive Selection: Three-phase selection: (a) mandatory chunks from anchor windows, (b) best remaining chunks with 60th percentile threshold, relative score drops >0.6, position diversity gaps, and character budget limits, (c) min_k guarantee 
        8. Language Detection & Translation: Detects non-Spanish content and translates to Spanish when needed.

        Parameters
        ----------
        text : str
            Raw document text from Spanish procurement documents

        Returns
        -------
        str
            Optimized context string with selected chunks joined by double newlines, ready for LLM prompting
        """
        
        def _default_queries():
            """Returns a set of default Spanish queries for multi-query retrieval."""

            spanish_queries = [
                # explicit objective anchors
                "objeto del contrato",
                "objeto del pliego",
                "tiene por objeto",
                "descripción del objeto del contrato",
                "finalidad del contrato",
                "objeto del procedimiento de contratación",
                "objetivos del contrato",
                "objeto de la contratación",
                "objeto del proyecto",

                # common procurement phrases
                "definir las obras de",
                "proyecto de ejecución",
                "servicio de suministro",
                "ejecución de obras",
                "mejora de vías urbanas",
                "suministro de",
            ]

            seen, out = set(), []
            for q in spanish_queries:
                q_norm = q.strip()
                if q_norm and q_norm not in seen:
                    seen.add(q_norm)
                    out.append(q_norm)
            return out

        def _retrieve_multiquery(retriever, queries, per_query_k):
            """Retrieve for each short query and keep the best-scoring hit per node (max fusion)."""
            best_by_id = {}
            for q in queries:
                try:
                    res = retriever.retrieve(q)
                except Exception:
                    res = []
                for n in res[:per_query_k]:
                    nid = getattr(getattr(n, "node", None), "node_id", None) or getattr(n, "node_id", None)
                    sc = float(getattr(n, "score", 0.0) or 0.0)
                    cur = best_by_id.get(nid)
                    if (cur is None) or (sc > float(getattr(cur, "score", 0.0) or 0.0)):
                        best_by_id[nid] = n

            return sorted(best_by_id.values(), key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)

        def _safe_detect(text):
            try:
                if text and text.strip():
                    return detect(text)
            except LangDetectException:
                return None
            return None
        
        # -------------------------------
        # 1) Text cleaning
        # -------------------------------
        clean_text = re.sub(r'[\uf0b7]', '', text or '')  # weird bullets
        clean_text = self._strip_toc_and_equipment(clean_text)
        clean_text = (clean_text or '').replace('\r\n','\n').replace('\r','\n')
        clean_text = re.sub(r'[ \t\f\v]+', ' ', clean_text)
        clean_text = re.sub(r'\n+', '\n', clean_text).strip()

        # -------------------------------
        # 2) Regex anchor detection
        # -------------------------------
        direct = self._find_object_snippet(clean_text)
        regex_window_text = None
        win_start = win_end = None
        if direct:
            pos = clean_text.lower().find(direct.lower())
            if pos != -1:
                # extract a window around the direct snippet
                win_start = max(0, pos - self.window_anchor_start)
                win_end = min(len(clean_text), pos +
                              len(direct) + self.window_anchor_end)
                regex_window_text = clean_text[win_start:win_end].strip()
            self._logger.info(f"Direct snippet found by regex: {direct}")
            self._logger.debug(
                f"Context window around direct snippet: {regex_window_text}")
        # direct = None
        # regex_window_text = None
        # win_start = win_end = None

        # -------------------------------
        # 3) Chunking + metadata
        # -------------------------------
        doc = Document(text=clean_text)
        nodes = self.node_parser.get_nodes_from_documents([doc])
        for i, n in enumerate(nodes):
            n.metadata = dict(n.metadata or {})
            n.metadata['idx'] = i

        top_k = min(len(nodes), self.max_k)
        if direct is not None:
            # if we have a direct snippet, we can reduce top_k to save budget
            top_k = max(self.min_k, min(top_k, 8))

        # -------------------------------
        # 4) Dense retrieval (query-by-example, else multi-query)
        # -------------------------------
        vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

        dense_nodes = []
        # Query-by-example: use the regex found snippet / window (best signal)
        if regex_window_text:
            q_snippet = regex_window_text[:250]
            self._logger.info(f"Using query-by-example with snippet: {q_snippet}")
            queries_used = [q_snippet]
            query_strategy = "regex_snippet"
            try:
                dense_nodes = vector_retriever.retrieve(q_snippet)
            except Exception:
                dense_nodes = []

        # multi-query
        if not dense_nodes:
            multi_q = _default_queries()
            queries_used = multi_q[:]
            query_strategy = "multiquery"
            per_query_k = max(3, min(6, top_k))
            dense_nodes = _retrieve_multiquery(vector_retriever, multi_q, per_query_k)

        combined_nodes = dense_nodes[:]

        # save dense-only top1 id (for telemetry)
        dense_only_top1_id = None
        if dense_nodes:
            try:
                dense_only_top1_id = dense_nodes[0].node.node_id
            except Exception:
                dense_only_top1_id = getattr(dense_nodes[0], "node_id", None)
        
        # -------------------------------
        # 5) Anchor window around retrieved anchor hits
        # -------------------------------
        required_ids = set()
        anchor_idx = None  # Initialize for metadata tracking
        # find which chunk contains the objective anchor
        if regex_window_text is not None and win_start is not None:
            for j, n in enumerate(nodes):
                chunk_text = n.get_content() if hasattr(
                    n, "get_content") else getattr(n, "text", "")
                if not chunk_text:
                    continue
                probe = regex_window_text[:80]

                if probe and probe.lower() in (chunk_text.lower()):
                    anchor_idx = j
                    break
                if RE_ANCHOR.search(chunk_text):
                    anchor_idx = j
                    break

            # Mark chunks around the found position as required
            if anchor_idx is None:
                anchor_idx = 0
            for k in range(max(0, anchor_idx - 1), min(len(nodes), anchor_idx + 2)):
                required_ids.add(nodes[k].node_id)

        #  add context window around any anchor found in dense nodes
        anchor_idx_retr = None
        for n in combined_nodes:
            if RE_ANCHOR.search(n.get_content() or ''):
                try:
                    anchor_idx_retr = n.node.metadata.get('idx', None)
                    self._logger.info(
                        f"Anchor found in retrieved node with idx={anchor_idx_retr}")
                except Exception:
                    anchor_idx_retr = (n.metadata or {}).get('idx', None)
                break
        window_nodes = []
        if anchor_idx_retr is not None:
            window_nodes = self._build_window_around_node(
                nodes, anchor_idx_retr, radius=self.anchor_window_radius_wide)
            for m in self._build_window_around_node(nodes, anchor_idx_retr, radius=self.anchor_window_radius_narrow):
                required_ids.add(m.node_id)

        # -------------------------------
        # 6) Optional BM25 + RRF
        # -------------------------------
        dense_scores = [float(getattr(n, "score", 0.0) or 0.0)
                        for n in dense_nodes] or [0.0]
        top_dense = max(dense_scores) if dense_scores else 0.0
        
        bm25_retriever = None
        need_bm25 = False

        def _init_bm25(capping=5):
            nonlocal bm25_retriever
            if bm25_retriever is None:
                try:
                    bm25_k = min(capping, top_k, len(nodes))  # cap
                    bm25_retriever = CleanedBM25Retriever(
                        nodes=nodes, similarity_top_k=bm25_k)
                except ValueError as e:
                    self._logger.warning(
                        f"BM25Retriever could not be initialized: {e}")
                    bm25_retriever = None

        if not self.enable_bm25:
            need_bm25 = False
            bm25_reasons = ["disabled_in_config"]
        else:
            bm25_reasons = []
            no_regex = (direct is None)
            low_dense = (top_dense < self.dense_confidence_thr)
            is_long = (len(nodes) > self.long_doc_chunk_thr)

            # only use BM25 if no regex anchor and (low dense confidence or long document)
            if no_regex and (low_dense or is_long):
                need_bm25 = True
                if low_dense:
                    bm25_reasons.append(
                        f"low_dense_confidence_{top_dense:.3f}")
                if is_long:
                    bm25_reasons.append(f"long_document_{len(nodes)}_chunks")

            # if there is an anchor and the config says not to use BM25 with anchor, we disable it
            if (not no_regex) and (not self.bm25_on_anchor):
                need_bm25 = False
                
        tokens = {
            # purpose / object
            "objeto", "objecte", "obxecto", "xede",
            # contract / pliego
            "contrato", "contracte", "plec", "pliego", "prego",
            # execution / project
            "ejecución", "execució", "execución", "exekuzio", "proyecto", "projecte", "proxecto", "proiektua",
            # service / supply
            "servicio", "servei", "servizo", "zerbitzu", "suministro", "subministrament", "subministración", "hornidura",
            # works
            "obras", "obres", "obrak",
            # improvement / roads
            "mejora", "millora", "mellora", "hobekuntza",
            "vías", "vies", "viais", "bideak", "viales", "urbanas", "urbanes", "urbanos", "hiritar",
        }

        query_bm25 = " ".join(sorted(tokens))

        bm25_nodes = []
        if need_bm25:
            _init_bm25()
            if bm25_retriever is not None:
                bm25_nodes = bm25_retriever.retrieve(query_bm25)
                if bm25_nodes:
                    self._logger.info(
                        f"Found {len(bm25_nodes)} BM25 nodes for fusion.")

        # penalize BM25 nodes that look like TOC/machinery lines and do RRF
        if bm25_nodes:
            self._logger.info("Doing RRF fusion with BM25 nodes.")
            for n in bm25_nodes:
                s = n.score or 0.0
                try:
                    txt = n.get_content() or ""
                except Exception:
                    txt = getattr(n.node, "text", "") or ""
                if RE_TOC_LINE.search(txt) or RE_MACHINERY_LINE.search(txt) or RE_DOT_LEADER.search(txt):
                    n.score = s * self.bm25_noise_penalty

            # Mapas de ranking
            bm25_rank_map = {}
            vec_rank_map = {}

            for i, n in enumerate(sorted(bm25_nodes, key=lambda x: x.score or 0.0, reverse=True)):
                bm25_rank_map[n.node.node_id] = i + 1
            for i, n in enumerate(sorted(dense_nodes, key=lambda x: x.score or 0.0, reverse=True)):
                vec_rank_map[n.node.node_id] = i + 1

            # RRF
            all_ids = set(bm25_rank_map) | set(vec_rank_map)
            combined_by_rrf = {}
            for nid in all_ids:
                r1 = bm25_rank_map.get(nid, 10**6)
                r2 = vec_rank_map.get(nid, 10**6)
                rrf_score = 1.0 / (self.rrf_k + r1) + 1.0 / (self.rrf_k + r2)

                pick = next((n for n in bm25_nodes if n.node.node_id == nid), None) or \
                    next((n for n in dense_nodes if n.node.node_id == nid), None)
                pick.score = rrf_score
                combined_by_rrf[nid] = pick

            # Pool tras RRF
            combined_nodes = heapq.nlargest(
                int(top_k * self.budget_on_top_k),
                list(combined_by_rrf.values()),
                key=lambda x: x.score or 0.0
            )
        else:
            # without BM25, just use dense nodes
            combined_nodes = dense_nodes[:]

        # -------------------------------
        # 7) Rerank
        # -------------------------------
        pre_rerank_nodes = combined_nodes[:]

        def _nid_safe(n):
            try:
                return n.node.node_id
            except Exception:
                return getattr(n, "node_id", None)

        pre_top1_id = _nid_safe(
            pre_rerank_nodes[0]) if pre_rerank_nodes else None

        # apply rerank only if enabled
        if self.enable_rerank:
            combined_nodes = self.rerank_nodes(
                combined_nodes, total_count=len(nodes))
            post_top1_id = _nid_safe(
                combined_nodes[0]) if combined_nodes else None
            rerank_applied = True
            rerank_changed_top1 = (
                pre_top1_id is not None and post_top1_id is not None and pre_top1_id != post_top1_id)
        else:
            post_top1_id = pre_top1_id
            rerank_applied = False
            rerank_changed_top1 = False

        # does the top1 come from BM25 or dense?
        top1_source = "none"
        top1_id = None
        if combined_nodes:
            try:
                top1_id = combined_nodes[0].node.node_id
            except Exception:
                top1_id = getattr(combined_nodes[0], "node_id", None)

        if top1_id is not None and bm25_nodes:
            bm25_ids = {n.node.node_id for n in bm25_nodes}
            top1_source = "bm25" if top1_id in bm25_ids else "dense"
        elif combined_nodes:
            top1_source = "dense"

        top1_changed_by_bm25 = (
            True if (top1_source == "bm25" and dense_only_top1_id is not None and top1_id != dense_only_top1_id)
            else False
        )

        # -------------------------------
        # 8) Adaptive selection
        # -------------------------------
        pool = list(combined_nodes)
        if window_nodes:
            pool.extend(self._wrap_nodes_with_score(
                window_nodes, base=self.window_nodes_base_score))
        pool = self._wrap_nodes_with_score(
            pool, base=self.pool_nodes_base_score)

        best_by_id = {}
        for n in pool:
            nid = _nid(n)
            sc = float(getattr(n, "score", 0.0) or 0.0)
            cur = best_by_id.get(nid)
            if cur is None or sc > float(getattr(cur, "score", 0.0) or 0.0):
                best_by_id[nid] = n
        pool = list(best_by_id.values())

        selected_chunks = self._select_context_chunks(
            pool,
            required_ids=required_ids,
            min_k=self.min_k,
            max_k=self.max_k,
            budget_chars=self.budget_chars,
            max_per_chunk=self.max_per_chunk,
            diversity_pos_gap=self.diversity_pos_gap,
            rel_drop=self.rel_drop,
            budget_soft=self.budget_soft,
        )

        # -------------------------------
        # 9) Language detection & translation to ES (if most non-ES)
        # -------------------------------
        lang_per_chunk = [_safe_detect(fragment)
                          for fragment in selected_chunks]
        detected = [l for l in lang_per_chunk if l is not None]
        non_es = sum(1 for l in detected if l != "es")

        # Only translate if most detected chunks aren't Spanish
        do_translate = (len(detected) > 0) and (
            non_es / len(detected) >= self.translation_threshold)

        if do_translate:
            translated = []
            for frag, lang in zip(selected_chunks, lang_per_chunk):
                if lang in ("ca", "gl", "eu") and self.multi_translator.available(lang):
                    translated.append(
                        self.multi_translator.translate(frag, lang))
                else:
                    translated.append(frag)
            selected_chunks = translated

        # Prepare metadata about the context preparation process
        context_metadata = {
            'bm25_used': len(bm25_nodes) > 0,
            'bm25_reasons': bm25_reasons if self.enable_bm25 else ["disabled_in_config"],
            'total_nodes': len(nodes),
            'regex_anchor_found': direct is not None,
            'regex_anchor_position': anchor_idx if regex_window_text is not None and win_start is not None else None,
            'retriever_anchor_found': anchor_idx_retr is not None,
            'retriever_anchor_position': anchor_idx_retr,
            'top_dense_score': float(top_dense),
            'top1_source': top1_source,
            'top1_changed_by_bm25': bool(top1_changed_by_bm25),
            'rerank_applied': bool(rerank_applied),
            'rerank_changed_top1': bool(rerank_changed_top1),
            'pre_rerank_top1_id': pre_top1_id,
            'post_rerank_top1_id': post_top1_id,
            'queries_used': queries_used,
            'query_strategy': query_strategy,  # "regex_snippet" | "multiquery"
        }

        self._logger.info(
            "CTX meta: bm25_used=%s reasons=%s regex=%s dense_top=%.3f top1_source=%s changed_by_bm25=%s total_nodes=%d rerank_applied=%s rerank_changed_top1=%s pre_rerank_top1_id=%s post_rerank_top1_id=%s",
            context_metadata['bm25_used'],
            context_metadata['bm25_reasons'],
            context_metadata['regex_anchor_found'],
            context_metadata['top_dense_score'],
            context_metadata['top1_source'],
            context_metadata['top1_changed_by_bm25'],
            context_metadata['total_nodes'],
            context_metadata['rerank_applied'],
            context_metadata['rerank_changed_top1'],
            context_metadata['pre_rerank_top1_id'],
            context_metadata['post_rerank_top1_id'],
        )

        return selected_chunks, context_metadata

    def _prepare_both_contexts(self, text):
        # rerank ON
        self.enable_rerank = True
        ctx_on, meta_on = self._prepare_context(text)
        # rerank OFF
        self.enable_rerank = False
        ctx_off, meta_off = self._prepare_context(text)
        # restore default ON
        self.enable_rerank = True
        return (ctx_on, meta_on), (ctx_off, meta_off)

    def extract(
        self,
        text: str,
        gold_objective: str = None,
        option: str = "generative",
        precomputed_context: tuple = None
    ):
        """
        Extracts the objective from the given text using either generative or extractive prompting, and computes BERT score if a gold objective is provided.

        Parameters
        ----------
        text : str
            The input text from which to extract the objective.
        gold_objective : str, optional
            The ground truth objective for BERT score calculation, defaults to None.
        option : str, optional
            The extraction method to use: "generative" or "extractive", defaults to "generative".
        precomputed_context : tuple, optional
            Tuple of (context chunks, metadata) to use instead of preparing context from text, defaults to None.

        Returns
        -------
        tuple
            A tuple containing:
            - extracted objective (str)
            - context length (int)
            - BERT precision (float)
            - BERT recall (float)
            - BERT F1 score (float)
            - token overlap precision (float)
            - token overlap recall (float)
            - token overlap F1 score (float)
            - context metadata (dict)
        """

        def _is_na(x):
            try:
                return pd.isna(x)
            except Exception:
                return x is None or (isinstance(x, float) and math.isnan(x))

        try:
            if precomputed_context is not None:
                context, context_metadata = precomputed_context
            else:
                context, context_metadata = self._prepare_context(text)

            context_joint = self.separator.join(context)
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
            
            def _parse_result(result):
                if "objeto del contrato no aparece de forma explícita en el contexto proporcionado" in result:
                    return "/"
                return result

            result_str = "" if _is_na(_parse_result(result)) else str(result).strip()

            self._logger.info(f"Objective ({option}): {result_str}")
            
            bert_P = bert_R = bert_F1 = None
            
            # calculate metrics only if gold is given
            if (
                gold_objective is not None
                and not _is_na(gold_objective)
                and isinstance(gold_objective, str)
                and gold_objective.strip()
                and result_str
                and result_str != '/'
            ):
                # Calculate BERT Score
                bert_P, bert_R, bert_F1 = self.get_bert_score(pd.DataFrame({
                    "PREDICTED": [result_str],
                    "GROUND": [gold_objective]
                }))
                # these are tensors, convert to float
                bert_P_val = float(bert_P[0]) if bert_P is not None else None
                bert_R_val = float(bert_R[0]) if bert_R is not None else None
                bert_F1_val = float(bert_F1[0]) if bert_F1 is not None else None
                
                # token overlap metrics
                token_P_val, token_R_val, token_F1_val = self.get_token_overlap_metrics(result_str, gold_objective)
                
                self._logger.info(
                    f"BERT Score - P: {bert_P_val:.4f}, R: {bert_R_val:.4f}, F1: {bert_F1_val:.4f}")
                self._logger.info(
                    f"Token Overlap - P: {token_P_val:.4f}, R: {token_R_val:.4f}, F1: {token_F1_val:.4f}")
            else:
                self._logger.info(
                    "No gold objective provided; skipping metric calculations.")
                bert_P_val = bert_R_val = bert_F1_val = None
                token_P_val = token_R_val = token_F1_val = None
                
            return result_str, len(context), bert_P_val, bert_R_val, bert_F1_val, token_P_val, token_R_val, token_F1_val, context_metadata
        except Exception as e:
            print(f"EXCEPTION in extract(): {e}")
            import pdb
            pdb.set_trace()
            return f"ERROR: {e}", 0, None, None, None, None, None, None, {}
    
    def extract_extractive(self, text: str) -> str:
        """Return only the extractive objective as text."""
        context = self._prepare_context(text)
        return self.extract(text, option="extractive", precomputed_context=context)[0]

    def extract_generative(self, text: str) -> str:
        """Return only the generative objective as text."""
        context = self._prepare_context(text)
        return self.extract(text, option="generative", precomputed_context=context)[0]

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

    def get_bert_score(self, df: pd.DataFrame):
        """Calculate the BERT score of the predictions on the ground truth.
        """

        P, R, F1 = score(df.PREDICTED.values.tolist(
        ), df.GROUND.values.tolist(),
            # lang='es',
            model_type="xlm-roberta-large",
        )  # , #model_type="microsoft/deberta-xlarge-mnli")#self.embedding_model_type)

        # Convert tensors to regular Python values to avoid serialization issues
        P = P.cpu().numpy() if hasattr(P, 'cpu') else P
        R = R.cpu().numpy() if hasattr(R, 'cpu') else R
        F1 = F1.cpu().numpy() if hasattr(F1, 'cpu') else F1

        return P, R, F1

    def get_token_overlap_metrics(self, predicted: str, ground_truth: str):
        """
        Calculate token-level precision, recall, and F1 based on literal token overlap.
        
        This metric measures the proportion of tokens from the ground truth that the model
        has managed to extract literally in its prediction.
        
        Parameters
        ----------
        predicted : str
            The predicted objective text
        ground_truth : str  
            The ground truth objective text
            
        Returns
        -------
        tuple
            (precision, recall, f1) where:
            - precision = correct_tokens / predicted_tokens
            - recall = correct_tokens / ground_truth_tokens  
            - f1 = 2 * precision * recall / (precision + recall)
        """
        import re
        
        def tokenize_spanish(text):
            """Simple tokenization for Spanish text"""
            stopwords = ["objeto", "contrato", "contratación", "objetivo", "pliego", "licitación"]
            if not text or pd.isna(text):
                return []
            # Convert to lowercase and split on whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', str(text).lower())
            return [token for token in tokens if len(token) > 0 and token not in stopwords]
        
        # Tokenize both texts
        pred_tokens = tokenize_spanish(predicted)
        gt_tokens = tokenize_spanish(ground_truth)
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            # Both empty - perfect match
            return 1.0, 1.0, 1.0
        elif len(pred_tokens) == 0:
            # No prediction but ground truth exists
            return 0.0, 0.0, 0.0
        elif len(gt_tokens) == 0:
            # Prediction exists but no ground truth
            return 0.0, 0.0, 0.0
        
        # Convert to sets for overlap calculation
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)
        
        # Calculate overlap
        correct_tokens = len(pred_set.intersection(gt_set))
        
        # Calculate metrics
        precision = correct_tokens / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = correct_tokens / len(gt_set) if len(gt_set) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        return precision, recall, f1

    def get_token_overlap_metrics_batch(self, df: pd.DataFrame):
        """
        Calculate token overlap metrics for a batch of predictions.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'PREDICTED' and 'GROUND' columns
            
        Returns
        -------
        tuple
            (precision_array, recall_array, f1_array) - numpy arrays with metrics for each sample
        """
        import numpy as np
        
        precisions = []
        recalls = []
        f1s = []
        
        for _, row in df.iterrows():
            p, r, f = self.get_token_overlap_metrics(row['PREDICTED'], row['GROUND'])
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            
        return np.array(precisions), np.array(recalls), np.array(f1s)

    def apply_to_dataframe(self, df: pd.DataFrame, mode="both", checkpoint_file=None, checkpoint_every=100):
        """Applies the objective extraction to a pandas DataFrame column with checkpoint support."""
        tqdm.pandas()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input.")
        if self.calculate_on not in df.columns:
            raise ValueError(
                f"Column '{self.calculate_on}' not found in DataFrame.")
        if self.evaluate_on and self.evaluate_on not in df.columns:
            raise ValueError(
                f"Gold objective column '{self.evaluate_on}' not found in DataFrame.")

        valid_modes = {"extractive", "generative", "both"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from {valid_modes}.")

        # Set up checkpoint file
        if checkpoint_file:
            import os
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(checkpoint_file)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Check if checkpoint exists and load it
            if os.path.exists(checkpoint_file):
                self._logger.info(f"Loading checkpoint from {checkpoint_file}")
                try:
                    checkpoint_df = pd.read_parquet(checkpoint_file)
                    # Find where to resume
                    processed_indices = set(checkpoint_df.index)
                    remaining_indices = [i for i in df.index if i not in processed_indices]
                    
                    if not remaining_indices:
                        self._logger.info("All rows already processed. Returning checkpoint data.")
                        return checkpoint_df
                    
                    self._logger.info(f"Resuming from index {min(remaining_indices)}. {len(remaining_indices)} rows remaining.")
                    df_remaining = df.loc[remaining_indices].copy()
                    processed_df = checkpoint_df.copy()
                except Exception as e:
                    self._logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                    df_remaining = df.copy()
                    processed_df = None
            else:
                df_remaining = df.copy()
                processed_df = None
        else:
            df_remaining = df.copy()
            processed_df = None

        def extract_with_metrics(row, option):
            """Helper function to extract and return all metrics as a Series"""
            text = row[self.calculate_on]
            gold = row[self.evaluate_on] if self.evaluate_on else None

            start_time = pd.Timestamp.now()
            obj, context_length, bert_P, bert_R, bert_F1, token_P, token_R, token_F1, context_metadata = self.extract(
                text, gold_objective=gold, option=option)
            end_time = pd.Timestamp.now()
            extraction_time = (end_time - start_time).total_seconds()

            result_dict = {
                f"{option}_objective": obj,
                f"{option}_context_length": context_length,
                f"{option}_bert_precision": bert_P,
                f"{option}_bert_recall": bert_R,
                f"{option}_bert_f1": bert_F1,
                f"{option}_token_precision": token_P,
                f"{option}_token_recall": token_R,
                f"{option}_token_f1": token_F1,
                f"{option}_time_seconds": extraction_time
            }

            # Add context metadata only once (for the first extraction type)
            if option == "extractive" or (option == "generative" and mode != "both"):
                result_dict.update({
                    'bm25_used': context_metadata.get('bm25_used', False),
                    'bm25_reasons': context_metadata.get('bm25_reasons', []),
                    'total_nodes': context_metadata.get('total_nodes', 0),
                    'regex_anchor_found': context_metadata.get('regex_anchor_found', False),
                    'regex_anchor_position': context_metadata.get('regex_anchor_position', None),
                    'retriever_anchor_found': context_metadata.get('retriever_anchor_found', False),
                    'retriever_anchor_position': context_metadata.get('retriever_anchor_position', None)
                })

            return pd.Series(result_dict)

        def process_both(row):
            text = row[self.calculate_on]
            gold = row[self.evaluate_on] if self.evaluate_on else None
            context, context_metadata = self._prepare_context(text)

            start_ext = pd.Timestamp.now()
            obj_ext, ctx_len_ext, bert_P_ext, bert_R_ext, bert_F1_ext, token_P_ext, token_R_ext, token_F1_ext, _ = self.extract(
                text, gold_objective=gold, option="extractive", precomputed_context=(context, context_metadata))
            end_ext = pd.Timestamp.now()
            ext_time = (end_ext - start_ext).total_seconds()

            start_gen = pd.Timestamp.now()
            obj_gen, ctx_len_gen, bert_P_gen, bert_R_gen, bert_F1_gen, token_P_gen, token_R_gen, token_F1_gen, _ = self.extract(
                text, gold_objective=gold, option="generative", precomputed_context=(context, context_metadata))
            end_gen = pd.Timestamp.now()
            gen_time = (end_gen - start_gen).total_seconds()

            return pd.Series({
                "extractive_objective": obj_ext,
                "extractive_bert_precision": bert_P_ext,
                "extractive_bert_recall": bert_R_ext,
                "extractive_bert_f1": bert_F1_ext,
                "extractive_token_precision": token_P_ext,
                "extractive_token_recall": token_R_ext,
                "extractive_token_f1": token_F1_ext,
                "extractive_time_seconds": ext_time,
                "generative_objective": obj_gen,
                "generative_bert_precision": bert_P_gen,
                "generative_bert_recall": bert_R_gen,
                "generative_bert_f1": bert_F1_gen,
                "generative_token_precision": token_P_gen,
                "generative_token_recall": token_R_gen,
                "generative_token_f1": token_F1_gen,
                "generative_time_seconds": gen_time,
                # Context metadata (shared between both extraction types)
                "context_length": ctx_len_gen | ctx_len_ext,
                'bm25_used': context_metadata.get('bm25_used', False),
                'bm25_reasons': context_metadata.get('bm25_reasons', []),
                'total_nodes': context_metadata.get('total_nodes', 0),
                'regex_anchor_found': context_metadata.get('regex_anchor_found', False),
                'regex_anchor_position': context_metadata.get('regex_anchor_position', None),
                'retriever_anchor_found': context_metadata.get('retriever_anchor_found', False),
                'retriever_anchor_position': context_metadata.get('retriever_anchor_position', None)
            })

        time_start = pd.Timestamp.now()

        # Process in batches for checkpointing
        if checkpoint_file and len(df_remaining) > checkpoint_every:
            self._logger.info(f"Processing {len(df_remaining)} rows in batches of {checkpoint_every}")
            
            # Process original df columns to maintain structure
            if processed_df is not None:
                results_list = [processed_df]
            else:
                results_list = []
            
            for i in range(0, len(df_remaining), checkpoint_every):
                batch_df = df_remaining.iloc[i:i+checkpoint_every].copy()
                self._logger.info(f"Processing batch {i//checkpoint_every + 1}: rows {i} to {min(i+checkpoint_every, len(df_remaining))}")
                
                if mode == "extractive":
                    batch_results = batch_df.progress_apply(
                        lambda row: extract_with_metrics(row, "extractive"), axis=1)
                elif mode == "generative":
                    batch_results = batch_df.progress_apply(
                        lambda row: extract_with_metrics(row, "generative"), axis=1)
                elif mode == "both":
                    batch_results = batch_df.progress_apply(process_both, axis=1)
                
                # Combine batch with original data
                batch_combined = pd.concat([batch_df, batch_results], axis=1)
                results_list.append(batch_combined)
                
                # Save checkpoint to the same output file
                combined_so_far = pd.concat(results_list, ignore_index=False)
                combined_so_far.to_parquet(checkpoint_file, index=True)
                self._logger.info(f"Checkpoint saved to {checkpoint_file}: {len(combined_so_far)} rows processed")
            
            # Final result
            final_df = pd.concat(results_list, ignore_index=False)
            
        else:
            # Process all at once (original behavior)
            if mode == "extractive":
                self._logger.info(
                    f"Applying extractive objective extraction to column '{self.calculate_on}'")
                results = df_remaining.progress_apply(
                    lambda row: extract_with_metrics(row, "extractive"), axis=1)

            elif mode == "generative":
                self._logger.info(
                    f"Applying generative objective extraction to column '{self.calculate_on}'")
                results = df_remaining.progress_apply(
                    lambda row: extract_with_metrics(row, "generative"), axis=1)

            elif mode == "both":
                self._logger.info(
                    f"Applying both extractive and generative extraction to column '{self.calculate_on}'")
                results = df_remaining.progress_apply(process_both, axis=1)

            # Combine with original df
            final_df = pd.concat([df_remaining, results], axis=1)
            
            # If we had processed data from checkpoint, combine it
            if processed_df is not None:
                final_df = pd.concat([processed_df, final_df], ignore_index=False)

        time_end = pd.Timestamp.now()
        self._logger.info(f"{mode.capitalize()} objective extraction completed in %.2f seconds",
                          (time_end - time_start).total_seconds())

        return final_df


def main():
    argparser = argparse.ArgumentParser(description="Objective Extractor")
    argparser.add_argument(
        "--config", type=str,
        default="src/rag/config/config.yaml",
        help="Path to the configuration file"
    )
    argparser.add_argument(
        "--path_to_parquet", type=str,
        default="/export/data_ml4ds/NextProcurement/PLACE/temporal/filtrados.parquet",
        help="Path to the input parquet file"
    )
    argparser.add_argument(
        "--path_save", type=str,
        default="/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace_withExtracted",
        help="Path to save the output parquet file"
    )
    argparser.add_argument(
        "--calculate_on", type=str,
        default="texto_tecnico",
        help="Column to calculate the objective on"
    )
    argparser.add_argument(
        "--evaluate_on", type=str,
        default="title",
        help="Column to evaluate the extracted objective against"
    )
    argparser.add_argument(
        "--llm_model_type", type=str,
        default="llama3.1:8b",
        help="LLM model type to use if not specified"
    )
    argparser.add_argument(
        "--llm_model_type_gen", type=str,
        default="mixtral:8x22b",
        help="LLM model type to use for generative extraction"
    )
    argparser.add_argument(
        "--llm_model_type_ex", type=str,
        default="falcon3:10b-instruct-fp16",
        help="LLM model type to use for extractive extraction"
    )
    argparser.add_argument(
        "--mode_extractive_generative", type=str,
        default="both",
        choices=["extractive", "generative", "both"],
        help="Mode of extraction: 'extractive', 'generative', or 'both'")
    argparser.add_argument(
        "--ollama_host", type=str,
        default="http://kumo01.tsc.uc3m.es:11434",
        help="Ollama host URL for LLM requests"
    )
    argparser.add_argument(
        "--enable_checkpoints", action="store_true",
        help="Enable checkpoint saving to the output file. Will save progress every --checkpoint_every rows."
    )
    argparser.add_argument(
        "--checkpoint_every", type=int,
        default=100,
        help="Save checkpoint every N rows (default: 100). Only used if --enable_checkpoints is set."
    )

    args = argparser.parse_args()

    if args.llm_model_type_gen is None:
        llm_model_type_gen = args.llm_model_type
        print(
            f"Using default LLM model type for generative extraction: {llm_model_type_gen}")
    else:
        llm_model_type_gen = args.llm_model_type_gen
        print(
            f"Using LLM model type for generative extraction: {llm_model_type_gen}")

    if args.llm_model_type_ex is None:
        llm_model_type_ex = args.llm_model_type
        print(
            f"Using default LLM model type for extractive extraction: {llm_model_type_ex}")
    else:
        llm_model_type_ex = args.llm_model_type_ex
        print(
            f"Using LLM model type for extractive extraction: {llm_model_type_ex}")

    extractor = ObjectiveExtractor(
        config_path=pathlib.Path(args.config),
        ollama_host=args.ollama_host,
        calculate_on=args.calculate_on,
        evaluate_on=args.evaluate_on,
        llm_model_type_ex=llm_model_type_ex,
        llm_model_type_gen=llm_model_type_gen,
    )

    # read parquet file
    df = pd.read_parquet(args.path_to_parquet)
    extractor._logger.info("Loaded dataframe with %d rows", len(df))

    extractor._logger.info(f"Creating save path: {args.path_save}")
    path_save = pathlib.Path(args.path_save)

    extractor._logger.info(
        f"Extracting objectives from {len(df)} rows in column '{args.calculate_on}'")
    
    # Set up checkpoint file if enabled
    checkpoint_file = None
    if args.enable_checkpoints:
        checkpoint_file = str(path_save)
        extractor._logger.info(f"Checkpoints enabled. Progress will be saved to: {checkpoint_file}")
        extractor._logger.info(f"Checkpoint frequency: every {args.checkpoint_every} rows")
    else:
        extractor._logger.info("Checkpoints disabled. Processing all rows at once.")
    
    df = extractor.apply_to_dataframe(
        df, 
        mode=args.mode_extractive_generative,
        checkpoint_file=checkpoint_file,
        checkpoint_every=args.checkpoint_every
    )

    # Save final result (if checkpoints were disabled or as final confirmation)
    if not args.enable_checkpoints:
        extractor._logger.info("Saving final dataframe to %s", path_save)
        df.to_parquet(path_save, index=False)
    else:
        extractor._logger.info("Final result already saved via checkpoints to %s", path_save)
    extractor._logger.info("Dataframe saved to %s", path_save)


if __name__ == "__main__":
    main()
