import os
import json
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CPV8ClassifierHF:
    """
    Clasificador CPV-8 basado en un modelo fine-tuned de Hugging Face.
    Carga un snapshot del repo (público), lee labels.json (códigos CPV-8)
    y opcionalmente un global_threshold.npy para usar el umbral global
    (estrategia precision_tilt).
    """

    def __init__(
        self,
        repo_id: Optional[str] = "erick4556/cpv8-bert-spanish-ft",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        # por si no hay global_threshold.npy en el repo
        default_threshold: float = 0.80,
        hf_local_dir: str = "/models/cpv8",   # dónde cachear el snapshot
        max_length: int = 384,                # longitud de tokenización para inferencia
        # limita threads en CPU (opcional)
        cpu_threads: Optional[int] = None
    ):
        # Logger
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Repo HF
        self.repo_id = repo_id
        if not self.repo_id:
            raise ValueError(
                "Debes especificar repo_id (p.ej. 'usuario/mi-modelo-cpv8').")

        # Control opcional de threads en CPU
        if cpu_threads is not None and cpu_threads > 0:
            try:
                torch.set_num_threads(cpu_threads)
                os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            except Exception as e:
                self.logger.warning(f"No se pudo fijar threads CPU: {e}")

        # Descarga snapshot (si está cacheado, no re-descarga)
        self.model_dir = snapshot_download(
            repo_id=self.repo_id,
            local_dir=hf_local_dir,
            local_dir_use_symlinks=False,
        )
        self.logger.info(f"Modelo descargado/listo en: {self.model_dir}")

        # Carga etiquetas CPV-8
        labels_path = os.path.join(self.model_dir, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"No se encontró labels.json en {self.model_dir}. "
                "Asegúrate de haber subido las etiquetas (CPV-8) junto al modelo."
            )
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)

        # Verificación básica: que parezcan CPV-8
        bad = [c for c in self.labels if not (
            isinstance(c, str) and c.isdigit() and len(c) == 8)]
        if bad:
            self.logger.warning(
                f"labels.json contiene códigos no-CPV8 (ej: {bad[:3]}) — revisa tu artefacto."
            )

        # Umbral global (precision_tilt) si fue exportado; si no, usa default
        global_thr_path = os.path.join(self.model_dir, "global_threshold.npy")
        if os.path.exists(global_thr_path):
            thr = np.load(global_thr_path)
            self.global_threshold: float = float(
                thr[0] if getattr(thr, "ndim", 0) > 0 else thr)
        else:
            self.global_threshold = float(default_threshold)
            self.logger.warning(
                f"No se encontró global_threshold.npy. Usando umbral por defecto={self.global_threshold:.3f}"
            )

        # Tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.labels),
            problem_type="multi_label_classification",
        )

        # Dispositivo
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Otros
        self.max_length = int(max_length)

        self.logger.info(
            f"Cargado modelo {self.repo_id} | etiquetas={len(self.labels)} | "
            f"umbral_global={self.global_threshold:.3f} | device={self.device}"
        )

    # --------- Utilidades internas ---------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _predict_single_logits(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.detach().cpu().numpy()[0]
        return logits

    def _top1_with_threshold(self, probs: np.ndarray) -> Tuple[str, float]:

        mask = probs >= self.global_threshold
        if mask.any():
            idx = int(np.argmax(probs * mask))
        else:
            idx = int(np.argmax(probs))
        return str(self.labels[idx]), float(probs[idx])

    # --------- API pública ---------
    def predict_one(self, text: str) -> Dict[str, str]:
        """
        Devuelve: {"original_text": <texto>, "cpv_predicted": "<CPV8>", "prob": <float>}
        """
        logits = self._predict_single_logits(text)
        probs = self._sigmoid(logits)
        code, prob = self._top1_with_threshold(probs)

        # Sanity check: 8 dígitos
        if not (code.isdigit() and len(code) == 8):
            self.logger.warning(
                f"Etiqueta no es 8 dígitos: '{code}' (revisa labels.json)")
        return {"original_text": text, "cpv_predicted": code}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Lote simple (secuencial). Para grandes volúmenes.
        """
        return [self.predict_one(t) for t in texts]
