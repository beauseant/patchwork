import json
import logging
import pickle
import shutil
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BaseModel:
    def __init__(
        self,
        model_dir: Union[str, Path],
        stop_words: list = [],
        word_min_len: int = 2,
        logger: logging.Logger = None,
    ):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.num_topics = None
        self.stop_words = stop_words
        self.word_min_len = word_min_len
        self._thetas_thr = 3e-3

        # Create sub-directories
        self._model_data_dir = self.model_dir / "model_data"
        self._model_data_dir.mkdir(parents=True, exist_ok=True)
        self._train_data_dir = self.model_dir / "train_data"
        self._train_data_dir.mkdir(parents=True, exist_ok=True)
        self._infer_data_dir = self.model_dir / "infer_data"
        self._infer_data_dir.mkdir(parents=True, exist_ok=True)
        self._test_data_dir = self.model_dir / "test_data"
        self._test_data_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(tempfile.gettempdir())

    @abstractmethod
    def _model_train(
        self, texts: List[str], num_topics: int, **kwargs
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        pass

    @abstractmethod
    def _model_predict(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def _createTMmodel(self):
        pass

    def _SaveThrFig(self, thetas32, plotFile):
        allvalues = np.sort(thetas32.flatten())
        step = max(1, int(np.round(len(allvalues) / 1000)))
        plt.semilogx(
            allvalues[::step],
            (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step],
        )
        plt.semilogx([self._thetas_thr, self._thetas_thr], [0, 100], "r")
        plt.savefig(plotFile)
        plt.close()

    def train(self, texts: List[str], ids: List[int], **kwargs):
        probs, topic_keys = self._model_train(texts, ids, **kwargs)
        self._save_train_texts(texts, ids)
        self._save_train_doctopics(probs)
        self._save_topickeys(topic_keys)

    def predict(self, texts: List[str]):
        probs = self._model_predict(texts)
        self._save_infer_doctopics(probs)

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Save methods ──────────────────────────────────────────────

    def _save_train_texts(self, texts: List[str], ids: List[int], sep="\t"):
        tmp = self._temp_dir / "corpus.txt"
        with tmp.open("w", encoding="utf8") as f:
            f.writelines([f"{n}{sep}0{sep}{t}\n" for n, t in zip(ids, texts)])
        shutil.copy(tmp, self._train_data_dir / "corpus.txt")
        self.logger.info("Saved train texts")

    def _save_train_doctopics(self, doctopics: np.ndarray, sep="\t"):
        tmp = self._temp_dir / "doc-topics.txt"
        with tmp.open("w", encoding="utf8") as f:
            f.writelines(
                [
                    f"{n}{sep}{n}{sep}{sep.join(t)}\n"
                    for n, t in enumerate(doctopics.astype(str))
                ]
            )
        shutil.copy(tmp, self._model_data_dir / "doc-topics.txt")
        self.logger.info("Saved train doctopics")

    def _save_infer_doctopics(self, doctopics: np.ndarray, sep="\t"):
        tmp = self._temp_dir / "doc-topics.txt"
        with tmp.open("w", encoding="utf8") as f:
            f.writelines(
                [
                    f"{n}{sep}{n}{sep}{sep.join(t)}\n"
                    for n, t in enumerate(doctopics.astype(str))
                ]
            )
        shutil.copy(tmp, self._infer_data_dir / "doc-topics.txt")
        self.logger.info("Saved infer doctopics")

    def _save_topickeys(self, topickeys: Dict[int, str]):
        tmp = self._temp_dir / "topic-keys.json"
        with tmp.open("w", encoding="utf8") as f:
            json.dump(topickeys, f)
        shutil.copy(tmp, self._model_data_dir / "topic-keys.json")
        self.logger.info("Saved topic keys")

    # ── Read methods ──────────────────────────────────────────────

    def read_doctopics(self):
        with (self._model_data_dir / "doc-topics.txt").open("r", encoding="utf8") as f:
            return np.loadtxt(f)[:, 2:]

    def read_topickeys(self) -> Dict[int, str]:
        with (self._model_data_dir / "topic-keys.json").open("r", encoding="utf8") as f:
            return json.load(f)

    def get_topics_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        return {k: v.split()[:n_words] for k, v in self.read_topickeys().items()}

    def show_words_per_topic(self, n_words=10) -> pd.DataFrame:
        topic_words = self.get_topics_words(n_words)
        df = pd.DataFrame.from_dict(
            topic_words, orient="index",
            columns=[f"Word_{i}" for i in range(n_words)],
        )
        df.index.name = "Topic"
        return df
