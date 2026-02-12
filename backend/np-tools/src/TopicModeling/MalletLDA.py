import logging
import pathlib
import shutil
from pathlib import Path
from subprocess import check_output
from typing import List, Union

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .BaseModel import BaseModel
from src.TopicModeling.tm_utils.tm_model import TMmodel
from src.TopicModeling.tm_utils.utils import file_lines


class MalletLDAModel(BaseModel):
    """
    Wrapper for training Mallet LDA topic models.
    Handles corpus import, training, TMmodel creation and inference.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        stop_words: list = [],
        word_min_len: int = 2,
        mallet_path="/app/Mallet/bin/mallet",
        logger: logging.Logger = None,
    ):
        super().__init__(model_dir, stop_words, word_min_len, logger)
        self._mallet_path = Path(mallet_path)
        self._temp_mallet_dir = self._temp_dir / Path(model_dir).stem
        self._temp_mallet_dir.mkdir(parents=True, exist_ok=True)

    def _create_corpus_mallet(
        self,
        texts: List[str],
        ids: List[int],
        predict: bool = False,
        path_model: str = None,
        mallet_path: str = None,
        save_temp = True
    ):
        """
        Generate .txt and .mallet files to train LDA model.
        Returns path to generated .mallet file.
        """
        # Define directories
        if save_temp:
            path_save = self._temp_mallet_dir
        else:
            path_save = pathlib.Path(path_model) / ("infer_data")
        if predict:
            texts_txt_path = path_save.joinpath(
                "corpus_predict.txt")
            texts_mallet_path = path_save.joinpath("corpus_predict.mallet")
        else:
            texts_txt_path = path_save.joinpath("corpus_train.txt")
            texts_mallet_path = path_save.joinpath(
                "corpus_train.mallet")
            
        if not mallet_path:
            mallet_path = self._mallet_path

        # Create corpus.txt
        self.logger.info("Creating corpus.txt...")
        with texts_txt_path.open("w", encoding="utf8") as fout:
            for i, t in zip(ids, texts):
                fout.write(f"{i} 0 {t}\n")
        self.logger.info(f"corpus.txt created")

        # Convert corpus.txt to corpus.mallet
        self.logger.info("Creating corpus.mallet...")
        
        texts_txt_path_tr = pathlib.Path(path_model) / "train_data/corpus.txt"
        texts_mallet_path_tr = pathlib.Path(path_model) / "train_data/corpus.mallet"
        cmd_pre = (
            f"{mallet_path} import-file "
            f"--input {texts_txt_path_tr} "
            f"--output {texts_mallet_path_tr} "
            f"--keep-sequence "
        )
        self.logger.info(f"Running pre: {cmd_pre}")
        check_output(args=cmd_pre, shell=True)

        cmd = (
            f"{mallet_path} import-file "
            f"--input {texts_txt_path} "
            f"--output {texts_mallet_path} "
            f"--keep-sequence "
            # f"--remove-stopwords "
        )
        if predict:
            inferencer_mallet = pathlib.Path(path_model) / "train_data" / "corpus.mallet" #/ "model_data" / "inferencer.mallet"
            cmd += f"--use-pipe-from {inferencer_mallet}"
        check_output(args=cmd, shell=True)
        self.logger.info(f"Running command: {cmd}")
        self.logger.info(f"corpus.mallet created")

        # Move info to desired location
        if save_temp:
            name = self._infer_data_dir if predict else self._train_data_dir
            texts_txt_path, texts_mallet_path = shutil.copy(
                texts_txt_path, name.joinpath("corpus.txt")), shutil.copy(
                texts_mallet_path, name.joinpath("corpus.mallet"))

        return Path(texts_mallet_path)

    def _model_train(
        self,
        texts: List[str],
        ids: List[int],
        num_topics: int,
        alpha: float = 5.0,
        optimize_interval: int = 10,
        num_threads: int = 4,
        num_iterations: int = 1000,
        doc_topic_thr: float = 0.0,
        texts_test: List[str] = None,
        ids_test: List[int] = None,
        **kwargs,
    ):
        self.num_topics = num_topics
        config_file = self._train_data_dir / "train.config"

        # Generate corpus.mallet
        texts_mallet_path = self._create_corpus_mallet(texts, ids, predict=False)

        # Save test data if available
        if texts_test is not None:
            texts_test_path = self._temp_mallet_dir / "corpus_test.txt"
            self.logger.info("Creating Test corpus.txt...")
            with texts_test_path.open("w", encoding="utf8") as fout:
                for i, t in zip(ids_test, texts_test):
                    fout.write(f"{i} 0 {t}\n")
            shutil.copy(texts_test_path, self._test_data_dir / "corpus.txt")

        # Generate train config
        with config_file.open("w", encoding="utf8") as fout:
            fout.write(f"input = {texts_mallet_path}\n")
            fout.write(f"num-topics = {self.num_topics}\n")
            fout.write(f"alpha = {alpha}\n")
            fout.write(f"optimize-interval = {optimize_interval}\n")
            fout.write(f"num-threads = {num_threads}\n")
            fout.write(f"num-iterations = {num_iterations}\n")
            fout.write(f"doc-topics-threshold = {doc_topic_thr}\n")
            fout.write(f"output-doc-topics = {self._temp_mallet_dir / 'doc-topics.txt'}\n")
            fout.write(f"word-topic-counts-file = {self._temp_mallet_dir / 'word-topic-counts.txt'}\n")
            fout.write(f"diagnostics-file = {self._temp_mallet_dir / 'diagnostics.xml'}\n")
            fout.write(f"xml-topic-report = {self._temp_mallet_dir / 'topic-report.xml'}\n")
            fout.write(f"output-topic-keys = {self._temp_mallet_dir / 'topic-keys.txt'}\n")
            fout.write(f"inferencer-filename = {self._temp_mallet_dir / 'inferencer.mallet'}\n")

        cmd = f"{self._mallet_path} train-topics "
        with config_file.open("r") as f:
            cmd += " ".join(
                [f"--{line.strip()}" for line in f.readlines()]
            ).replace("=", "")

        check_output(args=cmd, shell=True)
        self.logger.info("Finished training")

        # Copy outputs to model_data
        for fname in [
            "doc-topics.txt", "word-topic-counts.txt", "diagnostics.xml",
            "topic-report.xml", "topic-keys.txt", "inferencer.mallet",
        ]:
            shutil.copy(self._temp_mallet_dir / fname, self._model_data_dir / fname)

        pred = np.loadtxt(
            self._model_data_dir / "doc-topics.txt",
            usecols=range(2, self.num_topics + 2),
        )
        with (self._model_data_dir / "topic-keys.txt").open("r", encoding="utf8") as f:
            topic_keys = [t.strip().split("\t") for t in f.readlines()]
        topic_keys = {t[0]: t[-1] for t in topic_keys}

        # Create TMmodel
        self._createTMmodel()

        return pred, topic_keys

    def _model_predict(
        self, 
        texts: List[str],
        path_model: str = None,
        path_mallet: str = None,
        save_temp = True
    ):
        """
        Infer doctopic file of corpus using ntopics model

        Parameters:
        -----------
        corpus: list(str)
            Preprocessed corpus as a list of strings

        """
        self.logger.info(f"Mallet path from args: {path_mallet}")
        
        # Define directories
        if save_temp:
            temp_mallet_dir = self._temp_mallet_dir.joinpath('predicted_topics.txt')
            predicted_doctopic = self._infer_data_dir.joinpath(
            "predicted_topics.txt")
        else:
            temp_mallet_dir = pathlib.Path(path_model) / ("model_data") /'predicted_topics.txt'
            predicted_doctopic = temp_mallet_dir
        
        # Get inferencer
        if path_model:
            dir_inferencer = pathlib.Path(path_model) / "model_data" / "inferencer.mallet"
            if not dir_inferencer.is_file():
                self.logger.warning(f"Path provided does not leead to inferencer. Train model first.")
                return
        else:
            if self._temp_mallet_dir.joinpath("inferencer.mallet").exists():
                dir_inferencer = self._temp_mallet_dir.joinpath(
                    "inferencer.mallet")
            elif self._model_data_dir.joinpath("inferencer.mallet").exists():
                dir_inferencer = self._model_data_dir.joinpath("inferencer.mallet")
            else:
                self.logger.warning("There is no inferencer. Train model first.")
                return
            
        if not path_mallet:
            self.logger.info("Mallet path not provided. Using default path.")
            path_mallet = self._mallet_path
        
        self.logger.info(f"Path to mallet: {path_mallet}")

        self.logger.info("Infer topics")

        # Generate corpus.mallet
        texts_mallet_infer_path = self._create_corpus_mallet(
            texts, 
            ids = list(range(len(texts))),
            predict=True,
            path_model=path_model,
            save_temp=save_temp,
            mallet_path=path_mallet)
        
        # Infer topics
        cmd = (
            f"{path_mallet} infer-topics "
            f"--inferencer {dir_inferencer} "
            f"--input {texts_mallet_infer_path} "
            f"--output-doc-topics {temp_mallet_dir} "
        )
        self.logger.info(f"Running command to infer topics: {cmd}")
        check_output(args=cmd, shell=True)
        
        if save_temp:
            shutil.copy(temp_mallet_dir,predicted_doctopic)
            
        self.logger.info("Finished topic inference")

        pred = np.loadtxt(predicted_doctopic,
                          usecols=range(2, self.num_topics + 2))
        return pred

    def _createTMmodel(self):
        # Load and sparsify thetas
        thetas_file = self._model_data_dir / "doc-topics.txt"
        cols = list(np.arange(2, self.num_topics + 2))
        thetas32 = np.loadtxt(
            thetas_file, delimiter="\t", dtype=np.float32, usecols=cols
        )

        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm="l1")
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Build beta matrix and vocabulary from word-topic-counts
        wtcFile = self._model_data_dir / "word-topic-counts.txt"
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self.num_topics, vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))
        with wtcFile.open("r", encoding="utf8") as fin:
            for i, line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(":")[0])
                    cnt = int(counts.split(":")[1])
                    betas[tpc, i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas, axis=1, norm="l1")

        # Save vocabulary and frequencies
        vocabfreq_file = self._model_data_dir / "vocab_freq.txt"
        with vocabfreq_file.open("w", encoding="utf8") as fout:
            for w, freq in zip(vocab, term_freq):
                fout.write(f"{w}\t{int(freq)}\n")

        tm = TMmodel(TMfolder=self._model_data_dir / "TMmodel")
        tm.create(betas=betas, thetas=thetas32, alphas=alphas, vocab=vocab)

        # Remove doc-topics file
        thetas_file.unlink()

        return tm
