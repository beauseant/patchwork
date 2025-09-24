import logging
import pathlib
import warnings
from typing import List, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer


class Embedder(object):
    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 10,
        sg: int = 1,
        default_w2vec: str = "word2vec-google-news-300",
        default_bert: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_sequence_length: int = 384,
        logger: logging.Logger = None,
    ) -> None:

        # Set logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # Set Word2Vec parameters
        self._vector_size = vector_size
        self._window = window
        self._min_count = min_count
        self._sg = sg
        self._default_w2vec = default_w2vec
        # Set BERT parameters
        self._default_bert = default_bert
        self._max_sequence_length = max_sequence_length

    def _check_max_local_length(self,
                                max_seq_length: int,
                                texts: List[str]) -> None:
        """
        Checks if the longest document in the collection is longer than the maximum sequence length allowed by the model

        Parameters
        ----------
        max_seq_length: int
            Context of the transformer model used for the embeddings generation
        texts: list[str]
            The sentences to embed
        """

        max_local_length = np.max([len(t.split()) for t in texts])
        if max_local_length > max_seq_length:
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                          f"truncates to {max_seq_length} tokens.")
        return

    def _get_sentence_embedding(
        self,
        sent: list[str],
        sbert_model_to_load: str ="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
        w2vec_model: Word2Vec = None,
        max_seq_length: int = None,
        method: str = "word2vec"
    ) -> np.ndarray:
        """Get the vector representation of a sentence using a BERT or Word2Vec model. If Word2Vec is used, the input sentence is tokenized and the vector is the average of the vectors of the tokens in the sentence. If BERT is used, the sentence is embedded using the SBERT model specified.

        Parameters
        ----------
        sent: list[str]
            The sentence to get the vector representation.
        w2vec_model: Word2Vec
            The Word2Vec model to use.
        sbert_model_to_load: str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size: int (default=32)
            The batch size used for the computation
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        np.ndarray
            The vector representation of the sentence.
        """

        if method != "word2vec" and method != "bert":
            raise ValueError(
                "The method specified is not valid. Use 'word2vec' or 'bert'.")
        elif method == "word2vec":
            size = w2vec_model.vectors.shape[1]
            vec = np.zeros(size).reshape((1, size))
            count = 0
            for word in sent:
                try:
                    vec += w2vec_model[word].reshape((1, size))
                    count += 1.
                except KeyError:  # handling the case where the token is not in vocabulary
                    continue
            if count != 0:
                vec /= count
            return vec
        else:
            model = SentenceTransformer(sbert_model_to_load)

            if max_seq_length is not None:
                model.max_seq_length = max_seq_length

            #self._check_max_local_length(max_seq_length, sent)
            embeddings = model.encode(
                sent, show_progress_bar=True, batch_size=batch_size)  # .tolist()

            return embeddings

    def _train_w2vec_model(
        self,
        corpus_file: pathlib.Path,
        subword_level: bool = True
    ):
        """
        Train a Word2Vec model using the corpus in the file specified. The model is saved in the same directory as the corpus file with name "model_w2v_{corpus_file.stem}.model".

        Parameters
        ----------
        corpus_file: pathlib.Path
            Path to the file containing the corpus to train the model.

        Returns
        -------
        Word2Vec
            The trained Word2Vec model.
        """

        class IterableSentence_fromfile(object):
            def __init__(self, filename):
                self.__filename = filename

            def __iter__(self):
                for line in open(self.__filename):
                    # assume there's one sentence per line, tokens separated by whitespace
                    yield line.split()

        # Create a memory-friendly iterator
        sentences = IterableSentence_fromfile(corpus_file)

        if subword_level:
            model_w2v =  FastText(
                sentences,
                vector_size=self._vector_size,  # Dimensionality of the word vectors
                window=self._window,  # Context window size
                min_count=self._min_count,  # Ignores all words with total frequency lower than this
                sg=self._sg,  # Training algorithm: 1 for skip-gram; otherwise CBOW
                seed=42)
        else:
            model_w2v = Word2Vec(
                sentences,
                vector_size=self._vector_size,  # Dimensionality of the word vectors
                window=self._window,  # Context window size
                min_count=self._min_count,  # Ignores all words with total frequency lower than this
                sg=self._sg,  # Training algorithm: 1 for skip-gram; otherwise CBOW
                seed=42)

        path_save = corpus_file.parent / f"model_w2v_{corpus_file.stem}.model"
        model_w2v.save(path_save.as_posix())

        return model_w2v

    def infer_embeddings(
        self,
        embed_from: list[list[str]],
        method: str = "word2vec",
        model_path: str = None,
        do_train_w2vec: bool = False,
        corpus_file: pathlib.Path = None,
        subword_level: bool = True
    ) -> list:
        """
        Infer embeddings for a given list of sentences or words using the specified method.

        Parameters
        ----------
            embed_from: list[list[str]]
                A list of sentences or words for which embeddings need to be inferred.
            method: str, optional
                The method to use for inferring embeddings. Defaults to "word2vec".
            model_path str, optional
                The path to the pre-trained model file. Defaults to None.
            do_train_w2vec: bool, optional
                Whether to train a Word2Vec model. Defaults to False.
            corpus_file: pathlib.Path, optional
                The path to the corpus file for training the Word2Vec model. Defaults to None.

        Returns
        -------
            list: A list of embeddings for each sentence or word in embed_from.

        Raises
        ------
            FileNotFoundError
                If the model_path is not provided and do_train_w2vec is False.
        """

        # Get embeddings according to the method specified
        if method == "word2vec":
            self.logger.info(f"-- -- Calculating embeddings with Word2Vec...")
            # Load Word2Vec model. If do_train_w2vec is True, train the model and save it in the same directory as the corpus file with name "model_w2v_{corpus_file.stem}.model". Otherwise, load the model from the path specified in model_path. If model_path is None, load the default model.
            if model_path is None:
                if do_train_w2vec:
                    if corpus_file is None:
                        raise FileNotFoundError(
                            "The corpus file is required to train the Word2Vec model.")
                    model = self._train_w2vec_model(corpus_file)
                else:
                    model_path = self._default_w2vec

            if model_path is not None:
                if subword_level:
                    model = FastText.load(model_path.as_posix())
                else:
                    model = Word2Vec.load(model_path.as_posix())
                self.logger.info(f"Word2Vec model loaded from {model_path}")
            else:
                self.logger.error(f"Word2Vec could not be loaded from {model_path}")
            # If we are just getting the embedding of one word, return the embedding directly
            if len(embed_from) == 1 and len(embed_from[0].split()) == 1:
                self.logger.info(f"Getting embedding for word {embed_from[0][0]}")
                return model.wv[embed_from[0]]
            # If we are getting the embedding of one sentence, return the average of the embeddings of the words in the sentence
            elif len(embed_from) == 1 and len(embed_from[0].split()) > 1:
                self.logger.info(f"Getting embedding for sentence {embed_from[0]}")
                return self._get_sentence_embedding(
                        sent=embed_from[0], w2vec_model= model.wv, method=method)
            # If we are getting the embeddings of multiple sentences, return the embeddings of each sentence
            else:
                self.logger.info(f"Getting embeddings for {len(embed_from)} sentences")
                return [
                    self._get_sentence_embedding(
                        sent=sent, w2vec_model= model.wv, method=method)
                    for sent in embed_from]
        elif method == "bert":
            if len(embed_from) == 1:
                return self._get_sentence_embedding(
                    sent=embed_from[0], sbert_model_to_load=self._default_bert, method=method)
            else:
                return [
                    self._get_sentence_embedding(
                        sent=sent, sbert_model_to_load=self._default_bert, method=method)
                    for sent in embed_from]

    def bert_embeddings_from_df(
        self,
        df: Union[dd.DataFrame, pd.DataFrame],
        text_columns: List[str],
        sbert_model_to_load: str,
        batch_size: int = 32,
        use_dask=False
    ) -> Union[dd.DataFrame, pd.DataFrame]:
        """
        Creates SBERT Embeddings for each row in a dask or pandas dataframe and saves the embeddings in a new column.

        Parameters
        ----------
        df : Union[dd.DataFrame, pd.DataFrame]
            The dataframe containing the sentences to embed
        text_column : str
            The name of the column containing the text to embed
        sbert_model_to_load : str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size : int (default=32)
            The batch size used for the computation

        Returns
        -------
        df: Union[dd.DataFrame, pd.DataFrame]
            The dataframe with the original data and the generated embeddings
        """

        model = SentenceTransformer(sbert_model_to_load)
        model.max_seq_length = self._max_sequence_length
        
        def encode_text(text):
            embedding = model.encode(text,show_progress_bar=True, batch_size=batch_size)
            # Convert to string
            embedding = ' '.join(str(x) for x in embedding)
            return embedding

        for col in text_columns:
            self._check_max_local_length(self._max_sequence_length, df[col])

            col_emb = col.split(
                "_")[0]+"_embeddings" if len(text_columns) > 1 else "embeddings"

            if use_dask:
                df[col_emb] = df[col].apply(
                    encode_text, meta=('x', 'str'))
            else:
                df[col_emb] = df[col].apply(encode_text)

        return df
