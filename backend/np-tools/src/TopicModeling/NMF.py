from typing import List, Union

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import trange

from .BaseModel import BaseModel


class NMFModel(BaseModel):
    def _model_train(
        self,
        texts: List[str],
        num_topics: int,
        vectorizer: Union[str, TfidfVectorizer, CountVectorizer] = "tfidf",
        max_df: float = 0.8,
        min_df: Union[float, int] = 1,
    ):
        # Set num topics
        self.num_topics = num_topics

        # Set vectorizer
        if vectorizer == "tfidf":
            vectorizer = TfidfVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=max_df,
                min_df=min_df,
            )
        elif vectorizer == "count":
            vectorizer = CountVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=max_df,
                min_df=min_df,
            )

        self.vectorizer = vectorizer
        tfidf = self.vectorizer.fit_transform(texts)
        self.logger.info("Texts vectorized")
        self.model = NMF(n_components=self.num_topics, random_state=42)
        probs = self.model.fit_transform(tfidf)
        self.logger.info("Finished training")

        # Get topickeys
        topic_keys = dict()
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.model.components_):
            topic_keys[topic_idx] = " ".join(
                [feature_names[i] for i in np.argsort(topic)[:-21:-1]]
            )

        return probs, topic_keys

    def _model_predict(self, texts: List[str]):
        tfidf = self.vectorizer.transform(texts)
        pred = self.model.transform(tfidf)
        return pred
