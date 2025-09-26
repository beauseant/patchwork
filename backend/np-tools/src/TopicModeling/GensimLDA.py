from typing import List

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore

from .BaseModel import BaseModel


class GensimLDAModel(BaseModel):
    def _model_train(self, texts: List[str], num_topics: int, iterations=400):
        # Set num topics
        self.num_topics = num_topics

        # Convert texts to format
        texts = [t.split() for t in texts]
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.logger.info("Texts transformed")
        self.model = LdaModel(
            corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            iterations=iterations,
            random_state=42,
        )
        self.logger.info("Finished training")

        # Corpus predictions
        probs = np.zeros(shape=(len(corpus), self.num_topics))
        for d, doctopics in enumerate(self.model.get_document_topics(corpus)):
            for topic_idx, topic_prob in doctopics:
                probs[d][topic_idx] = topic_prob

        # Get topickeys
        topic_keys = dict()
        for topic_idx in range(self.num_topics):
            topic_keys[topic_idx] = " ".join(
                list(zip(*self.model.show_topic(topic_idx, 20)))[0]
            )

        return probs, topic_keys

    def _model_predict(self, texts: List[str]):
        # Convert texts to format
        texts = [t.split() for t in texts]
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        pred = np.zeros(shape=(len(texts), self.num_topics))
        for d, bow in enumerate(corpus):
            for topic_idx, topic_prob in self.model.get_document_topics(bow):
                pred[d][topic_idx] = topic_prob
        return pred
