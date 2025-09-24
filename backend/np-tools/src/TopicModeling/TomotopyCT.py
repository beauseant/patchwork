from typing import List

import numpy as np
import tomotopy as tp
from tqdm import tqdm, trange

from .BaseModel import BaseModel


class TomotopyCTModel(BaseModel):
    def _model_train(self, texts: List[str], num_topics: int, iterations=400):
        # Set num topics
        self.num_topics = num_topics

        # Set model
        self.model = tp.CTModel(k=self.num_topics, min_cf=0, min_df=10, seed=42)
        texts = [t.split() for t in texts]
        for text in texts:
            self.model.add_doc(text)
        self.logger.info("Texts loaded")
        # self.model.train(0)

        batch = 10
        progress = 0
        pbar = tqdm(total=iterations, desc="Cleaning corpus", leave=True)
        for i in range(0, iterations, batch):
            self.model.train(10)
            update = min(batch, iterations - i)
            progress += update
            pbar.set_description(
                f"Iteration:{progress}\tLL:{self.model.ll_per_word:.3f}", refresh=False
            )
            # pbar.refresh()
            pbar.update(update)
        pbar.close()
        self.logger.info("Finished training")

        # Corpus predictions
        probs = []
        for d in self.model.docs:
            probs.append(d.get_topic_dist())
        probs = np.array(probs)

        # Get topickeys
        topic_keys = dict()
        for topic_idx in range(num_topics):
            topic_keys[topic_idx] = " ".join(
                [word for word, _ in self.model.get_topic_words(topic_idx, top_n=20)]
            )

        return probs, topic_keys

    def _model_predict(self, texts: List[str]):
        texts = [t.split() for t in texts]
        doc_inst = [self.model.make_doc(text) for text in texts]
        topic_prob, log_ll = self.model.infer(doc_inst)
        pred = np.array(topic_prob)
        return pred
