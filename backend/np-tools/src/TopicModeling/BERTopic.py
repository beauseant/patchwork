from typing import List, Union

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
#from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import trange
from umap import UMAP
import shutil
from .BaseModel import BaseModel


class BERTopicModel(BaseModel):
    def _model_train(
        self,
        texts: List[str],
        num_topics: int,
        embedding_model: Union[str, SentenceTransformer, None] = None,
        umap_model: Union[UMAP, None] = None,
        hdbscan_model: Union[HDBSCAN, None] = None,
        vectorizer_model: Union[CountVectorizer, TfidfVectorizer, None] = None,
        ctfidf_model: Union[ClassTfidfTransformer, None] = None,
        representation_model: Union[MaximalMarginalRelevance, None] = None,
        verbose=True,
        texts_test=None,
    ):
        # Set num topics
        self.num_topics = num_topics
        
        # Save test data if available
        if texts_test is not None:
            texts_test_path = self._temp_mallet_dir.joinpath("corpus_test.txt")
            self.logger.info("Creating Test corpus.txt...")
            with texts_test_path.open("w", encoding="utf8") as fout:
                for i, t in enumerate(texts_test):
                    fout.write(f"{i} 0 {t}\n")
            self.logger.info(f"Test corpus.txt created")
            texts_test_path = shutil.copy(
                texts_test_path, self._test_data_dir.joinpath("corpus.txt"))

        # Step 1 - Extract embeddings
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

        # Step 2 - Reduce dimensionality
        if umap_model:
            self.umap_model = umap_model
        else:
            self.umap_model = UMAP(
                n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"
            )

        # Step 3 - Cluster reduced embeddings
        if hdbscan_model:
            self.hdbscan_model = hdbscan_model
        else:
            self.hdbscan_model = HDBSCAN(
                min_cluster_size=500,
                min_samples=2,
                metric="euclidean",
                prediction_data=True,
            )

        # Step 4 - Tokenize topics
        if vectorizer_model:
            self.vectorizer_model = vectorizer_model
        else:
            vectorizer_model = CountVectorizer(
                token_pattern=self.word_pattern,
                stop_words=self.stop_words,
                # ngram_range=(1, 2),
                # vocabulary=vocabulary,
                max_df=0.8,
                min_df=1,
            )

        # Step 5 - Create topic representation
        if ctfidf_model:
            self.ctfidf_model = ctfidf_model
        else:
            self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        # Step 6 - (Optional) Fine-tune topic representations with
        # a `bertopic.representation` model
        if representation_model:
            self.representation_model = representation_model
        else:
            self.representation_model = MaximalMarginalRelevance(
                diversity=0.3, top_n_words=20
            )

        self.model = BERTopic(
            language="multilingual",
            nr_topics=self.num_topics,
            low_memory=False,
            calculate_probabilities=False,
            seed_topic_list=None,
            embedding_model=self.embedding_model,  # Step 1 - Extract embeddings
            umap_model=self.umap_model,  # Step 2 - Reduce dimensionality
            hdbscan_model=self.hdbscan_model,  # Step 3 - Cluster reduced embeddings
            vectorizer_model=self.vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=self.ctfidf_model,  # Step 5 - Extract topic words
            representation_model=self.representation_model,  # Step 6 - (Optional) Fine-tune topic represenations
            verbose=verbose,
        )
        # self.model.fit(texts)
        self.model.calculate_probabilities = True
        topics, probs = self.model.fit_transform(texts)
        self.model.calculate_probabilities = False

        # Reduce outliers
        new_topics = self.model.reduce_outliers(
            texts, topics, probabilities=probs, strategy="embeddings", threshold=0.0
        )
        self.model.update_topics(
            docs=texts,
            topics=new_topics,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
        )
        self.num_topics = len(self.model.topic_labels_)
        self.logger.info("Finished training")

        # Get topickeys
        topic_keys = dict()
        for k, v in self.model.get_topics().items():
            topic_keys[k] = " ".join([el[0] for el in v])

        return probs, topic_keys

    def _model_predict(self, texts):
        self.model.calculate_probabilities = True
        topics, pred = self.model.transform(texts)
        self.model.calculate_probabilities = False
        return pred
