"""
This script offers a basic lemmatizer that leverages the spaCy library to lemmatize a given text.

Author: Lorena Calvo-Bartolomé
Date: 07/03/2024
"""

import configparser
import logging
from typing import List
import spacy
import pandas as pd

class Lemmatizer(object):
    def __init__(
        self,
        logger: logging.Logger = None,
        config_file: str = "/config/config.cf",
    ) -> None:

        # Set logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger(__name__)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)

        # By default uses these but they could be configured in the config file (they need to be also downloaded from spacy. The latter is done in the Dockerfile)
        self._nlp_es = spacy.load("es_dep_news_trf")
        self._nlp_en = spacy.load("en_core_web_trf")

    def lemmatize(
        self,
        text: List[str],
        language: str = "es",
    ) -> str:
        """
        Lemmatize a given text.

        Parameters
        ----------
        text : str
            The text to lemmatize.
        language : str, optional
            The language of the text. Default is "es" (Spanish).

        Returns
        -------
        str
            The lemmatized text.
        """

        if language == "es":
            nlp = self._nlp_es
        elif language == "en":
            nlp = self._nlp_en
        else:
            self._logger.error(
                f"Language not supported. Please use 'es' or 'en'.")
            raise ValueError(
                "Language not supported. Please use 'es' or 'en'.")

        # convert to dataframe for efficiency
        df_text = pd.DataFrame(text, columns=["text"])
        
        def lemmatize_text(text: str) -> str:
            doc = nlp(text)
            lemmatized_words = [token.lemma_ for token in doc]

            return " ".join(lemmatized_words)
        
        df_text["lemmatized_text"] = df_text["text"].apply(lemmatize_text)

        return df_text["lemmatized_text"].values.tolist()
