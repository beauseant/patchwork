"""
This  module provides 2 classes to handle NP Tools API responses and requests.

The NPToolsResponse class handles NP Tools API response and errors, while the NPToolsClient class handles requests to the NP Tools API.

Author: Lorena Calvo-Bartolomé
Date: 21/05/2023
"""

import logging
import os
from typing import List
from urllib.parse import urlencode

import requests
from src.core.clients.external.api_generic.client import Client
from src.core.clients.external.api_generic.response import Response


class NPToolsResponse(Response):
    """
    A class to handle Inferencer API response and errors.
    """

    def __init__(
        self,
        resp: requests.Response,
        logger: logging.Logger
    ) -> None:

        super().__init__(resp, logger)
        return


class NPToolsClient(Client):
    """
    A class to handle NP Tools API requests.
    """

    def __init__(
        self,
        logger: logging.Logger,
        timeout: int = 120
    ) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        timeout : int, optional
            The timeout of the request in seconds, by default 120.
        """

        super().__init__(logger, "NP Tools Client")

        # Get the NP Tools URL from the environment variables
        self.nptools_url = os.environ.get('NP_TOOLS_URL')
        self.timeout = timeout

        return

    def _do_request(
        self,
        type: str,
        url: str,
        timeout: int = 120,
        **params
    ) -> NPToolsResponse:
        """Sends a request to the Inferencer API and returns an object of the NPToolsResponse class.

        Parameters
        ----------
        type : str
            The type of the request.
        url : str
            The URL of the Inferencer API.
        timeout : int, optional
            The timeout of the request in seconds, by default 10.
        **params: dict
            The parameters of the request.

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        # Send request
        resp = super()._do_request(type, url, timeout, **params)

        # Parse NP Tools response
        inf_resp = NPToolsResponse(resp, self.logger)

        return inf_resp

    def get_embedding(
        self,
        text_to_embed: str,
        embedding_model: str,
        lang: str,
        model_for_embedding: str = None
    ) -> NPToolsResponse:
        """Get the embedding of a word using the given model.

        Parameters
        ----------
        word_to_embed : str
            The word to embed.
        embedding_model : str
            The model to use for embedding (either 'word2vec' or 'bert')
        model_for_embedding : str
            The model to use for embedding, i.e., the topic model on the basis of which the embeddings where generated,
        lang : str
            The language of the text to be lemmatized (es/en)

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        if model_for_embedding is None:
            params_ = {
                'text_to_embed': text_to_embed,
                'embedding_model': embedding_model,
                'lang': lang
            }
        else:
            params_ = {
                'text_to_embed': text_to_embed,
                'embedding_model': embedding_model,
                'model': model_for_embedding,
                'lang': lang
            }

        encoded_params = urlencode(params_)

        url_ = '{}/embedding/generate/?{}'.format(
            self.nptools_url, encoded_params)

        self.logger.info(f"-- -- get_embedding - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="post", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_embedding - Response: {resp}")

        return resp

    def get_lemmas(
        self,
        text_to_lemmatize: str,
        lang: str,
    ) -> NPToolsResponse:
        """Get the lemmas of a text.

        Parameters
        ----------
        text_to_lemmatize : str
            The word to lemmatize.
        embedding_model : str
        lang : str
            The language of the text to be lemmatized (es/en)

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_lemmatize': text_to_lemmatize,
            'lang': lang
        }

        encoded_params = urlencode(params_)

        url_ = '{}/lemmatization/extract/?{}'.format(
            self.nptools_url, encoded_params)

        self.logger.info(f"-- -- get_lemmas - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="post", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_lemmas - Response: {resp}")

        return resp

    def get_thetas(
        self,
        text_to_infer: List[str],
        model_for_infer: str,
    ) -> NPToolsResponse:
        """Get the thetas representation for a document based on a given trained topic model. 
        The format of the response from the NP Tools API is as follows:

        {
            "responseHeader": {
                "status": 200,
                "time": 2.7594828605651855
            },
            "response": [
                {
                "id": 0,
                "thetas": "t0|188 t1|244 t2|210 t3|249 t4|109"
                }
            ]
        }

        Parameters
        ----------
        text_to_infer : str
            Text to be inferred.
        model_for_infer : str
            The model to be used for inference.

        Returns
        -------
        NPToolsResponse: NPToolsResponse
            An object of the NPToolsResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_infer': ",".join(text_to_infer),
            'model_name': model_for_infer,
        }

        encoded_params = urlencode(params_)

        url_ = '{}/inference/predict/?{}'.format(
            self.nptools_url, encoded_params)

        self.logger.info(f"-- -- get_thetas - URL: {url_}")

        # Send request to NPtooler
        resp = self._do_request(
            type="post", url=url_, timeout=self.timeout, headers=headers_)

        self.logger.info(f"-- -- get_thetas - Response: {resp}")

        return resp
