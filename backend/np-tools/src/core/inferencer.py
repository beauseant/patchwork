"""
This module provides the class Inferencer, which consists of a wrapper for perfoming inference on a new unseen corpus. It contains specific implementations according to the trainer used for the generation of the topic model that is being used for inference. It is based in the inference process of the NP-Search-Tools project.

Author: Lorena Calvo-Bartolomé
Date: 18/03/2024
"""

import logging
import pathlib
import sys
from typing import List
import os
import random

import numpy as np
import pandas as pd # type: ignore
from sklearn.preprocessing import normalize # type: ignore
sys.path.append('../')
from src.TopicModeling.BaseModel import BaseModel

MALLET_HOME = "/opt/mallet"
JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
JAVA_8_PATH = "/usr/lib/jvm/java-8-openjdk-amd64/bin/java"

# Ensure Java 8 is set
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = f"{JAVA_HOME}/bin:{MALLET_HOME}/bin:{os.environ['PATH']}"

def sum_up_to(vector: np.ndarray, max_sum: int) -> np.ndarray:
    """It takes in a vector and a max_sum value and returns a NumPy array with the same shape as vector but with the values adjusted such that their sum is equal to max_sum.

    Parameters
    ----------
    vector: 
        The vector to be adjusted.
    max_sum: int
        Number representing the maximum sum of the vector elements.

    Returns:
    --------
    x: np.ndarray
        A NumPy array of the same shape as vector but with the values adjusted such that their sum is equal to max_sum.
    """
    x = np.array(list(map(np.int_, vector*max_sum))).ravel()
    pos_idx = list(np.where(x != 0)[0])
    while np.sum(x) != max_sum:
        idx = random.choice(pos_idx)
        x[idx] += 1
    return x


class Inferencer(object):
    """
    Wrapper for a NP-Search-Tools Topic Models Inferencer
    Assumes model is saved as a pickle file.
    """

    def __init__(
        self,
        logger: logging.Logger
    ) -> None:
        """
        Init Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Inferencer')

        return

    def transform_inference_output(
        self,
        thetas32: np.array,
        max_sum: int) -> List[dict]:
        """Saves the topic distribution for each document in text format (tXX|weightXX)

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents

        Returns
        -------
        List[dict]: List of dictionaries with the topic distribution for each document
        """

        self._logger.info(
            '-- Inference: Saving the topic distribution in text format')

        def get_doc_str_rpr(vector: np.array, max_sum: int) -> str:
            """Calculates the string representation of a document's topic proportions in the format 't0|100 t1|200 ...', so that the sum of the topic proportions is at most max_sum.

            Parameters
            ----------
            vector: numpy.array
                Array with the topic proportions of a document.
            max_sum: int
                Maximum sum of the topic proportions.

            Returns 
            -------
            rpr: str
                String representation of the document's topic proportions.
            """
            vector = sum_up_to(vector, max_sum)
            rpr = ""
            for idx, val in enumerate(vector):
                if val != 0:
                    rpr += "t" + str(idx) + "|" + str(val) + " "
            rpr = rpr.rstrip()
            return rpr

        if thetas32.ndim == 2:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32[row, :], max_sum) for row in range(len(thetas32))]
        elif thetas32.ndim == 1:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32, max_sum)]
        else:
            self._logger.error(
                f"-- -- Thetas32 has wrong number of dimensions when transforming inference output")
        ids = np.arange(len(thetas32))
        df = pd.DataFrame(list(zip(ids, doc_tpc_rpr)), columns=['id', 'thetas'])

        return df.to_dict(orient='records')

    def predict(
        self,
        model_for_infer_path: pathlib.Path,
        texts: List[str],
        max_sum: int = 1000,
        thetas_thr: float = 3e-3
    ):

        # Check if the model to perform inference on exists
        # The model is saved as pickle file, so we need to check if the file exists
        if not model_for_infer_path.is_dir():
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on is not valid -- Stop')
            return

        # Load model
        path_pickle = model_for_infer_path / 'model_data/model.pickle'
        model = BaseModel.load_model(
            path=path_pickle.as_posix()
        )

        self._logger.info(
            f'-- -- Model loaded from {model_for_infer_path}')

        infer_thetas = model._model_predict(
            texts=texts,
            path_model=model_for_infer_path,
            #path_mallet="/app/Mallet/bin/mallet",
            path_mallet=f"{MALLET_HOME}/bin/mallet",
            save_temp=False)

        self._logger.info(
            f'-- -- Inference performed on the provided texts: {infer_thetas}')

        thetas32_rpr = self.get_final_thetas(
            thetas32=infer_thetas,
            thetas_thr=thetas_thr,
            max_sum=max_sum)

        self._logger.info(
            f'-- -- Inference results transform into string representation: {thetas32_rpr}')

        return thetas32_rpr

    def get_final_thetas(
        self,
        thetas32: np.ndarray,
        thetas_thr: float,
        max_sum: int
    ) -> List[dict]:
        """
        Given the inferred document-topic proportions, it returns the final thetas in the desired format

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics in string representation
        """

        # Thresholding and normalization
        thetas32[thetas32 < thetas_thr] = 0
        if thetas32.ndim == 2:
            thetas32 = normalize(thetas32, axis=1, norm='l1')
        elif thetas32.ndim == 1:
            thetas32 = normalize(thetas32.reshape(1, -1), axis=1, norm='l1')

        # Transform thetas into string representation
        thetas32_rpr = self.transform_inference_output(thetas32, max_sum)

        return thetas32_rpr
