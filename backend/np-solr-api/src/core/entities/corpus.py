"""
This module is a class implementation to manage and hold all the information associated with a logical corpus.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Project))
"""

import configparser
import json
from typing import List
from gensim.corpora import Dictionary
import pathlib
import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from src.core.entities.utils import (convert_datetime_to_strftime,
                                     parseTimeINSTANT)


class Corpus(object):
    """
    A class to manage and hold all the information associated with a logical corpus.
    """

    def __init__(self,
                 path_to_raw: pathlib.Path,
                 logger=None,
                 config_file: str = "/config/config.cf") -> None:
        """Init method.

        Parameters
        ----------
        path_to_raw: pathlib.Path
            Path the raw corpus file.
        logger : logging.Logger
            The logger object to log messages and errors.
        config_file: str
            Path to the configuration file.
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Entity Corpus')

        if not path_to_raw.exists():
            self._logger.error(
                f"Path to raw data {path_to_raw} does not exist."
            )
        self.path_to_raw = path_to_raw
        self.name = path_to_raw.stem.lower()
        self.fields = None

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self._logger.info(f"Sections {cf.sections()}")
        if self.name + "-config" in cf.sections():
            section = self.name + "-config"
        else:
            self._logger.error(
                f"Corpus configuration {self.name} not found in config file.")
        self.id_field = cf.get(section, "id_field")
        self.title_field = cf.get(section, "title_field")
        self.date_field = cf.get(section, "date_field")
        self.MetadataDisplayed = cf.get(
            section, "MetadataDisplayed").split(",")
        self.SearcheableField = cf.get(section, "SearcheableField").split(",")
        if self.title_field in self.SearcheableField:
            self.SearcheableField.remove(self.title_field)
            self.SearcheableField.append("title")
        if self.date_field in self.SearcheableField:
            self.SearcheableField.remove(self.date_field)
            self.SearcheableField.append("date")

        return

    def get_docs_raw_info(self):
        """Extracts the information contained in the parquet file in a memory-efficient way
        using a generator instead of returning a full list.
        """
        ddf = dd.read_parquet(self.path_to_raw).fillna("")
        self._logger.info(ddf.head())

        # If the id_field is in the SearcheableField, adjust it
        if self.id_field in self.SearcheableField:
            self.SearcheableField.remove(self.id_field)
            self.SearcheableField.append("id")

        self._logger.info(f"SearcheableField {self.SearcheableField}")

        # Rename necessary fields
        # if there is already an "id" field that is different from self.id_field, rename it to "id_"
        if "id" in ddf.columns and "id" != self.id_field:
            ddf = ddf.rename(columns={"id": "id_"})
        ddf = ddf.rename(columns={
            self.id_field: "id",
            self.title_field: "title",
            self.date_field: "date"
        })

        self._logger.info(ddf.columns)
        dictionary = Dictionary()

        def process_partition(partition):
            """Processes a single partition of the dataframe"""
            partition["nwords_per_doc"] = partition["lemmas"].apply(lambda x: len(x.split()))
            partition["lemmas_"] = partition["lemmas"].apply(lambda x: x.split() if isinstance(x, str) else [])
            
            # Convert to BoW representation
            partition['bow'] = partition["lemmas_"].apply(
                lambda x: dictionary.doc2bow(x, allow_update=True) if x else []
            )
            partition['bow'] = partition['bow'].apply(
                lambda x: [(dictionary[id], count) for id, count in x] if x else []
            )
            partition['bow'] = partition['bow'].apply(lambda x: ' '.join([f'{word}|{count}' for word, count in x]) if x else None)
            
            partition = partition.drop(['lemmas_'], axis=1)

            # Convert embeddings (assume space-separated numbers)
            """
            partition["embeddings"] = partition["embeddings"].apply(
                lambda x: [float(val) for val in x.split()] if isinstance(x, str) else []
            )
            """
            #partition["embeddings"] =  partition["embeddings"].apply(lambda x: [float(val) for _, val in enumerate(x.split())])
            
            for col in partition.columns:
                partition[col] = partition[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

            # Convert date fields
            partition, cols = convert_datetime_to_strftime(partition)
            partition[cols] = partition[cols].applymap(parseTimeINSTANT)

            # Create SearcheableField
            partition['SearcheableField'] = partition[self.SearcheableField].apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )
            
            for record in partition.to_dict(orient="records"):
                yield record

        # Process and yield data partition by partition
        for partition in ddf.to_delayed():
            yield from process_partition(partition.compute())

    def get_corpora_update(
        self,
        id: int
    ) -> List[dict]:
        """Creates the json to update the 'corpora' collection in Solr with the new logical corpus information.
        """

        fields_dict = [{"id": id,
                        "corpus_name": self.name,
                        "corpus_path": self.path_to_raw.as_posix(),
                        "fields": self.fields,
                        "MetadataDisplayed": self.MetadataDisplayed,
                        "SearcheableFields": self.SearcheableField}]

        return fields_dict

    def get_corpora_SearcheableField_update(
        self,
        id: int,
        field_update: list,
        action: str
    ) -> List[dict]:

        json_lst = [{"id": id,
                    "SearcheableFields": {action: field_update},
                     }]

        return json_lst

    def get_corpus_SearcheableField_update(
        self,
        new_SearcheableFields: str,
        action: str
    ):

        ddf = dd.read_parquet(self.path_to_raw).fillna("")

        # Rename id-field to id, title-field to title and date-field to date
        # if there is already an "id" field that is different from self.id_field, rename it to "id_"
        if "id" in ddf.columns and "id" != self.id_field:
            ddf = ddf.rename(columns={"id": "id_"})
        ddf = ddf.rename(
            columns={self.id_field: "id",
                     self.title_field: "title",
                     self.date_field: "date"})

        with ProgressBar():
            df = ddf.compute(scheduler='processes')

        if action == "add":
            new_SearcheableFields = [
                el for el in new_SearcheableFields if el not in self.SearcheableField]
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = list(
                set(new_SearcheableFields + self.SearcheableField))
        elif action == "remove":
            if self.title_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.title_field)
                new_SearcheableFields.append("title")
            if self.date_field in new_SearcheableFields:
                new_SearcheableFields.remove(self.date_field)
                new_SearcheableFields.append("date")
            new_SearcheableFields = [
                el for el in self.SearcheableField if el not in new_SearcheableFields]

        df['SearcheableField'] = df[new_SearcheableFields].apply(
            lambda x: ' '.join(x.astype(str)), axis=1)

        not_keeps_cols = [el for el in df.columns.tolist() if el not in [
            "id", "SearcheableField"]]
        df = df.drop(not_keeps_cols, axis=1)

        # Create json from dataframe
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)

        new_list = []
        for d in json_lst:
            d["SearcheableField"] = {"set": d["SearcheableField"]}
            new_list.append(d)

        return new_list, new_SearcheableFields