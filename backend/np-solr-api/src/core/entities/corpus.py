"""
This module is a class implementation to manage and hold all the information associated with a logical corpus.

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Project))
"""

import ast
import configparser
import datetime
import json
import math
from typing import Any, List
from gensim.corpora import Dictionary
import pathlib
import pandas as pd

import pytz
import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from src.core.entities.utils import (convert_datetime_to_strftime,parseTimeINSTANT)
#from datetime import datetime
              

# def is_valid_xml_char_ordinal(i):
#     """
#     Defines whether char is valid to use in xml document
#     XML standard defines a valid char as::
#     Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
#     """
#     # conditions ordered by presumed frequency
#     return (
#         0x20 <= i <= 0xD7FF
#         or i in (0x9, 0xA, 0xD)
#         or 0xE000 <= i <= 0xFFFD
#         or 0x10000 <= i <= 0x10FFFF
#     )


# def clean_xml_string(s):
#     """
#     Cleans string from invalid xml chars
#     Solution was found there::
#     http://stackoverflow.com/questions/8733233/filtering-out-certain-bytes-in-python
#     """
#     return "".join(c for c in s if is_valid_xml_char_ordinal(ord(c)))
                       
                                     
# def convert_datetime_to_strftime(df):
#     """
#     Converts all columns of type datetime64[ns] in a dataframe to strftime format.
#     """
#     columns = []
#     for column in df.columns:
#         if df[column].dtype == "datetime64[ns]":
#             columns.append(column)
#             df[column] = df[column].dt.strftime("%Y-%m-%d %H:%M:%S")
#     return df, columns


# def parseTimeINSTANT(time):
#     """
#     Parses a string representing an instant in time and returns it as an Instant object.
#     """
#     format_string = '%Y-%m-%d %H:%M:%S'
#     if isinstance(time, str) and time != "foo":
#         dt = datetime.strptime(time, format_string)
#         dt_utc = dt.astimezone(pytz.UTC)
#         return clean_xml_string(dt_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
#     elif time == "foo":
#         return clean_xml_string("")
#     else:
#         if math.isnan(time):
#             return clean_xml_string("")
                                             

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
        """
        self.id_field = "place_id"
        self.title_field = "title"
        self.date_field = "date"
        self.MetadataDisplayed = ["id", "title", "date"]
        self.SearcheableField = ["id", "title", "date"]
        """


        return

    def _normalize_date_value(self, val: Any) -> Any:
        """
        Normaliza valores de fecha problemáticos y los convierte a None.

        Parameters
        ----------
        val : Any
            Valor a normalizar

        Returns
        -------
        Any or None
            El valor original si es válido, None si es un valor nulo
        """
        # Pandas NaT o NaN
        if pd.isna(val):
            return None

        # Numpy NaN
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return None

        # Strings que representan nulos
        if isinstance(val, str):
            val_normalized = val.strip().lower()
            null_strings = {'nan', 'nat', 'none', 'null',
                            '', 'n/a', 'na', 'n.a.', 'not available'}
            if val_normalized in null_strings:
                return None

        return val

    def _clean_record_for_solr(self, record: dict) -> dict:
        """
        Limpia un registro antes de enviarlo a Solr, eliminando valores nulos.

        Parameters
        ----------
        record : dict
            Diccionario con los datos del documento

        Returns
        -------
        dict
            Diccionario limpio sin valores nulos
        """
        cleaned = {}
        for key, value in record.items():
            # Normalizar el valor
            normalized_value = self._normalize_date_value(value)

            # Solo agregar si no es None
            if normalized_value is not None:
                cleaned[key] = normalized_value

        return cleaned

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
        self._logger.info(f"id_field: {self.id_field}")
        self._logger.info(f"title_field: {self.title_field}")
        self._logger.info(f"date_field: {self.date_field}")

        # Guardar los nombres originales
        original_id_field = self.id_field
        original_title_field = self.title_field
        original_date_field = self.date_field

        self._logger.info(f"Original columns: {ddf.columns.tolist()}")
        dictionary = Dictionary()

        def _maybe_literal_eval(val):
            """Convierte strings que parecen dicts/listas a objetos Python"""
            if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
                try:
                    parsed = ast.literal_eval(val)
                    return parsed
                except Exception:
                    return val
            return val

        def _parse_lotes_to_child_docs(lotes_data, parent_id):
            """Convierte el diccionario/JSON de lotes en una lista de child documents para Solr."""
            
            # Log inicial
            self._logger.debug(f"[LOTES] Processing lotes for parent_id: {parent_id[:80]}")
            self._logger.debug(f"[LOTES] Input type: {type(lotes_data)}")
            self._logger.debug(f"[LOTES] Input value: {str(lotes_data)[:300]}")
            
            if not lotes_data:
                self._logger.debug(f"[LOTES] Empty lotes_data, returning []")
                return []

            # Si es string, intentar parsearlo
            if isinstance(lotes_data, str):
                try:
                    lotes_data = json.loads(lotes_data)
                    self._logger.debug(f"[LOTES] Parsed from JSON string")
                except json.JSONDecodeError:
                    try:
                        lotes_data = ast.literal_eval(lotes_data)
                        self._logger.debug(f"[LOTES] Parsed from literal_eval")
                    except:
                        self._logger.warning(f"[LOTES] Could not parse lotes data for {parent_id[:80]}")
                        return []

            # Si no es un diccionario, devolver lista vacía
            if not isinstance(lotes_data, dict):
                self._logger.warning(f"[LOTES] lotes_data is not a dict, it's {type(lotes_data)}")
                return []

            child_docs = []

            # Caso 1: Diccionario plano con campos de lote
            if any(key in lotes_data for key in ["ItemClassificationCode", "TaxExclusiveAmount", "numLotes", "name"]):
                self._logger.debug(f"[LOTES] Case 1: Single lote in flat dict")
                
                child_doc = {
                    "doc_type": "lote",
                    "id": f"{parent_id}_lote_1"
                }

                # Agregar campos con valores
                if lotes_data.get("ItemClassificationCode"):
                    child_doc["lote_ItemClassificationCode"] = str(lotes_data["ItemClassificationCode"]).strip()
                    self._logger.debug(f"[LOTES]   Added ItemClassificationCode: {child_doc['lote_ItemClassificationCode']}")

                if lotes_data.get("TaxExclusiveAmount"):
                    try:
                        child_doc["lote_TaxExclusiveAmount"] = float(lotes_data["TaxExclusiveAmount"])
                        self._logger.debug(f"[LOTES]   Added TaxExclusiveAmount: {child_doc['lote_TaxExclusiveAmount']}")
                    except (ValueError, TypeError) as e:
                        self._logger.debug(f"[LOTES]   Could not convert TaxExclusiveAmount: {e}")

                if lotes_data.get("numLotes"):
                    try:
                        child_doc["lote_numLotes"] = int(lotes_data["numLotes"])
                        self._logger.debug(f"[LOTES]   Added numLotes: {child_doc['lote_numLotes']}")
                    except (ValueError, TypeError) as e:
                        self._logger.debug(f"[LOTES]   Could not convert numLotes: {e}")

                if lotes_data.get("name"):
                    child_doc["lote_name"] = str(lotes_data["name"]).strip()
                    self._logger.debug(f"[LOTES]   Added name: {child_doc['lote_name']}")

                # Solo agregar si tiene al menos un campo además de doc_type e id
                if len(child_doc) > 2:
                    child_docs.append(child_doc)
                    self._logger.info(f"[LOTES] ✓ Created 1 child document with {len(child_doc)-2} fields")
                else:
                    self._logger.warning(f"[LOTES] ✗ Lote had no valid data (all fields empty)")

            # Caso 2: Diccionario con lista de lotes
            elif "lotes" in lotes_data and isinstance(lotes_data["lotes"], list):
                self._logger.debug(f"[LOTES] Case 2: List of lotes, count: {len(lotes_data['lotes'])}")
                
                for idx, lote in enumerate(lotes_data["lotes"], start=1):
                    if not isinstance(lote, dict):
                        self._logger.debug(f"[LOTES]   Lote {idx} is not a dict, skipping")
                        continue

                    child_doc = {
                        "doc_type": "lote",
                        "id": f"{parent_id}_lote_{idx}"
                    }

                    if lote.get("ItemClassificationCode"):
                        child_doc["lote_ItemClassificationCode"] = str(lote["ItemClassificationCode"]).strip()

                    if lote.get("TaxExclusiveAmount"):
                        try:
                            child_doc["lote_TaxExclusiveAmount"] = float(lote["TaxExclusiveAmount"])
                        except (ValueError, TypeError):
                            pass

                    if lote.get("numLotes"):
                        try:
                            child_doc["lote_numLotes"] = int(lote["numLotes"])
                        except (ValueError, TypeError):
                            pass

                    if lote.get("name"):
                        child_doc["lote_name"] = str(lote["name"]).strip()

                    if len(child_doc) > 2:
                        child_docs.append(child_doc)
                        self._logger.debug(f"[LOTES]   ✓ Created child document {idx}")

                self._logger.info(f"[LOTES] ✓ Created {len(child_docs)} child documents from list")
            else:
                self._logger.warning(f"[LOTES] Unknown lotes structure, keys: {list(lotes_data.keys())}")

            return child_docs

        
        def process_partition(partition):
            """Processes a single partition of the dataframe"""
            
            self._logger.info(f"[PARTITION] Processing partition with {len(partition)} rows")
            self._logger.info(f"[PARTITION] Columns BEFORE rename: {partition.columns.tolist()}")
            
            # ===== RENOMBRAR COLUMNAS =====
            rename_dict = {}
            
            if "id" in partition.columns and original_id_field != "id":
                rename_dict["id"] = "id_original"
            
            if original_id_field in partition.columns:
                rename_dict[original_id_field] = "id"
            
            if original_title_field in partition.columns:
                rename_dict[original_title_field] = "title"
            
            if original_date_field in partition.columns:
                rename_dict[original_date_field] = "date"
            
            if rename_dict:
                partition = partition.rename(columns=rename_dict)
                self._logger.info(f"[PARTITION] Applied rename: {rename_dict}")
            
            self._logger.info(f"[PARTITION] Columns AFTER rename: {partition.columns.tolist()}")
            
            if 'id' not in partition.columns:
                self._logger.error(f"[PARTITION] CRITICAL: 'id' field not found after rename")
                return

            # ===== VERIFICAR CAMPO LOTES ANTES DE PROCESAMIENTO =====
            has_lotes = 'lotes' in partition.columns
            if has_lotes:
                sample_lotes = partition['lotes'].head(3)
                self._logger.info(f"[PARTITION] Campo 'lotes' existe")
                self._logger.info(f"[PARTITION] Sample lotes (first 3):")
                for i, val in enumerate(sample_lotes):
                    self._logger.info(f"[PARTITION]   Row {i}: type={type(val)}, value={str(val)[:200]}")
            else:
                self._logger.warning(f"[PARTITION] Campo 'lotes' NO existe en el DataFrame")

            for col in partition.columns:
                partition[col] = partition[col].apply(_maybe_literal_eval)

            # verify lotes
            if has_lotes:
                sample_lotes = partition['lotes'].head(3)
                self._logger.info(f"[PARTITION] Lotes AFTER literal_eval:")
                for i, val in enumerate(sample_lotes):
                    self._logger.info(f"[PARTITION]   Row {i}: type={type(val)}, value={str(val)[:200]}")

            # verify objective
            if 'generative_objective' in partition.columns:
                sample_obj = partition['generative_objective'].head(3)
                self._logger.info(f"[PARTITION] generative_objective sample (first 3):")
                for i, val in enumerate(sample_obj):
                    self._logger.info(f"[PARTITION]   Row {i}: type={type(val)}, is_list={isinstance(val, list)}, value={str(val)[:200]}")
                    if isinstance(val, list):
                        self._logger.warning(f"[PARTITION] generative_objective is a LIST at row {i}! Converting to string...")
                        partition.at[i, 'generative_objective'] = val[0] if len(val) > 0 else ""

            # Procesar lemmas y BoW
            partition["nwords_per_doc"] = partition["lemmas"].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0)
            partition["lemmas_"] = partition["lemmas"].apply(
                lambda x: x.split() if isinstance(x, str) else [])

            partition['bow'] = partition["lemmas_"].apply(
                lambda x: dictionary.doc2bow(x, allow_update=True) if x else []
            )
            partition['bow'] = partition['bow'].apply(
                lambda x: [(dictionary[id], count) for id, count in x] if x else []
            )
            partition['bow'] = partition['bow'].apply(
                lambda x: ' '.join([f'{word}|{count}' for word, count in x]) if x else None
            )
            

            partition = partition.drop(['lemmas_'], axis=1)

            # Procesar embeddings
            def _parse_embeddings(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return []
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, list):
                    return [float(v) for v in x]
                if isinstance(x, str):
                    try:
                        x = x.replace('[', '').replace(']', '').replace('\n', ' ')
                        return [float(val.strip()) for val in x.split() if val.strip()]
                    except (ValueError, AttributeError):
                        return []
                return []

            partition["embeddings"] = partition["embeddings"].apply(_parse_embeddings)

            # Convertir numpy arrays
            for col in partition.columns:
                partition[col] = partition[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
                

            # Convertir fechas
            partition, cols = convert_datetime_to_strftime(partition)
            if cols:
                partition[cols] = partition[cols].applymap(parseTimeINSTANT)

            # Convertir Timestamps a strings
            def _convert_timestamp_to_string(val):
                import pandas as pd
                from datetime import datetime, date
                
                if val is None:
                    return None
                if isinstance(val, np.ndarray):
                    return None
                try:
                    if pd.isna(val):
                        return None
                except (TypeError, ValueError):
                    pass
                if isinstance(val, str):
                    val_lower = val.strip().lower()
                    if val_lower in ['nan', 'nat', 'none', 'null', '', 'n/a', 'na']:
                        return None
                    return val
                if isinstance(val, pd.Timestamp):
                    return val.isoformat() + 'Z' if val.tz is None else val.isoformat()
                if isinstance(val, (datetime, date)):
                    return val.isoformat() + 'Z'
                if isinstance(val, (float, np.floating)):
                    try:
                        if np.isnan(val):
                            return None
                    except (TypeError, ValueError):
                        pass
                return val

            date_columns = ['award_date', 'project_start_date', 'tender_result_start_date', 'date']
            for date_col in date_columns:
                if date_col in partition.columns:
                    partition[date_col] = partition[date_col].apply(_convert_timestamp_to_string)

            # Procesar arrays (cpv, two_cpv)
            def _parse_array_field(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return []
                
                if isinstance(x, list):
                    out = []
                    for item in x:
                        # FLATTEN nested lists (e.g. [["90921000","90911200"]])
                        if isinstance(item, list):
                            out.extend([str(v).strip() for v in item if v and str(v).strip()])
                            continue

                        # UNWRAP list encoded as a string inside a list (e.g. ["['90921000','90911200']"])
                        if isinstance(item, str):
                            s = item.strip()
                            if s.startswith('['):
                                try:
                                    parsed = ast.literal_eval(s)
                                    if isinstance(parsed, list):
                                        out.extend([str(v).strip() for v in parsed if v and str(v).strip()])
                                        continue
                                except Exception:
                                    pass

                            if s:
                                out.append(s)
                            continue

                        # Any other scalar
                        if item is not None:
                            s = str(item).strip()
                            if s:
                                out.append(s)

                    return out
                
                if isinstance(x, str):
                    if x.strip().lower() in ['[]', '[[]]', 'nan', 'none', '']:
                        return []
                    if x.startswith('['):
                        try:
                            parsed = ast.literal_eval(x)
                            if isinstance(parsed, list):
                                return [str(item).strip() for item in parsed if item and str(item).strip()]
                        except:
                            pass
                    return [x.strip()] if x.strip() else []
                return []

            array_fields = ['cpv', 'two_cpv']
            for field in array_fields:
                if field in partition.columns:
                    partition[field] = partition[field].apply(_parse_array_field)
            
            # Create SearcheableField
            def _create_searcheable_field(row):
                values = []
                for col in self.SearcheableField:
                    if col in row and row[col] is not None:
                        val = row[col]
                        if isinstance(val, list):
                            values.extend([str(v) for v in val if v])
                        else:
                            val_str = str(val).strip()
                            if val_str and val_str.lower() not in ['nan', 'none']:
                                values.append(val_str)
                return ' '.join(values)

            partition['SearcheableField'] = partition.apply(_create_searcheable_field, axis=1)

            # Serialización JSON
            def _ensure_json_serializable(val):
                import pandas as pd
                from datetime import datetime, date
                
                if val is None:
                    return None
                if isinstance(val, np.ndarray):
                    return val.tolist() if val.size > 0 else []
                if isinstance(val, list):
                    return [_ensure_json_serializable(item) for item in val] if len(val) > 0 else []
                if isinstance(val, dict):
                    return {k: _ensure_json_serializable(v) for k, v in val.items()} if len(val) > 0 else {}
                if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(val)
                if isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                    return None if np.isnan(val) else float(val)
                if isinstance(val, np.bool_):
                    return bool(val)
                try:
                    if pd.isna(val):
                        return None
                except (TypeError, ValueError):
                    pass
                if isinstance(val, pd.Timestamp):
                    return val.isoformat() + 'Z' if val.tz is None else val.isoformat()
                if isinstance(val, (datetime, date)):
                    return val.isoformat() + 'Z'
                return val
            
            record_count = 0
            records_with_lotes = 0
            records_with_child_docs = 0
            
            for record in partition.to_dict(orient="records"):
                record_count += 1
                
                if 'id' not in record:
                    self._logger.error(f"[RECORD] Missing 'id' field at record {record_count}")
                    continue
                
                record["doc_type"] = "licitacion"
                
                # Convertir url dict a JSON string
                if "url" in record and isinstance(record["url"], dict):
                    record["url"] = json.dumps(record["url"])

                ## LOTES
                child_documents = None
                lotes_json_value = None
                
                if has_lotes and "lotes" in record:
                    lotes_raw = record["lotes"]
                    
                    if record_count <= 5:
                        self._logger.info(f"[RECORD {record_count}] Processing lotes")
                        self._logger.info(f"[RECORD {record_count}]   ID: {record['id'][:80]}")
                        self._logger.info(f"[RECORD {record_count}]   lotes type: {type(lotes_raw)}")
                        self._logger.info(f"[RECORD {record_count}]   lotes value: {str(lotes_raw)[:300]}")
                    
                    # Crear child documents si hay datos
                    if lotes_raw:  # Si no es None, "", [], o {}
                        records_with_lotes += 1
                        
                        child_docs = _parse_lotes_to_child_docs(lotes_raw, record["id"])
                        
                        if child_docs:
                            records_with_child_docs += 1
                            child_documents = child_docs
                            
                            if record_count <= 5:
                                self._logger.info(f"[RECORD {record_count}]   ✓ Created {len(child_docs)} child documents")
                                self._logger.info(f"[RECORD {record_count}]   First child: {child_docs[0]}")
                        else:
                            if record_count <= 5:
                                self._logger.warning(f"[RECORD {record_count}]   ✗ No child documents created")
                    
                    # serialize lotes to JSON for lotes_json field                    
                    if isinstance(lotes_raw, dict):
                        lotes_json_value = json.dumps(lotes_raw)
                    elif lotes_raw:
                        lotes_json_value = str(lotes_raw)
                    
                    # delete original lotes field to avoid confusion and because we have the info in child docs and lotes_json                    
                    del record["lotes"]
                
                cleaned_record = {}
                for k, v in record.items():
                    v_serializable = _ensure_json_serializable(v)
                    
                    if v_serializable is not None:
                        if isinstance(v_serializable, (list, dict)) and len(v_serializable) == 0:
                            continue
                        cleaned_record[k] = v_serializable

                record = cleaned_record
                
                # ===== AGREGAR CHILD DOCUMENTS Y LOTES_JSON AL RECORD LIMPIO =====
                if child_documents:
                    # Serializar child docs
                    child_documents = [
                        {k: _ensure_json_serializable(v) for k, v in child.items()}
                        for child in child_documents
                    ]
                    record["_childDocuments_"] = child_documents
                
                if lotes_json_value:
                    record["lotes_json"] = lotes_json_value

                yield record
            
            self._logger.info(f"[PARTITION] Summary:")
            self._logger.info(f"[PARTITION]   Total records processed: {record_count}")
            self._logger.info(f"[PARTITION]   Records with 'lotes' field: {records_with_lotes}")
            self._logger.info(f"[PARTITION]   Records with child documents: {records_with_child_docs}")

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
        #  if there is already an "id" field that is different from self.id_field, rename it to "id_"
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


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Test Corpus class")
    

#     corpus = Corpus(path_to_raw=pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/patchwork/data/cpv_5_2024.parquet"))
#     for doc in corpus.get_docs_raw_info():
#         print(doc)
#         import pdb; pdb.set_trace()
