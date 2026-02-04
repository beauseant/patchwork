import argparse
import pandas as  pd
import os
from pathlib import Path
from tqdm import tqdm        # asegúrate de: pip install tqdm
import ast 
import json

import swifter
import numpy as np

def extract_cpv(cpv):
    if isinstance(cpv, (list, np.ndarray)):  # If it's a list or numpy array
        cpv_clean = str(cpv[0]).strip("[]") if len(cpv) > 0 else np.nan
    elif isinstance(cpv, str):  # If it's already a string
        cpv_clean = cpv.strip("[]")
    elif pd.isna(cpv):  # If it's NaN, return NaN
        return np.nan
    else:  # If it's an unexpected type (e.g., a number), return NaN
        return np.nan

    # Now process the CPV codes safely
    cpv_list = cpv_clean.split(", ")
    cpv_list = [el.split(".")[0] for el in cpv_list]  # Remove the dot and everything after it
    # if any of the cpv codes is less than 8 characters, add as many zeros as needed at the beginning until it reaches 8 characters
    for i in range(len(cpv_list)):
        if len(cpv_list[i]) < 8:
            while len(cpv_list[i]) < 8:
                cpv_list[i] = "0" + cpv_list[i]
    return list(set([el[:2] for el in cpv_list])) if cpv_list else np.nan
 
def extract_complete_cpv(cpv):
    if isinstance(cpv, (list, np.ndarray)):  # If it's a list or numpy array
        cpv_clean = str(cpv[0]).strip("[]") if len(cpv) > 0 else np.nan
    elif isinstance(cpv, str):  # If it's already a string
        cpv_clean = cpv.strip("[]")
    elif pd.isna(cpv):  # If it's NaN, return NaN
        return np.nan
    else:  # If it's an unexpected type (e.g., a number), return NaN
        return np.nan

    # Now process the CPV codes safely
    cpv_list = cpv_clean.split(", ")
    cpv_list = [el.split(".")[0] for el in cpv_list]  # Remove the dot and everything after it
    # if any of the cpv codes is less than 8 characters, add as many zeros as needed at the beginning until it reaches 8 characters
    for i in range(len(cpv_list)):
        if len(cpv_list[i]) < 8:
            while len(cpv_list[i]) < 8:
                cpv_list[i] = "0" + cpv_list[i]
    return list(set([el for el in cpv_list])) if cpv_list else np.nan

def unify_colname(col):
    return ".".join([el for el in col if el])


def procesar_parquet(df: pd.DataFrame, fichero: Path, destino: Path) -> None:
    df_RED = pd.read_parquet(fichero)
    ruta = destino / fichero.name

    #import ipdb ; ipdb.set_trace()

    df_merged = pd.merge(df, df_RED, on='place_id')
    df_merged.to_parquet (str(ruta))


def extract_lotes (row ):
    columna_id = ('ContractFolderStatus', 'ProcurementProjectLot', 'ID')
    columna_lotes = ('ContractFolderStatus', 'ProcurementProjectLot')
    valor = str(row[columna_id])
    lote = row.loc[columna_lotes]
    
    salida = {}

    if '[nan]' in (valor):
        salida['ItemClassificationCode'] = ''
        salida['TaxExclusiveAmount'] = ''
        salida['numLotes'] = ''
        salida['name'] = ''

    else:
        salida = {}        

        if 'RequiredCommodityClassificationlote' in lote['ProcurementProject'].to_dict().keys():
            salida['ItemClassificationCode'] = lote['ProcurementProject']['RequiredCommodityClassification']['ItemClassificationCode'].iloc[0][0]
            salida['TaxExclusiveAmount'] = lote['ProcurementProject']['BudgetAmount']['TaxExclusiveAmount'].iloc[0][0]

            try:
                salida['numLotes'] = len (ast.literal_eval (lote['ID'].iloc[0][0]))
                salida['name'] = ast.literal_eval (lote['ProcurementProject']['Name'].iloc[0][0])
            except:
                salida['numLotes'] = 1
                salida['name'] = lote['ProcurementProject']['Name'].iloc[0][0]         
        

    return json.dumps (salida)


def readMetadatos ( file ):
    with open( file ) as f:
        d = json.load(f)
        return d

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convierte a texto los ficheros PDF pasados como parámetro",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("parquet_origen_raw", help="Ruta al fichero Parquet de entrada, la descarga en bruto, ej: /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/DatosBruto_outsiders.parquet")
    parser.add_argument("parquet_dir_origen_red", help="Ruta al fichero Parquet de entrada, el reducido, ej:/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_outsiders_2024_chunks/part_0000.parquet")
    parser.add_argument("metadatos", help="Fichero json con los metadatos para add")
    parser.add_argument("parquet_dir_destino", help="Ruta al fichero Parquet de salida, será el reducido con la columna")
    args = parser.parse_args()

    df_PLACE = pd.read_parquet(args.parquet_origen_raw)


    columna_id = ('ContractFolderStatus', 'ProcurementProjectLot', 'ID')
    # Creamos la columna lotes:
    #df_PLACE['lotes'] = df_PLACE.swifter.apply(extract_lotes, axis=1)    
    #ordenamos para evitar PerformanceWarning: indexing past lexsort depth may impact performance.
    df_PLACE = df_PLACE.sort_index(axis=1)
    df_PLACE['lotes'] = df_PLACE.apply(extract_lotes, axis=1)

    df_PLACE = df_PLACE.rename(columns={"id":"place_id"})

    df_PLACE.columns = [unify_colname(col) for col in df_PLACE.columns]

    # rename 'ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode' to 'cpv'
    df_PLACE = df_PLACE.rename(columns={"ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode": "cpv"})

    df_PLACE["two_cpv"] = df_PLACE["cpv"].apply(extract_cpv)
    #df_PLACE["two_cpv"].value_counts()
    df_PLACE["cpv"] = df_PLACE["cpv"].apply(extract_complete_cpv)

    metadatos = readMetadatos (args.metadatos)
    metadatos.append('lotes')
    metadatos.append('place_id')
    metadatos.append('cpv')
    metadatos.append('two_cpv')
    metadatos =  list (set (metadatos))

    if 'ContractFolderStatus.ProcurementProject.PlannedPeriod.StartDate' in metadatos:
        df_PLACE["ContractFolderStatus.ProcurementProject.PlannedPeriod.StartDate"] = pd.to_datetime(df_PLACE["ContractFolderStatus.ProcurementProject.PlannedPeriod.StartDate"], errors="coerce")

    if 'ContractFolderStatus.TenderResult.AwardDate' in metadatos:
        if 'ContractFolderStatus.TenderResult.AwardDate' in df_PLACE.columns:
            df_PLACE["ContractFolderStatus.TenderResult.AwardDate"] = pd.to_datetime(df_PLACE["ContractFolderStatus.TenderResult.AwardDate"], errors="coerce")
        else:
            df_PLACE['ContractFolderStatus.TenderResult.AwardDate']=np.nan

    if 'ContractFolderStatus.TenderResult.StartDate' in metadatos:
        if 'ContractFolderStatus.TenderResult.StartDate' in df_PLACE.columns:
            df_PLACE["ContractFolderStatus.TenderResult.StartDate"] = pd.to_datetime(df_PLACE["ContractFolderStatus.TenderResult.StartDate"], errors="coerce")
        else:
            df_PLACE['ContractFolderStatus.TenderResult.StartDate']=np.nan
    # si queremos ver fechas que han dado error: error = df_PLACE[df_PLACE["ContractFolderStatus.ProcurementProject.PlannedPeriod.StartDate"].isna()]
    parquets = sorted(Path(args.parquet_dir_origen_red).glob("*.parquet"))        # lista de rutas .parquet


    #en caso de no existir alguna columna de metadatos, la crea con nulos:
    df_red = df_PLACE.reindex (columns=metadatos)
    destino = Path (args.parquet_dir_destino)
    destino.mkdir(parents=True, exist_ok=True)
    for fichero in tqdm(parquets,
                    desc="Procesando ficheros parquet",
                    unit="fichero"):
            
        procesar_parquet(df_red, fichero, destino)

    

    '''

    df_PLACE = df_PLACE[['place_id', args.columna]]


    parquets = sorted(Path(args.parquet_dir_origen_red).glob("*.parquet"))        # lista de rutas .parquet

    destino = Path (args.parquet_dir_destino)
    destino.mkdir(parents=True, exist_ok=True)

    for fichero in tqdm(parquets,
                    desc="Procesando ficheros parquet",
                    unit="fichero"):
        procesar_parquet(df_PLACE, fichero, destino)

    
    df = pd.read_parquet ('/export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/DatosBruto_insiders_muestra.parquet')

    columna_id = ('ContractFolderStatus', 'ProcurementProjectLot', 'ID')
    # Creamos la columna lotes:
    df['lotes'] = df.apply(extract_lotes, axis=1)
    '''
    
 #python addColumnToDF_21Jan.py /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/DatosBruto_insiders.parquet /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_chunks /export/usuarios_ml4ds/sblanco/solid-octo-waddle/metaDataToAdd.json /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/prueba/    


