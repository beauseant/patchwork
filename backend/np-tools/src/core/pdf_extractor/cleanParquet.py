import pandas as pd
import argparse
import os
from src.utils import clean_string


def cargar_parquet(fichero):
    try:
        # Cargar el archivo Parquet en un dataframe
        df = pd.read_parquet(fichero)
        print(f"Fichero {fichero} cargado correctamente.")
        return df
    except Exception as e:
        print(f"Error al cargar el fichero {fichero}: {e}")
        exit(1)

def guardar_parquet(df, fichero_salida):
    try:
        # Comprobar si el directorio del archivo de salida existe
        directorio = os.path.dirname(fichero_salida)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio)  # Crear el directorio si no existe

        # Guardar el dataframe en un archivo Parquet
        df.to_parquet(fichero_salida, index=False)
        print(f"Fichero guardado en {fichero_salida}")
    except Exception as e:
        print(f"Error al guardar el fichero {fichero_salida}: {e}")
        exit(1)

def main():
    # Crear el parser
    parser = argparse.ArgumentParser(description="Cargar un fichero Parquet, procesarlo y guardarlo en otro Parquet.")
    
    # Definir los argumentos para los ficheros de entrada y salida
    parser.add_argument('fichero_entrada', type=str, help="Ruta al archivo Parquet de entrada")
    parser.add_argument('fichero_salida', type=str, help="Ruta al archivo Parquet de salida")
    
    # Parsear los argumentos
    args = parser.parse_args()
    
    # Cargar el archivo Parquet de entrada
    dataframe = cargar_parquet(args.fichero_entrada)

    dataframe['clean_text'] = dataframe['extracted'].map (clean_string)
    print(dataframe.head())
    # Guardar el dataframe en el archivo Parquet de salida
    guardar_parquet(dataframe, args.fichero_salida)

if __name__ == "__main__":
    main()
