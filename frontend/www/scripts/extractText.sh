#!/bin/bash

# 1. VERIFICAR QUE SE HA PROPORCIONADO UN PARÁMETRO
# Comprueba si el número de argumentos ($#) no es igual a 1.
# Si no se pasa exactamente un argumento, muestra un mensaje de error y termina.
if [ "$#" -ne 1 ]; then
    echo "Error: Debes proporcionar la ruta a un archivo PDF como único parámetro."
    echo "Uso: $0 mi_archivo.pdf"
    exit 1
fi

SERVIDOR=$(<servidor.cnf)

# Asignar el primer argumento a la variable PDF_FILE para mayor claridad.
PDF_FILE="$1"

# 2. COMPROBAR SI EL ARCHIVO EXISTE Y ES UN PDF
# Verifica si el archivo proporcionado realmente existe.
if [ ! -f "$PDF_FILE" ]; then
    echo "Error: El archivo '$PDF_FILE' no se ha encontrado."
    exit 1
fi
# Comprueba si la extensión del archivo es '.pdf'.
if [[ "$PDF_FILE" != *.pdf ]]; then
    echo "Error: El archivo '$PDF_FILE' no parece ser un documento PDF."
    exit 1
fi

# 3. EXTRAER EL NOMBRE DEL ARCHIVO SIN LA EXTENSIÓN
# `basename` elimina la ruta y deja solo el nombre del archivo.
# El segundo argumento de `basename` ('.pdf') elimina esa extensión.
DIR_NAME="data/"$(basename "$PDF_FILE" .pdf)
FILE_NAME=$(basename "$PDF_FILE")
# 4. COMPROBAR Y BORRAR EL DIRECTORIO SI YA EXISTE
# El flag '-d' comprueba si existe un directorio con ese nombre.
if [ -d "$DIR_NAME" ]; then
    echo "El directorio '$DIR_NAME' ya existe. Borrándolo..."
    # 'rm -r' borra el directorio y todo su contenido de forma recursiva.
    # 'rm -rf' forzaría el borrado sin preguntar, úsalo con precaución.
    rm -rf "$DIR_NAME"
fi

# 5. CREAR EL NUEVO DIRECTORIO
echo "Creando el directorio '$DIR_NAME'..."
mkdir "$DIR_NAME"

# 6. CONVERTIR EL PDF A TEXTO DENTRO DEL NUEVO DIRECTORIO
# Define el nombre del archivo de texto de salida.
OUTPUT_TEXT_FILE="$DIR_NAME/$FILE_NAME.txt"
ERROR_FILE="$DIR_NAME/error"
echo "Convirtiendo '$PDF_FILE' a texto en '$OUTPUT_TEXT_FILE'..."
# Llama a pdftotext.
# El primer argumento es el PDF de entrada.
# El segundo es la ruta del archivo de texto de salida.
#pdftotext "$PDF_FILE" "$OUTPUT_TEXT_FILE"

curl -X 'POST' \
"${SERVIDOR}/pdf/extract_text/" \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F "file=@${PDF_FILE};type=application/pdf" \
-o "${OUTPUT_TEXT_FILE}"\



# Verificamos si la conversión fue exitosa
if [ $? -eq 0 ]; then
  php scripts/readjson.php "$OUTPUT_TEXT_FILE" 
  #echo "La conversión fue exitosa. El archivo de texto está en $file_name/$file_name.txt"
  touch "$DIR_NAME/finalizado"
else
  touch "$DIR_NAME/error"
fi
