#!/bin/bash


# 1. VERIFICAR QUE SE HA PROPORCIONADO UN PARÁMETRO
# Comprueba si el número de argumentos ($#) no es igual a 1.
# Si no se pasa exactamente un argumento, muestra un mensaje de error y termina.
if [ "$#" -ne 1 ]; then
    echo "Error: Debes proporcionar la ruta a un archivo PDF como único parámetro."
    echo "Uso: $0 mi_archivo.pdf"
    exit 1
fi


DIR_NAME="data/"$(basename "$1" .pdf)
FILE_NAME=$(basename "$1")
OBJ="$DIR_NAME/obj"
OBJFILE="$DIR_NAME/obj/obj.txt"



TEXT="\"text\": \"Se hace un contrato de prestacioón de servicios para tocar el trombon los śábados a la salida de misa\""

FICHERO_A_ESPERAR="$DIR_NAME/$FILE_NAME.txt"
#RAWTEXT="$(php scripts/readjson.php ${FICHERO_A_ESPERAR})"
php scripts/readjson.php "$FICHERO_A_ESPERAR" > /tmp/datos.json
echo $FICHERO_A_ESPERAR
echo $RAWTEXT


# --- Configuración ---
TIMEOUT_MINUTOS=1
# ---------------------

# Convertimos el timeout de minutos a segundos para la comparación
TIMEOUT_SEGUNDOS=$((TIMEOUT_MINUTOS * 60))

# Obtenemos el tiempo actual en segundos desde la época (timestamp)
TIEMPO_INICIO=$(date +%s)

echo "⏳ Esperando a que se cree el fichero '${FICHERO_A_ESPERAR}'..."
echo "   (Timeout: ${TIMEOUT_MINUTOS} minutos)"

# Bucle infinito que romperemos nosotros desde dentro
while true; do
    # PRIMERA COMPROBACIÓN: ¿Existe el fichero?
    if [ -f "$FICHERO_A_ESPERAR" ]; then
        echo "" # Salto de línea para un formato limpio
        echo "✅ ¡Fichero encontrado en '${FICHERO_A_ESPERAR}'!"
        mkdir $OBJ
        curl -X 'POST' \
            'http://kumo01.tsc.uc3m.es:112/objective/extract/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d @/'tmp/datos.json' \
            -o "${OBJFILE}"\

        touch "${OBJ}/finalizado"
        #rm -f /tmp/datos.json
        exit 0 # Termina el script con éxito
    fi

    # SEGUNDA COMPROBACIÓN: ¿Ha pasado el tiempo de timeout?
    TIEMPO_ACTUAL=$(date +%s)
    TIEMPO_TRANSCURRIDO=$((TIEMPO_ACTUAL - TIEMPO_INICIO))

    if [ "$TIEMPO_TRANSCURRIDO" -ge "$TIMEOUT_SEGUNDOS" ]; then
        echo "" # Salto de línea
        echo "🚨 ERROR: Se ha superado el tiempo de espera de ${TIMEOUT_MINUTOS} minutos."
        echo "   El fichero '${FICHERO_A_ESPERAR}' no fue creado a tiempo."
        touch "${OBJ}/error"
        #rm -f /tmp/datos.json

        exit 1 # Termina el script con un código de error
    fi

    # Si no se cumple ninguna de las condiciones, esperamos 1 segundo antes de volver a comprobar.
    # Esto evita que el bucle consuma el 100% de la CPU.
    sleep 1
done