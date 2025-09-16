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
CPV="$DIR_NAME/cpv"
CPVFILE="$DIR_NAME/cpv/cpv.txt"
FICHERO_A_ESPERAR="$DIR_NAME/obj/obj.txt"


echo $FICHERO_A_ESPERAR


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
        mkdir $CPV
        curl -X 'POST' \
            'http://kumo01.tsc.uc3m.es:112/objective/extract/' \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d @/'tmp/datos.json' \
            -d "file=@${FICHERO_A_ESPERAR};type=application/pdf" \
            -o "${CPVFILE}"\

        touch "${CPV}/finalizado"
        exit 0 # Termina el script con éxito
    fi


    # SEGUNDA COMPROBACIÓN: ¿Ha pasado el tiempo de timeout?
    TIEMPO_ACTUAL=$(date +%s)
    TIEMPO_TRANSCURRIDO=$((TIEMPO_ACTUAL - TIEMPO_INICIO))

    if [ "$TIEMPO_TRANSCURRIDO" -ge "$TIMEOUT_SEGUNDOS" ]; then
        echo "" # Salto de línea
        echo "🚨 ERROR: Se ha superado el tiempo de espera de ${TIMEOUT_MINUTOS} minutos."
        echo "   El fichero '${FICHERO_A_ESPERAR}' no fue creado a tiempo."
        touch "${CPV}/error"
        #rm -f /tmp/datos.json

 B MN       exit 1 # Termina el script con un código de error
    fi

    # Si no se cumple ninguna de las condiciones, esperamos 1 segundo antes de volver a comprobar.
    # Esto evita que el bucle consuma el 100% de la CPU.
    sleep 1
done