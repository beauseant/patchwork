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
CRIT="$DIR_NAME/criterios"
CRITFILE="$DIR_NAME/criterios/crit.txt"
CRITFILEADJ="$DIR_NAME/criterios/adj.txt"
CRITFILESOL="$DIR_NAME/criterios/sol.txt"
CRITFILEESP="$DIR_NAME/criterios/esp.txt"
FICHERO_A_ESPERAR="$DIR_NAME/$FILE_NAME.txt_text.json"
SERVIDOR=$(<servidor.cnf)



echo $FICHERO_A_ESPERAR


# --- Configuración ---
TIMEOUT_MINUTOS=5
# ---------------------

# Convertimos el timeout de minutos a segundos para la comparación
TIMEOUT_SEGUNDOS=$((TIMEOUT_MINUTOS * 60))

# Obtenemos el tiempo actual en segundos desde la época (timestamp)
TIEMPO_INICIO=$(date +%s)

echo "⏳ Esperando a que se cree el fichero '${FICHERO_A_ESPERAR}'..."
echo "   (Timeout: ${TIMEOUT_MINUTOS} minutos)"

# Bucle infinito que romperemos nosotros desde dentro
mkdir $CRIT
while true; do
    # PRIMERA COMPROBACIÓN: ¿Existe el fichero?
    if [ -f "$FICHERO_A_ESPERAR" ]; then
        echo "" # Salto de línea para un formato limpio
        echo "✅ ¡Fichero encontrado en '${FICHERO_A_ESPERAR}'!"     
        TEXTO=$(<$FICHERO_A_ESPERAR)      
        curl -X 'POST' \
            "${SERVIDOR}/metadata/extract/" \
            -H 'accept: application/json' \
            -H 'Content-Type: application/json' \
            -d "${TEXTO}" \
            -o "${CRITFILE}"\

        if [ $? -eq 0 ]; then
            #el curl puede terminar bien, pero contener un mensaje tipo     "error": "Failed to process the text",
            #si es así se borra el fichero:        
            if grep -qF "error" "$CRITFILE"; then
                echo "Error encontrado en '$CRITFILE'. Borrando el archivo..."
                rm "$CRITFILE"
                echo "Archivo borrado."
                touch "${CRIT}/error"
                touch "${CRIT}/error_rest"
            else
                echo "No se encontró el error en '$CRITFILE'. El archivo no ha sido modificado."
                cat "$CRITFILE" | jq -r '.criterios_adjudicacion' > "${CRITFILEADJ}"
                cat "$CRITFILE" | jq -r '.criterios_solvencia' > "${CRITFILESOL}"
                cat "$CRITFILE" | jq -r '.condiciones_especiales' > "${CRITFILEESP}"
                touch "${CRIT}/finalizado"

            fi
        else
            touch "${CRIT}/error"
            touch "${CRIT}/error_curl"

        fi
        exit 0 # Termina el script con éxito
    fi

    # SEGUNDA COMPROBACIÓN: ¿Ha pasado el tiempo de timeout?
    TIEMPO_ACTUAL=$(date +%s)
    TIEMPO_TRANSCURRIDO=$((TIEMPO_ACTUAL - TIEMPO_INICIO))

    if [ "$TIEMPO_TRANSCURRIDO" -ge "$TIMEOUT_SEGUNDOS" ]; then
        echo "" # Salto de línea
        echo "🚨 ERROR: Se ha superado el tiempo de espera de ${TIMEOUT_MINUTOS} minutos."
        echo "   El fichero '${FICHERO_A_ESPERAR}' no fue creado a tiempo."
        touch "${CRIT}/error"
        touch "${CRIT}/error_timeout"        
        #rm -f /tmp/datos.json
        exit 1 # Termina el script con un código de error
    fi

    # Si no se cumple ninguna de las condiciones, esperamos 1 segundo antes de volver a comprobar.
    # Esto evita que el bucle consuma el 100% de la CPU.
    sleep 1
done