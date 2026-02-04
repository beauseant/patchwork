#!/bin/bash
#SBATCH --job-name=insiders_job         # Nombre descriptivo
#SBATCH --output=columns_%j.out        # Archivo de salida (%j = ID del job)
#SBATCH --error=columns_%j.err         # Archivo de error
#SBATCH --partition=batch            # Partición (ajusta si usas otra)
#SBATCH --gres=gpu:0                 # Solicitar 0 GPU 
#SBATCH --qos=cpu
#SBATCH --nodes=1                    # 1 Nodo
#SBATCH --ntasks=1                   # 1 Tarea
#SBATCH --cpus-per-task=4            # 4 CPUs (ajusta según lo que necesite tu script)
#SBATCH --mem=100G                     # 4 GB de RAM
#SBATCH --time=10:00:00              # Límite de tiempo (1 hora)
#SBATCH --chdir=/export/usuarios_ml4ds/sblanco/solid-octo-waddle/  # Directorio de trabajo

# 1. Cargar el entorno virtual
# Usamos la ruta absoluta para evitar ambigüedades
echo "Activando entorno virtual..."
source /export/usuarios_ml4ds/sblanco/slurm/gputest/myenv/bin/activate

# 2. Verificaciones de depuración (opcional, pero recomendado)
echo "Ejecutando en el host: $(hostname)"
echo "Python actual: $(which python)"  # Debería apuntar a .../myenv/bin/python

# 3. Ejecutar tu script
# 'srun' es recomendable para que Slurm pueda monitorizar el paso exacto
echo "Lanzando mambo.py..."
srun python addColumnToDF_21Jan.py /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/DatosBruto_insiders.parquet /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_chunks /export/usuarios_ml4ds/sblanco/solid-octo-waddle/metaDataToAdd.json /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_insiders_2024_MetadatosEnero2026_2
#srun python addColumnToDF_21Jan.py /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/DatosBruto_outsiders.parquet /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_outsiders_2024_chunks /export/usuarios_ml4ds/sblanco/solid-octo-waddle/metaDataToAdd.json /export/data_ml4ds/NextProcurement/Junio_2025/pliegosPlace/red_data_outsiders_2024_MetadatosEnero2026
# Al terminar el script, el entorno se desactiva solo al cerrarse el job,
# pero no pasa nada si pones 'deactivate' al final.
