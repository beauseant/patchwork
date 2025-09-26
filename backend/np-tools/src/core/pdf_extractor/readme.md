Para extraer un sólo PDF, en ese directorio:

python processOnePDF.py --pdf_path /home/sblanco/pliegos/data/2024/0068930-24/PCAPE.pdf --path_save /home/sblanco/tmp/ --output

El --output, si aparece, lo que hace es volcar en pantalla en formato json el raw_text del PDF ya limpio (*) En caso de no aparecer el parámetro (output) , no lo muestra en pantalla. Entiendo que eso es lo más cómodo para ti, pero puedo hacer con esa salida lo que quieras.

En cualquier caso, siempre se volcará el fichero de salida indicado en path_save.