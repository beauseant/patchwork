# patchwork
https://github.com/nextprocurement/NP-Backend-Dockers/wiki/NP-Tools-API
http://kumo01.tsc.uc3m.es:112/


Instalación del frontend:

El frontend es un proyecto independiente creado con PHP y boostrap encapsulados dentro de un Docker. Para su instalación debemos seguir los siguientes pasos.

1) Modifica frontend/.env para que use los datos de un repositorio local que tengas en la máquina. Uploads es el lugar donde se guardarán los PDF y data la base de datos, el texto extraído etc

El contenido será algo como:

PROJECT_PATH_UPLOADS=./ruta/a/tus/uploads
PROJECT_PATH_DATA=./ruta/a/tus/datos

2) Instala las librerias externas:
    cd frontend/www/
    npm run build
3) Ejecutar el docker:
    cd frontend/
    docker compose up -d
