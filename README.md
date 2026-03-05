```markdown
# Instalación del Frontend

El **frontend** es un proyecto independiente desarrollado en **PHP** utilizando **Bootstrap** para la interfaz de usuario.  
La aplicación se distribuye encapsulada dentro de un **contenedor Docker**, lo que permite desplegar el entorno de ejecución de forma sencilla y reproducible.

A continuación se describen los pasos necesarios para realizar la instalación y puesta en marcha del frontend.

---

## 1. Configuración de variables de entorno

Antes de iniciar el contenedor, es necesario configurar las rutas locales donde la aplicación almacenará los datos persistentes.

Edita el archivo:

```

frontend/.env

````

En este fichero se deben definir las rutas a los directorios locales donde se almacenarán:

- **Uploads**: directorio donde se guardarán los archivos PDF subidos al sistema.
- **Data**: directorio donde se almacenará la base de datos, el texto extraído de los documentos y otros datos generados por la aplicación.

Ejemplo de configuración:

```env
PROJECT_PATH_UPLOADS=./ruta/a/tus/uploads
PROJECT_PATH_DATA=./ruta/a/tus/datos
````

> ⚠️ Asegúrate de que los directorios especificados existen en tu máquina y tienen permisos de escritura.

---

## 2. Instalación de dependencias del frontend

El proyecto utiliza **Node.js** para compilar los recursos del frontend (por ejemplo, estilos o scripts).

Desde el directorio correspondiente ejecuta:

```bash
cd frontend/www/
npm run build
```

Este comando instalará las dependencias necesarias y generará los archivos estáticos requeridos para la aplicación.

---

## 3. Configuración de autenticación HTTP (Apache)

Para proteger el acceso al frontend, es necesario crear los archivos de autenticación de **Apache**:

* `.htaccess`
* `.htpasswd`

El archivo `.htpasswd` contendrá los usuarios y contraseñas autorizados para acceder a la aplicación.

Para generar las contraseñas se recomienda utilizar las herramientas oficiales de Apache, por ejemplo:

```bash
htpasswd -c .htpasswd usuario
```

Este comando solicitará la contraseña y generará el hash correspondiente.

> Puedes añadir usuarios adicionales ejecutando el mismo comando sin la opción `-c`.

---

## 4. Ejecución del contenedor Docker

Una vez completados los pasos anteriores, se puede iniciar el contenedor del frontend mediante **Docker Compose**.

```bash
cd frontend/
docker compose up -d
```

Este comando:

* Construirá la imagen del contenedor (si es necesario)
* Iniciará el servicio en segundo plano
* Montará los volúmenes definidos para persistir los datos

---

## 5. Verificación del despliegue

Tras iniciar el contenedor, el frontend debería estar disponible en la URL configurada en el `docker-compose.yml` o en el puerto expuesto por el contenedor.

Para comprobar que el contenedor se está ejecutando correctamente:

```bash
docker ps
```

También puedes consultar los logs con:

```bash
docker compose logs -f
```

---

