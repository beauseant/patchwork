# Imagen base con PHP 8.2 y Apache
FROM php:8.2-apache

RUN apt-get update  && apt-get install -y \
    poppler-utils \
    jq


# Instalar extensiones necesarias y activar mod_rewrite
RUN docker-php-ext-install mysqli pdo pdo_mysql \
    && a2enmod rewrite

# Configuración de PHP personalizada
RUN echo "display_errors = on\n upload_max_filesize = 50M\npost_max_size = 50M\nmemory_limit = 256M\n" > /usr/local/etc/php/conf.d/uploads.ini

# Directorio de trabajo
WORKDIR /var/www/html

# Exponer puerto
EXPOSE 80
