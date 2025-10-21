<?php
// proxy.php

// Establecemos la cabecera de contenido como JSON
header('Content-Type: application/json');

// Obtenemos la URL de destino desde el parámetro 'url'
$targetUrl = $_GET['url'] ?? null;

if (!$targetUrl) {
    http_response_code(400); // Bad Request
    echo json_encode(['error' => 'No se ha proporcionado una URL de destino.']);
    exit;
}

// -----------------------------------------------------------------
// VALIDACIÓN DE SEGURIDAD (¡Importante!)
// -----------------------------------------------------------------
// Para evitar que tu proxy sea usado para atacar otros sitios,
// valida que la URL solo apunte al host permitido.
$allowedHost = 'kumo01.tsc.uc3m.es';
$parsedUrl = parse_url($targetUrl);

if ($parsedUrl === false || !isset($parsedUrl['host']) || $parsedUrl['host'] !== $allowedHost) {
    http_response_code(403); // Forbidden
    echo json_encode(['error' => 'Host no permitido.']);
    exit;
}

// Inicializar cURL
$ch = curl_init();

// Configurar las opciones de cURL
curl_setopt($ch, CURLOPT_URL, $targetUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); // Devolver la respuesta como string
curl_setopt($ch, CURLOPT_HTTPHEADER, array(
    'accept: application/json' // La cabecera que tu API requiere
));
curl_setopt($ch, CURLOPT_FAILONERROR, true); // Fallar si el código HTTP es >= 400

// Ejecutar la petición
$response = curl_exec($ch);
$httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);

// Manejar errores de cURL
if (curl_errno($ch)) {
    http_response_code(500); // Internal Server Error
    echo json_encode(['error' => 'Error en el proxy cURL: ' . curl_error($ch)]);
} else {
    // Si todo fue bien, devolvemos la respuesta de la API
    // El proxy ya ha establecido la cabecera 'Content-Type: application/json'
    http_response_code($httpcode); // Reenviar el código de estado original
    echo $response;
}

// Cerrar cURL
curl_close($ch);
?>