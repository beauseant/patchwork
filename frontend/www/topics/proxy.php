<?php
// Permitir peticiones desde cualquier origen (para tu desarrollo local)
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");

// Obtener la URL real a la que queremos llamar de los parámetros de la petición
$targetUrl = isset($_GET['url']) ? $_GET['url'] : '';

if (empty($targetUrl)) {
    http_response_code(400);
    echo json_encode(["error" => "No target URL provided."]);
    exit;
}

// Iniciar una sesión cURL para hacer la petición desde el servidor
$ch = curl_init();

// Configurar las opciones de cURL
curl_setopt($ch, CURLOPT_URL, $targetUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true); // Devuelve la respuesta como string
curl_setopt($ch, CURLOPT_HEADER, false); // No incluir las cabeceras en la respuesta

// Ejecutar la petición
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

// Comprobar si hubo errores en cURL
if (curl_errno($ch)) {
    http_response_code(500);
    echo json_encode(["error" => "cURL Error: " . curl_error($ch)]);
} else {
    // Enviar la respuesta de la API de vuelta a nuestro JavaScript
    http_response_code($http_code);
    echo $response;
}

// Cerrar la sesión cURL
curl_close($ch);
?>