<?php
header('Content-Type: application/json');

$apiUrl = 'http://kumo01.tsc.uc3m.es:9083/corpora/getAllSearchableFileds/';

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $apiUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
$httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($response === false || $httpcode != 200) {
    http_response_code(500);
    echo json_encode(['error' => 'No se pudo conectar a la API de campos de búsqueda.']);
    exit;
}

echo $response;
?>