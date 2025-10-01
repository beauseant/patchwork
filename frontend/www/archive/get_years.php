<?php
header('Content-Type: application/json');
include ('includes/utils.php');

// Validamos que el parámetro 'corpus' exista
if (!isset($_GET['corpus']) || empty($_GET['corpus'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Parámetro "corpus" no especificado.']);
    exit;
}

$corpus = urlencode($_GET['corpus']); // Codificamos por si tiene caracteres especiales

$servidor =  getServer();
$apiUrl = $servidor . "/queries/getAllYears/?corpus_collection={$corpus}";

$response = @file_get_contents($apiUrl);

if ($response === FALSE) {
    http_response_code(500);
    echo json_encode(['error' => 'No se pudo conectar a la API de años.']);
    exit;
}

echo $response;
?>