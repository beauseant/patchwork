<?php
header('Content-Type: application/json');
include ('includes/utils.php');

$servidor =  getServer();
$apiUrl = $servidor . 'corpora/listAllCorpus/';
$response = @file_get_contents($apiUrl); // Usamos @ para suprimir warnings si la API falla

if ($response === FALSE) {
    http_response_code(500);
    echo json_encode(['error' => 'No se pudo conectar a la API de corpus.']);
    exit;
}

echo $response;
?>