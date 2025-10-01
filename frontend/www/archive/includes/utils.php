<?php
function getCountByYear($data, $year) {
    // Buscar el año en los datos
    foreach($data as $key => $value) {
        if ($value["year"] === $year) {
            return $value["count"];
        }
    }

    // Si no se encuentra el año, devolver 0 o un mensaje de error
    return -1; // o puedes devolver 'Año no encontrado'
}

function getYears ($corpus){
    $corpus = urlencode($corpus); // Codificamos por si tiene caracteres especiales
    $apiUrl = "http://kumo01.tsc.uc3m.es:9083/queries/getAllYears/?corpus_collection={$corpus}";

    $response = @file_get_contents($apiUrl);

    if ($response === FALSE) {
        http_response_code(500);
        echo json_encode(['error' => 'No se pudo conectar a la API de años.']);
        exit;
    }

    return $response;

}

?>
