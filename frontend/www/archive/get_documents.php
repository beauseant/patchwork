<?php
header('Content-Type: application/json');

include ('includes/utils.php');


// --- Parámetros de DataTables ---
// DataTables envía 'start' para el offset y 'length' para el número de filas.
$start = isset($_GET['start']) ? intval($_GET['start']) : 0;
$rows = isset($_GET['length']) ? intval($_GET['length']) : 10;
// DataTables envía un contador 'draw' para evitar ataques XSS y sincronizar peticiones.
$draw = isset($_GET['draw']) ? intval($_GET['draw']) : 0;

// --- Parámetros personalizados ---
if (!isset($_GET['corpus']) || !isset($_GET['year'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Faltan los parámetros "corpus" o "year".']);
    exit;
}
$corpus = urlencode($_GET['corpus']);
$year = intval($_GET['year']);


// --- Llamada a la API ---
$apiUrl = "http://kumo01.tsc.uc3m.es:9083/queries/getDocsByYear/?corpus_collection={$corpus}&year={$year}&start={$start}&rows={$rows}";
$apiResponse = @file_get_contents($apiUrl);

if ($apiResponse === FALSE) {
    http_response_code(500);
    echo json_encode(['error' => 'No se pudo obtener los documentos de la API.']);
    exit;
}

$data = json_decode($apiResponse, true);

// --- Normalizar datos (Añadir 'NA' si falta una clave) ---
$normalizedData = [];
foreach ($data as $doc) {
    $normalizedData[] = [
        'id' => $doc['id'] ?? 'NA',
        'title' => $doc['title'] ?? 'NA',
        'cpv' => $doc['cpv'][0] ?? 'NA',
        'generated_objective' => $doc['generated_objective'] ?? 'NA',
        'criterios_adjudicacion' => $doc['criterios_adjudicacion'] ?? 'NA',
        'criterios_solvencia' => $doc['criterios_solvencia'] ?? 'NA', // Esta puede faltar
        'condiciones_especiales' => $doc['condiciones_especiales'] ?? 'NA'
    ];
}

// --- Construir la respuesta para DataTables ---
// NOTA: Tu API no devuelve el número total de registros. Para una paginación correcta,
// DataTables necesita 'recordsTotal' y 'recordsFiltered'.
// Una solución es usar el 'count' que obtienes en la llamada de años.
// Por simplicidad aquí, lo dejaremos como un número grande o lo calcularemos si fuera posible.
// Lo ideal sería que la API de documentos también devolviera el total.
// Como apaño, podrías almacenar el count del año en la sesión o pasarlo como parámetro.
// Por ahora, asumiremos un total grande para que la paginación funcione visualmente.
#$totalRecords = 1000; // ¡OJO! Esto es un valor placeholder.

#sacamos, para el año solicitado el número de registros:
#sacamos todos los años disponibles junto al número de documentos de cada uno:
$dataYears = getYears ($corpus);
$totalRecords = getCountByYear (json_decode ($dataYears, true), $year);

$response = [
    "draw" => $draw,
    "recordsTotal" => $totalRecords, // Total de registros sin filtrar
    "recordsFiltered" => $totalRecords, // Total de registros después de filtrar (no implementamos filtro)
    "data" => $normalizedData
];

echo json_encode($response);
?>