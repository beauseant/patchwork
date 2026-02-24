<?php
header('Content-Type: application/json');

include ('includes/utils.php');

$servidor =  getServer();


// --- 1. Recoger todos los parámetros ---
$start = $_GET['start'] ?? 0;
$rows = $_GET['length'] ?? 10;
$draw = $_GET['draw'] ?? 0;
$searchValue = $_GET['search']['value'] ?? '';
$corpus = $_GET['corpus'] ?? '';
$year = $_GET['year'] ?? '';
$searchableField = $_GET['searchable_field'] ?? '*';
$totalRecords = $_GET['records_total'] ?? 0;
$sortBy = $_GET['sort_by_order'] ?? 'date:desc';
$keyword = !empty($searchValue) ? $searchValue : '*';

// --- 2. Construir la URL para la API ---
$apiBaseUrl = $servidor . '/queries/getDocsByYearAugmented/';
$queryParams = [
    'corpus_collection' => $corpus, 'start' => $start, 'rows' => $rows,
    'sort_by_order' => $sortBy, 'start_year' => $year, 'keyword' => $keyword,
    'searchable_field' => $searchableField
];
$apiUrl = $apiBaseUrl . '?' . http_build_query($queryParams);



// --- 3. Llamar a la API ---
$ch = curl_init($apiUrl);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 10);
curl_setopt($ch, CURLOPT_TIMEOUT, 20);
$apiResponse = curl_exec($ch);
$httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

// --- 4. Comprobar la respuesta ---
if ($apiResponse === false || $httpcode != 200) {
    echo json_encode(["draw" => $draw, "recordsTotal" => 0, "recordsFiltered" => 0, "data" => [], "error" => "API connection failed with code: $httpcode"]);
    exit;
}

$data = json_decode($apiResponse, true);

// Comprobación de seguridad final
if ($data === null) {
    $jsonError = json_last_error_msg();
    echo json_encode(["draw" => $draw, "recordsTotal" => 0, "recordsFiltered" => 0, "data" => [], "error" => "FATAL: JSON Decode failed. Error: " . $jsonError,"url"=>$apiUrl]);
    exit;
}

/*echo json_encode(["draw" => $draw, "recordsTotal" => 0, "recordsFiltered" => 0, "year"=> $apiUrl,"data" => [], "error" => count ($data)]);
exit;*/

// --- 5. Normalizar los datos ---
$normalizedData = [];
foreach ($data as $doc) {
    // -----------------------------------------------------------------
    // !!! LA SOLUCIÓN FINAL !!!
    // Comprobamos si el documento es válido (si tiene un 'title').
    // Si no lo tiene, simplemente lo ignoramos y pasamos al siguiente.
    // -----------------------------------------------------------------
    if (isset($doc['title'])) {
        $normalizedData[] = [
            'id' => $doc['id'], #el id existe siempre, espero
	    'link' => '<a target="_blank" href="'. $doc['link'] .'">'. $doc['link'] .'</a>',
            'title' => $doc['title'], // Ya no necesitamos '?? NA' aquí porque sabemos que existe
            //'cpv' => isset($doc['cpv'][0]) ? str_replace(["['", "']", "' '"], ['', '', ', '], $doc['cpv'][0]) : 'NA',
	    'cpv' =>  isset($doc['cpv'][0]) ? implode(', ',$doc['cpv']):'NA',
            //'cpv_predicted' =>  isset($doc['cpv_predicted'][0]) ? implode(', ',$doc['cpv_predicted']):'NA',
            'generated_objective' => $doc['generated_objective'] ?? 'NA',
            'criterios_adjudicacion' => $doc['criterios_adjudicacion'] ?? 'NA',
            'criterios_solvencia' => $doc['criterios_solvencia'] ?? 'NA',
            'condiciones_especiales' => $doc['condiciones_especiales'] ?? 'NA'
        ];
    }
}

// --- 6. Enviar la respuesta final mollete es por debug---
$response = [
    "draw" => intval($draw),
    "recordsTotal" => intval($totalRecords),
    "recordsFiltered" => intval($totalRecords),
    "data" => $normalizedData,
   "mollete" => $apiUrl
];

echo json_encode($response);
?>
