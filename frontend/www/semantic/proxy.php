<?php
// Establecer la cabecera de respuesta como JSON
header('Content-Type: application/json');

// --- CONFIGURACIÓN ---
// URL base de tu API.
$api_base_url = 'http://kumo01.tsc.uc3m.es:9083';

// --- LÓGICA DEL PROXY ---

// Verificar si se ha proporcionado una 'action'
if (!isset($_GET['action'])) {
    echo json_encode(['error' => 'No action specified']);
    exit;
}

$action = $_GET['action'];
$query_params = [];
$target_url = '';

// Construir la URL de destino y los parámetros basados en la 'action'
switch ($action) {
    case 'listCorpus':
        $target_url = $api_base_url . '/corpora/listAllCorpus/';
        break;

    case 'listModels':
        if (!isset($_GET['corpus_col'])) {
            echo json_encode(['error' => 'Missing corpus_col parameter']);
            exit;
        }
        $query_params['corpus_col'] = $_GET['corpus_col'];
        $target_url = $api_base_url . '/corpora/listCorpusModels/';
        break;

    case 'getSimilarDocs':
        // Validar parámetros requeridos
        $required_params = ['corpus_collection', 'cpv', 'granularity', 'text_to_infer', 'start', 'rows'];
        foreach ($required_params as $param) {
            if (!isset($_GET[$param])) {
                echo json_encode(['error' => "Missing $param parameter"]);
                exit;
            }
            $query_params[$param] = $_GET[$param];
        }
        $target_url = $api_base_url . '/queries/getDocsSimilarToFreeTextTM/';
        break;

    // --- NUEVO ENDPOINT AÑADIDO ---
    case 'getSimilarDocsEmb':
        // Validar parámetros requeridos (keyword es opcional)
        $required_params = ['corpus_collection', 'free_text', 'start', 'rows'];
        foreach ($required_params as $param) {
            if (!isset($_GET[$param])) {
                echo json_encode(['error' => "Missing $param parameter"]);
                exit;
            }
            $query_params[$param] = $_GET[$param];
        }
        // Añadir keyword si existe, si no, se envía vacía
        $query_params['keyword'] = isset($_GET['keyword']) ? $_GET['keyword'] : '';
        
        $target_url = $api_base_url . '/queries/getDocsSimilarToFreeTextEmb/';
        break;
    // --- FIN DE LA ADICIÓN ---

    case 'getMetadata':
        // Validar parámetros requeridos
        $required_params = ['corpus_collection', 'doc_id'];
        foreach ($required_params as $param) {
            if (!isset($_GET[$param])) {
                echo json_encode(['error' => "Missing $param parameter"]);
                exit;
            }
            $query_params[$param] = $_GET[$param];
        }
        $target_url = $api_base_url . '/queries/getMetadataDocById/';
        break;

    default:
        echo json_encode(['error' => 'Invalid action specified']);
        exit;
}

// Añadir los parámetros a la URL si existen
if (!empty($query_params)) {
    $target_url .= '?' . http_build_query($query_params);
}

// --- EJECUCIÓN DE cURL ---

// Inicializar cURL
$ch = curl_init();

// Configurar las opciones de cURL
curl_setopt($ch, CURLOPT_URL, $target_url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1); // Devolver la respuesta como string
curl_setopt($ch, CURLOPT_TIMEOUT, 30); // Timeout de 30 segundos
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'accept: application/json' // Cabecera requerida por tu API
]);

// Ejecutar la solicitud
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);

// Manejar errores de cURL
if (curl_errno($ch)) {
    echo json_encode(['error' => 'cURL Error: ' . curl_error($ch)]);
    exit;
}

// Manejar errores de respuesta HTTP de la API
if ($http_code >= 400) {
    echo json_encode([
        'error' => 'API Error',
        'http_code' => $http_code,
        'response_body' => $response // Devolver el cuerpo del error de la API
    ]);
    exit;
}

// Cerrar cURL
curl_close($ch);

// Devolver la respuesta de la API al frontend
echo $response;

?>

