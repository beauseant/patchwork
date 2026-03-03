<?php
/**
 * API Proxy — procurement dashboard (PLACE, multi-año)
 * Lee el JSON generado por generate_dashboard_json.py.
 * Cambia DATA_FILE para apuntar a la ruta correcta en tu servidor.
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

define('DATA_FILE', __DIR__ . '/dashboard_data.json');

if (file_exists(DATA_FILE)) {
    $mtime = filemtime(DATA_FILE);
    $etag  = '"' . md5($mtime . DATA_FILE) . '"';

    header('Last-Modified: ' . gmdate('D, d M Y H:i:s \G\M\T', $mtime));
    header('ETag: ' . $etag);
    header('Cache-Control: public, max-age=300');

    $ifNoneMatch = $_SERVER['HTTP_IF_NONE_MATCH']    ?? '';
    $ifModSince  = $_SERVER['HTTP_IF_MODIFIED_SINCE'] ?? '';
    if ($ifNoneMatch === $etag ||
        ($ifModSince && strtotime($ifModSince) >= $mtime)) {
        http_response_code(304);
        exit;
    }

    echo file_get_contents(DATA_FILE);
    exit;
}

http_response_code(503);
echo json_encode([
    'error'      => 'dashboard_data.json no encontrado. Ejecuta generate_dashboard_json.py primero.',
    'as_of'      => null,
    'years'      => [],
    'indicators' => new stdClass(),
]);
