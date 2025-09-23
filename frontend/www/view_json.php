<?php
// borrar_documento.php
header('Content-Type: application/json');
$response = ['status' => 'error', 'message' => 'Petición no válida.'];

include ('includes/extractMetada.php'); 

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['doc'])) {
    $doc = $_POST['doc'];
    $response['status'] = 'success';
    $response['message'] = @file_get_contents($doc);
    echo json_encode($response);
}
?>