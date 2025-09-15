<?php
// borrar_documento.php

header('Content-Type: application/json');
$response = ['status' => 'error', 'message' => 'Petición no válida.'];

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['stored_name'])) {
    $stored_name = $_POST['stored_name'];
    $dbFile = 'uploads/database.sqlite';
    $uploadDir = 'uploads/';
    $filePath = $uploadDir . $stored_name;

    // Validar que el nombre del archivo no contenga caracteres maliciosos (ej. '..')
    if (strpos($stored_name, '/') !== false || strpos($stored_name, '\\') !== false) {
        $response['message'] = 'Nombre de archivo no válido.';
        echo json_encode($response);
        exit();
    }

    try {
        $pdo = new PDO('sqlite:' . $dbFile);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

        // Iniciar transacción para garantizar la integridad de los datos
        $pdo->beginTransaction();

        // 1. Encontrar todos los IDs de documento asociados con este archivo físico
        $stmt_find = $pdo->prepare("SELECT id FROM documents WHERE stored_name = ?");
        $stmt_find->execute([$stored_name]);
        $doc_ids = $stmt_find->fetchAll(PDO::FETCH_COLUMN);

        if (!empty($doc_ids)) {
            // Crear placeholders para la consulta IN (?, ?, ?)
            $placeholders = implode(',', array_fill(0, count($doc_ids), '?'));
            
            // 2. Borrar todos los metadatos asociados a esos IDs
            $stmt_meta = $pdo->prepare("DELETE FROM metadatos WHERE document_id IN ($placeholders)");
            $stmt_meta->execute($doc_ids);

            // 3. Borrar todos los registros de documento asociados con este archivo
            $stmt_docs = $pdo->prepare("DELETE FROM documents WHERE stored_name = ?");
            $stmt_docs->execute([$stored_name]);
        }
        
        // 4. Si todo ha ido bien en la BBDD, confirmar la transacción
        $pdo->commit();

        // 5. Ahora, borrar el archivo físico del servidor
        if (file_exists($filePath)) {
            unlink($filePath);
        }

        $response['status'] = 'success';
        $response['message'] = 'Archivo y registros eliminados correctamente.';

    } catch (Exception $e) {
        // Si algo falla, revertir la transacción
        if (isset($pdo) && $pdo->inTransaction()) {
            $pdo->rollBack();
        }
        $response['message'] = 'Error en la base de datos: ' . $e->getMessage();
    }
}

echo json_encode($response);
?>