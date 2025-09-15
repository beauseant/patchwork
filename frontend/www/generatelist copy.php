<?php
// api_datos.php

$dbFile = 'uploads/database.sqlite';
$pdo = new PDO('sqlite:'. $dbFile);
$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

// Consulta y procesamiento de datos (sin cambios)
$query = "SELECT d.id, d.original_name, d.stored_name, d.upload_date, d.doc_type, m.metadata_key, m.metadata_value FROM documents d LEFT JOIN metadatos m ON d.id = m.document_id ORDER BY d.upload_date DESC";
$stmt = $pdo->query($query);
$results = $stmt->fetchAll(PDO::FETCH_ASSOC);

$documents = [];
foreach ($results as $row) {
    $docId = $row['id'];
    if (!isset($documents[$docId])) {
        $documents[$docId] = [
            'id' => $docId, // **NUEVO: Guardamos el ID del documento**
            'original_name' => $row['original_name'],
            'stored_name'   => $row['stored_name'],
            'upload_date'   => $row['upload_date'],
            'doc_type'      => $row['doc_type'],
            'metadata'      => []
        ];
    }
    if ($row['metadata_key']) {
        $documents[$docId]['metadata'][] = [ 'key' => $row['metadata_key'], 'value' => $row['metadata_value'] ];
    }
}

$documentosTecnicos = array_filter($documents, fn($doc) => $doc['doc_type'] === 'tecnico');
$documentosAdministrativos = array_filter($documents, fn($doc) => $doc['doc_type'] === 'administrativo');

function checkErrorText ($doc){
    $cadena_a_buscar = '<title>404';

    if (file_exists($doc)) {
        // 3. Lee todo el contenido del archivo en una sola cadena.
        // Se usa @ para suprimir warnings si el archivo no es legible, lo manejamos a continuación.
        $contenido = @file_get_contents($doc);
        // 4. Comprueba si la lectura fue exitosa (si no, devuelve false).
        if ($contenido !== false) {
            // 5. Busca la posición de la cadena. strpos() es más rápido que otras funciones para solo verificar existencia.
            if (strpos($contenido, $cadena_a_buscar) !== false) {
                $salida = 'ERROR REST';
            } else {
                $salida = '<a href="' . $doc . '"/>VER</a>';
            }

        } else {
            $salida = 'ERROR PERMISOS';
        }
    } else {
        $salida = 'ERROR NO ENCONTRADO';
    }
    return $salida;
}
function checkExtractText ( $doc  ){
    $uploadDir = 'data/';
    $salida = 'PENDIENTE';
    $dir = str_replace ('.pdf', '', $doc);

    if (is_dir($uploadDir . '/' .  $dir)){
        $salida = 'PROCESANDO';
    }
    if (is_file($uploadDir . '/' .  $dir . '/' . 'error')){
        $salida = 'ERROR';
    }

    if (is_file($uploadDir . '/' .  $dir . '/' . $doc . '.txt')){
        $salida =  checkErrorText($uploadDir . '/' .  $dir . '/' . $doc . '.txt');
    }

    #$salida =  ($uploadDir . '/' .  $dir . '/' . $doc . '.txt');
    return $salida;
}

// Función para generar una tabla
function generateTable($title, $data) {
    echo "<h2>$title</h2>";
    if (empty($data)) {
        echo '<div class="alert alert-info">No hay documentos de este tipo.</div>';
        return;
    }
    echo '<div class="table-responsive"><table class="table table-striped table-bordered table-hover">';
    // **CAMBIO: Añadimos la columna "Acciones"**
    echo '<thead class="table-dark"><tr><th>Nombre Original</th><th>MD5</th><th>Fecha de Subida</th><th>Metadatos</th><th>Texto</th><th>Acciones</th></tr></thead>';
    echo '<tbody>';
    foreach ($data as $doc) {
        $date = new DateTime($doc['upload_date']);
        echo '<tr>';
        echo '<td>' . htmlspecialchars($doc['original_name']) . '</td>';
        echo '<td>' . htmlspecialchars($doc['stored_name']) . '</td>';
        echo '<td>' . $date->format('d/m/Y H:i:s') . '</td>';
        echo '<td>';
        if (!empty($doc['metadata'])) {
            echo '<ul class="list-unstyled mb-0">';
            foreach ($doc['metadata'] as $meta) {

                echo '<li><strong>' . htmlspecialchars($meta['key']) . ':</strong> ' . htmlspecialchars($meta['value']) . '</li>';
            }
            echo '</ul>';
        } else {
            echo '<small class="text-muted">Sin metadatos</small>';
        }
        echo '</td>';
        echo '<td>';
        echo (checkExtractText ($doc['stored_name']));

        echo '</td>';
        // **CAMBIO: Nueva celda de "Acciones" con los botones y sus datos**
        echo '<td>';
        
        // Botón Ver - con todos los datos necesarios para el modal
        echo '<button class="btn btn-sm btn-info me-2 btn-view" 
                data-bs-toggle="modal" 
                data-bs-target="#viewModal"
                data-original-name="' . htmlspecialchars($doc['original_name']) . '"
                data-stored-name="' . htmlspecialchars($doc['stored_name']) . '"
                data-upload-date="' . $date->format('d/m/Y H:i:s') . '"
                data-doc-type="' . htmlspecialchars($doc['doc_type']) . '"
                data-metadata=\'' . json_encode($doc['metadata']) . '\'>
                Ver
              </button>';

        // Botón Borrar - con los datos para identificar el archivo
        echo '<button class="btn btn-sm btn-danger btn-delete"
                data-bs-toggle="modal"
                data-bs-target="#deleteModal"
                data-stored-name="' . htmlspecialchars($doc['stored_name']) . '"
                data-original-name="' . htmlspecialchars($doc['original_name']) . '">
                Borrar
              </button>';
        
        echo '</td>';
        echo '</tr>';
    }
    echo '</tbody></table></div>';
}

generateTable('📑 Documentos Administrativos', $documentosAdministrativos);
echo '<hr class="my-4">';
generateTable('🛠️ Documentos Técnicos', $documentosTecnicos);
?>