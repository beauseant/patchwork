<?php
// Este archivo es un "endpoint": solo se encarga de la lógica de datos y la presentación de las tablas.

$dbFile = 'uploads/database.sqlite';
$pdo = new PDO('sqlite:'. $dbFile);
$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

// 1. Consulta SQL (sin cambios)
$query = "
    SELECT 
        d.id, d.original_name, d.stored_name, d.upload_date, d.doc_type,
        m.metadata_key, m.metadata_value
    FROM documents d
    LEFT JOIN metadatos m ON d.id = m.document_id
    ORDER BY d.upload_date DESC
";
$stmt = $pdo->query($query);
$results = $stmt->fetchAll(PDO::FETCH_ASSOC);

// 2. Procesar y agrupar resultados (sin cambios)
$documents = [];
foreach ($results as $row) {
    $docId = $row['id'];
    if (!isset($documents[$docId])) {
        $documents[$docId] = [
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

// 3. Separar en arrays (sin cambios)
$documentosTecnicos = array_filter($documents, fn($doc) => $doc['doc_type'] === 'tecnico');
$documentosAdministrativos = array_filter($documents, fn($doc) => $doc['doc_type'] === 'administrativo');

// 4. Función para generar una tabla (sin cambios)
function generateTable($title, $data) {
    echo "<h2>$title</h2>";
    if (empty($data)) {
        echo '<div class="alert alert-info">No hay documentos de este tipo.</div>';
        return;
    }
    echo '<div class="table-responsive"><table class="table table-striped table-bordered table-hover">';
    echo '<thead class="table-dark"><tr><th>Nombre Original</th><th>Fecha de Subida</th><th>Metadatos Solicitados</th><th>Enlace al Archivo</th></tr></thead>';
    echo '<tbody>';
    foreach ($data as $doc) {
        $date = new DateTime($doc['upload_date']);
        echo '<tr>';
        echo '<td>' . htmlspecialchars($doc['original_name']) . '</td>';
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
        echo '<td><a href="uploads/' . htmlspecialchars($doc['stored_name']) . '" target="_blank" class="btn btn-sm btn-outline-primary">Ver Archivo</a></td>';
        echo '</tr>';
    }
    echo '</tbody></table></div>';
}

// 5. Renderizar ambas tablas (sin cambios)
generateTable('📑 Documentos Administrativos', $documentosAdministrativos);
echo '<hr class="my-4">';
generateTable('🛠️ Documentos Técnicos', $documentosTecnicos);

?>