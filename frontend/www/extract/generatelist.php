<?php
// api_datos.php
header('Content-Type: application/json');
include ('includes/extractMetada.php'); 




try {
    $dbFile = 'uploads/database.sqlite';
    $pdo = new PDO('sqlite:'. $dbFile);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // 1. Consulta para obtener todos los documentos y sus metadatos asociados
    $query = "
        SELECT 
            d.id, d.original_name, d.stored_name, d.upload_date, d.doc_type,
            m.metadata_key, m.metadata_value, m.document_id
        FROM documents d
        LEFT JOIN metadatos m ON d.id = m.document_id
        ORDER BY d.upload_date DESC
    ";
    $stmt = $pdo->query($query);
    $results = $stmt->fetchAll(PDO::FETCH_ASSOC);

    // 2. Procesar los resultados para agrupar los metadatos por documento
    $documents = [];
    foreach ($results as $row) {
        $docId = $row['id'];
        if (!isset($documents[$docId])) {
            $documents[$docId] = [
                'original_name' => $row['original_name'],
                'stored_name'   => $row['stored_name'],
                'upload_date'   => $row['upload_date'],
                'doc_type'      => $row['doc_type'],
                'document_id'   => $row['document_id'],
                'metadata'      => []
            ];
        }
        if ($row['metadata_key']) {
            $documents[$docId]['metadata'][] = ['key' => $row['metadata_key'], 'value' => $row['metadata_value']];
        }
    }

    // 3. Preparar los arrays de datos finales para el JSON
    $data_administrativos = [];
    $data_tecnicos = [];

    
    $estado = array ('SOLICITADO','PENDIENTE','PROCESANDO' );

    foreach ($documents as $doc) {
        // Formatear la fecha
        $date = new DateTime($doc['upload_date']);
        $formatted_date = $date->format('d/m/Y H:i:s');

        // Formatear los metadatos como una lista HTML
        $metadata_html = '';
        if (!empty($doc['metadata'])) {
            $metadata_html .= '<ul class="list-unstyled mb-0">';
            foreach ($doc['metadata'] as $meta) {
                #Actualizamos el objeto del contrato si es la primera vez que se solicita.
                #para las siguientes ya s ha grabado en la base de datos:                           
                if (  ( $meta['key'] == 'Objeto del contrato') && ( in_array ($meta['value'] , $estado) ))    {        
                    $objetoC = checkMetadato ($doc['stored_name'], 'OBJ');
                    //$data = implode(",", $objetoC);
                    $meta['value'] = $objetoC;
                    //$query ='UPDATE metadatos SET metadata_value="' . $objetoC . '" WHERE document_id='. $doc['document_id'] .'AND metadata_key="Objeto del contrato"';
                    //error_log ($query, 3, '/tmp/log');
                    $rst = $pdo->query('UPDATE metadatos SET metadata_value="' . $objetoC . '" WHERE document_id='. $doc['document_id'] .' AND metadata_key="Objeto del contrato"');
                    //error_log ('objeto', 3, '/tmp/log');              

                }
                if (( $meta['key'] == 'CPV') && ( in_array ($meta['value'] , $estado))) {                    
                    $objetoCPV = checkMetadato ($doc['stored_name'], 'CPV');                                        
                    $objetoCPV  = str_replace('cpv:','',str_replace('"','',$objetoCPV));

                    $meta['value'] = $objetoCPV;
                    $rst = $pdo->query('UPDATE metadatos SET metadata_value="' . $objetoCPV . '" WHERE document_id='. $doc['document_id'] .' AND metadata_key="CPV"');
                }
                    
                if (( $meta['key'] == 'Criterios de adjudicación' ) && ( in_array ($meta['value'] , $estado))){                    
                    $criterio = checkMetadato ($doc['stored_name'], 'CRITADJ');
                    $meta['value'] = $criterio;   
                    $meta['value'] = $criterio;                                     
                    $rst = $pdo->query('UPDATE metadatos SET metadata_value="' . $criterio . '" WHERE document_id='. $doc['document_id'] .' AND metadata_key="Criterios de adjudicación"');
                }
                if (( $meta['key'] == 'Solvencia económica y técnica' ) && ( in_array ($meta['value'] , $estado))){  
                    $criterio = checkMetadato ($doc['stored_name'], 'CRITSOL');                  
                    $meta['value'] = $criterio;            
                    $rst = $pdo->query('UPDATE metadatos SET metadata_value="' . $criterio . '" WHERE document_id='. $doc['document_id'] .' AND metadata_key="Solvencia económica y técnica"');        
                }

                if (( $meta['key'] == 'Condiciones de ejecución especiales' ) && ( in_array ($meta['value'] , $estado))){                    
                    $criterio = checkMetadato ($doc['stored_name'], 'CRITESP');          
                    $meta['value'] = $criterio;                    
                    $rst = $pdo->query('UPDATE metadatos SET metadata_value="' . $criterio . '" WHERE document_id='. $doc['document_id'] .' AND metadata_key="Condiciones de ejecución especiales"');        
                }             

                $metadata_html .= '<li><strong>' . htmlspecialchars($meta['key'], ENT_QUOTES, 'UTF-8') . ':</strong> ' . 
                                            htmlspecialchars(mb_strimwidth( $meta['value'],0,60,'...'), ENT_QUOTES, 'UTF-8') . 
                                            '<i class="bi bi-info-circle" data-bs-toggle="tooltip" data-bs-placement="bottom" title="' .  $meta['value']. '"></i>'.
                                   '</li>';
            }
            $metadata_html .= '</ul>';
        } else {
            $metadata_html = '<small class="text-muted">Sin metadatos</small>';
        }
        
        // Formatear los botones de acciones con todos los atributos data-* necesarios
        $actions_html = '<button class="btn btn-outline-secondary  btn-view"  
                            data-original-name="' . htmlspecialchars($doc['original_name'], ENT_QUOTES, 'UTF-8') . '"
                            data-stored-name="' . htmlspecialchars($doc['stored_name'], ENT_QUOTES, 'UTF-8') . '"
                            data-upload-date="' . $formatted_date . '"
                            data-doc-type="' . htmlspecialchars($doc['doc_type'], ENT_QUOTES, 'UTF-8') . '"
                            data-metadata=\'' . htmlspecialchars(json_encode($doc['metadata']), ENT_QUOTES, 'UTF-8') . '\'>
                            <i class="bi bi-arrow-down-right-circle icon fs-5"></i>
                         </button>                        
                         <button class="btn btn-outline-danger btn-delete"
                            data-stored-name="' . htmlspecialchars($doc['stored_name'], ENT_QUOTES, 'UTF-8') . '"
                            data-original-name="' . htmlspecialchars($doc['original_name'], ENT_QUOTES, 'UTF-8') . '">
                            <i class="bi bi-trash icon fs-5"></i>
                         </button>';

        // Construir el objeto de la fila para DataTables
        
        $rowData = [
            "original_name" => htmlspecialchars($doc['original_name'], ENT_QUOTES, 'UTF-8'),
            "md5"           => substr ($doc['stored_name'],0,12) . '...',
            "upload_date"   => $formatted_date,
            "metadata"      => $metadata_html,
            "texto"         => checkExtractText ($doc['stored_name']),
            "actions"       => $actions_html
        ];

        
        
        // Añadir la fila al array del tipo de documento correspondiente
        if ($doc['doc_type'] === 'administrativo') {
            $data_administrativos[] = $rowData;
        } else {
            $data_tecnicos[] = $rowData;
        }
    }

    #error_log ( 'y', 3, '/tmp/log'); 
    // 4. Devolver la estructura JSON final que DataTables espera
    echo json_encode([
        'administrativos' => $data_administrativos,
        'tecnicos' => $data_tecnicos
    ]);

} catch (Exception $e) {
    // En caso de error, devolver un JSON de error
    http_response_code(500);
    echo json_encode(['error' => $e->getMessage()]);
}
?>
