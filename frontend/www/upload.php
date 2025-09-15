<?php
// --- CONFIGURACIÓN ---
$uploadDir = 'uploads/';
$dbFile = $uploadDir. 'database.sqlite';

// Preparamos la respuesta JSON por defecto
$response = ['status' => 'error', 'message' => 'Petición no válida.'];




// --- LÓGICA PRINCIPAL ---
if ($_SERVER["REQUEST_METHOD"] == "POST") {

    if (!isset($_POST['docType']) || empty($_POST['docType'])) {
        $response['message'] = 'Error: No se ha especificado el tipo de documento.';
        header('Content-Type: application/json');
        echo json_encode($response);
        exit();
    }
    
    $docType = $_POST['docType'];
    $datosAExtraer = isset($_POST['datosAExtraer']) && is_array($_POST['datosAExtraer']) ? $_POST['datosAExtraer'] : [];

    if (!is_dir($uploadDir)) {
        mkdir($uploadDir, 0755, true);
    }

    if (isset($_FILES['pdfFile']) && $_FILES['pdfFile']['error'] == UPLOAD_ERR_OK) {
        
        $fileTmpPath = $_FILES['pdfFile']['tmp_name'];
        $originalFileName = basename($_FILES['pdfFile']['name']);
        
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        $mime = finfo_file($finfo, $fileTmpPath);
        finfo_close($finfo);

        if ($mime == 'application/pdf') {
            
            $fileMD5 = md5_file($fileTmpPath);
            $fileExtension = pathinfo($originalFileName, PATHINFO_EXTENSION);
            $newFileName = $fileMD5 . '.' . $fileExtension;
            $destPath = $uploadDir . $newFileName;
            
            // **CAMBIO 1: Mapa para traducir los valores de los checkboxes a texto legible**
            $metadataMap = [
                'criterios_adjudicacion' => 'Criterios de adjudicación',
                'solvencia' => 'Solvencia económica y técnica',
                'condiciones_ejecucion' => 'Condiciones de ejecución especiales',
                'condiciones_desempate' => 'Condiciones de desempate',
                'objeto_contrato' => 'Objeto del contrato',
                'cpv' => 'CPV'
            ];

            try {
                $pdo = new PDO('sqlite:' . $dbFile);
                $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

                // Definiciones de tablas
                $pdo->exec("CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    upload_date DATETIME NOT NULL,
                    doc_type TEXT NOT NULL 
                )");

                // **CAMBIO 2: Crear la nueva tabla de metadatos**
                $pdo->exec("CREATE TABLE IF NOT EXISTS metadatos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    metadata_key TEXT NOT NULL,
                    metadata_value TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )");

                // Comprobar duplicados (lógica sin cambios)
                $stmtCheck = $pdo->prepare("SELECT id FROM documents WHERE stored_name = :stored_name AND doc_type = :doc_type");
                $stmtCheck->execute([':stored_name' => $newFileName, ':doc_type' => $docType]);

                if ($stmtCheck->fetch()) {
                    $response['message'] = 'Error: Este archivo ya fue subido como documento de tipo "' . htmlspecialchars($docType) . '".';
                } else {
                    // **CAMBIO 3: Iniciar una transacción**
                    $pdo->beginTransaction();

                    if (move_uploaded_file($fileTmpPath, $destPath)) {
                        
                        // 1. Insertar en la tabla 'documents'
                        $stmtDoc = $pdo->prepare(
                            "INSERT INTO documents (original_name, stored_name, upload_date, doc_type) VALUES (:original_name, :stored_name, :upload_date, :doc_type)"
                        );
                        $stmtDoc->execute([
                            ':original_name' => $originalFileName,
                            ':stored_name'   => $newFileName,
                            ':upload_date'   => date('Y-m-d H:i:s'),
                            ':doc_type'      => $docType
                        ]);
                        
                        // **CAMBIO 4: Obtener el ID del documento que acabamos de insertar**
                        $lastDocId = $pdo->lastInsertId();

                        // 2. Insertar en la tabla 'metadatos' por cada opción seleccionada
                        if ($lastDocId && !empty($datosAExtraer)) {
                            $stmtMeta = $pdo->prepare(
                                "INSERT INTO metadatos (document_id, metadata_key, metadata_value) VALUES (:doc_id, :meta_key, :meta_value)"
                            );

                            foreach ($datosAExtraer as $key) {
                                $value = 'SOLICITADO';
                                if (isset($metadataMap[$key])) {     
                                    #Si hemos pedido el objeto del contrato lanzamos un script que espera en segundo plano
                                    #a que se haya creado el fichero con el texto del documento, o un timeout a los x minutos:
                                    if ($key == 'objeto_contrato' ){  
                                        exec( getcwd() . '/scripts/extractObjective.sh ' . getcwd() . '/'. $destPath  . '> /dev/null &'); // no $output
                                    }

                                    $stmtMeta->execute([
                                        ':doc_id' => $lastDocId,
                                        ':meta_key' => $metadataMap[$key], // Usamos el texto legible
                                        ':meta_value' =>  $value
                                    ]);
                                }
                            }
                        }

                        // **CAMBIO 5: Confirmar la transacción si todo ha ido bien**
                        $pdo->commit();

                        #shell_exec (getcwd() . '/scripts/extractText.sh ' . getcwd() . '/'. $destPath  . ' &');
                        exec( getcwd() . '/scripts/extractText.sh ' . getcwd() . '/'. $destPath  . '> /dev/null &'); // no $output

                        $response['status'] = 'success';
                        $response['message'] = '¡Archivo y metadatos registrados con éxito!';
                        #$response['message'] = (getcwd() . '/scripts/extractText.sh ' . $destPath  . ' &');
                        $response['data'] = [
                            'originalName' => htmlspecialchars($originalFileName),
                            'storedName' => htmlspecialchars($newFileName)
                        ];

                        


                    } else {
                        $pdo->rollBack(); // Revertir si falla el movimiento del archivo
                        $response['message'] = 'Hubo un error al mover el archivo subido.';
                    }
                }
            } catch (Exception $e) {
                // **CAMBIO 6: Revertir la transacción en caso de cualquier error**
                if (isset($pdo) && $pdo->inTransaction()) {
                    $pdo->rollBack();
                }
                if (file_exists($destPath)) {
                    unlink($destPath); 
                }
                $response['message'] = 'Error crítico: ' . $e->getMessage();
            }
        } else {
            $response['message'] = 'Error: El archivo no es un PDF válido.';
        }
    } else {
        $response['message'] = 'No se recibió ningún archivo o hubo un error en la subida.';
    }
}

// DEVOLVER RESPUESTA JSON
header('Content-Type: application/json');
echo json_encode($response);
exit();
?>