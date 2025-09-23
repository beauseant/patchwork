<?php


function rrmdir($src) {
    $dir = opendir($src);
    while(false !== ( $file = readdir($dir)) ) {
        if (( $file != '.' ) && ( $file != '..' )) {
            $full = $src . '/' . $file;
            if ( is_dir($full) ) {
                rrmdir($full);
            }
            else {
                unlink($full);
            }
        }
    }
    closedir($dir);
    rmdir($src);
}


function checkErrorText ($doc, $cadena_a_buscar){
    #$cadena_a_buscar = '<title>404';

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
                $salida = '<button type="button"  class="btn-viewjson btn btn-link" data-doc="' . $doc . '">Ver</button>';
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
        $salida =  checkErrorText($uploadDir . '/' .  $dir . '/' . $doc . '.txt', '<title>404');
    }

    #$salida =  ($uploadDir . '/' .  $dir . '/' . $doc . '.txt');
    return $salida;
}


function readJsonMetadata ( $file ){

    $json = file_get_contents($file);
    if ($json === false) {
        die('Error reading the JSON file');
    }

    $json_data = json_decode($json, true); 
    if ($json_data === null) {
        die('Error decoding the JSON file');
    }

    return $json_data['response'];
}

function checkMetadato ( $doc, $tipo  ){
    
    $uploadDir = 'data/';    
    $dir = str_replace ('.pdf', '', $doc);
    $salida = 'PENDIENTE';
    $filedata = '';

    switch ($tipo) {
        case "CPV":
            $dir = $dir . '/cpv/';
            $filedata = $dir . '/cpv.txt';
            break;
        case "OBJ" :
            $dir = $dir . '/obj/';
            $filedata = $dir . '/obj.txt';
            break;
    }

    $dir  = $uploadDir . '/' .  $dir;    
    $filedata = $uploadDir . '/' .  $filedata;

    if (is_dir($dir )){
        $salida = 'PROCESANDO';
    }
    if (is_file( $dir . 'error')){
        $salida = 'ERROR';
    }  

    if (is_file($dir .  'error_curl')){
        $salida = 'ERROR SERVIDOR';
    }        

    if (is_file( $dir . 'error_rest')){
        $salida = 'ERROR REST';
    }     
    
    if (is_file( $dir . 'error_timeout')){
        $salida = 'ERROR TIMEOUT';
    }  

    if (is_file($filedata )){
        $salida = readJsonMetadata ($filedata);
        #$salida = file_get_contents ($filedata, true);
        #$salida = 'contenido';
    }  
    if ($salida==''){
        $salida = '-';
    }

    
    return $salida;
}



?>