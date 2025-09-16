<?php
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
                $salida = '<a href="' . $doc . '" target="_blank" />VER</a>';
            }

        } else {
            $salida = 'ERROR PERMISOS';
        }
    } else {
        $salida = 'ERROR NO ENCONTRADO';
    }
    return $salida;
}

/*
function checkErrorObj ($doc){
    $cadena_a_buscar = '400 Bad Request:'; 

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
                
                $salida = $contenido;
            }

        } else {
            $salida = 'ERROR PERMISOS';
        }
    } else {
        $salida = 'ERROR NO ENCONTRADO';
    }
    return $salida;
}*/

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


function checkObjeto ( $doc  ){
    $uploadDir = 'data/';
    $salida = 'PENDIENTE';
    $dir = str_replace ('.pdf', '', $doc);

    if (is_dir($uploadDir . '/' .  $dir . '/obj')){
        $salida = 'PROCESANDO';
    }
    if (is_file($uploadDir . '/' .  $dir . '/obj/' . 'error')){
        $salida = 'ERROR';
    }
    
    if (is_file($uploadDir . '/' .  $dir . '/obj/obj.txt')){
        $salida =  checkErrorText($uploadDir . '/' .  $dir . '/obj/obj.txt','400 Bad Request:');
    }

    #$salida =  ($uploadDir . '/' .  $dir . '/' . $doc . '.txt');*/
    return $salida;
}


?>