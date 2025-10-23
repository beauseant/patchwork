<?php

function pingHost ($serverfile = 'servidor.cnf'){   
    
    try {
        $contenido = @file_get_contents($serverfile );        
        $url = $contenido . '/ping';
        $curl = curl_init($url);
        curl_setopt($curl, CURLOPT_RETURNTRANSFER, 1);
        $data = curl_exec($curl);
        $datasearch = json_decode($data, true);
        if ( $datasearch ) {
            #$salida = array('OK' => $datasearch['status']);
            $salida = array('OK' => $datasearch);
        }
        else {
            $salida = array('NOOK' => 'Error reading file');
        }
        curl_close($curl);        

    } catch (Exception $e) { 
        $salida = array('NOOK' => $e->getMessage()); 
    }
    
    return $salida;
}


?>