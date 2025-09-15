
<?php 
function readJson ( $file ){

    $json = file_get_contents($file);
    if ($json === false) {
        die('Error reading the JSON file');
    }

    $json_data = json_decode($json, true); 

    if ($json_data === null) {
        die('Error decoding the JSON file');
    }
    return $json_data;
}

print (json_encode (readJson ($argv[1])['response']));
#print 	'{ "text": "string"}';
?>