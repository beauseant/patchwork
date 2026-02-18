
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
    $myArray = json_encode($json_data['text'], JSON_UNESCAPED_UNICODE);
    file_put_contents($file . '_text.puro', $myArray);
    return $myArray;
}

readJson ($argv[1]);
#print 	'{ "text": "string"}';
?>

