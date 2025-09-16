<?php 

$json = file_get_contents('data/da09df8bb30a86a503a9e0e8c8871c54/da09df8bb30a86a503a9e0e8c8871c54.pdf.txt_text.json');
if ($json === false) {
    die('Error reading the JSON file');
}

$json_data = json_decode($json, true); 

if ($json_data === null) {
    die('Error decoding the JSON file');
}

echo "<pre>";
print_r($json_data);
echo "</pre>";
echo "----------";
$myArray = json_encode($json_data, JSON_UNESCAPED_UNICODE);
print_r ($myArray);
file_put_contents('data/da09df8bb30a86a503a9e0e8c8871c54/da09df8bb30a86a503a9e0e8c8871c54.pdf.txt_text2.json', $myArray);

?>