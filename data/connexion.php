<?php

try {
$bdd = new PDO('mysql:host=localhost;dbname=c3mdialloDB','c3mdialloesp','MdialloE16#');
}
catch (Exception $e) {
die('Erreur : ' . $e->getMessage());
}
if($bdd){
	echo 'ok';
}
?>
