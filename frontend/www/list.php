<?php include 'includes/header.php'; ?>
<?php include 'includes/sidebar.php'; ?>

<!-- Contenido principal -->
<main class="col-md-9 col-lg-10 p-4">

    <div class="container mt-5">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1 class="mb-0">Listado de Documentos</h1>
            <a href="load.php" class="btn btn-success">Ir a Subir Archivo</a>
        </div>
        
        <div id="data-container">
            <p class="text-center">Cargando datos...</p>
        </div>
    </div>
</main>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const dataContainer = document.getElementById('data-container');
        
        const fetchData = () => {
            // La URL ahora apunta directamente a nuestro nuevo script de datos.
            fetch('generatelist.php')
                .then(response => {
                    if (!response.ok) throw new Error('La respuesta de la red no fue correcta');
                    return response.text();
                })
                .then(html => {
                    dataContainer.innerHTML = html;
                })
                .catch(error => {
                    console.error('Hubo un problema con la operación de fetch:', error);
                    dataContainer.innerHTML = '<div class="alert alert-danger">No se pudo actualizar la información.</div>';
                });
        };

        // 1. Cargar los datos la primera vez que se visita la página.
        fetchData();

        // 2. Establecer el intervalo para refrescar cada 10 segundos.
        setInterval(fetchData, 10000);
    });
</script>
