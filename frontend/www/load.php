
    <style>
        .hidden-options, .hidden { display: none; }
        #drop-area { border: 2px dashed #ccc; border-radius: 8px; padding: 30px; text-align: center; color: #666; cursor: pointer; transition: border-color 0.3s, background-color 0.3s; }
        #drop-area.highlight { border-color: #0d6efd; background-color: #f0f8ff; }
        #drop-area p { margin: 0; font-size: 1.1em; }
        #file-name { margin-top: 10px; font-weight: bold; color: #333; }
    </style>


<?php include 'includes/header.php'; ?>
<?php include 'includes/sidebar.php'; ?>
<?php include 'includes/utils.php'; ?>

        <main class="main-content p-4">
            <div class="container-fluid mt-5">                
                        <?php
                            $salida =  pingHost();                            
                            if (array_key_exists('NOOK', $salida)) {
                                echo '
                                    <div class="alert alert-danger" role="alert">
                                        <p>Error en servidor: '. $salida['NOOK'] . ' No es posible subir archivos</p>' .
                                    '</div>
                                ';
                                include 'includes/footer.php';
                                exit();
                            }
                            if (array_key_exists('OK', $salida)) {      
                                echo '
                                    <div class="alert alert-success" role="alert">
                                        <p>Conexión correcta con el servidor: '. $salida['OK']['service'] . '/'. $salida['OK']['timestamp']  . '</p>' .
                                    '</div>
                                ';                                    
      
                            }
                         ?>
                </div>
                <div class="card">
                    <div class="card-header">
                        <h2>Subir Documento para Extracción de Datos</h2>
                    </div>
                    <div class="card-body">
                        <form id="upload-form" enctype="multipart/form-data">

                            <div class="mb-3">
                                <label class="form-label">**1. Arrastra un PDF aquí o haz clic para seleccionarlo:**</label>
                                <div id="drop-area">
                                    <p>Suelta tu archivo PDF aquí</p>
                                    <small class="text-muted">Solo se permiten archivos .pdf</small>
                                    <div id="file-name"></div>
                                </div>
                                <input type="file" id="pdfFile" name="pdfFile" accept="application/pdf" required class="d-none">
                            </div>

                            <div class="mb-3">
                                <label class="form-label">**2. Indica el tipo de documento:**</label>
                                <div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input" type="radio" name="docType" id="tipoAdministrativo" value="administrativo" required>
                                        <label class="form-check-label" for="tipoAdministrativo">Administrativo</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input" type="radio" name="docType" id="tipoTecnico" value="tecnico">
                                        <label class="form-check-label" for="tipoTecnico">Técnico</label>
                                    </div>
                                </div>
                            </div>

                            <div id="opcionesAdministrativo" class="mb-3 hidden-options"></div>
                            <div id="opcionesTecnico" class="mb-3 hidden-options"></div>

                            <hr>
                            <button type="submit" class="btn btn-primary w-100">Subir y Procesar Documento</button>
                        
                        </form>

                        <div id="progress-wrapper" class="mt-3 hidden">
                            <div class="progress">
                                <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                            </div>
                        </div>
                        
                        <div id="upload-status" class="mt-3"></div>

                    </div>
                </div>
            </div>
    </main>
</div>     <!--<div class="wrapper d-flex"> -->
<script>
document.addEventListener('DOMContentLoaded', function () {
    // Referencias a elementos del DOM
    const form = document.getElementById('upload-form');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('pdfFile');
    const fileNameDisplay = document.getElementById('file-name');
    const statusDiv = document.getElementById('upload-status');
    const progressWrapper = document.getElementById('progress-wrapper');
    const progressBar = document.getElementById('progress-bar');
    
    // --- LÓGICA DE DRAG & DROP Y SELECCIÓN (sin cambios) ---
    // ... (El código anterior para handleFiles, preventDefaults, etc. sigue siendo válido)
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => dropArea.addEventListener(eventName, preventDefaults, false));
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    ['dragenter', 'dragover'].forEach(eventName => dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false));
    ['dragleave', 'drop'].forEach(eventName => dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false));
    dropArea.addEventListener('drop', handleDrop, false);
    function handleDrop(e) { handleFiles(e.dataTransfer.files); }
    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === "application/pdf") {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                fileNameDisplay.textContent = `Archivo seleccionado: ${file.name}`;
            } else {
                fileNameDisplay.textContent = "Error: Por favor, selecciona un archivo PDF.";
                fileInput.value = ''; // Limpiar input
            }
        }
    }
    
    // --- NUEVO: LÓGICA DE SUBIDA CON AJAX ---
    form.addEventListener('submit', function(e) {
        e.preventDefault(); // Evitar la recarga de la página

        // Validar que hay un archivo
        if (fileInput.files.length === 0) {
            statusDiv.innerHTML = `<div class="alert alert-danger">Por favor, selecciona un archivo PDF.</div>`;
            return;
        }

        // Limpiar estado anterior y mostrar barra de progreso
        statusDiv.innerHTML = '';
        progressWrapper.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';

        const formData = new FormData(form);
        const xhr = new XMLHttpRequest();

        // Configurar la petición
        xhr.open('POST', 'upload.php', true);

        // 1. Escuchar el progreso de la subida
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = percentComplete + '%';
                progressBar.textContent = percentComplete + '%';
            }
        });

        // 2. Escuchar cuando la subida se ha completado
        xhr.onload = function() {
            progressWrapper.classList.add('hidden'); // Ocultar barra de progreso

            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    let alertClass = response.status === 'success' ? 'alert-success' : 'alert-danger';
                    
                    let message = `<strong>${response.message}</strong>`;
                    if (response.status === 'success') {
                       message += `<br><small>Guardado como: ${response.data.storedName}. '</small>`;
                    }

                    statusDiv.innerHTML = `<div class="alert ${alertClass}">${message}</div>`;
                    
                    if(response.status === 'success') {
                        form.reset(); // Limpiar el formulario si todo fue bien
                        fileNameDisplay.textContent = '';
                    }

                } catch (error) {
                    statusDiv.innerHTML = `<div class="alert alert-danger">Error al procesar la respuesta del servidor.</div>`;
                }
            } else {
                statusDiv.innerHTML = `<div class="alert alert-danger">Error en el servidor: ${xhr.status}</div>`;
            }
        };

        // 3. Escuchar por errores de red
        xhr.onerror = function() {
            progressWrapper.classList.add('hidden');
            statusDiv.innerHTML = `<div class="alert alert-danger">Error de red al intentar subir el archivo.</div>`;
        };

        // Enviar el formulario
        xhr.send(formData);
    });

    // --- Lógica para opciones dinámicas (sin cambios) ---
    // ...
});
</script>

<script>
    document.getElementById('opcionesAdministrativo').innerHTML = `
        <label class="form-label">**3. Elige los datos a extraer (Administrativo):**</label>
        <div class="form-check"><input class="form-check-input" type="checkbox" name="datosAExtraer[]" value="criterios_adjudicacion" id="check1" ><label class="form-check-label" for="check1">Criterios de adjudicación</label></div>
        <div class="form-check"><input class="form-check-input" type="checkbox" name="datosAExtraer[]" value="solvencia" id="check2"><label class="form-check-label" for="check2">Solvencia económica y técnica</label></div>
    `;
    document.getElementById('opcionesTecnico').innerHTML = `
        <label class="form-label">**3. Elige los datos a extraer (Técnico):**</label>
        <div class="form-check"><input class="form-check-input" type="checkbox" name="datosAExtraer[]" value="objeto_contrato" id="check5"><label class="form-check-label" for="check5">Objeto del contrato</label></div>
        <div class="form-check"><input class="form-check-input" type="checkbox" name="datosAExtraer[]" value="cpv" id="check6" ><label class="form-check-label" for="check6">CPV</label></div>
    `;
    const tipoAdministrativo = document.getElementById('tipoAdministrativo');
    const tipoTecnico = document.getElementById('tipoTecnico');
    const opcionesAdmin = document.getElementById('opcionesAdministrativo');
    const opcionesTecnico = document.getElementById('opcionesTecnico');
    function toggleOptions() {
        if (tipoAdministrativo.checked) { opcionesAdmin.style.display = 'block'; opcionesTecnico.style.display = 'none'; }
        else if (tipoTecnico.checked) { opcionesAdmin.style.display = 'none'; opcionesTecnico.style.display = 'block'; }
    }
    tipoAdministrativo.addEventListener('change', toggleOptions);
    tipoTecnico.addEventListener('change', toggleOptions);
</script>
<?php include 'includes/footer.php'; ?>
