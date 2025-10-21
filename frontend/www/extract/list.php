<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>


<main class="main-content p-4">
    <div class="container-fluid mt-5">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1 class="mb-0">Uploaded Documents (Enriched)</h1>
            <a href="load.php" class="btn btn-success">Upload archive</a>
        </div>



        <div class="accordion" id="accordionDocs">
        <div class="accordion-item">
            <h2 class="accordion-header" id="headingOne">
            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                <h2 class="mt-2">📑 Administrative Documents (PCAP)</h2>
            </button>
            </h2>
            <div id="collapseOne" class="accordion-collapse collapse show" aria-labelledby="headingOne" data-bs-parent="#accordionExample">
            <div class="accordion-body">
                <table id="tabla-administrativos" class="table table-striped table-bordered" style="width:90%">
                    <thead>
                        <tr>
                            <th>Original name</th>
                            <th>MD5</th>
                            <th>Upload date</th>
                            <th>Metadata</th>
                            <th>Text</th>
                            <th style="width: 120px;">Tender</th>
                        </tr>
                    </thead>
                    <tbody>
                        </tbody>
                </table>
            </div>
            </div>
        </div>
        <div class="accordion-item">
            <h2 class="accordion-header" id="headingTwo">
            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                <h2 class="mt-2"> 🛠️ Technical Documents (PCPT)</h2>
            </button>
            </h2>
            <div id="collapseTwo" class="accordion-collapse collapse" aria-labelledby="headingTwo" data-bs-parent="#accordionExample">
            <div class="accordion-body">
                <table id="tabla-tecnicos" class="table table-striped table-bordered" style="width:90%">
                    <thead>
                        <tr>
                            <th>Original name</th>
                            <th>MD5</th>                    
                            <th>Upload date</th>
                            <th>Metadata</th>
                            <th>Text</th>                    
                            <th style="width: 120px;">Tender</th>
                        </tr>
                    </thead>
                    <tbody>
                        </tbody>
                </table>        
            </div>
            </div>
        </div>
        </div>    
    </div>



    <div class="modal fade" id="viewModal" tabindex="-1" aria-labelledby="viewModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-xl">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="viewModalLabel">Detalles del Documento</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class="row">
                <div class="col-md-4">
                    <h5>Información</h5>
                    <p><strong>Original Name:</strong> <span id="modal-original-name"></span></p>
                    <p><strong>MD5:</strong> <span id="modal-doc-type"></span></p>
                    <p><strong>Upload date:</strong> <span id="modal-upload-date"></span></p>
                    <hr>
                    <h5>Metadata:</h5>
                    <div id="modal-metadata-list"></div>
                </div>
                <div class="col-md-8">
                    <h5>View PDF</h5>
                    <iframe id="pdf-viewer" src="" frameborder="0"></iframe>
                </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="modal fade" id="deleteModal" tabindex="-1" aria-labelledby="deleteModalLabel" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="deleteModalLabel">Confirmar Borrado</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <p>¿Estás seguro de que quieres borrar el archivo <strong id="delete-file-name"></strong>?</p>
            <p class="text-danger small">Esta acción es irreversible y eliminará el archivo físico y todos sus registros asociados de la base de datos.</p>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
            <button type="button" class="btn btn-danger" id="confirmDeleteBtn">Sí, Borrar</button>
          </div>
        </div>
      </div>
    </div>


    <div class="modal fade" id="viewjsonmodal" tabindex="-1" aria-labelledby="viewModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-xl">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="viewModalLabel">Texto extraído</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">      

                 <div class="border p-3">
            <pre class="text-break" style="white-space: pre-wrap;"><code id="show-file-name"></code></pre>
        </div>
            
            
            
          
        <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Salir</button>
        </div>
        
        </div>

        </div>
      </div>
    </div>





</main>

</div>     <!--<div class="wrapper d-flex"> -->


    <script>
    $(document).ready(function() {

        // Guardamos las instancias de los modales de Bootstrap para poder controlarlos.
        const viewModal = new bootstrap.Modal(document.getElementById('viewModal'));
        const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
        const viewJsonModal = new bootstrap.Modal(document.getElementById('viewjsonmodal'));

        // Configuración común para ambas tablas DataTables.
        const dataTableConfig = {
            ajax: {
                url: 'generatelist.php',
                dataSrc: '' // La fuente de datos se especifica al inicializar cada tabla.
            },
            columns: [
                { data: 'original_name' },
                { data: 'md5' },
                { data: 'upload_date' },
                { data: 'metadata' },
                { data: 'texto' },
                { data: 'actions', orderable: false, searchable: false } // Columna de botones no se puede ordenar ni buscar.
            ],
            /*language: {
                url: 'https://cdn.datatables.net/plug-ins/2.0.8/i18n/es-ES.json'
            },*/
            responsive: true,
            destroy: true ,
            filter:true,
        };

        // Inicializar la tabla de Documentos Administrativos.
        let tablaAdmin = $('#tabla-administrativos').DataTable({
            ...dataTableConfig,
            ajax: { ...dataTableConfig.ajax, dataSrc: 'administrativos' }
        });

        // Inicializar la tabla de Documentos Técnicos.
        let tablaTecnicos = $('#tabla-tecnicos').DataTable({
            ...dataTableConfig,
            ajax: { ...dataTableConfig.ajax, dataSrc: 'tecnicos' }
        });

        // Función para refrescar ambas tablas.
        const refreshTables = () => {
            tablaAdmin.ajax.reload(null, false); // 'false' evita que se reinicie la paginación.
            tablaTecnicos.ajax.reload(null, false);
        };
        
        // Establecer el intervalo de refresco automático.
        setInterval(refreshTables, 10000);
        

        // --- LÓGICA DE LOS MODALES (Manejadores de eventos con jQuery) ---
        
        // Se usa la delegación de eventos en el cuerpo de la tabla para que funcione con la paginación de DataTables.
        $('table.dataTable tbody').on('click', '.btn-view', function () {
            const data = $(this).data(); // jQuery extrae todos los atributos data-* en un objeto.

            const dataprueba = $(this).data('originalName');
            //console.log(dataprueba);

            // Rellenar el modal de "Ver" con los datos del botón.
            $('#modal-original-name').text(data.originalName);
            $('#modal-doc-type').text(data.docType);
            $('#modal-upload-date').text(data.uploadDate);
            $('#pdf-viewer').attr('src', 'uploads/' + data.storedName);
            
            const metadataList = $('#modal-metadata-list');
            metadataList.empty(); // Limpiar metadatos anteriores.
            
            // Los metadatos vienen como un string JSON, hay que parsearlos.
            const metadata = (typeof data.metadata === 'string') ? JSON.parse(data.metadata) : data.metadata;

            if (metadata && metadata.length > 0) {
                const ul = $('<ul class="list-unstyled"></ul>');
                metadata.forEach(item => {
                    ul.append(`<li><strong>${item.key}:</strong> ${item.value}</li>`);
                });
                metadataList.append(ul);
            } else {
                metadataList.append('<p><small class="text-muted">Sin metadatos</small></p>');
            }

            // Mostrar el modal.
            viewModal.show();
        });

        // Delegación de eventos para el botón BORRAR.
        $('table.dataTable tbody').on('click', '.btn-delete', function () {
            const data = $(this).data();
            $('#delete-file-name').text(data.storedName);
            $('#confirmDeleteBtn').data('storedName', data.storedName);
            deleteModal.show();
        });

        $('table.dataTable tbody').on('click', '.btn-viewjson', function () {
            const doc = $(this).data('doc');
            $.post('view_json.php', { doc: doc }, function(response) {
                if (response.status === 'success') {
                    var jsonString = response.message;
                    //var jsonPretty = JSON.stringify(JSON.parse(jsonString),null,2);  
                    var jsonPretty = JSON.stringify(JSON.parse(jsonString), false, "\t");  
                    
                    $('#show-file-name').text(jsonPretty);

                    viewJsonModal.show();
                } else {
                    alert('Error al mostrar: ' + response.message);
                }
            }, 'json').fail(function() {
                alert('Error de comunicación con el servidor.');
            });            
            //$('#show-file-name').text(doc);
            //$('#show-file-name').text('hola');
            //$('#confirmDeleteBtn').data('storedName', data.storedName);
            //viewJsonModal.show();
        });


        // Evento para el botón de CONFIRMAR BORRADO.
        $('#confirmDeleteBtn').on('click', function() {
            const storedName = $(this).data('stored-name');
            
            $.post('delete_document.php', { stored_name: storedName }, function(response) {
                if (response.status === 'success') {
                    deleteModal.hide();
                    refreshTables();
                } else {
                    alert('Error al borrar: ' + response.message);
                }
            }, 'json').fail(function() {
                alert('Error de comunicación con el servidor.');
            });
        });
    });
    </script>
<?php include '../includes/footer.php'; ?>

