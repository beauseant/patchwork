<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php';?>
<?php include '../includes/utils.php'; ?>

   <main class="main-content p-4">
        <div class="container-fluid mt-5">
            <?php
                            $salida =  pingHost();                            
                            if (array_key_exists('NOOK', $salida)) {
                                echo '
                                    <div class="alert alert-danger" role="alert">
                                        <p>Error en servidor: '. $salida['NOOK'] . ' No es posible mostrar resultados</p>' .
                                    '</div>
                                ';
                                include '../includes/footer.php';
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
           
        <h2 class="mb-4">Pre-Enriched Tender Documents<button type="button" class="btn btn-text" data-bs-toggle="modal" data-bs-target="#infoModal"> [+]</button> </h2>

        <div class="row mb-3">
            <div class="col-md-6">
                <label for="corpusSelector" class="form-label"><strong>Corpus:</strong></label>
                <select id="corpusSelector" disabled class="form-select"></select>
            </div>
            <div class="col-md-6">
                <label for="yearSelector" class="form-label"><strong>Año:</strong></label>
                <select id="yearSelector" class="form-select"></select>
            </div>
        </div>
        
        <div class="row mb-3">
             <div class="col-md-6">
                <label for="searchFieldSelector" class="form-label"><strong>Buscar en campo:</strong></label>
                <select id="searchFieldSelector" class="form-select">
                </select>
            </div>
        </div>


    <div>
        Columnas a mostrar:<a class="toggle-vis" data-column="0">Id</a> - <a class="toggle-vis" data-column="1">Title</a> - <a class="toggle-vis" data-column="2">CPV</a> - <a class="toggle-vis" data-column="3">Generated objective</a> - <a class="toggle-vis" data-column="4">Award criteria</a> - <a class="toggle-vis" data-column="5">Solvency criteria</a> - <a class="toggle-vis" data-column="6">Special conditions</a>
    </div>

    
        <table id="licitacionesTable" class="table table-striped table-borderless table-hover  table-responsive-sm" style="width:80%">

            <thead>
                <tr>
                    <th>Link</th>                    
                    <th>Title</th>                    
                    <th>CPV</th>
                    <th >Generated objective</th>
                    <th>Award criteria</th>
                    <th>Solvency criteria</th>
                    <th>Special conditions</th>
                    <th>View</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>


                 <!-- Modal -->
                    <div class="modal fade modal-lg" id="infoModal" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
                    <div class="modal-dialog">
                        <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="exampleModalLabel">Pre-Enriched Tender Documents</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <i style="font-size:30px;"class="bi bi-tools"></i><br>
                            This page contains procurement documents that have been pre-processed and enriched in advance. <br>
                            These tenders form part of the reference corpus used for topic modeling and semantic similarity search.</br>
                            The displayed metadata includes contract objectives, CPV codes, award criteria, solvency requirements, and special conditions.

                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                        </div>
                    </div>
                    </div>

    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>

    <script>
        $(document).ready(function() {
            let table;

            
            // 1. Cargar Corpus y CAMPOS DE BÚSQUEDA al iniciar
            $.ajax({
                url: 'get_corpus.php',
                success: function(corpora) {
                    const corpusSelector = $('#corpusSelector');
                    corpora.forEach(c => corpusSelector.append(`<option value="${c}">${c}</option>`));
                    corpusSelector.trigger('change');
                }
            });
            
            $.ajax({
                url: 'get_searchable_fields.php',
                success: function(fields) {
                    const searchFieldSelector = $('#searchFieldSelector');
                    fields.forEach(f => searchFieldSelector.append(`<option value="${f}">${f}</option>`));
                }
            });

            // 2. Cargar Años cuando cambia el Corpus
            $('#corpusSelector').on('change', function() {
                const selectedCorpus = $(this).val();
                $.ajax({
                    url: 'get_years.php',
                    data: { corpus: selectedCorpus },
                    dataType: 'json',
                    success: function(years) {
                        const yearSelector = $('#yearSelector');
                        yearSelector.empty();
                        //years.sort((a, b) => b.year - a.year).forEach(item => {
                        years.sort((a, b) => b.count - a.count).forEach(item => {

                            // Guardamos el total de documentos en un atributo de datos
                            yearSelector.append(`<option value="${item.year}" data-count="${item.count}">${item.year} (${item.count} docs)</option>`);
                        });
                        yearSelector.trigger('change');
                    }
                });
            });

            // 3. Cuando cambia el Año o el CAMPO DE BÚSQUEDA, se recarga la tabla
            $('#yearSelector, #searchFieldSelector').on('change', function() {
                // Solo recargamos si la tabla ya está inicializada
                if (table) {
                    table.draw();
                }
            });
            
            // Función para inicializar o recargar la tabla
            function initializeDataTable() {
                const selectedCorpus = $('#corpusSelector').val();
                const selectedYear = $('#yearSelector').val();

                if (table) {
                    table.destroy();
                }

                table = $('#licitacionesTable').DataTable({
                    "processing": true,
                    "serverSide": true, // VOLVEMOS A ACTIVAR EL MODO SERVIDOR
                    "ajax": {
                        "url": "get_documents.php",
                        "type": "GET",
                        "data": function(d) {
                            // 'd' contiene los parámetros de DataTables (start, length, search, order)
                            // Añadimos nuestros parámetros personalizados
                            d.corpus = $('#corpusSelector').val();
                            d.year = $('#yearSelector').val();
                            d.searchable_field = $('#searchFieldSelector').val();
                            d.records_total = $('#yearSelector option:selected').data('count');

                            // Traducimos la ordenación de DataTables al formato de la API
                            if (d.order && d.order.length > 0) {
                                const colIndex = d.order[0].column;
                                const colDir = d.order[0].dir;
                                const colName = d.columns[colIndex].data; // Obtenemos el nombre del campo
                                if (colName) { // Evitamos ordenar la columna "Ver"
                                     d.sort_by_order = `${colName}:${colDir}`;
                                }
                            }
                            // DataTables envía el término de búsqueda en d.search.value
                        }
                    },
                    "columns": [ // MUY IMPORTANTE: el 'name' o 'data' debe coincidir con el de la API
                        {"data":"link"},
                        { "data": "title" },
                        { "data": "cpv" },
			//{ "data": "cpv_predicted"},
                        { "data": "generated_objective" },
                        { "data": "criterios_adjudicacion" },
                        { "data": "criterios_solvencia" },
                        { "data": "condiciones_especiales" },
                        { 
                            "data": null,
                            "orderable": false,
                            "searchable": false,
                            "defaultContent": '<button class="btn btn-primary btn-sm btn-ver">Ver</button>'
                        }
                    ],
                    "scrollX": true,
                    /*"language": {
                        "url": "https://cdn.datatables.net/plug-ins/1.13.7/i18n/es-ES.json"
                    }*/
                });
            }

            



            // La primera carga de la tabla la dispara el 'change' del selector de año
            $('#yearSelector').on('change', initializeDataTable);

           // --- CÓDIGO DEL MODAL CORREGIDO Y OPTIMIZADO ---

            // 1. Obtenemos la instancia del modal UNA SOLA VEZ, cuando la página carga.
            const detalleModal = new bootstrap.Modal(document.getElementById('detalleModal'));

            // 2. Creamos el "escuchador" de eventos para los botones "Ver"
            $('#licitacionesTable tbody').on('click', '.btn-ver', function() {
                const row = table.row($(this).parents('tr'));
                const data = row.data();
                //console.log (data); 
                
                // Verificamos que 'data' no sea undefined antes de usarlo
                if (data) {
                    const modalContentHtml = `
                        <p><strong>Title:</strong> ${data.title}</p>
                        <hr>
                        <p><strong>CPV:</strong> ${data.cpv}</p>
                        <hr>
                        <p><strong>Generated objective:</strong></p>
                        <p>${data.generated_objective}</p>
                        <hr>
                        <p><strong>Award criteria:</strong></p>
                        <p style="white-space: pre-wrap;">${data.criterios_adjudicacion}</p>
                        <hr>
                        <p><strong>Solvency criteria:</strong></p>
                        <p style="white-space: pre-wrap;">${data.criterios_solvencia}</p>
                        <hr>
                        <p><strong>Special conditions:</strong></p>
                        <p style="white-space: pre-wrap;">${data.condiciones_especiales}</p>
                    `;

                    $('#modalTitle').text(data.id);
                    $('#modalBody').html(modalContentHtml);

                    // 3. Simplemente mostramos el modal que ya teníamos preparado.
                    detalleModal.show();
                }
            });

        document.querySelectorAll('a.toggle-vis').forEach((el) => {

            table = $('#licitacionesTable').DataTable();

            el.addEventListener('click', function (e) {
                e.preventDefault();
        
                let columnIdx = e.target.getAttribute('data-column');
                let column = table.column(columnIdx);
        
                // Toggle the visibility
                column.visible(!column.visible());
            });
        });



        });
    </script>


    <div class="modal fade" id="detalleModal" tabindex="-1" aria-labelledby="modalTitle" aria-hidden="true">
        <div class="modal-dialog modal-lg modal-dialog-scrollable">
            <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="modalTitle">Detalles de la Licitación</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body" id="modalBody">
                </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
            </div>
            </div>
        </div>
    </div>    
    
<?php include '../includes/footer.php'; ?>
