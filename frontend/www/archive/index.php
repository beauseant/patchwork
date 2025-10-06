<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>
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
           
        <h1 class="mb-4">Documentos de Licitaciones</h1>

        <div class="row mb-3">
            <div class="col-md-6">
                <label for="corpusSelector" class="form-label"><strong>Corpus:</strong></label>
                <select id="corpusSelector" class="form-select"></select>
            </div>
            <div class="col-md-6">
                <label for="yearSelector" class="form-label"><strong>Año:</strong></label>
                <select id="yearSelector" class="form-select"></select>
            </div>
        </div>
<div>
        Columnas a mostrar: <a class="toggle-vis" data-column="0">Id</a> - <a class="toggle-vis" data-column="1">Título</a> - <a class="toggle-vis" data-column="2">CPV</a> - <a class="toggle-vis" data-column="3">Objetivo</a> - <a class="toggle-vis" data-column="4">C.Adjudicación</a> - <a class="toggle-vis" data-column="5">C. Solvencia</a> - <a class="toggle-vis" data-column="5">C. Especiales</a>
    </div>
        <table id="licitacionesTable" class="table table-striped table-bordered display responsive nowrap" style="width:100%">
            <thead>
                <tr>
                    <th>Id</th>
                    <th>Título</th>
                    <th>CPV</th>
                    <th>Objetivo Generado</th>
                    <th>Criterios Adjudicación</th>
                    <th>Criterios Solvencia</th>
                    <th>Condiciones Especiales</th>
                    <th>Ver</th> </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>




  <script>



 
    $(document).ready(function() {
        let table; 

        // 1. Cargar Corpus (sin cambios)
        $.ajax({
            url: 'get_corpus.php',
            // ... (código existente)
            success: function(corpora) {
                const corpusSelector = $('#corpusSelector');
                corpusSelector.empty();
                corpora.forEach(function(corpus) {
                    corpusSelector.append(`<option value="${corpus}">${corpus}</option>`);
                });
                corpusSelector.trigger('change');
            }
        });

        // 2. Cargar Años (sin cambios)
        $('#corpusSelector').on('change', function() {
            // ... (código existente)
            const selectedCorpus = $(this).val();
            if (selectedCorpus) {
                $.ajax({
                    url: 'get_years.php',
                    data: { corpus: selectedCorpus },
                    // ... (código existente)
                    success: function(years) {
                        const yearSelector = $('#yearSelector');
                        yearSelector.empty();
                        years.sort((a, b) => b.year - a.year);
                        years.forEach(function(item) {
                            yearSelector.append(`<option value="${item.year}">${item.year} (${item.count} docs)</option>`);
                        });
                        yearSelector.trigger('change');
                    }
                });
            }
        });

        // 3. Cargar DataTable (MODIFICADO)
        $('#yearSelector').on('change', function() {
            const selectedCorpus = $('#corpusSelector').val();
            const selectedYear = $(this).val();

            if (selectedCorpus && selectedYear) {
                if (table) {
                    table.destroy();
                }

                table = $('#licitacionesTable').DataTable({
                    "searching": false, // Desactiva el cuadro de búsqueda
                    "ordering": false,  // Desactiva la ordenación por columnas
                    "processing": true,
                    "serverSide": true,
                    "pageLength": 50,
                    "scrollY": '800px',
                    "ajax": {
                        "url": "get_documents.php",
                        "type": "GET",
                        "data": function(d) {
                            d.corpus = selectedCorpus;
                            d.year = selectedYear;
                        }
                    },
                    "columns": [ // MODIFICADO: Añadida la nueva columna al final
                        { "data": "id" },
                        { "data": "title" },
                        { "data": "cpv" },
                        { "data": "generated_objective" },
                        { "data": "criterios_adjudicacion" },
                        { "data": "criterios_solvencia" },
                        { "data": "condiciones_especiales" },
                        { 
                            "data": null, // No viene de una columna de datos específica
                            "defaultContent": '<button class="btn btn-primary btn-sm btn-ver">Ver</button>',
                            "orderable": false, // Esta columna no se puede ordenar
                            "searchable": false // Esta columna no se puede buscar
                        }
                    ],
                    "language": {
                        "url": "https://cdn.datatables.net/plug-ins/1.13.7/i18n/es-ES.json"
                    }
                });
            }
        });

        

        // 4. GESTOR DE EVENTOS PARA EL MODAL (AÑADIDO)
        // Usamos '.tbody' para delegar el evento, asegurando que funcione
        // incluso para los botones creados dinámicamente en otras páginas de la tabla.
        $('#licitacionesTable tbody').on('click', '.btn-ver', function() {
            // Obtenemos la fila a la que pertenece el botón pulsado
            const row = table.row($(this).parents('tr'));
            const data = row.data(); // Obtenemos el objeto con todos los datos de la fila
            
            // Construimos el contenido HTML para el cuerpo del modal
            const modalContentHtml = `
                <p><strong>Título:</strong> ${data.title}</p>
                <hr>
                <p><strong>CPV:</strong> ${data.cpv}</p>
                <hr>
                <p><strong>Objetivo Generado:</strong></p>
                <p>${data.generated_objective}</p>
                <hr>
                <p><strong>Criterios de Adjudicación:</strong></p>
                <p style="white-space: pre-wrap;">${data.criterios_adjudicacion}</p>
                <hr>
                <p><strong>Criterios de Solvencia:</strong></p>
                <p style="white-space: pre-wrap;">${data.criterios_solvencia}</p>
                <hr>
                <p><strong>Condiciones Especiales:</strong></p>
                <p style="white-space: pre-wrap;">${data.condiciones_especiales}</p>
            `;

            // Rellenamos el modal con los datos
            $('#modalTitle').text(data.title);
            $('#modalBody').html(modalContentHtml);

            // Mostramos el modal
            const myModal = new bootstrap.Modal(document.getElementById('detalleModal'));
            myModal.show();
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
