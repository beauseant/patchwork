<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visor de Licitaciones</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <style>
        /* Estilo para que las celdas no se expandan demasiado */
        #licitacionesTable td {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
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

<table id="licitacionesTable" class="table table-striped table-bordered" style="width:100%">
    <thead>
        <tr>
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

    

    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>

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
                    "processing": true,
                    "serverSide": true,
                    "ajax": {
                        "url": "get_documents.php",
                        "type": "GET",
                        "data": function(d) {
                            d.corpus = selectedCorpus;
                            d.year = selectedYear;
                        }
                    },
                    "columns": [ // MODIFICADO: Añadida la nueva columna al final
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


    </body>
</html>