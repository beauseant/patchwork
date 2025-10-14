<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visor de Licitaciones</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <style>
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
        
        <div class="row mb-3">
             <div class="col-md-6">
                <label for="searchFieldSelector" class="form-label"><strong>Buscar en campo:</strong></label>
                <select id="searchFieldSelector" class="form-select">
                    <option value="*">Todos los campos</option>
                </select>
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
                    <th>Ver</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <div class="modal fade" id="detalleModal" tabindex="-1" aria-labelledby="modalTitle" aria-hidden="true">
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
                        years.sort((a, b) => b.year - a.year).forEach(item => {
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
                        { "data": "title" },
                        { "data": "cpv" },
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
                    "language": {
                        "url": "https://cdn.datatables.net/plug-ins/1.13.7/i18n/es-ES.json"
                    }
                });
            }
            
            // La primera carga de la tabla la dispara el 'change' del selector de año
            $('#yearSelector').on('change', initializeDataTable);

            // 4. Gestor del modal (sin cambios)
            $('#licitacionesTable tbody').on('click', '.btn-ver', function() {
                const data = table.row($(this).parents('tr')).data();
                // ... (código del modal sin cambios)
            });
        });
    </script>
</body>
</html>