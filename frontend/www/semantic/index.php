<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>
<?php include '../includes/utils.php'; ?>


    <div class="container mt-4">

        <?php 

                $salida =  pingHost(getcwd(). '/servidor.cnf' );                            
                if (array_key_exists('NOOK', $salida)) {
                    echo '
                        <div class="alert alert-danger" role="alert">
                            <p>Error en servidor: '. $salida['NOOK'] . ' No es posible subir archivos</p>' .
                        '</div></div>
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

        <div class="card">
            <h5 class="card-header">Corpus Search Tool</h5>
            <div class="card-body p-4 p-md-5">

                <!-- PASO 1: SELECCIÓN DE CORPUS -->
                <div id="step1_corpus" class="mb-3">
                    <label for="corpus_select" class="form-label fw-bold">1. Select Corpus:</label>
                    <select id="corpus_select" class="form-select form-select-lg">
                        <option value="">Loading corpus...</option>
                    </select>
                </div>

                <!-- PASO 2: FORMULARIO PRINCIPAL (Oculto) -->
                <div id="step2_main_form" style="display: none;">
                    <hr class="my-4">
                    <div class="mb-3">
                        <label for="text_input" class="form-label fw-bold">2. Enter the text:</label>
                        <textarea id="text_input" class="form-control" rows="3" placeholder="Escriba su consulta aquí..."></textarea>
                    </div>
                    <div class="mb-3">
                        <label for="query_type" class="form-label fw-bold">3. Select search type:</label>
                        <select id="query_type" class="form-select">
                            <option value="">Select...</option>
                            <option value="topic-based">Topic-Based</option>
                            <option value="semantic-similarity">Semantic Similarity</option>
                        </select>
                    </div>
                </div>

                <!-- PASO 3: OPCIONES TOPIC-BASED (Oculto) -->
                <div id="step3_topic_options" class="row g-3" style="display: none;">
                    <div class="col-md-6 mb-3">
                        <label for="cpv_select" class="form-label fw-bold">4. CPV:</label>
                        <select id="cpv_select" class="form-select">
                            <option value="">Loading models...</option>
                        </select>
                    </div>
                    <div class="col-md-6 mb-3">
                        <label for="granularity_select" class="form-label fw-bold">5. Granularity:</label>
                        <select id="granularity_select" class="form-select">
                            <option value="">First, select a CPV....</option>
                        </select>
                    </div>
                </div>

                <!-- PASO 3b: OPCIONES SEMANTIC SIMILARITY (Oculto) -->
                <div id="step3_semantic_options" class="row g-3" style="display: none;">
                    <div class="col-md-12 mb-3">
                        <label for="keyword_input" class="form-label fw-bold">4. Keyword to refine (optional):</label>
                        <input type="text" id="keyword_input" class="form-select" placeholder="Añada una keyword si lo desea...">
                    </div>
                </div>


                <!-- BOTÓN DE ENVÍO (Oculto) -->
                <button id="submit_button" class="btn btn-primary btn-lg w-100 mt-3" style="display: none;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-search me-2" viewBox="0 0 16 16">
                        <path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"/>
                    </svg>
                    Search
                </button>

                <!-- ÁREA DE RESULTADOS (Oculta) -->
                <div id="results_area" class="mt-5" style="display: none;">
                    <hr>
                    <h4 class="mb-3 text-center">Search Results</h4>
                    
                    <!-- Loader -->
                    <div id="loader" class="loader"></div>

                    <!-- Gráfico y Tabla -->
                    <div id="results_content" style="display: none;">
                        <div class="row" id="topic_chart_row"> <!-- ID AÑADIDO -->
                            <!-- Fila 1: Gráfico Donut -->
                            <div class="col-lg-8 col-md-10 mx-auto mb-4"> <!-- Centrado -->
                                <h5 class="text-center">Topic Distribution</h5>
                                <!-- Contenedor flex para gráfico y leyenda -->
                                <div class="d-flex justify-content-center align-items-center flex-wrap p-3" style="border: 1px solid #ddd; border-radius: 8px; background-color: #fff;">
                                    <div id="donut_chart"></div> <!-- El SVG irá aquí -->
                                    <div id="donut_legend" class="ms-md-4 mt-3 mt-md-0" style="max-height: 360px; overflow-y: auto; font-size: 0.9rem;"></div> <!-- La leyenda irá aquí -->
                                </div>
                            </div>
                        </div>
                        <div class="row">
                             <!-- Fila 2: Tabla de Documentos -->
                            <div class="col-12"> <!-- Ancho completo -->
                                <h5 class="text-center">Similar Documents</h5>
                                <!-- Paginación de API -->
                                <div id="api_pagination" class="d-flex justify-content-between align-items-center mb-2">
                                    <button id="prev_page" class="btn btn-outline-secondary btn-sm">Anterior</button>
                                    <span id="page_info" class="text-muted">Página 1</span>
                                    <button id="next_page" class="btn btn-outline-secondary btn-sm">Siguiente</button>
                                </div>
                                <div class="table-responsive">
                                    <table id="results_table" class="table table-striped table-bordered" style="width:100%">
                                        <!-- El contenido se generará con DataTables -->
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
</div>
<script src="../public/assets/js/d3.min.js"></script>
<script src="app.js"></script>
<?php include '../includes/footer.php'; ?>