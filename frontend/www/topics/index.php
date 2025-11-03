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
            <h5 class="card-header">Topic Viewer</h5>
            <div class="card-body p-4 p-md-5">        
                <div class="row g-3 p-3 border rounded mb-4 bg-light shadow-sm">
                    <div class="col-md-4">
                        <label for="selectCorpus" class="form-label fw-bold">1. Select Corpus:</label>
                        <select id="selectCorpus" class="form-select" disabled>
                            <option selected>Loading...</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label for="selectCPV" class="form-label fw-bold">2. Select CPV:</label>
                        <select id="selectCPV" class="form-select" disabled>
                            <option selected>Waiting for corpus...</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label for="selectGranularity" class="form-label fw-bold">3. Select Granularity:</label>
                        <select id="selectGranularity" class="form-select" disabled>
                            <option selected>Waiting for CPV...</option>
                        </select>
                    </div>
                </div>

                <div id="treemap-container" class="border rounded p-2" style="min-height: 600px; position: relative;">
                    <div id="loading-spinner" class="d-none text-center mt-5">
                        <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2 fs-5">Loading chart data...</p>
                    </div>
                    </div>

                <div class="modal fade" id="topicModal" tabindex="-1" aria-labelledby="topicModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="topicModalLabel">Topic Details</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Cerrar"></button>
                            </div>
                            <div class="modal-body">
                                <h5>📊 Topic Statistics</h5>
                                <ul id="topic-stats" class="list-group list-group-flush mb-3">
                                    </ul>
                                
                                <h5 class="mt-3">🔑 Top Words</h5>
                                <p id="topic-words" class="p-2 bg-light rounded text-muted" style="font-size: 0.9rem;">
                                    </p>
                                
                                <h5 class="mt-3">📑 Top Documents</h5>
                                <div id="topic-docs-loading" class="text-center d-none p-3">
                                    <div class="spinner-border spinner-border-sm" role="status"></div>
                                    <span class="ms-2">Loading documents..</span>
                                </div>
                                <div class="accordion" id="topic-docs-accordion">
                                    </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div> 
        </div>
    </div></div>
   <script src="../public/assets/js/d3.min.js"></script>
    <script src="app.js"></script>
<?php include '../includes/footer.php'; ?>