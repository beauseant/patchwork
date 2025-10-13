<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>
<?php include '../includes/utils.php'; ?>


<?php 
        $restServer = @file_get_contents( 'servidor.cnf' );
?>

<div class="container mt-4">

    <header class="text-center mb-4">
        <h1>📊 Visualizador de Tópicos con Treemap</h1>
        <p class="lead">Selecciona un corpus y un modelo para generar la visualización automáticamente.</p>
    </header>

    <div class="row g-3 mb-4 p-3 bg-light border rounded">
        <div class="col-md-6">
            <label for="corpus-select" class="form-label"><strong>1. Selecciona un Corpus</strong></label>
            <select id="corpus-select" class="form-select" disabled>
                <option selected>Cargando corpus...</option>
            </select>
        </div>
        <div class="col-md-6">
            <label for="model-select" class="form-label"><strong>2. Selecciona un Modelo</strong></label>
            <select id="model-select" class="form-select" disabled>
                <option selected>Esperando selección de corpus...</option>
            </select>
        </div>
    </div>

    <div class="row">
        <div class="col-12">
            <div id="treemap-container">
                <div class="loading-overlay d-none">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <svg id="treemap-svg" width="100%" height="100%"></svg>
            </div>
             <p class="text-center text-muted mt-2">Haz clic sobre cualquier área del gráfico para ver sus detalles.</p>
        </div>
    </div>
</div>

<div class="modal fade" id="topic-details-modal" tabindex="-1" aria-labelledby="modal-title" aria-hidden="true">
  <div class="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="modal-title">Detalles del Tópico</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body" id="modal-details-content">
        </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
      </div>
    </div>
  </div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<script>
document.addEventListener('DOMContentLoaded', () => {
    const corpusSelect = document.getElementById('corpus-select');
    const modelSelect = document.getElementById('model-select');
    const treemapContainer = document.getElementById('treemap-container');
    const loadingOverlay = treemapContainer.querySelector('.loading-overlay');
    
    const topicModalElement = document.getElementById('topic-details-modal');
    const topicModal = new bootstrap.Modal(topicModalElement);
    const modalTitle = document.getElementById('modal-title');
    const modalDetailsContent = document.getElementById('modal-details-content');

    const PROXY_URL = 'proxy.php?url=';
    const API_BASE_URL = '<?php echo $restServer; ?>';

    const showLoading = (isLoading) => {
        loadingOverlay.classList.toggle('d-none', !isLoading);
    };

    const fetchCorpora = async () => {
        try {
            const target = `${API_BASE_URL}/corpora/listAllCorpus/`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`Error de red`);
            const corpora = await response.json();
            
            corpusSelect.innerHTML = '<option value="" selected disabled>Selecciona uno...</option>';
            corpora.forEach(corpus => {
                corpusSelect.appendChild(new Option(corpus, corpus));
            });
            corpusSelect.disabled = false;
        } catch (error) {
            corpusSelect.innerHTML = '<option selected>Error al cargar</option>';
            console.error("Error fetching corpora:", error);
        }
    };

    const fetchModels = async (corpusName) => {
        if (!corpusName) return;
        modelSelect.disabled = true;
        modelSelect.innerHTML = '<option selected>Cargando modelos...</option>';
        d3.select("#treemap-svg").selectAll("*").remove();
        try {
            const target = `${API_BASE_URL}/corpora/listCorpusModels/?corpus_col=${corpusName}`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`Error de red`);
            const models = await response.json();

            modelSelect.innerHTML = '<option value="" selected disabled>Selecciona uno...</option>';
            models.forEach(model => {
                modelSelect.appendChild(new Option(model, model));
            });
            modelSelect.disabled = false;
        } catch (error) {
            modelSelect.innerHTML = '<option selected>Error al cargar</option>';
            console.error("Error fetching models:", error);
        }
    };
    
    const generateTreemap = async () => {
        const modelCollection = modelSelect.value;
        if (!modelCollection) return;
        
        showLoading(true);
        d3.select("#treemap-svg").selectAll("*").remove();

        try {
            const target = `${API_BASE_URL}/queries/getModelInfo/?model_collection=${modelCollection}`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`Error de red`);
            const topicData = await response.json();
            drawTreemap(topicData);
        } catch (error) {
            console.error("Error fetching model info:", error);
            alert("Error al cargar los datos del modelo.");
        } finally {
            showLoading(false);
        }
    };

    const drawTreemap = (data) => {
        const width = treemapContainer.clientWidth;
        const height = treemapContainer.clientHeight;
        const svg = d3.select("#treemap-svg");

        const root = d3.hierarchy({ children: data })
            .sum(d => d.alphas)
            .sort((a, b) => b.value - a.value);

        d3.treemap().size([width, height]).padding(2)(root);
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        const nodes = svg.selectAll("g").data(root.leaves()).join("g")
            .attr("transform", d => `translate(${d.x0},${d.y0})`);

        nodes.append("rect")
            .attr("class", "node-rect")
            .attr("width", d => d.x1 - d.x0)
            .attr("height", d => d.y1 - d.y0)
            .attr("fill", d => colorScale(d.data.id))
            .on("click", (event, d) => displayTopicDetails(d.data));
            
        nodes.append("text").attr("class", "node-label")
            .selectAll("tspan").data(d => d.data.tpc_labels.split(" ").slice(0, 4))
            .join("tspan").attr("x", 5).attr("y", (d, i) => 15 + i * 12).text(d => d);
    };

    // --- FUNCIÓN MODIFICADA ---
    const displayTopicDetails = (topic) => {
        modalTitle.textContent = topic.tpc_labels;

        const topWordsHtml = topic.top_words_betas.split(' ')
            .map(wordPair => {
                const [word, beta] = wordPair.split('|');
                return `<li>${word.replace(/_/g, ' ')}: <span class="badge bg-secondary">${parseFloat(beta).toFixed(2)}</span></li>`;
            }).join('');

        // 1. Mostrar contenido estático y placeholder de carga para los documentos
        modalDetailsContent.innerHTML = `
            <p><strong>Descripción:</strong> ${topic.tpc_descriptions.replace(/_/g, ' ')}</p>
            <h6><i class="bi bi-bar-chart-fill"></i> Estadísticas (ID: ${topic.id})</h6>
            <ul class="list-group list-group-flush mb-3">
                <li class="list-group-item d-flex justify-content-between align-items-center">Alphas (Peso)<span class="badge bg-primary rounded-pill">${topic.alphas.toFixed(4)}</span></li>
                <li class="list-group-item d-flex justify-content-between align-items-center">Entropía<span class="badge bg-primary rounded-pill">${topic.topic_entropy.toFixed(4)}</span></li>
                <li class="list-group-item d-flex justify-content-between align-items-center">Coherencia<span class="badge bg-primary rounded-pill">${topic.topic_coherence.toFixed(4)}</span></li>
                <li class="list-group-item d-flex justify-content-between align-items-center">Documentos Activos<span class="badge bg-primary rounded-pill">${topic.ndocs_active}</span></li>
            </ul>
            <h6><i class="bi bi-tags-fill"></i> Palabras Principales</h6>
            <ul class="list-unstyled" style="column-count: 2;">${topWordsHtml}</ul>
            <h6 class="mt-3"><i class="bi bi-file-earmark-text-fill"></i> Top Documents</h6>
            <div id="top-docs-container">
                <div class="d-flex align-items-center text-muted">
                    <strong>Cargando documentos...</strong>
                    <div class="spinner-border spinner-border-sm ms-auto" role="status" aria-hidden="true"></div>
                </div>
            </div>
        `;
        
        topicModal.show();

        // 2. Cargar asíncronamente los documentos
        const fetchTopDocs = async () => {
            try {
                const corpus = corpusSelect.value;
                const model = modelSelect.value;
                const topicId = topic.id;
                
                const target = `${API_BASE_URL}/queries/getTopicTopDocs/?corpus_collection=${corpus}&model_name=${model}&topic_id=${topicId}&start=1&rows=10`;
                const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);

                if (!response.ok) throw new Error('Error en la respuesta del servidor');

                // Leer como texto para limpiar posible coma final (JSON inválido)
                const responseText = await response.text();
                const cleanJsonText = responseText.trim().replace(/,\s*\]$/, ']');
                const topDocs = JSON.parse(cleanJsonText);
                
                const topDocsContainer = document.getElementById('top-docs-container');
                if (topDocs && topDocs.length > 0) {
                    const docsHtml = topDocs.map(doc => `
                        <li class="list-group-item d-flex justify-content-between align-items-start">
                            <div class="ms-2 me-auto" style="overflow-wrap: break-word;">
                                <a href="${doc.id}" target="_blank" rel="noopener noreferrer" class="fw-bold">${doc.id}</a>
                                <div class="small text-muted">Nº Palabras: ${doc.num_words_per_doc}</div>
                            </div>
                            <span class="badge bg-info rounded-pill ms-2" title="Relevancia del Tópico">${doc.topic_relevance}</span>
                        </li>
                    `).join('');
                    topDocsContainer.innerHTML = `<ul class="list-group">${docsHtml}</ul>`;
                } else {
                    topDocsContainer.innerHTML = `<p class="text-muted">No se encontraron documentos para este tópico.</p>`;
                }
            } catch (error) {
                console.error("Error fetching top docs:", error);
                const topDocsContainer = document.getElementById('top-docs-container');
                if (topDocsContainer) {
                    topDocsContainer.innerHTML = `<p class="text-danger">No se pudieron cargar los documentos.</p>`;
                }
            }
        };

        fetchTopDocs();
    };

    corpusSelect.addEventListener('change', (e) => fetchModels(e.target.value));
    modelSelect.addEventListener('change', () => { if (modelSelect.value) generateTreemap(); });

    fetchCorpora();
});
</script>

<?php include '../includes/footer.php'; ?>