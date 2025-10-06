<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador de Tópicos (Treemap)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        /* Estilos para el treemap y la interacción */
        #treemap-container {
            width: 100%;
            height: 70vh; /* Altura relativa a la ventana */
            min-height: 500px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }
        .node-rect {
            stroke: #fff;
            stroke-width: 2px;
            transition: fill 0.3s ease;
        }
        .node-rect:hover {
            opacity: 0.8;
            cursor: pointer;
        }
        .node-label {
            font-size: 12px;
            fill: white;
            pointer-events: none; /* Para que el texto no interfiera con el clic en el rectángulo */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
        }
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.7);
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>

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
  <div class="modal-dialog modal-lg modal-dialog-centered">
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
    // --- Referencias a Elementos del DOM ---
    const corpusSelect = document.getElementById('corpus-select');
    const modelSelect = document.getElementById('model-select');
    const treemapContainer = document.getElementById('treemap-container');
    const loadingOverlay = treemapContainer.querySelector('.loading-overlay');
    
    // --- Referencias al Modal de Bootstrap ---
    const topicModalElement = document.getElementById('topic-details-modal');
    const topicModal = new bootstrap.Modal(topicModalElement);
    const modalTitle = document.getElementById('modal-title');
    const modalDetailsContent = document.getElementById('modal-details-content');

    // --- Configuración del Proxy ---
    const PROXY_URL = 'proxy.php?url=';
    const API_BASE_URL = 'http://kumo01.tsc.uc3m.es:9083';

    // Función para mostrar/ocultar el spinner
    const showLoading = (isLoading) => {
        loadingOverlay.classList.toggle('d-none', !isLoading);
    };

    // 1. Cargar la lista de Corpus al iniciar
    const fetchCorpora = async () => {
        try {
            const target = `${API_BASE_URL}/corpora/listAllCorpus/`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const corpora = await response.json();
            
            corpusSelect.innerHTML = '<option value="" selected disabled>Selecciona uno...</option>';
            corpora.forEach(corpus => {
                const option = document.createElement('option');
                option.value = corpus;
                option.textContent = corpus;
                corpusSelect.appendChild(option);
            });
            corpusSelect.disabled = false;
        } catch (error) {
            corpusSelect.innerHTML = '<option selected>Error al cargar</option>';
            console.error("Error fetching corpora:", error);
        }
    };

    // 2. Cargar modelos cuando se selecciona un corpus
    const fetchModels = async (corpusName) => {
        if (!corpusName) return;
        modelSelect.disabled = true;
        modelSelect.innerHTML = '<option selected>Cargando modelos...</option>';
        d3.select("#treemap-svg").selectAll("*").remove(); // Limpiar gráfico anterior
        try {
            const target = `${API_BASE_URL}/corpora/listCorpusModels/?corpus_col=${corpusName}`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const models = await response.json();

            modelSelect.innerHTML = '<option value="" selected disabled>Selecciona uno...</option>';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
            modelSelect.disabled = false;
        } catch (error) {
            modelSelect.innerHTML = '<option selected>Error al cargar</option>';
            console.error("Error fetching models:", error);
        }
    };
    
    // 3. Generar el Treemap (se llama al seleccionar modelo)
    const generateTreemap = async () => {
        const modelCollection = modelSelect.value;
        if (!modelCollection) return;
        
        showLoading(true);
        d3.select("#treemap-svg").selectAll("*").remove(); // Limpiar SVG anterior

        try {
            const target = `${API_BASE_URL}/queries/getModelInfo/?model_collection=${modelCollection}`;
            const response = await fetch(`${PROXY_URL}${encodeURIComponent(target)}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const topicData = await response.json();
            
            drawTreemap(topicData);

        } catch (error) {
            console.error("Error fetching model info:", error);
            alert("Error al cargar los datos del modelo.");
        } finally {
            showLoading(false);
        }
    };

    // Función principal de D3.js para dibujar el gráfico
    const drawTreemap = (data) => {
        const width = treemapContainer.clientWidth;
        const height = treemapContainer.clientHeight;
        const svg = d3.select("#treemap-svg");

        const rootData = { name: "root", children: data };
        const root = d3.hierarchy(rootData)
            .sum(d => d.alphas)
            .sort((a, b) => b.value - a.value);

        d3.treemap().size([width, height]).padding(2)(root);
        
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        const nodes = svg.selectAll("g")
            .data(root.leaves())
            .join("g")
            .attr("transform", d => `translate(${d.x0},${d.y0})`);

        nodes.append("rect")
            .attr("class", "node-rect")
            .attr("width", d => d.x1 - d.x0)
            .attr("height", d => d.y1 - d.y0)
            .attr("fill", d => colorScale(d.data.id))
            .on("click", (event, d) => {
                displayTopicDetails(d.data); // MODIFICADO: Llama a la función del modal
            });
            
        nodes.append("text")
            .attr("class", "node-label")
            .selectAll("tspan")
            .data(d => d.data.tpc_labels.split(" ").slice(0, 4))
            .join("tspan")
            .attr("x", 5)
            .attr("y", (d, i) => 15 + i * 12)
            .text(d => d);
    };

    // 4. Mostrar detalles en la ventana modal
    const displayTopicDetails = (topic) => {
        // Actualizar el título del modal
        modalTitle.textContent = topic.tpc_labels;

        // Formatear las palabras principales
        const topWordsHtml = topic.top_words_betas.split(' ')
            .map(wordPair => {
                const [word, beta] = wordPair.split('|');
                return `<li>${word.replace(/_/g, ' ')}: <span class="badge bg-secondary">${parseFloat(beta).toFixed(2)}</span></li>`;
            }).join('');

        // Generar el contenido HTML para el cuerpo del modal
        modalDetailsContent.innerHTML = `
            <p><strong>Descripción:</strong> ${topic.tpc_descriptions.replace(/_/g, ' ')}</p>
            
            <h6><i class="bi bi-bar-chart-fill"></i> Estadísticas del Tópico (ID: ${topic.id})</h6>
            <ul class="list-group list-group-flush mb-3">
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Alphas (Peso)
                    <span class="badge bg-primary rounded-pill">${topic.alphas.toFixed(4)}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Entropía
                    <span class="badge bg-primary rounded-pill">${topic.topic_entropy.toFixed(4)}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Coherencia
                    <span class="badge bg-primary rounded-pill">${topic.topic_coherence.toFixed(4)}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Documentos Activos
                    <span class="badge bg-primary rounded-pill">${topic.ndocs_active}</span>
                </li>
            </ul>

            <h6><i class="bi bi-tags-fill"></i> Palabras Principales (Top Words)</h6>
            <ul class="list-unstyled" style="column-count: 2;">${topWordsHtml}</ul>
            
            <h6 class="mt-3"><i class="bi bi-file-earmark-text-fill"></i> Top Documents</h6>
            <p class="text-muted"><i>(Endpoint para documentos no especificado)</i></p>
        `;
        
        // Mostrar el modal
        topicModal.show();
    };

    // --- Event Listeners ---
    corpusSelect.addEventListener('change', (e) => {
        fetchModels(e.target.value);
    });
    
    // MODIFICADO: Ahora genera el gráfico directamente
    modelSelect.addEventListener('change', () => {
        if (modelSelect.value) {
            generateTreemap();
        }
    });

    // Iniciar la carga de datos
    fetchCorpora();
});
</script>

</body>
</html>