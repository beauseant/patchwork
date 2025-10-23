document.addEventListener('DOMContentLoaded', () => {

    // --- 1. CONFIGURACIÓN Y SELECTORES ---
    const PROXY_URL = 'proxy.php';

async function cargarArchivo() {
    try {
        const response = await fetch('servidor.cnf');
        const text = await response.text();
        console.log(text);
        return text;
        // Aquí puedes hacer algo más con el texto
    } catch (error) {
        console.error("Error:", error);
    }
}
cargarArchivo().then(text => {
                console.log("Texto fuera de la función:", text); // Aquí tienes el texto fuera de la función

                console.log(text);

                const API_BASE_URL = text;
                //const API_BASE_URL = text;

                // Selectores de los dropdowns
                const selectCorpus = document.getElementById('selectCorpus');
                const selectCPV = document.getElementById('selectCPV');
                const selectGranularity = document.getElementById('selectGranularity');
                
                // Contenedores del gráfico
                const treemapContainer = document.getElementById('treemap-container');
                const loadingSpinner = document.getElementById('loading-spinner');
                
                // Elementos del Modal
                const topicModal = new bootstrap.Modal(document.getElementById('topicModal'));
                const modalLabel = document.getElementById('topicModalLabel');
                const modalStats = document.getElementById('topic-stats');
                const modalWords = document.getElementById('topic-words');
                const modalDocsAccordion = document.getElementById('topic-docs-accordion');
                const modalDocsLoading = document.getElementById('topic-docs-loading');

                // Almacén de estado temporal
                let currentCorpusData = {}; // Guarda la respuesta de /listCorpusModels
                let currentTreemapData = []; // Guarda la respuesta de /getModelInfo

                // --- 2. FUNCIÓN HELPER PARA FETCH VÍA PROXY ---
                
                /**
                 * Realiza una llamada GET a un endpoint de la API a través del proxy.
                 * @param {string} endpoint El endpoint de la API (ej: '/corpora/listAllCorpus/')
                 */
                async function fetchData(endpoint) {
                    // Codificamos la URL completa para pasarla como parámetro al proxy
                    const encodedUrl = encodeURIComponent(API_BASE_URL + endpoint);
                    try {
                        const response = await fetch(`${PROXY_URL}?url=${encodedUrl}`);
                        if (!response.ok) {
                            const errorData = await response.json();
                            throw new Error(`Error HTTP ${response.status}: ${errorData.error || response.statusText}`);
                        }
                        return await response.json();
                    } catch (error) {
                        console.error('Error al fetchear datos:', error);
                        alert(`Error de red: ${error.message}. Revisa la consola y el proxy.php.`);
                        return null;
                    }
                }

                // --- 3. CADENA DE CARGA DE DATOS (WORKFLOW) ---

                /**
                 * PASO 1: Carga la lista inicial de Corpus.
                 */
                async function loadCorpus() {
                    const corpusList = await fetchData('/corpora/listAllCorpus/');
                    if (corpusList) {
                        populateSelect(selectCorpus, corpusList, 'Select a corpus');
                        selectCorpus.disabled = false;
                    }
                }

                /**
                 * PASO 2: Carga los CPVs cuando se selecciona un Corpus.
                 */
                async function loadCPVs() {
                    const corpus = selectCorpus.value;
                    if (!corpus) return;

                    // Resetear los siguientes pasos
                    resetSelect(selectCPV, 'Loading CPVs...');
                    resetSelect(selectGranularity, 'Waiting CPV...');
                    clearTreemap();
                    
                    const cpvData = await fetchData(`/corpora/listCorpusModels/?corpus_col=${corpus}`);
                    if (cpvData) {
                        currentCorpusData = cpvData; // Guardar datos para el paso 3
                        const cpvKeys = Object.keys(cpvData); // ["45", "48", ...]
                        populateSelect(selectCPV, cpvKeys, 'Select  CPV');
                        selectCPV.disabled = false;
                    }
                }

                /**
                 * PASO 3: Carga la Granularidad cuando se selecciona un CPV.
                 */
                function loadGranularity() {
                    const cpv = selectCPV.value;
                    if (!cpv) return;

                    // Resetear
                    resetSelect(selectGranularity, 'Loading granularity...');
                    clearTreemap();

                    // Extraer granularidad de los datos guardados en el paso 2
                    // currentCorpusData es: { "45": [ {"high": 12}, {"low": 6} ] }
                    try {
                        const granularities = currentCorpusData[cpv].map(item => Object.keys(item)[0]); // ["high", "low"]
                        
                        if (granularities.length > 0) {
                            populateSelect(selectGranularity, granularities, 'Select granularity');
                            selectGranularity.disabled = false;
                        } else {
                            resetSelect(selectGranularity, 'There is no granularity');
                        }
                    } catch (error) {
                        console.error("Error parsing granularity:", error, currentCorpusData);
                        resetSelect(selectGranularity, 'Error loading');
                    }
                }

                /**
                 * PASO 4: Carga y dibuja el Treemap.
                 */
                async function loadTreemap() {
                    const corpus = selectCorpus.value; 
                    const cpv = selectCPV.value;
                    const granularity = selectGranularity.value;

                    if (!corpus || !cpv || !granularity) return;

                    clearTreemap();
                    loadingSpinner.classList.remove('d-none'); // Mostrar spinner

                    const data = await fetchData(`/queries/getModelInfo/?cpv=${cpv}&granularity=${granularity}`);
                    
                    loadingSpinner.classList.add('d-none'); // Ocultar spinner

                    if (data && data.length > 0) {
                        currentTreemapData = data; // Guardar para el modal
                        
                        // D3 necesita un objeto "raíz" (root) que contenga los "hijos" (children)
                        const rootData = {
                            name: "root",
                            children: data
                        };
                        drawTreemap(rootData);
                    } else {
                        treemapContainer.innerHTML = '<p class="text-center text-muted fs-5 mt-5">No se encontraron datos para esta selección.</p>';
                    }
                }

                // --- 4. LÓGICA DE D3 (DIBUJAR EL TREEMAP) ---

                function drawTreemap(data) {
                    clearTreemap();

                    const width = treemapContainer.clientWidth;
                    const height = 600; // Puedes ajustar esta altura

                    // Crear el SVG
                    const svg = d3.select("#treemap-container")
                        .append("svg")
                        .attr("viewBox", `0 0 ${width} ${height}`) // Hacerlo responsive
                        .style("font", "11px sans-serif");

                    // 1. Crear la Jerarquía
                    // Le decimos a D3 que el valor (tamaño) de cada nodo será la propiedad 'alphas'
                    const root = d3.hierarchy(data)
                        .sum(d => d.alphas) 
                        .sort((a, b) => b.value - a.value); // Ordenar de mayor a menor

                    // 2. Crear el Layout del Treemap
                    d3.treemap()
                        .size([width, height])
                        .padding(2) // Espacio entre rectángulos
                        .round(true)
                        (root); // Aplica el layout a nuestros datos

                    // Escala de color
                    const color = d3.scaleOrdinal(d3.schemeTableau10); // Una paleta de colores categórica

                    // 3. Dibujar los Nodos (hojas)
                    const leaf = svg.selectAll("g")
                        .data(root.leaves()) // Solo nos interesan las "hojas" (los tópicos)
                        .join("g")
                        .attr("transform", d => `translate(${d.x0},${d.y0})`)
                        .style("cursor", "pointer")
                        .on("click", (event, d) => handleNodeClick(event, d.data)); // d.data es el objeto original del tópico

                    // Añadir tooltip (usando el title nativo o el de Bootstrap)
                    leaf.append("title")
                        .text(d => `${d.data.tpc_labels}\nDescripción: ${d.data.tpc_descriptions}\nAlphas: ${d.data.alphas.toFixed(4)}`);

                    // Dibujar los rectángulos
                    leaf.append("rect")
                        .attr("id", d => (d.leafUid = `leaf-${d.data.id}`))
                        .attr("fill", d => color(d.data.tpc_labels))
                        .attr("fill-opacity", 0.7)
                        .attr("width", d => d.x1 - d.x0)
                        .attr("height", d => d.y1 - d.y0)
                        .attr("stroke", "#555");

                    // Añadir el texto (con recorte para que no se salga)
                    leaf.append("clipPath")
                        .attr("id", d => (d.clipUid = `clip-${d.data.id}`))
                        .append("use")
                        .attr("xlink:href", d => `#${d.leafUid}`);

                    /*leaf.append("text")
                        .attr("clip-path", d => `url(#${d.clipUid})`)
                        .selectAll("tspan")
                        // Dividimos el label para que quepa mejor
                        .data(d => d.data.tpc_labels.split(/(?=[A-Z][^A-Z])/g) || [d.data.id]) // Divide por mayúsculas o usa ID
                        .join("tspan")
                        .attr("x", 4) // Pequeño padding
                        .attr("y", (d, i) => 13 + i * 11) // Salto de línea
                        .text(d => d)
                        .attr("fill", "black")
                        .style("font-weight", "bold");*/

                    leaf.append("text")
                        .attr("clip-path", d => `url(#${d.clipUid})`)
                        .selectAll("tspan")
                        // --- INICIO DE LA MODIFICACIÓN ---
                        .data(d => {
                            // 1. Procesamos el título como antes
                            const titleLines = d.data.tpc_labels.split(/(?=[A-Z][^A-Z])/g) || [d.data.id];
                            
                            // 2. Procesamos las descripciones
                            //    Las separamos por coma y cogemos solo las 3 primeras para que quepan
                            const descWords = d.data.tpc_descriptions.split(', ').slice(0, 13);
                            
                            // 3. Devolvemos un array combinado
                            //    Guardamos el número de líneas del título para estilarlas después
                            d.titleLineCount = titleLines.length; 
                            return [...titleLines, ...descWords];
                        })
                        // --- FIN DE LA MODIFICACIÓN ---
                        .join("tspan")
                        .attr("x", 4) // Pequeño padding
                        .attr("y", (d, i) => 13 + i * 11) // Salto de línea (funciona igual)
                        .text(d => d)
                        .attr("fill", "black")
                        // --- Ajuste de estilo (Opcional pero recomendado) ---
                        .style("font-weight", (d, i, nodes) => {
                            // Accedemos al dato del nodo padre (leaf)
                            const leafData = d3.select(nodes[i].parentNode).datum();
                            // Si el índice (i) es menor que el nº de líneas del título, es 'bold'
                            return (i < leafData.titleLineCount) ? 'bold' : 'normal';
                        })
                        .style("font-size", (d, i, nodes) => {
                            const leafData = d3.select(nodes[i].parentNode).datum();
                            // Hacemos la letra de la descripción un poco más pequeña
                            return (i < leafData.titleLineCount) ? '11px' : '10px';
                        });            
                    
                }

                // --- 5. LÓGICA DEL MODAL ---

                /**
                 * Se llama al hacer click en un nodo del treemap.
                 * @param {object} event El evento de click
                 * @param {object} data El objeto del tópico (ej: {id: "t1", alphas: ...})
                 */
                function handleNodeClick(event, data) {
                    
                    // 1. Limpiar modal anterior
                    modalLabel.textContent = `Details: ${data.tpc_labels} (ID: ${data.id})`;
                    modalStats.innerHTML = '';
                    modalWords.textContent = '';
                    modalDocsAccordion.innerHTML = '';
                    modalDocsLoading.classList.add('d-none');

                    // 2. Poblar estadísticas (topic_statits)
                    const stats = {
                        'Alphas': data.alphas.toFixed(6),
                        'Topic Entropy': data.topic_entropy.toFixed(6),
                        'Topic Coherence': data.topic_coherence.toFixed(6),
                        'Active Documents': data.ndocs_active
                    };
                    for (const [key, value] of Object.entries(stats)) {
                        const li = document.createElement('li');
                        li.className = 'list-group-item d-flex justify-content-between align-items-center';
                        li.innerHTML = `${key} <span class="badge bg-primary rounded-pill">${value}</span>`;
                        modalStats.appendChild(li);
                    }

                    // 3. Poblar Top Words (tpc_words)
                    // Formatear la cadena "palabra|beta palabra|beta ..."
                    const wordsHtml = data.top_words_betas.split(' ')
                                        .map(w => w.split('|'))
                                        .map(parts => `<strong>${parts[0]}</strong> <span class="text-body-secondary">(${parseFloat(parts[1]).toFixed(1)})</span>`)
                                        .join(' &middot; '); // Separador
                    modalWords.innerHTML = wordsHtml;

                    // 4. Mostrar modal
                    topicModal.show();

                    // 5. Cargar documentos (asíncrono)
                    loadTopicDocs(data.id);
                }

                /**
                 * Carga y muestra los documentos principales en el acordeón del modal.
                 * @param {string} topicId El ID del tópico (ej: "t1")
                 */
                async function loadTopicDocs(topicId) {
                    modalDocsLoading.classList.remove('d-none');
                    modalDocsAccordion.innerHTML = '';

                    // Recoger los valores actuales de los selectores
                    const corpus = selectCorpus.value;
                    const cpv = selectCPV.value;
                    const granularity = selectGranularity.value;

                    //el topicId viene con t0, t1.. pero para hacer la consulta debemos quitar el t, de lo contrario devuelve datos
                    //incorrectos.
                    topicId = topicId.replace ("t","");

                    const endpoint = `/queries/getTopicTopDocs/?corpus_collection=${corpus}&cpv=${cpv}&granularity=${granularity}&topic_id=${topicId}`;
                    const docs = await fetchData(endpoint);
                    

                    modalDocsLoading.classList.add('d-none');

                    if (docs && docs.length > 0) {
                        docs.forEach((doc, index) => {
                            // Crear un ID único para el acordeón
                            const accordionId = `doc-${topicId}-${index}`;
                            
                            const accordionItem = `
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="heading-${accordionId}">
                                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-${accordionId}" aria-expanded="false" aria-controls="collapse-${accordionId}">
                                            Documento #${index + 1} (Relevancia: ${doc.topic_relevance})
                                        </button>
                                    </h2>
                                    <div id="collapse-${accordionId}" class="accordion-collapse collapse" aria-labelledby="heading-${accordionId}" data-bs-parent="#topic-docs-accordion">
                                        <div class="accordion-body">
                                            <p><strong>Objetivo Generado:</strong></p>
                                            <p class="text-muted" style="font-size: 0.9rem; white-space: pre-wrap;">${doc.generated_objective}</p>
                                            <hr>
                                            <p class="mb-0 small"><strong>ID:</strong> <a href="${doc.id}" target="_blank" rel="noopener noreferrer">${doc.id}</a></p>
                                            <p class="mb-0 small"><strong>Nº Palabras:</strong> ${doc.num_words_per_doc}</p>
                                        </div>
                                    </div>
                                </div>
                            `;
                            modalDocsAccordion.innerHTML += accordionItem;
                        });
                    } else {
                        modalDocsAccordion.innerHTML = '<p class="text-muted text-center">No se encontraron documentos para este tópico.</p>';
                    }
                }

                // --- 6. FUNCIONES UTILITARIAS ---

                /**
                 * Rellena un <select> con opciones.
                 * @param {HTMLSelectElement} selectEl El elemento select
                 * @param {string[]} options Array de strings para las opciones
                 * @param {string} placeholder Texto del placeholder (value="")
                 */
                function populateSelect(selectEl, options, placeholder) {
                    selectEl.innerHTML = `<option value="">${placeholder}</option>`;
                    options.forEach(option => {
                        const opt = document.createElement('option');
                        opt.value = option;
                        opt.textContent = option;
                        selectEl.appendChild(opt);
                    });
                }

                /**
                 * Resetea un <select> a su estado inicial.
                 * @param {HTMLSelectElement} selectEl El elemento select
                 * @param {string} message Mensaje de espera
                 */
                function resetSelect(selectEl, message) {
                    selectEl.innerHTML = `<option value="">${message}</option>`;
                    selectEl.disabled = true;
                }

                /**
                 * Limpia el contenedor del treemap.
                 */
                function clearTreemap() {
                    treemapContainer.innerHTML = ''; // Limpiar SVG anterior
                    loadingSpinner.classList.add('d-none'); // Asegurarse de que el spinner esté oculto
                }

                // --- 7. EVENT LISTENERS ---
                selectCorpus.addEventListener('change', loadCPVs);
                selectCPV.addEventListener('change', loadGranularity);
                selectGranularity.addEventListener('change', loadTreemap);

                // --- INICIAR APP ---
                loadCorpus(); // Iniciar la cadena de carga
    });
});