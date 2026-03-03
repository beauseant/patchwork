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
    // Variables globales
        let selectedCorpus = '';
        let corpusModelsData = null; // Almacenará la respuesta de listCorpusModels
        let dtInstance = null; // Almacenará la instancia de DataTables
        
        // Variables de paginación de API
        let currentPage = 0; // La API usa 'start', así que esto será un índice (0, 1, 2...)
        const rowsPerPage = 10; // Fijo por tu especificación

        // --- INICIALIZACIÓN ---
        
        $(document).ready(function() {
            // 1. Cargar la lista inicial de corpus
            loadCorpora();

            // 2. Configurar los event listeners
            $('#corpus_select').on('change', handleCorpusSelect);
            $('#query_type').on('change', handleQueryTypeSelect);
            $('#cpv_select').on('change', handleCpvSelect);
            $('#submit_button').on('click', handleSubmit);

            // 3. Listeners de paginación de API
            $('#prev_page').on('click', function() {
                if (currentPage > 0) {
                    currentPage--;
                    handleSubmit();
                }
            });

            $('#next_page').on('click', function() {
                currentPage++;
                handleSubmit();
            });
        });

        // --- MANEJO DE ESTADO (Loader) ---

        function showLoader(show) {
            if (show) {
                $('#loader').show();
                $('#results_content').hide(); // Ocultar contenido mientras carga
                $('#submit_button').prop('disabled', true).text('Buscando...');
            } else {
                $('#loader').hide();
                $('#results_content').show(); // Mostrar contenido
                $('#submit_button').prop('disabled', false).html('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-search me-2" viewBox="0 0 16 16"><path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001c.03.04.062.078.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1.007 1.007 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0z"/></svg> Buscar');
            }
        }

        // --- LÓGICA DE FORMULARIO ---

        function loadCorpora() {
            $.getJSON('proxy.php?action=listCorpus')
                .done(function(data) {
                    const $select = $('#corpus_select');
                    $select.empty().append('<option value="">Select a corpus...</option>');
                    // data es un array de strings, ej: ["combined_data_..."]
                    data.forEach(function(corpusName) {
                        $select.append($('<option>', {
                            value: corpusName,
                            text: corpusName
                        }));
                    });
                })
                .fail(function(jqXHR, textStatus, errorThrown) {
                    console.error("Error al cargar corpus:", textStatus, errorThrown, jqXHR.responseJSON);
                    alert("Error loading the corpus list. Check the console.");
                });
        }

        function handleCorpusSelect() {
            selectedCorpus = $(this).val();
            // Ocultar todo lo que sigue
            $('#step2_main_form, #step3_topic_options, #submit_button, #results_area').hide();
            $('#query_type').val(''); // Resetear
            
            if (selectedCorpus) {
                // Mostrar el siguiente paso
                $('#step2_main_form').slideDown();
                // Precargar los modelos para este corpus
                loadCorpusModels(selectedCorpus);
            }
        }

        function loadCorpusModels(corpusName) {
            const $cpvSelect = $('#cpv_select');
            $cpvSelect.empty().append('<option value="">Cargando modelos...</option>').prop('disabled', true);
            
            $.getJSON('proxy.php?action=listModels', { corpus_col: corpusName })
                .done(function(data) {
                    // data es un objeto, ej: {"45": [{"high": 12}, {"low": 6}]}
                    corpusModelsData = data; // Guardar globalmente
                    
                    $cpvSelect.empty().append('<option value="">Select a CPV...</option>');
                    
                    // Las claves del objeto son los CPV
                    const cpvKeys = Object.keys(corpusModelsData);
                    cpvKeys.forEach(function(cpv) {
                        $cpvSelect.append($('<option>', {
                            value: cpv,
                            text: cpv
                        }));
                    });
                    $cpvSelect.prop('disabled', false);
                })
                .fail(function(jqXHR, textStatus, errorThrown) {
                    console.error("Error loading models:", textStatus, errorThrown, jqXHR.responseJSON);
                    $cpvSelect.empty().append('<option value="">Error al cargar</option>');
                    alert("Error loading CPV models. Check the console.");
                });
        }

        function handleQueryTypeSelect() {
            const queryType = $(this).val();
            $('#submit_button').hide(); // Ocultar botón por defecto
            
            if (queryType === 'topic-based') {
                $('#step3_topic_options').slideDown();
                $('#step3_semantic_options').slideUp(); // Ocultar el otro
                $('#submit_button').show(); // Mostrar botón solo si todo está listo
            } else if (queryType === 'semantic-similarity') {
                $('#step3_topic_options').slideUp(); // Ocultar el otro
                $('#step3_semantic_options').slideDown(); // Mostrar keyword
                $('#submit_button').show(); // Mostrar botón
            } else {
                $('#step3_topic_options').slideUp();
                $('#step3_semantic_options').slideUp();
            }
        }

        function handleCpvSelect() {
            const selectedCpv = $(this).val();
            const $granularitySelect = $('#granularity_select');
            
            $granularitySelect.empty();
            
            if (selectedCpv && corpusModelsData[selectedCpv]) {
                // corpusModelsData[selectedCpv] es un array, ej: [{"high": 12}, {"low": 6}]
                const granularities = corpusModelsData[selectedCpv];
                
                $granularitySelect.append('<option value="">Select granularity...</option>');
                
                granularities.forEach(function(granularityObj) {
                    // El key del objeto es el nombre, ej: "high"
                    const granularityName = Object.keys(granularityObj)[0];
                    $granularitySelect.append($('<option>', {
                        value: granularityName,
                        text: granularityName
                    }));
                });
            } else {
                $granularitySelect.append('<option value="">First, Select a CPV...</option>');
            }
        }
        // --- LÓGICA DE BÚSQUEDA Y RESULTADOS ---

        async function handleSubmit() {
            const queryType = $('#query_type').val();
            const text = $('#text_input').val();
            
            if (!text) {
                alert("Please complete the main text field.");
                return;
            }

            if (currentPage === 0) {
                $('#results_area').slideDown();
            }
            showLoader(true);
            $('#page_info').text(`Página ${currentPage + 1}`);
            $('#prev_page').prop('disabled', currentPage === 0);
            const start = currentPage * rowsPerPage;

            if (queryType === 'topic-based') {
                const cpv = $('#cpv_select').val();
                const granularity = $('#granularity_select').val();

                if (!cpv || !granularity) {
                    alert("Please complete all fields: CPV and granularity.");
                    showLoader(false);
                    return;
                }
                
                $('#topic_chart_row').show();

                try {
                    // Hacer ambas peticiones en paralelo
                    const [data, topicsMap] = await Promise.all([
                        $.getJSON('proxy.php?action=getSimilarDocs', {
                            corpus_collection: selectedCorpus,
                            cpv: cpv,
                            granularity: granularity,
                            text_to_infer: text,
                            start: start,
                            rows: rowsPerPage
                        }),
                        $.getJSON('proxy.php?action=getTopicsLabels', {
                            cpv: cpv,
                            granularity: granularity,
                        })
                    ]);

		    const diccionario = {};
		    topicsMap.forEach(item => {
			    diccionario[item.id] = item.tpc_labels;
		    });

		    //console.log(diccionario);
		    const resultado = data.topics.split(' ').map(item => {
		    // Separamos el "tX" del valor numérico
		   const [id, valor] = item.split('|');

		   // Verificamos si ese ID existe en tus datos
		    if (diccionario[id]) {
		        // Obtenemos el label y reemplazamos espacios por guiones bajos
		        // Usamos una expresión regular / /g para reemplazar TODOS los espacios
		        const labelLimpia = diccionario[id].replace(/ /g, '_');
		        // Retornamos el formato solicitado: tX (label)|valor
		        return `${id}_(${labelLimpia})|${valor}`;
		    }

		    // Si no tenemos info para ese ID (ej: t0, t1), devolvemos el original
			    return item;
		     }).join(' '); // Unimos todo de nuevo con espacios

	             // Resultado
		    console.log(resultado);

		    data.topics=resultado;
                    // data = { topics: "...", mostSimilar: [...] }
                    
                    if (currentPage === 0) {
                        drawDonutChart(data.topics);
			            console.log (data.topics);
                    }
                    
                    if (data.mostSimilar && data.mostSimilar.length > 0) {
                        fetchAndDisplayDocuments(data.mostSimilar);
                        $('#next_page').prop('disabled', false); 
                    } else {
                        alert("No se encontraron más resultados.");
                        $('#next_page').prop('disabled', true);
                        if (currentPage === 0) {
                             $('#results_table').DataTable().clear().draw();
                        }
                        showLoader(false);
                    }
                    
                    } catch (error) {
                        console.error("Error:", error);
                        alert("Error while searching. Check the console.");
                        showLoader(false);
                    }
            
            } else if (queryType === 'semantic-similarity') {
                // Lógica para Semantic Similarity
                const keyword = $('#keyword_input').val(); // Es opcional, puede ir vacío
                
                $('#topic_chart_row').hide(); // Ocultar el gráfico de tópicos

                $.getJSON('proxy.php?action=getSimilarDocsEmb', {
                    corpus_collection: selectedCorpus,
                    free_text: text,
                    keyword: keyword,
                    start: start,
                    rows: rowsPerPage
                })
                .done(function(data) {
                    // data = { mostSimilar: [...] }
                    // NO hay 'topics' aquí
                    
                    console.log(data);
		    
                    
                    if (data.mostSimilar && data.mostSimilar.length > 0) {
                        fetchAndDisplayDocuments(data.mostSimilar);
                        $('#next_page').prop('disabled', false);
                    } else {
                        alert("No se encontraron más resultados.");
                        $('#next_page').prop('disabled', true);
                        if (currentPage === 0) {
                             $('#results_table').DataTable().clear().draw();
                        }
                        showLoader(false);
                    }
                })
                .fail(function(jqXHR, textStatus, errorThrown) {
                    console.error("Error en getSimilarDocsEmb:", textStatus, errorThrown, jqXHR.responseJSON);
                    alert("Error al realizar la búsqueda. Revise la consola.");
                    showLoader(false);
                });
            }
        }

function drawDonutChart(topicsString) {
            // topicsString = "t0|76 t1|194 t2|39..."
            
            // 1. Parsear los datos
            let totalValue = 0;
            const topicData = topicsString.split(' ').map(topicPair => {
                const parts = topicPair.split('|');
                const value = +parts[1]; // Convertir a número
                totalValue += value;
                return {
                    label: parts[0],
                    value: value
                };
            });

            // 2. Añadir el tópico "Restante" para completar 1000
            if (totalValue < 1000) {
                topicData.push({
                    label: "Resto",
                    value: 1000 - totalValue
                });
            }

            // 3. Configuración del gráfico D3
            $('#donut_chart').empty(); // Limpiar gráfico anterior
            
            const width = 360,
                  height = 360,
                  margin = 40;
            
            const radius = Math.min(width, height) / 2 - margin;

            const svg = d3.select("#donut_chart")
                .append("svg")
                    .attr("width", width)
                    .attr("height", height)
                .append("g")
                    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

            // 4. Esquema de color
            const color = d3.scaleOrdinal(d3.schemeCategory10);

            // 5. Generador de pie
            const pie = d3.pie()
                .value(d => d.value)
                .sort(null); // No ordenar, mantener el orden de la API

            const data_ready = pie(topicData);

            // 6. Generador de arcos (para el donut)
            const arc = d3.arc()
                .innerRadius(radius * 0.5) // Radio interno
                .outerRadius(radius * 0.8); // Radio externo
            
            // 6b. Generador de arcos para la posición de los labels (porcentajes)
            const labelArc = d3.arc()
                .innerRadius(radius * 0.65) // Posición dentro del arco
                .outerRadius(radius * 0.65);

            // 7. Tooltip
            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("position", "absolute")
                .style("z-index", "10")
                .style("visibility", "hidden")
                .style("background", "#fff")
                .style("border", "1px solid #ccc")
                .style("border-radius", "5px")
                .style("padding", "8px")
                .style("font-size", "12px");

            // 8. Dibujar los arcos
            svg.selectAll('path')
                .data(data_ready)
                .enter()
                .append('path')
                .attr('d', arc)
                .attr('fill', d => color(d.data.label))
                .attr("stroke", "white")
                .style("stroke-width", "2px")
                .style("opacity", 0.8)
                /*.on("mouseover", function(d) {
                    d3.select(this).style("opacity", 1);
		    
                    return tooltip.style("visibility", "visible").text(`${d.data.label}: ${d.data.value} (de 1000)`);
                })
                .on("mousemove", function() {
                    return tooltip.style("top", (d3.event.pageY - 10) + "px").style("left", (d3.event.pageX + 10) + "px");
                })
                .on("mouseout", function() {
                    d3.select(this).style("opacity", 0.8);
                    return tooltip.style("visibility", "hidden");
                });*/
            
            // 9. Añadir texto de porcentaje
            svg.selectAll('text.percentage')
               .data(data_ready)
               .enter()
               .append('text')
               .attr('class', 'percentage')
               .attr('transform', d => `translate(${labelArc.centroid(d)})`)
               .attr('dy', '0.35em')
               .style('text-anchor', 'middle')
               .style('font-size', '12px')
               .style('fill', 'white')
               .style('font-weight', 'bold')
               .text(d => {
                   const percent = (d.data.value / 1000) * 100;
                   // Solo mostrar si es lo suficientemente grande (ej: > 3%)
                   return percent > 3 ? `${percent.toFixed(0)}%` : ''; 
               });

            // 10. Crear la leyenda
            const legend = d3.select("#donut_legend");
            legend.html(''); // Limpiar leyenda anterior

            const legendItems = legend.selectAll('.legend-item')
                .data(topicData) // Usar topicData original
                .enter()
                .append('div')
                .attr('class', 'legend-item d-flex align-items-center mb-1');

            legendItems.append('span')
                .style('width', '15px')
                .style('height', '15px')
                .style('background-color', d => color(d.label))
                .style('border-radius', '3px')
                .style('margin-right', '8px');
                
            legendItems.append('span')
                .text(d => `${d.label}: ${d.value} (${((d.value/1000)*100).toFixed(1)}%)`);
        }

        function fetchAndDisplayDocuments(documents) {
            // documents es el array 'mostSimilar'
            // [ {id: "...", generated_objective: "...", score: ...}, ... ]
            
            // Creamos un array de promesas para las llamadas de metadatos
            const metadataPromises = documents.map(doc => {
                return $.getJSON('proxy.php?action=getMetadata', {
                    corpus_collection: selectedCorpus,
                    doc_id: doc.id // El proxy se encargará de encodear si es necesario
                }).then(metadataArray => {
                    // La API devuelve un array, tomamos el primer elemento
                    const metadata = metadataArray[0];
                    
                    // Combinamos los datos del documento original (score, objective)
                    // con los nuevos metadatos (title, cpv, etc.)
                    return {
                        ...metadata, // id, cpv, title, generated_objective, cpv_predicted...
                        score: doc.score, // Añadimos el score de la búsqueda
                        link: doc.link
                        // El 'generated_objective' de metadata sobrescribirá el de 'doc',
                        // pero según tu API, parecen ser el mismo.
                    };
                });
            });

            // Esperar a que todas las llamadas de metadatos terminen
            Promise.all(metadataPromises)
                .then(fullDocumentsData => {
                    // 'fullDocumentsData' es ahora un array de objetos completos
                    initializeDataTable(fullDocumentsData);
                    showLoader(false); // Ocultar loader cuando la tabla esté lista
                })
                .catch(error => {
                    console.error("Error al obtener metadatos:", error);
                    alert("Error al cargar los detalles de los documentos. Revise la consola.");
                    showLoader(false);
                });
        }

        function initializeDataTable(data) {
            // Si la tabla ya existe, la destruimos para recrearla
            if (dtInstance) {
                dtInstance.destroy();
            }

            // Definir las columnas
            const columns = [
                { title: "Enlace", data: "link", render: (d) => `<a href="${d}" target="_blank" title="${d}">${d.substring(0, 50)}...</a>` },
                { title: "Título", data: "title" },
                { title: "CPV (Predicho)", data: "cpv_predicted" },
                { title: "Score", data: "score", render: (d) => d.toFixed(2) }, // Formatear score
                { title: "Objetivo", data: "generated_objective" }
            ];

            // Inicializar DataTables
            dtInstance = $('#results_table').DataTable({
                data: data,
                columns: columns,
                responsive: true,                
                paging: false, // Usamos la paginación de DataTables
                pageLength: 5, // Mostrar 5 por página
                lengthMenu: [5, 10, 25],
                searching: false, // Deshabilitamos la búsqueda de DataTables
                info: false // Deshabilitamos la info de DataTables
            });
        }
    });
});
