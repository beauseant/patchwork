<?php include '../includes/header.php'; ?>
<?php include '../includes/sidebar.php'; ?>
<?php include '../includes/utils.php'; ?>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>

<div class="container mt-4">

      <div class="card">


            <h5 class="card-header">Indicadores</h5>
            <div class="card-body p-4 p-md-5">

            <!-- Panel selector -->
            <div class="panel-selector">
              <div class="panel-selector-title">
                <i class="bi bi-sliders me-1"></i> Visible panels — click to toggle
              </div>
              <div class="panel-pills" id="panel-pills"></div>
            </div>




            <!-- Cards grid -->
            <div id="panels-grid">
              <div class="state-overlay" id="loading-state">
                <div class="spinner"></div>
                Loading indicators…
              </div>
            </div>


            

      </div>



</div>
<!-- Global tooltip -->
<div class="d3-tooltip" id="tooltip"></div>



<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

<script>
        /* ================================================================
          dashboard.js — fully data-driven, easy to extend
          ================================================================ */

        const PROXY_URL   = 'proxy.php';
        const ACCENT_POOL = ['#3b82f6','#10b981','#f59e0b','#8b5cf6',
                            '#ec4899','#06b6d4','#f97316','#84cc16'];

        // persisted visibility state (localStorage)
        const STORAGE_KEY = 'dash_visible_panels';

        let globalData   = null;
        let visiblePanels = null;          // Set of indicator ids

        /* ── Tooltip helper ─────────────────────────────────────────────── */
        const tip = document.getElementById('tooltip');
        function showTip(html, e) {
          tip.innerHTML = html;
          tip.classList.add('visible');
          moveTip(e);
        }
        function moveTip(e) {
          const x = e.clientX + 14, y = e.clientY - 10;
          tip.style.left = x + 'px';
          tip.style.top  = y + 'px';
        }
        function hideTip() { tip.classList.remove('visible'); }

        /* ── Number formatting ──────────────────────────────────────────── */
        const fmt1    = d3.format('.1f');
        const fmtPct  = v => fmt1(v) + '%';
        const fmtK    = d3.format(',.0f');

        /* ── Fetch data ─────────────────────────────────────────────────── */
        async function loadData() {
          const res  = await fetch(PROXY_URL);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        }

        /* ── Bootstrap panels state from localStorage ───────────────────── */
        function initVisibility(ids) {
          const stored = localStorage.getItem(STORAGE_KEY);
          if (stored) {
            const parsed = JSON.parse(stored);
            // only keep ids that actually exist in current data
            visiblePanels = new Set(ids.filter(id => parsed.includes(id)));
            // if nothing visible (e.g. first load with different data), show all
            if (visiblePanels.size === 0) visiblePanels = new Set(ids);
          } else {
            visiblePanels = new Set(ids);
          }
        }
        function saveVisibility() {
          localStorage.setItem(STORAGE_KEY, JSON.stringify([...visiblePanels]));
        }

        /* ── Build pill selector ────────────────────────────────────────── */
        function buildPills(indicators) {
          const container = document.getElementById('panel-pills');
          container.innerHTML = '';
          const ids = Object.keys(indicators);
          ids.forEach((id, i) => {
            const ind   = indicators[id];
            const color = ACCENT_POOL[i % ACCENT_POOL.length];
            const pill  = document.createElement('button');
            pill.className = 'panel-pill' + (visiblePanels.has(id) ? ' active' : '');
            pill.dataset.id = id;
            pill.innerHTML = `<span class="pill-dot" style="color:${color}"></span>${ind.title}`;
            pill.style.setProperty('--pill-color', color);
            pill.addEventListener('click', () => togglePanel(id, indicators));
            container.appendChild(pill);
          });
        }

        function togglePanel(id, indicators) {
          if (visiblePanels.has(id)) {
            if (visiblePanels.size === 1) return; // always keep at least 1
            visiblePanels.delete(id);
          } else {
            visiblePanels.add(id);
          }
          saveVisibility();
          buildPills(indicators);
          renderGrid(indicators);
        }

        /* ── Main render ────────────────────────────────────────────────── */
        function renderGrid(indicators) {
          const grid = document.getElementById('panels-grid');
          grid.innerHTML = '';
          const ids = Object.keys(indicators);

          ids.forEach((id, i) => {
            if (!visiblePanels.has(id)) return;
            const ind   = indicators[id];
            const color = ACCENT_POOL[i % ACCENT_POOL.length];
            const card  = buildCard(ind, color, i + 1);
            grid.appendChild(card);
          });

          // now draw each chart (after DOM is ready)
          ids.forEach((id, i) => {
            if (!visiblePanels.has(id)) return;
            const ind   = indicators[id];
            const color = ACCENT_POOL[i % ACCENT_POOL.length];
            renderChart(ind, color);
          });
        }

        /* ── Build card shell ───────────────────────────────────────────── */
        function buildCard(ind, color, num) {
          const card = document.createElement('div');
          card.className = 'ind-card';
          card.id = `card-${ind.id}`;
          card.style.animationDelay = `${(num-1) * 0.07}s`;

          // KPI value
          let kpiHtml = '';
          if (ind.by_lots)                  kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(ind.by_lots.percentage)}</div><div class="kpi-unit">of lots — single bidder</div>`;
          else if (ind.percentage_published) kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(ind.percentage_published)}</div><div class="kpi-unit">published on TED</div>`;
          else if (ind.combined_joint)       kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(ind.combined_joint.percentage)}</div><div class="kpi-unit">joint procurement</div>`;
          else if (ind.by_invitation_limit)  kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(ind.by_invitation_limit[0].percentage)}</div><div class="kpi-unit">open procedures</div>`;

          // optional toggle for indicator 1 (lots vs budget)
          const toggleHtml = ind.by_lots && ind.by_budget ? `
            <div class="view-toggle" id="toggle-${ind.id}">
              <button class="active" data-view="lots">By Lots</button>
              <button data-view="budget">By Budget</button>
            </div>` : '';

          card.innerHTML = `
            <div class="card-header-custom">
              <div>
                <div class="card-index">Indicator ${num}</div>
                <div class="card-title">${ind.title}</div>
                <div class="card-subtitle">${ind.subtitle}</div>
              </div>
              <div class="card-kpi">${kpiHtml}</div>
            </div>
            <div class="card-body-custom">
              ${toggleHtml}
              <div class="chart-area" id="chart-${ind.id}"></div>
            </div>`;

          // Wire toggle buttons
          if (ind.by_lots && ind.by_budget) {
            card.addEventListener('click', e => {
              const btn = e.target.closest('[data-view]');
              if (!btn) return;
              const toggle = card.querySelector('.view-toggle');
              toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
              btn.classList.add('active');
              drawSingleBidder(ind, color, btn.dataset.view);
            });
          }

          return card;
        }

        /* ── Route to the right chart renderer ─────────────────────────── */
        function renderChart(ind, color) {
          switch(ind.chart_type) {
            case 'donut_trend':  drawSingleBidder(ind, color, 'lots');  break;
            case 'hbar':         drawHBar(ind, color);                  break;
            case 'area_trend':   drawAreaTrend(ind, color);             break;
            case 'grouped_bar':  drawGroupedBar(ind, color);            break;
            default:             drawGenericBar(ind, color);            break;
          }
        }

        /* ── Chart: Donut + line (Indicator 1) ─────────────────────────── */
        function drawSingleBidder(ind, color, view) {
          const container = document.getElementById(`chart-${ind.id}`);
          container.innerHTML = '';
          const src = view === 'budget' ? ind.by_budget : ind.by_lots;

          const W = container.clientWidth || 480;
          const H = 230;

          // layout: donut on left, line on right
          const donutW = 200;
          const lineW  = W - donutW - 16;
          const cx = donutW / 2, cy = H / 2, r = 72, innerR = 46;

          const svg = d3.select(container).append('svg')
            .attr('width', W).attr('height', H);

          // ── Donut ──
          const donut = svg.append('g').attr('transform', `translate(${cx},${cy})`);
          const pie = d3.pie().value(d => d.value).sort(null);
          const arc = d3.arc().innerRadius(innerR).outerRadius(r);

          const slices = pie([
            { label: 'Single bidder', value: src.percentage },
            { label: 'Multiple bidders', value: 100 - src.percentage }
          ]);

          donut.selectAll('path')
            .data(slices).enter().append('path')
            .attr('d', arc)
            .attr('fill', (d,i) => i === 0 ? color : 'rgba(255,255,255,.07)')
            .attr('stroke', '#0b0f1a').attr('stroke-width', 2)
            .on('mouseover', (e,d) => showTip(`<strong>${d.data.label}</strong><br>${fmtPct(d.data.value)}`, e))
            .on('mousemove', moveTip)
            .on('mouseout', hideTip)
            .style('cursor','pointer');

          // centre label
          donut.append('text')
            .attr('text-anchor','middle').attr('dy','.15em')
            .attr('fill', color)
            .style('font-family','Syne,sans-serif').style('font-weight','800').style('font-size','22px')
            .text(fmtPct(src.percentage));
          donut.append('text')
            .attr('text-anchor','middle').attr('dy','1.5em')
            .attr('fill','#64748b').style('font-size','10px')
            .text(view === 'budget' ? 'by budget' : 'by lots');

          // ── Line / trend ──
          const lx = donutW + 16;
          const lGroup = svg.append('g').attr('transform', `translate(${lx}, 16)`);
          const lW = lineW, lH = H - 36;

          const xScale = d3.scaleLinear()
            .domain(d3.extent(src.trend, d => d.year))
            .range([0, lW - 8]);
          const yScale = d3.scaleLinear()
            .domain([0, d3.max(src.trend, d => d.percentage) * 1.15])
            .range([lH, 0]);

          // grid lines
          lGroup.selectAll('.gl').data(yScale.ticks(4)).enter().append('line')
            .attr('class','grid-line')
            .attr('x1',0).attr('x2', lW-8)
            .attr('y1', d => yScale(d)).attr('y2', d => yScale(d));

          // area
          const area = d3.area()
            .x(d => xScale(d.year)).y0(lH).y1(d => yScale(d.percentage))
            .curve(d3.curveCatmullRom);
          const line = d3.line()
            .x(d => xScale(d.year)).y(d => yScale(d.percentage))
            .curve(d3.curveCatmullRom);

          const gradId = `grad-${ind.id}-${view}`;
          const defs = svg.append('defs');
          const grad = defs.append('linearGradient').attr('id', gradId).attr('x1','0').attr('x2','0').attr('y1','0').attr('y2','1');
          grad.append('stop').attr('offset','0%').attr('stop-color', color).attr('stop-opacity',.25);
          grad.append('stop').attr('offset','100%').attr('stop-color', color).attr('stop-opacity',0);

          lGroup.append('path').datum(src.trend)
            .attr('fill', `url(#${gradId})`).attr('d', area);
          lGroup.append('path').datum(src.trend)
            .attr('fill','none').attr('stroke', color).attr('stroke-width',2.5).attr('d', line);

          // dots
          lGroup.selectAll('circle').data(src.trend).enter().append('circle')
            .attr('cx', d => xScale(d.year)).attr('cy', d => yScale(d.percentage))
            .attr('r', 3).attr('fill', color).attr('stroke','#0b0f1a').attr('stroke-width',1.5)
            .style('cursor','pointer')
            .on('mouseover', (e,d) => showTip(`<strong>${d.year}</strong><br>${fmtPct(d.percentage)}`, e))
            .on('mousemove', moveTip).on('mouseout', hideTip);

          // x axis labels
          lGroup.selectAll('.xl').data(src.trend.filter((_,i) => i % 2 === 0)).enter().append('text')
            .attr('class','axis-label')
            .attr('x', d => xScale(d.year)).attr('y', lH + 14)
            .attr('text-anchor','middle')
            .text(d => d.year);
        }

        /* ── Chart: Horizontal bar (Indicator 2) ───────────────────────── */
        function drawHBar(ind, color) {
          const container = document.getElementById(`chart-${ind.id}`);
          container.innerHTML = '';
          const data = ind.by_invitation_limit;
          const W = container.clientWidth || 480;
          const barH = 34, gap = 10;
          const H = data.length * (barH + gap) + 10;
          const labelW = 96;
          const chartW = W - labelW - 56;

          const xScale = d3.scaleLinear().domain([0, 100]).range([0, chartW]);

          const svg = d3.select(container).append('svg').attr('width', W).attr('height', H);
          const g   = svg.append('g').attr('transform', `translate(${labelW}, 0)`);

          data.forEach((d,i) => {
            const y = i * (barH + gap);

            // bg track
            g.append('rect').attr('x',0).attr('y',y).attr('width',chartW).attr('height',barH)
              .attr('fill','rgba(255,255,255,.04)').attr('rx',6);

            // filled bar
            g.append('rect').attr('x',0).attr('y',y)
              .attr('width', xScale(d.percentage)).attr('height',barH)
              .attr('fill', color).attr('opacity', 1 - i * 0.18).attr('rx',6)
              .style('cursor','pointer')
              .on('mouseover', e => showTip(`<strong>${d.category}</strong><br>${fmtK(d.count)} procedures — ${fmtPct(d.percentage)}`, e))
              .on('mousemove', moveTip).on('mouseout', hideTip);

            // category label
            svg.append('text').attr('class','axis-label')
              .attr('x', labelW - 8).attr('y', y + barH/2 + 4)
              .attr('text-anchor','end')
              .attr('fill','#94a3b8')
              .style('font-size','12px')
              .text(d.category);

            // pct label
            g.append('text').attr('class','axis-label')
              .attr('x', xScale(d.percentage) + 8).attr('y', y + barH/2 + 4)
              .attr('fill', color).style('font-size','12px').style('font-weight','600')
              .text(fmtPct(d.percentage));
          });
        }

        /* ── Chart: Area trend (Indicator 3) ───────────────────────────── */
        function drawAreaTrend(ind, color) {
          const container = document.getElementById(`chart-${ind.id}`);
          container.innerHTML = '';
          const data = ind.trend;
          const W = container.clientWidth || 480;
          const H = 200;
          const margin = { top:10, right:16, bottom:28, left:38 };
          const iW = W - margin.left - margin.right;
          const iH = H - margin.top - margin.bottom;

          const svg = d3.select(container).append('svg').attr('width',W).attr('height',H);
          const g   = svg.append('g').attr('transform',`translate(${margin.left},${margin.top})`);

          const xScale = d3.scalePoint().domain(data.map(d=>d.period)).range([0,iW]).padding(.1);
          const yScale = d3.scaleLinear().domain([0, Math.max(100, d3.max(data,d=>d.percentage)*1.2)]).range([iH,0]);

          // grid
          g.selectAll('.gl').data(yScale.ticks(4)).enter().append('line')
            .attr('class','grid-line')
            .attr('x1',0).attr('x2',iW)
            .attr('y1',d=>yScale(d)).attr('y2',d=>yScale(d));

          // area + line
          const gradId = `grad3-${ind.id}`;
          const defs = svg.append('defs');
          const grad = defs.append('linearGradient').attr('id',gradId).attr('x1','0').attr('x2','0').attr('y1','0').attr('y2','1');
          grad.append('stop').attr('offset','0%').attr('stop-color',color).attr('stop-opacity',.3);
          grad.append('stop').attr('offset','100%').attr('stop-color',color).attr('stop-opacity',0);

          const area = d3.area().x(d=>xScale(d.period)).y0(iH).y1(d=>yScale(d.percentage)).curve(d3.curveCatmullRom);
          const line = d3.line().x(d=>xScale(d.period)).y(d=>yScale(d.percentage)).curve(d3.curveCatmullRom);

          g.append('path').datum(data).attr('fill',`url(#${gradId})`).attr('d',area);
          g.append('path').datum(data).attr('fill','none').attr('stroke',color).attr('stroke-width',2.5).attr('d',line);

          // dots
          g.selectAll('circle').data(data).enter().append('circle')
            .attr('cx',d=>xScale(d.period)).attr('cy',d=>yScale(d.percentage))
            .attr('r',4).attr('fill',color).attr('stroke','#0b0f1a').attr('stroke-width',2)
            .style('cursor','pointer')
            .on('mouseover',(e,d)=>showTip(`<strong>${d.period}</strong><br>${fmtPct(d.percentage)} published`,e))
            .on('mousemove',moveTip).on('mouseout',hideTip);

          // axes
          const xAxis = d3.axisBottom(xScale).tickSize(0).tickPadding(8);
          const yAxis = d3.axisLeft(yScale).ticks(4).tickFormat(d=>d+'%').tickSize(0).tickPadding(6);

          g.append('g').attr('transform',`translate(0,${iH})`).call(xAxis)
            .selectAll('text').attr('class','axis-label').attr('fill','#64748b');
          g.append('g').call(yAxis)
            .selectAll('text').attr('class','axis-label').attr('fill','#64748b');
          g.selectAll('.domain').remove();

          // reference line at current value
          g.append('line')
            .attr('x1',0).attr('x2',iW)
            .attr('y1',yScale(ind.percentage_published)).attr('y2',yScale(ind.percentage_published))
            .attr('stroke',color).attr('stroke-width',1).attr('stroke-dasharray','4,3').attr('opacity',.5);
        }

        /* ── Chart: Grouped bar (Indicator 4) ──────────────────────────── */
        function drawGroupedBar(ind, color) {
          const container = document.getElementById(`chart-${ind.id}`);
          container.innerHTML = '';
          const W = container.clientWidth || 480;
          const H = 220;
          const margin = { top:10, right:16, bottom:44, left:42 };
          const iW = W - margin.left - margin.right;
          const iH = H - margin.top - margin.bottom;

          // Two groups: summary + distribution
          const summaryData = [
            { label: 'UTE Awards',       value: ind.ute_awards.percentage,         color: color },
            { label: 'Subcontracting',   value: ind.with_subcontracting.percentage, color: ACCENT_POOL[3] },
            { label: 'Combined Joint',   value: ind.combined_joint.percentage,      color: ACCENT_POOL[1] },
          ];

          const svg = d3.select(container).append('svg').attr('width',W).attr('height',H);
          const g   = svg.append('g').attr('transform',`translate(${margin.left},${margin.top})`);

          const x0 = d3.scaleBand().domain(summaryData.map(d=>d.label)).range([0,iW]).padding(.3);
          const yScale = d3.scaleLinear().domain([0,Math.ceil(d3.max(summaryData,d=>d.value)/10)*10+5]).range([iH,0]);

          // grid
          g.selectAll('.gl').data(yScale.ticks(4)).enter().append('line')
            .attr('class','grid-line').attr('x1',0).attr('x2',iW)
            .attr('y1',d=>yScale(d)).attr('y2',d=>yScale(d));

          // bars
          g.selectAll('.bar').data(summaryData).enter().append('rect')
            .attr('class','bar')
            .attr('x', d=>x0(d.label)).attr('y', d=>yScale(d.value))
            .attr('width',x0.bandwidth())
            .attr('height', d=>iH - yScale(d.value))
            .attr('fill', d=>d.color).attr('rx',5)
            .style('cursor','pointer')
            .on('mouseover',(e,d)=>showTip(`<strong>${d.label}</strong><br>${fmtPct(d.value)}`,e))
            .on('mousemove',moveTip).on('mouseout',hideTip);

          // value labels above bars
          g.selectAll('.blabel').data(summaryData).enter().append('text')
            .attr('class','axis-label')
            .attr('x', d=>x0(d.label)+x0.bandwidth()/2)
            .attr('y', d=>yScale(d.value)-5)
            .attr('text-anchor','middle')
            .attr('fill','#94a3b8').style('font-size','11px')
            .text(d=>fmtPct(d.value));

          // x axis
          const xAxis = d3.axisBottom(x0).tickSize(0).tickPadding(8);
          g.append('g').attr('transform',`translate(0,${iH})`).call(xAxis)
            .selectAll('text').attr('class','axis-label').attr('fill','#64748b')
            .style('font-size','11px');
          g.selectAll('.domain').remove();

          // avg subcontracting rate annotation
          g.append('text')
            .attr('x', iW).attr('y', 4)
            .attr('text-anchor','end')
            .attr('fill','#64748b')
            .style('font-size','10px')
            .text(`Avg. subcontracting rate: ${ind.with_subcontracting.avg_subcontracting_rate}%`);
        }

        /* ── Generic fallback bar ───────────────────────────────────────── */
        function drawGenericBar(ind, color) {
          const container = document.getElementById(`chart-${ind.id}`);
          container.innerHTML = `<div class="state-overlay" style="min-height:120px">
            <i class="bi bi-bar-chart" style="font-size:1.4rem;opacity:.4"></i>
            No chart renderer for type: ${ind.chart_type}
          </div>`;
        }

        /* ── Init ───────────────────────────────────────────────────────── */
        async function init() {
          try {
            const resp = await loadData();
            globalData = resp;

    

            const indicators = resp.indicators;
            const ids        = Object.keys(indicators);

            initVisibility(ids);
            buildPills(indicators);
            renderGrid(indicators);

          } catch(err) {
            document.getElementById('panels-grid').innerHTML = `
              <div class="state-overlay" style="grid-column:1/-1;color:#ef4444">
                <i class="bi bi-exclamation-triangle"></i>
                Failed to load data: ${err.message}
              </div>`;
          }
        }

        // Re-render on resize (debounced)
        let resizeTimer;
        window.addEventListener('resize', () => {
          clearTimeout(resizeTimer);
          resizeTimer = setTimeout(() => {
            if (globalData) renderGrid(globalData.indicators);
          }, 200);
        });

  init();
</script>

<?php include '../includes/footer.php'; ?>