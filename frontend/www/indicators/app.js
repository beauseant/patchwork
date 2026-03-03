/* ================================================================
   app.js — PLACE dashboard, multi-año
   ================================================================ */

const PROXY_URL   = 'proxy.php';
const ACCENT_POOL = ['#3b82f6','#10b981','#f59e0b','#8b5cf6',
                     '#ec4899','#06b6d4','#f97316','#84cc16'];

const DS_COLORS = {
  total:     '#b35656',
  insiders:  '#3b82f6',
  outsiders: '#10b981',
  minors:    '#f59e0b',
};
const DS_LABELS = {
  total:     'Total',
  insiders:  'Insiders',
  outsiders: 'Outsiders',
  minors:    'Minors',
};

const STORAGE_KEY  = 'dash_visible_panels_place';
const YEAR_KEY     = 'dash_selected_year';

let globalData    = null;
let visiblePanels = null;
let selectedYear  = null;   // string, e.g. "2025"

/* ── Tooltip ─────────────────────────────────────────────────────── */
const tip = document.getElementById('tooltip');
function showTip(html, e) { tip.innerHTML = html; tip.classList.add('visible'); moveTip(e); }
function moveTip(e) { tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY-10)+'px'; }
function hideTip() { tip.classList.remove('visible'); }

/* ── Formatters ──────────────────────────────────────────────────── */
const fmt1   = d3.format('.1f');
const fmtPct = v => fmt1(v) + '%';
const fmtK   = d3.format(',.0f');
const fmtM   = v => {
  if (Math.abs(v) >= 1e9) return d3.format('.2f')(v/1e9) + ' B€';
  if (Math.abs(v) >= 1e6) return d3.format('.1f')(v/1e6) + ' M€';
  return fmtK(v) + ' €';
};

/* ── Fetch ───────────────────────────────────────────────────────── */
async function loadData() {
  const res = await fetch(PROXY_URL);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/* ── Visibility persistence ──────────────────────────────────────── */
function initVisibility(ids) {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    const parsed = JSON.parse(stored);
    visiblePanels = new Set(ids.filter(id => parsed.includes(id)));
    if (visiblePanels.size === 0) visiblePanels = new Set(ids);
  } else {
    visiblePanels = new Set(ids);
  }
}
function saveVisibility() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...visiblePanels]));
}

/* ── Year selector ───────────────────────────────────────────────── */
function buildYearSelector(years) {
  const container = document.getElementById('year-selector');
  if (!container) return;
  container.innerHTML = '';

  const storedYear = localStorage.getItem(YEAR_KEY);
  // default: último año disponible
  selectedYear = years.map(String).includes(storedYear)
    ? storedYear
    : String(years[years.length - 1]);

  years.forEach(y => {
    const btn = document.createElement('button');
    btn.className = 'year-pill' + (String(y) === selectedYear ? ' active' : '');
    btn.textContent = y;
    btn.dataset.year = y;
    btn.addEventListener('click', () => {
      selectedYear = String(y);
      localStorage.setItem(YEAR_KEY, selectedYear);
      container.querySelectorAll('.year-pill').forEach(b =>
        b.classList.toggle('active', b.dataset.year === selectedYear)
      );
      renderGrid(globalData.indicators);
    });
    container.appendChild(btn);
  });
}

/* ── Panel pills ─────────────────────────────────────────────────── */
function buildPills(indicators) {
  const container = document.getElementById('panel-pills');
  container.innerHTML = '';
  Object.keys(indicators).forEach((id, i) => {
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
    if (visiblePanels.size === 1) return;
    visiblePanels.delete(id);
  } else {
    visiblePanels.add(id);
  }
  saveVisibility();
  buildPills(indicators);
  renderGrid(indicators);
}

/* ── Get data for selected year (falls back gracefully) ──────────── */
function yearData(ind) {
  return (ind.by_year && ind.by_year[selectedYear]) || {};
}

/* ── Render grid ─────────────────────────────────────────────────── */
function renderGrid(indicators) {
  const grid = document.getElementById('panels-grid');
  grid.innerHTML = '';
  const ids = Object.keys(indicators);

  ids.forEach((id, i) => {
    if (!visiblePanels.has(id)) return;
    const ind   = indicators[id];
    const color = ACCENT_POOL[i % ACCENT_POOL.length];
    grid.appendChild(buildCard(ind, color, i + 1));
  });

  ids.forEach((id, i) => {
    if (!visiblePanels.has(id)) return;
    renderChart(indicators[id], ACCENT_POOL[i % ACCENT_POOL.length]);
  });
}

/* ── Card shell ──────────────────────────────────────────────────── */
function buildCard(ind, color, num) {
  const card = document.createElement('div');
  card.className = 'ind-card';
  card.id = `card-${ind.id}`;
  card.style.animationDelay = `${(num-1)*0.07}s`;

  const yd = yearData(ind);

  let kpiHtml = '';
  if (ind.id === 'total_procurement') {
    const total = yd.total_licitaciones ?? ind.total_licitaciones;
    kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtK(total ?? 0)}</div>
               <div class="kpi-unit">total tenders</div>`;
  } else if (ind.id === 'single_bidder') {
    const pct = yd.pct_count_total ?? ind.pct_count_total;
    kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(pct ?? 0)}</div>
               <div class="kpi-unit">single bidder (total)</div>`;
  } else if (ind.id === 'ted_publication') {
    const pct = yd.pct_count_insiders ?? ind.pct_count_insiders;
    kpiHtml = `<div class="kpi-value" style="color:${color}">${fmtPct(pct ?? 0)}</div>
               <div class="kpi-unit">insiders published on TED</div>`;
  } else if (ind.id === 'decision_speed') {
    const d = yd.media_total ?? ind.media_total;
    kpiHtml = `<div class="kpi-value" style="color:${color}">${fmt1(d ?? 0)}</div>
               <div class="kpi-unit">avg. days (total)</div>`;
  }

  const hasToggle = ['single_bidder','ted_publication','total_procurement'].includes(ind.id);
  const toggleHtml = hasToggle ? `
    <div class="view-toggle" id="toggle-${ind.id}">
      <button class="active" data-view="count">By no. of tenders</button>
      <button data-view="budget">By budget</button>
    </div>` : '';

  card.innerHTML = `
    <div class="card-header-custom">
      <div>
        <div class="card-index">Indicator ${num}</div>
        <div class="card-title">${ind.title}</div>
        <div class="card-subtitle">${ind.subtitle} <span class="year-badge">${selectedYear}</span></div>
      </div>
      <div class="card-kpi">${kpiHtml}</div>
    </div>
    <div class="card-body-custom">
      ${toggleHtml}
      <div class="chart-area" id="chart-${ind.id}"></div>
    </div>`;

  if (hasToggle) {
    card.addEventListener('click', e => {
      const btn = e.target.closest('[data-view]');
      if (!btn) return;
      card.querySelector('.view-toggle').querySelectorAll('button')
        .forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderChartView(ind, color, btn.dataset.view);
    });
  }
  return card;
}

/* ── Chart routing ───────────────────────────────────────────────── */
function renderChart(ind, color) { renderChartView(ind, color, 'count'); }

function renderChartView(ind, color, view) {
  const yd = yearData(ind);
  switch (ind.chart_type) {
    case 'grouped_bar_procurement':   drawTotalProcurement(ind, yd, color, view); break;
    case 'grouped_bar_single_bidder': drawSingleBidder(ind, yd, color, view);     break;
    case 'grouped_bar_ted':           drawTED(ind, yd, color, view);              break;
    case 'grouped_bar_decision':      drawDecisionSpeed(ind, yd, color);          break;
    default:
      document.getElementById(`chart-${ind.id}`).innerHTML =
        `<div class="state-overlay">Sin renderer para: ${ind.chart_type}</div>`;
  }
}

/* ── Indicador 0 ─────────────────────────────────────────────────── */
function drawTotalProcurement(ind, yd, color, view) {
  const isCount = view === 'count';
  const data    = isCount ? yd.by_count : yd.by_budget;
  if (!data) return;
  const series = ['total','insiders','outsiders','minors'].map(k => ({
    key: k, label: DS_LABELS[k], color: DS_COLORS[k], values: data[k] ?? []
  }));
  drawGroupedBarsInto({
    targetEl:   document.getElementById(`chart-${ind.id}`),
    series,
    quarters:   ind.quarters,
    yLabel:     isCount ? 'No. of tenders' : 'Budget (ex. VAT)',
    fmtFn:      isCount ? fmtK : fmtM,
    clearFirst: true,
  });
}

/* ── Indicador 1 ─────────────────────────────────────────────────── */
function drawSingleBidder(ind, yd, color, view) {
  const isCount = view === 'count';
  const data    = isCount ? yd.by_count : yd.by_budget;
  if (!data) return;

  const container = document.getElementById(`chart-${ind.id}`);
  container.innerHTML = '';

  if (yd.cobertura) {
    const cob = yd.cobertura;
    const note = document.createElement('div');
    note.className = 'coverage-note';
    note.innerHTML = `Bidder info coverage — 
      Total: ${fmtPct(cob.total ?? 0)} &nbsp;|&nbsp;
      Insiders: ${fmtPct(cob.insiders ?? 0)} &nbsp;|&nbsp;
      Outsiders: ${fmtPct(cob.outsiders ?? 0)} &nbsp;|&nbsp;
      Minors: ${fmtPct(cob.minors ?? 0)}`;
    container.appendChild(note);
  }

  const series = ['total','insiders','outsiders','minors'].map(k => ({
    key: k, label: DS_LABELS[k], color: DS_COLORS[k], values: data[k] ?? []
  }));
  drawGroupedBarsInto({
    targetEl: container,
    series,
    quarters: ind.quarters,
    yLabel:   isCount ? '% single bidder (count)' : '% single bidder (budget)',
    fmtFn:    fmtPct,
  });
}

/* ── Indicador 3 ─────────────────────────────────────────────────── */
function drawTED(ind, yd, color, view) {
  const isCount = view === 'count';
  const data    = isCount ? yd.by_count : yd.by_budget;
  if (!data) return;
  const series = ['insiders','outsiders'].map(k => ({
    key: k, label: DS_LABELS[k], color: DS_COLORS[k], values: data[k] ?? []
  }));
  drawGroupedBarsInto({
    targetEl:   document.getElementById(`chart-${ind.id}`),
    series,
    quarters:   ind.quarters,
    yLabel:     isCount ? '% with TED (count)' : '% budget with TED',
    fmtFn:      fmtPct,
    clearFirst: true,
  });
}

/* ── Indicador 6 ─────────────────────────────────────────────────── */
function drawDecisionSpeed(ind, yd, color) {
  const container = document.getElementById(`chart-${ind.id}`);
  container.innerHTML = '';

  if (yd.media_total != null) {
    const row = document.createElement('div');
    row.className = 'decision-stats-row';
    [
      { key: 'total',    val: yd.media_total,    n: yd.n_obs?.total    },
      { key: 'insiders', val: yd.media_insiders, n: yd.n_obs?.insiders },
      { key: 'minors',   val: yd.media_minors,   n: yd.n_obs?.minors   },
    ].forEach(s => {
      if (s.val == null) return;
      row.innerHTML += `
        <div class="decision-stat">
          <span class="ds-dot" style="background:${DS_COLORS[s.key]}"></span>
          <span style="color:${DS_COLORS[s.key]};font-weight:700;font-size:1.1rem">${fmt1(s.val)}</span>
          <span class="ds-label"> days — ${DS_LABELS[s.key]}</span>
          ${s.n != null ? `<span class="ds-obs">(n=${fmtK(s.n)})</span>` : ''}
        </div>`;
    });
    container.appendChild(row);
  }

  if (!yd.media_dias) return;
  const series = ['total','insiders','minors']
    .filter(k => yd.media_dias[k])
    .map(k => ({ key: k, label: DS_LABELS[k], color: DS_COLORS[k], values: yd.media_dias[k] }));

  drawGroupedBarsInto({
    targetEl: container,
    series,
    quarters: ind.quarters,
    yLabel:   'Avg. days',
    fmtFn:    v => fmt1(v) + ' days',
  });
}

/* ── Core grouped bar renderer ───────────────────────────────────── */
function drawGroupedBarsInto({ targetEl, series, quarters, yLabel, fmtFn, clearFirst = false }) {
  if (clearFirst) targetEl.innerHTML = '';

  // legend
  const legendEl = document.createElement('div');
  legendEl.className = 'ds-legend';
  series.forEach(s => {
    legendEl.innerHTML += `<span class="ds-dot" style="background:${s.color}"></span>
                           <span class="ds-label" style="color:${s.color};font-weight: bold;">${s.label}</span>`;
  });
  targetEl.appendChild(legendEl);

  const svgWrap = document.createElement('div');
  targetEl.appendChild(svgWrap);

  const W = targetEl.clientWidth || targetEl.parentElement?.clientWidth || 520;
  const H = 220;
  const margin = { top: 14, right: 16, bottom: 36, left: 52 };
  const iW = W - margin.left - margin.right;
  const iH = H - margin.top - margin.bottom;

  const svg = d3.select(svgWrap).append('svg').attr('width', W).attr('height', H);
  const g   = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x0 = d3.scaleBand().domain(quarters).range([0, iW]).padding(0.22);
  const x1 = d3.scaleBand().domain(series.map(s => s.key)).range([0, x0.bandwidth()]).padding(0.06);

  const allVals = series.flatMap(s => s.values.filter(v => v != null));
  if (allVals.length === 0) {
    svgWrap.innerHTML = '<div class="state-overlay" style="min-height:100px">No data for this year</div>';
    return;
  }
  const yMax   = d3.max(allVals) * 1.18;
  const yScale = d3.scaleLinear().domain([0, yMax]).range([iH, 0]);

  g.selectAll('.gl').data(yScale.ticks(4)).enter().append('line')
    .attr('class','grid-line')
    .attr('x1',0).attr('x2',iW)
    .attr('y1',d=>yScale(d)).attr('y2',d=>yScale(d));

  const qGroups = g.selectAll('.qg').data(quarters).enter().append('g')
    .attr('class','qg')
    .attr('transform', d => `translate(${x0(d)},0)`);

  series.forEach(s => {
    qGroups.append('rect')
      .attr('x',      x1(s.key))
      .attr('y',      (d, i) => s.values[i] != null ? yScale(s.values[i]) : iH)
      .attr('width',  x1.bandwidth())
      .attr('height', (d, i) => s.values[i] != null ? iH - yScale(s.values[i]) : 0)
      .attr('fill',   s.color)
      .attr('rx', 3).attr('opacity', 0.85)
      .style('cursor','pointer')
      .on('mouseover', (e, d) => {
        const qi = quarters.indexOf(d);
        const v  = s.values[qi];
        showTip(`<strong>${d} — ${s.label}</strong><br>${v != null ? fmtFn(v) : 'Sin dato'}`, e);
      })
      .on('mousemove', moveTip)
      .on('mouseout',  hideTip);
  });

  g.append('g').attr('transform',`translate(0,${iH})`)
    .call(d3.axisBottom(x0).tickSize(0).tickPadding(8))
    .selectAll('text').attr('class','axis-label').attr('fill','#64748b').style('font-size','11px');
  g.selectAll('.domain').remove();

  g.append('g')
    .call(d3.axisLeft(yScale).ticks(4).tickFormat(v => {
      if (yMax >= 1e9) return d3.format('.1f')(v/1e9)+'B';
      if (yMax >= 1e6) return d3.format('.0f')(v/1e6)+'M';
      if (yMax >= 1e3) return d3.format('.0f')(v/1e3)+'k';
      return d3.format('.0f')(v);
    }).tickSize(0).tickPadding(6))
    .selectAll('text').attr('class','axis-label').attr('fill','#64748b').style('font-size','10px');
  g.selectAll('.domain').remove();

  svg.append('text')
    .attr('transform','rotate(-90)')
    .attr('x', -(H/2)).attr('y', 12)
    .attr('text-anchor','middle')
    .attr('fill','#64748b').style('font-size','10px')
    .text(yLabel);
}

/* ── Init ────────────────────────────────────────────────────────── */
async function init() {
  try {
    const resp       = await loadData();
    globalData       = resp;
    const indicators = resp.indicators;
    const years      = resp.years ?? [];
    const ids        = Object.keys(indicators);

    buildYearSelector(years);
    initVisibility(ids);
    buildPills(indicators);
    renderGrid(indicators);

  } catch (err) {
    document.getElementById('panels-grid').innerHTML = `
      <div class="state-overlay" style="grid-column:1/-1;color:#ef4444">
        <i class="bi bi-exclamation-triangle"></i>
        Failed to load data: ${err.message}
      </div>`;
  }
}

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (globalData) renderGrid(globalData.indicators);
  }, 200);
});

init();
