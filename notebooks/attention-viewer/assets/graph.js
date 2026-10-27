'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const number = value => Number(value).toLocaleString();
  const projectionName = value => ({pca: 'PCA', tsne: 't-SNE', umap: 'UMAP'})[value];
  const pretty = value => String(value).replaceAll('_', ' ');
  const canvas = $('graph-canvas'), ctx = canvas.getContext('2d');
  const state = {data: null, edges: [], neighbors: null, selected: null, page: 0, k: 10, zoom: 1, panX: 0, panY: 0, sites: new Map(), generation: 0, edgeGeneration: 0, neighborGeneration: 0};
  const palette = ['#1e1b4b', '#0f766e', '#b45309', '#be185d', '#0369a1', '#7e22ce', '#4d7c0f', '#a16207', '#57534e', '#9f1239'];
  let controller, edgeController, neighborController, clusterController, projectionController, drag, queued = false;
  let projectionGeneration = 0;
  const projections = new Map();
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  Object.assign(state, {clusters: null, metadata: null, cohort: null, clusterFilter: null, clusterGeneration: 0});
  let clusterSettings = {};
  const ageBinLabels = ['<1 month', '1–<12 months', '12–<36 months', '36–<60 months', '60+ months'];
  const ageBinColors = ['#4338ca', '#2563eb', '#14b8a6', '#84cc16', '#fbbf24'];
  function clusterName(label) { return label < 0 ? 'Noise / unassigned' : `Cluster ${label + 1}`; }
  function clusterColor(label) { return label < 0 ? '#a1a1aa' : `hsl(${(label * 137.508 + 250) % 360} 58% 43%)`; }
  function ageText(value) { return value === null || value === undefined ? 'Unknown' : `${Number(value).toFixed(2)} months (approx.)`; }
  function matchesOverlay(index) {
    const age = state.metadata?.[index]?.age_months;
    const min = $('age-min').value === '' ? -Infinity : Number($('age-min').value), max = $('age-max').value === '' ? Infinity : Number($('age-max').value);
    return (state.clusterFilter === null || state.clusters?.labels[index] === state.clusterFilter) && (age === null || age === undefined ? $('age-include-missing').checked : age >= min && age <= max);
  }
  function pointColor(index) {
    const mode = $('graph-color').value, node = state.data.nodes[index], metadata = state.metadata?.[index];
    if (mode === 'cluster') return state.clusters ? clusterColor(state.clusters.labels[index]) : '#a1a1aa';
    if (mode === 'site') return state.sites.get(node.site);
    const value = metadata?.[mode];
    if (value === null || value === undefined) return '#a1a1aa';
    if (mode === 'age_months') { const ratio = Math.min(1, value / (state.ageMax || 1)); return `hsl(${245 - 200 * ratio} 68% 46%)`; }
    return state.categories.get(value) || '#a1a1aa';
  }
  function renderLegend() {
    const target = $('graph-legend'); target.replaceChildren();
    if (!state.data) return;
    const mode = $('graph-color').value;
    let items = [];
    if (mode === 'cluster') items = state.clusters ? [...new Set(state.clusters.labels)].sort((a,b) => a-b).map(label => [clusterName(label), clusterColor(label)]) : [['Run clustering to assign groups', '#a1a1aa']];
    else if (mode === 'site') items = [...state.sites].map(([label, color]) => [pretty(label), color]);
    else if (mode === 'age_months') {
      const ramp = el('span', `0 — ${(state.ageMax || 0).toFixed(1)} months (approx.)`); ramp.prepend(el('i', undefined, 'legend-ramp')); target.append(ramp); items = [['Unknown age', '#a1a1aa']];
    } else {
      const values = [...new Set((state.metadata || []).map(row => row[mode]).filter(Boolean))].sort();
      state.categories = new Map(values.map((value, i) => [value, palette[i % palette.length]]));
      items = [...state.categories, ['Unknown', '#a1a1aa']];
    }
    for (const [label, color] of items) { const item = el('span', label), dot = el('i'); dot.style.background = color; item.prepend(dot); target.append(item); }
    state.colors = state.data.nodes.map((_, i) => pointColor(i));
    state.highlighted = state.data.nodes.map((_, i) => matchesOverlay(i));
    const shown = state.highlighted.reduce((count, value) => count + Number(value), 0);
    const known = (state.metadata || []).filter(row => row.age_months !== null).length;
    $('overlay-status').textContent = `${number(shown)} of ${number(state.data.nodes.length)} slides highlighted · ${number(known)} with recorded age${state.clusterFilter === null ? '' : ` · highlighting ${clusterName(state.clusterFilter)}`}.`;
  }
  function methodFields() {
    const method = $('cluster-method').value;
    for (const field of document.querySelectorAll('[data-method]')) { const active = field.dataset.method === method; field.hidden = !active; field.querySelectorAll('input').forEach(input => { input.disabled = !active; }); }
    $('cluster-run').disabled = !state.data || method === 'none';
  }
  function resetClusters() {
    window.dispatchEvent(new CustomEvent('graph-clusters', {detail:null}));
    clusterController?.abort(); state.clusterGeneration++; state.clusters = null; state.clusterFilter = null;
    $('cluster-total').textContent = '—'; $('cluster-summaries').replaceChildren(el('p', 'Run a clustering method to compare groups.', 'list-message')); renderLegend(); updateLayout(); drawSoon();
  }
  function clusterOptions() {
    const method = $('cluster-method').value;
    if (method === 'leiden') return {leiden_k: $('cluster-k').value, resolution: $('cluster-resolution').value};
    if (method === 'hdbscan') return {min_cluster_size: $('cluster-min-size').value, min_samples: $('cluster-min-samples').value};
    return {n_clusters: $('cluster-count').value};
  }
  async function loadClusters() {
    if (!state.data || $('cluster-method').value === 'none' || !$('cluster-controls').reportValidity()) return;
    clusterController?.abort(); clusterController = new AbortController();
    const signal = clusterController.signal, generation = ++state.clusterGeneration, cohort = state.cohort, method = $('cluster-method').value;
    window.dispatchEvent(new CustomEvent('graph-clusters', {detail:null}));
    state.clusters = null; state.clusterFilter = null; renderLegend(); updateLayout(); drawSoon();
    $('cluster-total').textContent = '…'; $('cluster-summaries').replaceChildren(el('p', 'Computing clusters…', 'list-message'));
    $('cluster-status').textContent = `Computing ${$('cluster-method').selectedOptions[0].textContent} over all slide embeddings…`;
    try {
      const result = await poll(`/api/graph/clusters?${params({method, ...clusterOptions()})}`, signal);
      if (generation !== state.clusterGeneration || cohort !== state.cohort) return;
      window.dispatchEvent(new CustomEvent('graph-clusters', {detail:{result,organ:$('graph-organ').value,branch:$('graph-branch').value}}));
      state.clusters = result; renderSummaries(); renderLegend(); updateSelectionInfo(); updateLayout(); drawSoon();
      $('cluster-status').textContent = `${number(result.n_clusters)} clusters · ${number(result.noise_count)} noise/unassigned slides · ${result.method} · ${Object.entries(result.parameters).map(([key,value]) => `${pretty(key)} ${value}`).join(' · ')}. Projection and overlay changes preserve these assignments.`;
    } catch (error) { if (error.name !== 'AbortError') { $('cluster-status').textContent = error.message; $('cluster-summaries').replaceChildren(el('p', error.message, 'list-message')); $('cluster-total').textContent = '—'; } }
  }
  function renderSummaries() {
    if (!state.clusters) return;
    const table = el('table', undefined, 'cluster-table'), head = el('thead'), tr = el('tr');
    for (const title of ['Cluster', 'Slides', 'Mapped patients', 'Patient age: median [IQR], months', 'Patient age distribution', 'Missing age / unmapped slides', 'Sites (slides)']) tr.append(el('th', title));
    head.append(tr); table.append(head); const body = el('tbody');
    for (const summary of state.clusters.summaries) {
      const row = el('tr'), identity = el('td'), button = el('button', clusterName(summary.cluster), 'qc-button'); button.type = 'button'; button.dataset.variant = 'outline'; button.setAttribute('aria-pressed', String(state.clusterFilter === summary.cluster));
      button.addEventListener('click', () => { state.clusterFilter = state.clusterFilter === summary.cluster ? null : summary.cluster; renderSummaries(); renderLegend(); drawSoon(); }); identity.append(button); row.append(identity, el('td', number(summary.slides)), el('td', number(summary.patients)));
      row.append(el('td', summary.median_age_months === null ? 'Unknown' : `${summary.median_age_months.toFixed(2)} [${summary.age_q1.toFixed(2)}–${summary.age_q3.toFixed(2)}]`));
      const age = el('td'), bar = el('span', undefined, 'age-bar'); bar.setAttribute('role', 'img'); bar.setAttribute('aria-label', summary.age_bins.map((count,i) => `${ageBinLabels[i]}: ${count}`).join('; '));
      summary.age_bins.forEach((count,i) => { const segment = el('i'); segment.style.width = `${100 * count / (summary.patients_with_age || 1)}%`; segment.style.background = ageBinColors[i]; segment.title = `${ageBinLabels[i]}: ${count} patients`; bar.append(segment); }); age.append(bar, el('div', summary.age_bins.map((count,i) => `${ageBinLabels[i]}: ${count}`).join(' · '))); row.append(age);
      row.append(el('td', `${number(summary.patients_missing_age)} patients / ${number(summary.unmapped_slides)} slides`), el('td', Object.entries(summary.sites).map(([site,count]) => `${pretty(site)} ${number(count)}`).join(' · '))); body.append(row);
    }
    table.append(body); $('cluster-summaries').replaceChildren(table); $('cluster-total').textContent = `${number(state.clusters.n_clusters)} clusters`;
  }
  function selectionDetails(index) {
    const node = state.data.nodes[index], metadata = state.metadata?.[index];
    return [['Organ', pretty(node.organ)], ['Site', pretty(node.site)], ['Case', node.case_id], ['Graph row', index], ['Cluster', state.clusters ? clusterName(state.clusters.labels[index]) : 'Not computed'], ['Age', ageText(metadata?.age_months)], ['Sex', metadata?.sex || 'Unknown']];
  }
  function updateSelectionInfo() {
    const dl = $('graph-selection').querySelector('dl'); if (!dl || state.selected === null || !state.data) return;
    dl.replaceChildren(); for (const [label, value] of selectionDetails(state.selected)) dl.append(el('dt', label), el('dd', String(value)));
  }
  function el(tag, text, className) { const item = document.createElement(tag); if (text !== undefined) item.textContent = text; if (className) item.className = className; return item; }
  function status(text, error = false) { $('graph-status').textContent = text; $('graph-status').style.color = error ? 'var(--error)' : ''; }
  function params(extra = {}) { return new URLSearchParams({organ: $('graph-organ').value, branch: $('graph-branch').value, ...extra}); }
  async function json(url, signal) { const response = await fetch(url, {signal}); const data = await response.json(); if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`); return data; }
  async function poll(url, signal, waiting) {
    for (;;) {
      const data = await json(url, signal);
      if (data.status !== 'computing') return data;
      waiting?.();
      await new Promise((resolve, reject) => {
        const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
        const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 1200);
        signal.addEventListener('abort', abort, {once: true});
        if (signal.aborted) abort();
      });
    }
  }
  function isGrouped() { return (state.viewProjection || $('graph-projection').value) === 'grouped'; }
  function updateLayout(animate = true) {
    const from = state.points;
    const grouped = isGrouped(), mode = $('graph-group').value;
    $('graph-group-field').hidden = $('group-help').hidden = !grouped;
    $('graph-edge-label').textContent = grouped ? 'Show selected links' : 'Show links';
    if (!state.data) return;
    state.layout = null; state.coords = state.data.coords;
    if (grouped) {
      const labels = state.data.nodes.map((node, i) => mode === 'cluster' ? (state.clusters ? clusterName(state.clusters.labels[i]) : 'Awaiting clustering') : mode === 'site' ? pretty(node.site) : state.metadata?.[i]?.[mode] || 'Unknown');
      state.layout = groupedSquareLayout(labels, Math.max(1, canvas.clientWidth - 70) / Math.max(1, canvas.clientHeight - 70));
      state.coords = state.layout.coords;
      $('group-help').textContent = mode === 'cluster' && !state.clusters ? 'Run clustering to arrange one square per cluster. All slides are shown together until clusters are available.' : mode !== 'cluster' && mode !== 'site' && !state.metadata ? 'Loading grouping metadata… All slides are shown in the Unknown square until metadata is available.' : `${number(state.layout.groups.length)} groups · one point per slide · small gaps between squares. Color by is independent of Group by; positions do not encode similarity.`;
      $('projection-title').textContent = `Grouped squares · ${$('graph-group').selectedOptions[0].textContent}`;
    } else {
      const variance = state.data.variance.map(v => `${(v * 100).toFixed(1)}%`).join(' / ');
      $('projection-title').textContent = state.data.projection === 'pca' ? `PCA · PC1 / PC2 (${variance})` : `${projectionName(state.data.projection)} · Dimensions 1 / 2`;
    }
    state.zoom = 1; state.panX = state.panY = 0;
    state.transition = animate && from && from.length === state.coords.length && !reducedMotion.matches ? {from, radius: state.radius || 2, started: performance.now()} : null;
    drawSoon();
  }
  function fit() { state.transition = null; state.zoom = 1; state.panX = state.panY = 0; drawSoon(); }
  async function changeProjection() {
    projectionController?.abort();
    const generation = ++projectionGeneration, projection = $('graph-projection').value;
    if (!state.data) { load(); return; }
    if (projection === 'grouped') { state.viewProjection = projection; updateLayout(); status(`${number(state.data.nodes.length)} mapped slides. Select a point to inspect its complete k-neighbor list.`); return; }
    projectionController = new AbortController();
    const signal = projectionController.signal, cohort = state.cohort;
    try {
      let data = projections.get(projection);
      if (!data) {
        status(`Preparing ${projectionName(projection)} coordinates… The current view stays available.`);
        data = await poll(`/api/graph?${params({projection})}`, signal);
      }
      if (generation !== projectionGeneration || cohort !== state.cohort) return;
      // Join by slide identity so an API ordering change cannot move the wrong slide.
      const lookup = new Map(data.nodes.map((node, i) => [node.id, data.coords[i]]));
      const coords = state.data.nodes.map(node => lookup.get(node.id));
      if (coords.some(point => !point)) throw new Error('Projection is missing slides from the current view.');
      projections.set(projection, data);
      state.data = {...data, nodes: state.data.nodes, coords};
      state.viewProjection = projection; updateLayout();
      status(`${number(data.nodes.length)} mapped slides. Select a point to inspect its complete k-neighbor list.`);
    } catch (error) {
      if (error.name !== 'AbortError') { $('graph-projection').value = state.viewProjection; status(error.message, true); }
    }
  }
  function screen(index) {
    const [x, y] = state.coords[index], b = state.bounds;
    return [(x - b.cx) * b.scale * state.zoom + canvas.clientWidth / 2 + state.panX, -(y - b.cy) * b.scale * state.zoom + canvas.clientHeight / 2 + state.panY];
  }
  function drawSoon() { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; draw(); }); }
  function draw() {
    const width = canvas.clientWidth, height = canvas.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * dpr)) canvas.width = Math.round(width * dpr);
    if (canvas.height !== Math.round(height * dpr)) canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    if (!state.data) return;
    const xs = state.coords.map(p => p[0]), ys = state.coords.map(p => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    state.bounds = {cx: (xmin + xmax) / 2, cy: (ymin + ymax) / 2, scale: Math.min((width - 70) / (xmax - xmin || 1), (height - 70) / (ymax - ymin || 1))};
    const transition = state.transition;
    const progress = transition ? Math.min(1, (performance.now() - transition.started) / 750) : 1;
    const eased = progress * progress * (3 - 2 * progress);
    state.points = state.data.nodes.map((_, index) => {
      const target = screen(index);
      if (!transition) return target;
      const from = transition.from[index];
      return [from[0] + (target[0] - from[0]) * eased, from[1] + (target[1] - from[1]) * eased];
    });
    const radius = state.layout ? state.bounds.scale * state.zoom * 0.49 : 2;
    state.radius = transition ? transition.radius + (radius - transition.radius) * eased : radius;
    if (transition && progress < 1) drawSoon();
    else { state.transition = null; if (transition) drawSoon(); }
    const line = (a, b) => { const p = state.points[a], q = state.points[b]; ctx.moveTo(...p); ctx.lineTo(...q); };
    if ($('graph-show-edges').checked && (!isGrouped() || state.selected !== null)) {
      ctx.strokeStyle = 'rgba(30,27,75,0.035)'; ctx.lineWidth = 0.6; ctx.beginPath();
      if (!isGrouped() && !transition) for (const [a, b] of state.edges) line(a, b); ctx.stroke();
      if (state.neighbors) { ctx.strokeStyle = 'rgba(30,27,75,0.3)'; ctx.lineWidth = 1; ctx.beginPath(); for (const index of state.neighbors.indices) line(state.selected, index); ctx.stroke(); }
    }
    const neighbors = new Set(state.neighbors?.indices || []);
    state.data.nodes.forEach((node, index) => {
      const [x, y] = state.points[index]; if (x < -10 || y < -10 || x > width + 10 || y > height + 10) return;
      ctx.globalAlpha = (state.selected === null || neighbors.has(index) || index === state.selected ? 0.9 : 0.2) * (state.highlighted?.[index] ? 1 : 0.08);
      ctx.fillStyle = state.colors?.[index] || '#a1a1aa'; ctx.beginPath(); ctx.arc(x, y, index === state.selected ? Math.max(5, state.radius) : state.radius, 0, Math.PI * 2); ctx.fill();
      if (index === state.selected) { ctx.globalAlpha = 1; ctx.strokeStyle = '#18181b'; ctx.lineWidth = 2; ctx.stroke(); }
    });
    ctx.globalAlpha = 1;
    if (!transition && state.layout && $('cluster-labels').checked) {
      ctx.font = '500 11px InterVariable, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const group of state.layout.groups) {
        const first = state.points[group.indices[0]], size = group.side * state.bounds.scale * state.zoom;
        const text = `${group.label} (${number(group.indices.length)})`, w = ctx.measureText(text).width + 8;
        if (size < w || size < 24) continue;
        const x = first[0] + (size - state.bounds.scale * state.zoom) / 2, y = first[1] + size / 2;
        ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.fillRect(x-w/2, y-9, w, 18); ctx.fillStyle = '#18181b'; ctx.fillText(text, x, y);
      }
    }
    if (!transition && !state.layout && state.clusters && $('cluster-labels').checked) {
      const centers = new Map();
      state.points.forEach(([x,y], i) => { const label = state.clusters.labels[i]; if (label < 0) return; if (!centers.has(label)) centers.set(label, [0,0,0]); const point = centers.get(label); point[0] += x; point[1] += y; point[2]++; });
      ctx.font = '500 12px InterVariable, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (const [label, [sx,sy,count]] of centers) { const x=sx/count,y=sy/count; if(x<0||y<0||x>width||y>height) continue; const text=`C${label+1}`, w=ctx.measureText(text).width+12; ctx.fillStyle='rgba(255,255,255,0.9)'; ctx.fillRect(x-w/2,y-10,w,20); ctx.fillStyle=clusterColor(label); ctx.fillText(text,x,y); }
    }
  }
  function hit(event) {
    if (!state.points) return null;
    const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    let best = null, distance = 81;
    state.points.forEach(([px, py], index) => { const d = (px - x) ** 2 + (py - y) ** 2; if (d < distance) { distance = d; best = index; } });
    return best;
  }
  async function load() {
    projectionController?.abort(); projectionGeneration++; projections.clear(); state.transition = null;
    state.viewProjection = $('graph-projection').value;
    const cohort = `${$('graph-organ').value}:${$('graph-branch').value}`, changed = cohort !== state.cohort;
    if (changed) { resetClusters(); state.metadata = null; state.cohort = cohort; }
    controller?.abort(); edgeController?.abort(); neighborController?.abort();
    controller = new AbortController(); const signal = controller.signal, generation = ++state.generation;
    state.edgeGeneration++; state.neighborGeneration++;
    state.data = null; state.edges = []; state.selected = null; state.neighbors = null; state.points = null;
    $('graph-loading').hidden = false; $('graph-loading').textContent = 'Preparing all slide coordinates…';
    $('graph-export').disabled = true; $('apply-k').disabled = true; $('cluster-run').disabled = true; $('graph-neighbors').replaceChildren(); $('graph-search-results').replaceChildren(); $('graph-selection').replaceChildren(el('p', 'Click a point or search for a slide.', 'list-message'));
    $('neighbor-heading').hidden = $('neighbor-pagination').hidden = true; $('graph-edge-status').textContent = 'Waiting for projection…'; drawSoon();
    try {
      const projection = isGrouped() ? 'pca' : $('graph-projection').value;
      const data = await poll(`/api/graph?${params({projection})}`, signal, () => {
        $('graph-loading').textContent = `Computing ${projectionName(projection)} for every slide. The first run can take a few minutes; results are cached.`;
      });
      if (generation !== state.generation) return;
      state.data = data; projections.set(projection, data);
      $('graph-count').textContent = `${number(data.nodes.length)} liver slides`;
      $('graph-k').max = data.nodes.length - 1;
      state.k = Math.max(1, Math.min(Number($('graph-k').value) || 10, data.nodes.length - 1)); $('graph-k').value = state.k;
      $('k-help').textContent = `Choose any integer from 1 to ${number(data.nodes.length - 1)}. Neighbors use cosine distance in the original 512-dimensional space; projections do not change links.`;
      updateLayout();
      state.sites = new Map([...new Set(data.nodes.map(n => n.site))].sort().map((site, i) => [site, palette[i % palette.length]]));
      renderLegend();
      $('graph-provenance').textContent = `Branch: ${pretty(data.provenance.branch)} · checkpoint ${data.provenance.checkpoint_sha256} · seed ${data.seed} · t-SNE perplexity ${data.perplexity} · UMAP neighbors ${data.umap_neighbors}, min distance ${data.umap_min_dist}. Coverage: ${number(data.nodes.length)} / ${number(data.catalog_count)} liver slides; ${number(data.excluded)} have no completed mapped embedding.`;
      $('graph-loading').hidden = true; $('graph-export').disabled = false; $('apply-k').disabled = false; fit();
      status(`${number(data.nodes.length)} mapped slides. Select a point to inspect its complete k-neighbor list.`);
      const n = data.nodes.length;
      $('cluster-k').max = Math.min(n - 1, Math.floor(clusterSettings.max_leiden_edges / n));
      for (const id of ['cluster-min-size','cluster-min-samples','cluster-count']) $(id).max = n;
      methodFields(); search(); loadEdges();
      if (changed && $('cluster-method').value !== 'none') loadClusters();
      if (!state.metadata) {
        const metadata = await json(`/api/graph/metadata?${params()}`, signal);
        if (generation !== state.generation) return;
        const lookup = new Map(metadata.records.map(row => [row.id,row]));
        state.metadata = data.nodes.map(node => lookup.get(node.id) || {age_months:null, sex:null, age_group:null});
        state.ageMax = Math.max(0, ...state.metadata.map(row => row.age_months || 0));
        $('age-provenance').textContent = [metadata.age_note, ...metadata.warnings].join(' ');
        renderLegend(); updateSelectionInfo(); updateLayout(); drawSoon();
      }
    } catch (error) { if (error.name !== 'AbortError') { $('graph-loading').textContent = error.message; status(error.message, true); } }
  }
  async function loadEdges() {
    edgeController?.abort(); edgeController = new AbortController(); const signal = edgeController.signal, generation = ++state.edgeGeneration;
    state.edges = []; $('graph-edge-status').textContent = `Computing graph for k = ${number(state.k)}…`; drawSoon();
    try {
      const result = await poll(`/api/graph/edges?${params({k: state.k})}`, signal);
      if (generation !== state.edgeGeneration) return;
      state.edges = result.edges;
      $('graph-edge-status').textContent = result.sampled ? `${number(result.displayed_edges)} of ${number(result.total_edges)} directed links shown (ranks sampled). Select a slide for all ${number(result.k)} neighbors.` : `${number(result.total_edges)} directed links · k = ${number(result.k)}`;
      drawSoon();
    } catch (error) { if (error.name !== 'AbortError') { $('graph-edge-status').textContent = error.message; status(error.message, true); } }
  }
  async function select(index) {
    if (!state.data) return;
    neighborController?.abort(); neighborController = new AbortController(); const signal = neighborController.signal, generation = ++state.neighborGeneration;
    state.selected = index; state.neighbors = null; state.page = 0; drawSoon();
    const node = state.data.nodes[index];
    $('graph-selection').replaceChildren(el('h3', node.slide_id)); const dl = el('dl');
    for (const [label, value] of selectionDetails(index)) dl.append(el('dt', label), el('dd', String(value)));
    const link = el('a', 'Open slide attention', 'qc-button'); link.href = `/#${new URLSearchParams({slide: node.id, branch: $('graph-branch').value})}`;
    $('graph-selection').append(dl, link);
    $('neighbor-heading').hidden = false; $('neighbor-pagination').hidden = true;
    $('neighbor-count').textContent = '…'; $('graph-neighbors').replaceChildren(el('p', 'Finding nearest neighbors…', 'list-message'));
    try {
      const data = await json(`/api/graph/neighbors/${encodeURIComponent(node.id)}?${params({k: state.k})}`, signal);
      if (generation !== state.neighborGeneration) return;
      state.neighbors = data; renderNeighbors(); drawSoon();
    } catch (error) { if (error.name !== 'AbortError') { $('graph-neighbors').replaceChildren(el('p', error.message, 'list-message')); status(error.message, true); } }
  }
  function resultButton(index, subtitle) {
    const node = state.data.nodes[index], button = el('button', undefined, 'graph-result'); button.type = 'button';
    button.append(el('strong', node.slide_id), el('span', subtitle)); button.addEventListener('click', () => select(index)); return button;
  }
  function renderNeighbors() {
    const data = state.neighbors, start = state.page * 25, end = Math.min(start + 25, data.indices.length);
    $('neighbor-count').textContent = number(data.k); $('graph-neighbors').replaceChildren();
    for (let i = start; i < end; i++) { const node = state.data.nodes[data.indices[i]]; $('graph-neighbors').append(resultButton(data.indices[i], `#${i + 1} · cosine distance ${data.distances[i].toFixed(5)} · ${pretty(node.site)} · ${node.case_id}`)); }
    $('neighbor-pagination').hidden = false; $('neighbor-prev').disabled = start === 0; $('neighbor-next').disabled = end === data.indices.length; $('neighbor-page').textContent = `${number(start + 1)}–${number(end)}`;
  }
  function search() {
    $('graph-search-results').replaceChildren(); if (!state.data) return;
    const query = $('graph-search').value.trim().toLowerCase(); if (!query) return;
    const matches = [];
    state.data.nodes.forEach((node, i) => { if (`${node.slide_id} ${node.case_id}`.toLowerCase().includes(query)) matches.push(i); });
    $('graph-search-results').append(el('p', `${number(matches.length)} matches${matches.length > 20 ? ' · first 20 shown; refine search' : ''}`, 'graph-help'));
    for (const index of matches.slice(0, 20)) { const node = state.data.nodes[index]; $('graph-search-results').append(resultButton(index, `${pretty(node.site)} · ${node.case_id}`)); }
  }
  function download() {
    const data = state.data, header = ['id', 'slide_id', 'organ', 'site', 'case_id', 'projection', 'x', 'y', 'branch', 'checkpoint_sha256', 'selected_neighbor_rank', 'cosine_distance', 'cluster_method', 'cluster_parameters', 'cluster_seed', 'cluster_label', 'age_months_approx', 'sex', 'age_group', 'highlighted', 'group_by', 'group'];
    const ranks = new Map((state.neighbors?.indices || []).map((index, i) => [index, i]));
    const groupLabels = [];
    for (const group of state.layout?.groups || []) for (const index of group.indices) groupLabels[index] = group.label;
    const rows = data.nodes.map((node, i) => { const rank = ranks.get(i), metadata = state.metadata?.[i]; return [node.id, node.slide_id, node.organ, node.site, node.case_id, isGrouped() ? 'grouped' : data.projection, ...state.coords[i], data.provenance.branch, data.provenance.checkpoint_sha256, rank === undefined ? '' : rank + 1, rank === undefined ? '' : state.neighbors.distances[rank], state.clusters?.method || '', state.clusters ? JSON.stringify(state.clusters.parameters) : '', state.clusters?.seed ?? '', state.clusters?.labels[i] ?? '', metadata?.age_months ?? '', metadata?.sex || '', metadata?.age_group || '', matchesOverlay(i), isGrouped() ? $('graph-group').value : '', groupLabels[i] ?? '']; });
    const csv = [header, ...rows].map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'})); const a = el('a'); a.href = url; a.download = `liver-${data.provenance.branch}-${isGrouped() ? 'grouped' : data.projection}-mapping.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  window.addEventListener('graph-highlight-cluster', event => { if (!state.clusters) return; state.clusterFilter = state.clusterFilter === event.detail ? null : event.detail; renderSummaries(); renderLegend(); drawSoon(); });
  $('graph-controls').addEventListener('submit', event => { event.preventDefault(); if (!state.data || !$('graph-controls').reportValidity()) return; state.k = Number($('graph-k').value); loadEdges(); if (state.selected !== null) select(state.selected); });
  $('graph-projection').addEventListener('change', changeProjection);
  $('graph-group').addEventListener('change', () => updateLayout());
  $('graph-branch').addEventListener('change', load);
  $('graph-search').addEventListener('input', search);
  $('graph-fit').addEventListener('click', fit);
  $('graph-show-edges').addEventListener('change', drawSoon);
  $('graph-export').addEventListener('click', download);
  $('cluster-controls').addEventListener('submit', event => { event.preventDefault(); loadClusters(); });
  $('cluster-method').addEventListener('change', () => { resetClusters(); methodFields(); $('cluster-status').textContent = $('cluster-method').value === 'none' ? 'Clustering disabled.' : 'Adjust the parameters and select Run clustering.'; });
  for (const input of document.querySelectorAll('#cluster-controls input')) input.addEventListener('input', () => { resetClusters(); $('cluster-status').textContent = 'Parameters changed. Select Run clustering to apply.'; });
  for (const id of ['graph-color','age-min','age-max','age-include-missing','cluster-labels']) $(id).addEventListener('input', () => { renderLegend(); drawSoon(); });
  $('overlay-reset').addEventListener('click', () => { $('age-min').value = $('age-max').value = ''; $('age-include-missing').checked = true; state.clusterFilter = null; renderSummaries(); renderLegend(); drawSoon(); });
  $('neighbor-prev').addEventListener('click', () => { state.page--; renderNeighbors(); });
  $('neighbor-next').addEventListener('click', () => { state.page++; renderNeighbors(); });
  canvas.addEventListener('wheel', event => {
    event.preventDefault(); const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left - canvas.clientWidth / 2, y = event.clientY - rect.top - canvas.clientHeight / 2;
    const next = Math.max(0.25, Math.min(100, state.zoom * Math.exp(-event.deltaY * 0.001))), ratio = next / state.zoom;
    state.panX = x - (x - state.panX) * ratio; state.panY = y - (y - state.panY) * ratio; state.zoom = next; drawSoon();
  }, {passive: false});
  canvas.addEventListener('pointerdown', event => { drag = {x: event.clientX, y: event.clientY, px: state.panX, py: state.panY, moved: false}; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => {
    if (drag) { const dx = event.clientX - drag.x, dy = event.clientY - drag.y; drag.moved ||= Math.hypot(dx, dy) > 4; if (drag.moved) { state.panX = drag.px + dx; state.panY = drag.py + dy; drawSoon(); } $('graph-tooltip').hidden = true; return; }
    const index = hit(event), tooltip = $('graph-tooltip'); tooltip.hidden = index === null;
    if (index !== null) { const node = state.data.nodes[index]; tooltip.textContent = `${node.slide_id} · ${pretty(node.site)} · Case ${node.case_id} · ${ageText(state.metadata?.[index]?.age_months)}${state.clusters ? ` · ${clusterName(state.clusters.labels[index])}` : ''}`; const rect = canvas.getBoundingClientRect(); tooltip.style.left = `${Math.max(8, Math.min(event.clientX - rect.left + 12, rect.width - tooltip.offsetWidth - 8))}px`; tooltip.style.top = `${Math.max(8, Math.min(event.clientY - rect.top + 12, rect.height - tooltip.offsetHeight - 8))}px`; }
  });
  canvas.addEventListener('pointerup', event => { if (drag && !drag.moved) { const index = hit(event); if (index !== null) select(index); } drag = null; });
  canvas.addEventListener('pointercancel', () => { drag = null; });
  canvas.addEventListener('pointerleave', () => { $('graph-tooltip').hidden = true; });
  canvas.addEventListener('keydown', event => { if (event.key === 'Home') fit(); });
  new ResizeObserver(() => { state.transition = null; if (isGrouped() && state.data) updateLayout(false); else drawSoon(); }).observe(canvas);
  async function initialize() {
    try {
      const overview = await json('/api/graph/catalog');
      clusterSettings = await json('/api/graph/clustering');
      $('cluster-method').value = clusterSettings.default_method;
      for (const [id,key] of [['cluster-k','leiden_k'],['cluster-resolution','resolution'],['cluster-min-size','min_cluster_size'],['cluster-min-samples','min_samples'],['cluster-count','n_clusters']]) $(id).value = clusterSettings[key];
      for (const organ of overview.organs) $('graph-organ').add(new Option(`${pretty(organ.organ)} (${number(organ.count)})`, organ.organ));
      $('graph-organ').value = overview.default_organ;
      function branches() { const organ = overview.organs.find(item => item.organ === $('graph-organ').value); $('graph-branch').replaceChildren(); organ.branches.forEach((branch, i) => $('graph-branch').add(new Option(pretty(branch), i))); $('graph-branch').disabled = false; load(); }
      $('graph-organ').disabled = overview.organs.length === 1; $('graph-k').value = overview.default_k; $('graph-organ').addEventListener('change', branches); branches();
    } catch (error) { $('graph-loading').textContent = error.message; status(error.message, true); }
  }
  initialize();
})();
