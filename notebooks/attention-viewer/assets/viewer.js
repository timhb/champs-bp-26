'use strict';
const $ = id => document.getElementById(id);
const number = value => Number(value).toLocaleString();
const pretty = value => value.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
const state = {key: null, detail: null, heat: null, branch: 0, offset: 0, total: 0, rows: [], request: 0, branchRequest: 0, selectedTile: null};
let slideAbort, branchAbort, searchAbort, searchTimer, drawing = false, grid = new Map();
const canvas = $('attention-canvas');
const context = canvas.getContext('2d');
const MAX_MAG = 200;
const colors = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
const colorTable = Array.from({length: 256}, (_, i) => {
  const value = i / 255 * 4, index = Math.min(3, Math.floor(value)), t = value - index;
  const rgb = colors[index].map((v, c) => Math.round(v + (colors[index + 1][c] - v) * t));
  return `rgb(${rgb.join(',')})`;
});
const viewer = OpenSeadragon({
  id: 'slide-viewer', showNavigationControl: false, showNavigator: true, navigatorPosition: 'TOP_LEFT',
  visibilityRatio: 0.5, constrainDuringPan: true, minZoomImageRatio: 0.7,
  maxZoomPixelRatio: MAX_MAG / 40, animationTime: 0.4, blendTime: 0.1,
  timeout: 60000, imageLoaderLimit: 8,
  gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true},
});
function status(message, error = false) {
  $('app-status').textContent = message;
  $('app-status').dataset.error = String(error);
}
function message(title, description, loading = false, error = false) {
  const box = $('viewer-message');
  box.replaceChildren(); box.hidden = false; box.classList.toggle('error', error);
  if (loading) { const spinner = document.createElement('span'); spinner.className = 'spinner'; spinner.setAttribute('aria-hidden', 'true'); box.append(spinner); }
  const heading = document.createElement('strong'), text = document.createElement('span');
  heading.textContent = title; text.textContent = description; box.append(heading, text);
  if (error && state.key) {
    const retry = document.createElement('button'); retry.className = 'qc-button'; retry.textContent = 'Try again';
    retry.addEventListener('click', () => loadSlide(state.key)); box.append(retry);
  }
}
async function fetchJSON(url, signal) {
  const response = await fetch(url, {signal});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function node(tag, className, text) {
  const element = document.createElement(tag); if (className) element.className = className;
  if (text !== undefined) element.textContent = text; return element;
}
async function searchSlides(autoSelect = false) {
  searchAbort?.abort(); searchAbort = new AbortController();
  const signal = searchAbort.signal;
  $('slide-list').setAttribute('aria-busy', 'true');
  const params = new URLSearchParams({q: $('search-input').value, organ: $('organ-filter').value, site: $('site-filter').value, condition: $('condition-filter').value, offset: state.offset});
  try {
    const data = await fetchJSON(`/api/slides?${params}`, signal);
    state.rows = data.slides; state.total = data.total;
    const selectedCondition = $('condition-filter').value;
    $('condition-filter').replaceChildren(new Option('All conditions', ''));
    for (const condition of data.conditions) $('condition-filter').add(new Option(`${pretty(condition.value)} (${number(condition.count)})`, condition.value));
    if (selectedCondition && !data.conditions.some(condition => condition.value === selectedCondition)) $('condition-filter').add(new Option(`${pretty(selectedCondition)} (0)`, selectedCondition));
    $('condition-filter').value = selectedCondition;
    $('condition-filter').disabled = !data.conditions.length && !selectedCondition;
    if ($('site-filter').options.length === 1) {
      for (const site of data.sites) { const option = new Option(pretty(site), site); $('site-filter').add(option); }
    }
    $('catalog-count').textContent = number(data.total);
    $('slide-list').replaceChildren();
    if (!data.slides.length) $('slide-list').append(node('p', 'list-message', 'No matching slides. Try a different search or filter.'));
    for (const slide of data.slides) {
      const button = node('button', 'slide-option'); button.type = 'button'; button.dataset.key = slide.id;
      button.setAttribute('aria-current', String(slide.id === state.key));
      button.append(node('strong', '', slide.slide_id), node('span', '', `${pretty(slide.site)} · ${pretty(slide.organ)}`));
      button.addEventListener('click', () => loadSlide(slide.id)); $('slide-list').append(button);
    }
    $('page-label').textContent = data.total ? `${number(state.offset + 1)}–${number(state.offset + data.slides.length)}` : '0 results';
    $('previous-page').disabled = state.offset === 0;
    $('next-page').disabled = state.offset + 40 >= data.total;
    if (autoSelect && data.slides.length) {
      const fragment = new URLSearchParams(location.hash.slice(1));
      const key = !state.key ? fragment.get('slide') || data.slides[0].id : data.slides.some(slide => slide.id === state.key) ? state.key : data.slides[0].id;
      const branch = !state.key ? Number(fragment.get('branch') || 0) : 0;
      if (key !== state.key) loadSlide(key, Number.isInteger(branch) ? branch : 0);
    } else if (!state.detail && !state.key) status(`${number(data.total_available)} slides available. Select a slide to begin.`);
  } catch (error) {
    if (error.name === 'AbortError') return;
    $('slide-list').replaceChildren(node('p', 'list-message', error.message)); status(error.message, true);
  } finally { if (!signal.aborted) $('slide-list').setAttribute('aria-busy', 'false'); }
}
function markSlide() {
  for (const button of document.querySelectorAll('.slide-option')) button.setAttribute('aria-current', String(button.dataset.key === state.key));
}
async function loadSlide(key, initialBranch = 0) {
  slideAbort?.abort(); branchAbort?.abort(); slideAbort = new AbortController();
  const signal = slideAbort.signal, requestID = ++state.request;
  window.casePanel.load(key);
  state.branchRequest++; state.key = key; state.detail = null; state.heat = null; state.selectedTile = null; grid.clear();
  viewer.close(); queueDraw(); markSlide();
  $('magnification-controls').hidden = true; $('scale-text').textContent = '—'; $('scale-bar').style.width = '0';
  $('reference-count').textContent = '—';
  $('reference-findings').replaceChildren(node('p', 'list-message', 'Loading reference findings…'));
  $('export-button').disabled = true; $('branch-count').textContent = '—'; $('branch-select').disabled = true;
  $('branch-list').replaceChildren(node('p', 'list-message', 'Computing complete-slide attention…'));
  $('top-tiles').replaceChildren(node('p', 'list-message', 'Top tiles will appear after loading.'));
  $('tile-tooltip').hidden = true; $('tile-stat').textContent = 'Loading tiles…';
  const row = state.rows.find(item => item.id === key);
  $('slide-title').textContent = row?.slide_id || 'Loading slide…';
  $('slide-subtitle').textContent = row ? `${pretty(row.site)} · ${pretty(row.organ)}` : '';
  $('active-branch-label').textContent = 'Loading attention…';
  message('Loading slide attention', 'Reading Virchow2 features and scoring all finding branches on CPU.', true);
  status('Computing attention over the complete slide bag. The first load may take a moment.');
  try {
    const detail = await fetchJSON(`/api/slides/${encodeURIComponent(key)}`, signal);
    if (requestID !== state.request) return;
    state.detail = detail;
    $('slide-title').textContent = detail.slide_id;
    $('organ-label').textContent = `${detail.organ.toUpperCase()} · H&E`;
    $('slide-subtitle').textContent = `${pretty(detail.site)} · Case ${detail.case_id}`;
    setMagnificationControls();
    renderReferences(detail.reference);
    $('tile-stat').textContent = `${number(detail.n_tiles)} tiles · ${number(detail.dimensions[0])} × ${number(detail.dimensions[1])} px`;
    $('location-stat').textContent = 'Full slide · Level-0 coordinates';
    $('branch-count').textContent = detail.branches.length;
    renderDetails(detail);
    viewer.open(`/dzi/${encodeURIComponent(key)}.dzi`);
    await selectBranch(Math.max(0, Math.min(detail.branches.length - 1, initialBranch)));
  } catch (error) {
    if (error.name === 'AbortError' || requestID !== state.request) return;
    message('Could not load this slide', error.message, false, true); status(error.message, true);
    $('branch-list').replaceChildren(node('p', 'list-message', 'Select another slide or try again.'));
  }
}
function renderReferences(reference) {
  const target = $('reference-findings'); target.replaceChildren();
  $('reference-count').textContent = reference.findings.length;
  target.append(node('p', 'reference-source', reference.source + (reference.split ? ` · ${pretty(reference.split)}` : '')));
  if (reference.findings.length) {
    const chips = node('div', 'reference-chips');
    for (const label of reference.findings) chips.append(node('span', 'reference-chip', pretty(label)));
    target.append(chips);
  } else {
    const known = Object.values(reference.states).some(value => value === 'negative');
    target.append(node('p', 'list-message', known ? 'No positive reference findings.' : 'No definite reference findings available for this WSI.'));
  }
  if (reference.registry_findings.length) {
    const details = document.createElement('details'), summary = node('summary', '', 'Registry annotations');
    const chips = node('div', 'reference-chips');
    for (const label of reference.registry_findings) chips.append(node('span', 'reference-chip secondary', pretty(label)));
    details.className = 'registry-findings'; details.append(summary, chips); target.append(details);
  }
  target.append(node('p', 'reference-note', reference.note));
}
function setMagnificationControls() {
  const target = $('magnification-controls'); target.replaceChildren();
  const objective = state.detail.objective;
  viewer.maxZoomPixelRatio = objective ? MAX_MAG / objective : 5;
  for (const magnification of [null, 1.25, 5, 10, 20, 40, 100, MAX_MAG]) {
    const button = node('button', '', magnification === null ? 'Fit' : `${magnification}×`);
    button.type = 'button'; button.dataset.mag = magnification === null ? 'fit' : String(magnification);
    button.setAttribute('aria-label', magnification === null ? 'Fit slide to view' : `Zoom to ${magnification}×`);
    button.setAttribute('aria-pressed', 'false');
    button.disabled = magnification !== null && !objective;
    button.addEventListener('click', () => {
      const item = viewer.world.getItemAt(0); if (!item) return;
      state.selectedTile = null;
      if (magnification === null) viewer.viewport.goHome();
      else viewer.viewport.zoomTo(item.imageToViewportZoom(1) * magnification / objective);
      viewer.viewport.applyConstraints(); queueDraw();
    });
    target.append(button);
  }
}
function updateScale() {
  const item = viewer.world.getItemAt(0); if (!item || !state.detail) return;
  const p0 = item.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, 0));
  const p1 = item.viewerElementToImageCoordinates(new OpenSeadragon.Point(100, 0));
  const imagePixelsPerScreenPixel = (p1.x - p0.x) / 100;
  if (!(imagePixelsPerScreenPixel > 0)) return;
  const magnification = state.detail.objective ? state.detail.objective / imagePixelsPerScreenPixel : null;
  const parts = [];
  const mpp = state.detail.mpp_x;
  if (mpp) {
    const micronsPerPixel = imagePixelsPerScreenPixel * mpp;
    const target = micronsPerPixel * 120;
    const nice = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000].find(n => n >= target * 0.5) || 50000;
    $('scale-bar').style.width = `${nice / micronsPerPixel}px`;
    parts.push(nice >= 1000 ? `${nice / 1000} mm` : `${nice} µm`);
  } else $('scale-bar').style.width = '0';
  parts.push(magnification ? `~${magnification.toFixed(1)}×` : 'Magnification unavailable');
  $('scale-text').textContent = parts.join(' · ');
  const atHome = Math.abs(viewer.viewport.getZoom() / viewer.viewport.getHomeZoom() - 1) < 0.01;
  for (const button of $('magnification-controls').children) {
    const selected = button.dataset.mag === 'fit' ? atHome : magnification && Math.abs(magnification / Number(button.dataset.mag) - 1) < 0.02;
    button.setAttribute('aria-pressed', String(Boolean(selected)));
  }
}
function renderDetails(detail) {
  const target = $('model-details'); target.replaceChildren();
  for (const [label, value] of Object.entries({Encoder: detail.encoder, Aggregator: 'CLAM-MB', 'Patch size': '224 px at 20×', Footprint: `${detail.patch_size} px at level 0`, Tiles: number(detail.n_tiles), Epoch: detail.epoch, Checkpoint: detail.checkpoint})) {
    target.append(node('dt', '', label), node('dd', '', value));
  }
}
function renderBranches() {
  const preserveFocus = document.activeElement?.classList.contains('branch-option');
  const target = $('branch-list'); target.replaceChildren();
  $('branch-select').replaceChildren(); $('branch-select').disabled = false;
  for (const branch of state.detail.branches) {
    const button = node('button', 'branch-option'); button.type = 'button'; button.dataset.branch = branch.id;
    $('branch-select').add(new Option(pretty(branch.label), String(branch.id), false, branch.id === state.branch));
    button.setAttribute('aria-pressed', String(branch.id === state.branch));
    const line = node('div', 'branch-line');
    const metrics = node('span', 'branch-metrics');
    const reference = node('span', 'reference-badge', {positive: 'Label 1', negative: 'Label 0', masked: 'Masked', conflicting: 'Conflict', unknown: 'Unknown'}[branch.reference] || 'Unknown');
    reference.dataset.state = branch.reference || 'unknown';
    reference.title = branch.reference === 'positive' ? 'Positive report-derived reference label' : branch.reference === 'negative' ? 'Negative target in the champion reference dataset' : 'No definite reference label for this branch';
    metrics.append(reference, node('span', '', `${(branch.score * 100).toFixed(1)}%`));
    line.append(node('strong', '', pretty(branch.label)), metrics);
    const track = node('span', 'score-track'), fill = node('span', 'score-fill');
    fill.style.width = `${branch.score * 100}%`; track.append(fill); button.append(line, track);
    button.addEventListener('click', () => selectBranch(branch.id)); target.append(button);
  }
  if (preserveFocus) target.children[state.branch]?.focus({preventScroll: true});
}
async function selectBranch(branch) {
  if (!state.detail) return;
  branchAbort?.abort(); branchAbort = new AbortController();
  const signal = branchAbort.signal, requestID = ++state.branchRequest, key = state.key;
  state.branch = branch; state.heat = null; state.selectedTile = null; grid.clear(); queueDraw(); renderBranches();
  const finding = state.detail.branches[branch];
  $('active-branch-label').textContent = pretty(finding.label);
  $('export-button').disabled = true; $('tile-tooltip').hidden = true;
  $('top-tiles').replaceChildren(node('p', 'list-message', 'Loading branch attention…'));
  status(`Loading ${pretty(finding.label).toLowerCase()} attention…`);
  try {
    const heat = await fetchJSON(`/api/slides/${encodeURIComponent(key)}/attention/${branch}`, signal);
    if (requestID !== state.branchRequest || key !== state.key) return;
    state.heat = heat; buildGrid(); renderTopTiles(); queueDraw();
    $('export-button').disabled = false;
    history.replaceState(null, '', `#${new URLSearchParams({slide: key, branch})}`);
    if (viewer.world.getItemCount()) $('viewer-message').hidden = true;
    status(`${number(state.detail.n_tiles)} tiles · ${state.detail.branches.length} finding branches · Attention normalized over the full slide.`);
  } catch (error) {
    if (error.name === 'AbortError' || requestID !== state.branchRequest) return;
    message('Could not load branch attention', error.message, false, true); status(error.message, true);
  }
}
function buildGrid() {
  grid.clear(); const size = state.heat.patch_size;
  state.heat.coords.forEach(([x, y], index) => {
    for (let gx = Math.floor(x / size); gx <= Math.floor((x + size - 1) / size); gx++) {
      for (let gy = Math.floor(y / size); gy <= Math.floor((y + size - 1) / size); gy++) {
        const key = `${gx},${gy}`; if (!grid.has(key)) grid.set(key, []); grid.get(key).push(index);
      }
    }
  });
}
function renderTopTiles() {
  const target = $('top-tiles'); target.replaceChildren();
  state.heat.top_tiles.forEach((index, rank) => {
    const button = node('button', 'patch-option'); button.type = 'button';
    button.setAttribute('aria-label', `Locate attention tile ${rank + 1}, percentile ${(state.heat.percentiles[index] * 100).toFixed(1)}`);
    const image = document.createElement('img'); image.alt = `H&E tile ranked ${rank + 1}`;
    image.src = `/api/slides/${encodeURIComponent(state.key)}/patch/${index}.jpg`; image.loading = 'lazy';
    const caption = node('div', 'patch-info');
    caption.append(node('strong', '', `${(state.heat.percentiles[index] * 100).toFixed(1)}%ile`), node('span', '', state.heat.weights[index].toExponential(1)));
    button.append(image, node('span', 'patch-rank', String(rank + 1)), caption);
    button.addEventListener('click', () => locateTile(index)); target.append(button);
  });
}
function locateTile(index) {
  const item = viewer.world.getItemAt(0); if (!item || !state.heat) return;
  const [x, y] = state.heat.coords[index], size = state.heat.patch_size;
  state.selectedTile = index;
  viewer.viewport.fitBounds(item.imageToViewportRectangle(x - 2 * size, y - 2 * size, size * 5, size * 5)); queueDraw();
}
function queueDraw() {
  if (drawing) return; drawing = true;
  requestAnimationFrame(() => { drawing = false; draw(); });
}
function draw() {
  const box = $('slide-viewer').getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.round(box.width * dpr), height = Math.round(box.height * dpr);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; canvas.style.width = `${box.width}px`; canvas.style.height = `${box.height}px`; }
  context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, box.width, box.height);
  const item = viewer.world.getItemAt(0);
  if (!item || !state.heat || !$('overlay-visible').checked) return;
  const origin = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
  const edge = item.imageToViewerElementCoordinates(new OpenSeadragon.Point(state.heat.patch_size, 0));
  const tileSize = edge.x - origin.x, scale = tileSize / state.heat.patch_size;
  const cutoff = Number($('threshold').value) / 100;
  context.globalAlpha = Number($('opacity').value) / 100;
  state.heat.coords.forEach(([x, y], index) => {
    const rank = state.heat.percentiles[index]; if (rank < cutoff) return;
    const left = origin.x + x * scale, top = origin.y + y * scale;
    if (left + tileSize < 0 || top + tileSize < 0 || left > box.width || top > box.height) return;
    context.fillStyle = colorTable[Math.round(rank * 255)]; context.fillRect(left, top, tileSize, tileSize);
  });
  context.globalAlpha = 1;
  if (state.selectedTile !== null) {
    const [x, y] = state.heat.coords[state.selectedTile];
    context.strokeStyle = 'white'; context.lineWidth = 3;
    context.strokeRect(origin.x + x * scale, origin.y + y * scale, tileSize, tileSize);
  }
}
viewer.addHandler('open', () => { if (state.heat) $('viewer-message').hidden = true; $('magnification-controls').hidden = false; updateScale(); queueDraw(); });
viewer.addHandler('open-failed', event => { message('Could not open the slide image', event.message || 'The slide image could not be read.', false, true); status('Slide image failed to load.', true); });
for (const event of ['animation', 'resize', 'viewport-change']) viewer.addHandler(event, () => { queueDraw(); updateScale(); });
new ResizeObserver(queueDraw).observe($('slide-viewer'));
new OpenSeadragon.MouseTracker({element: viewer.canvas, moveHandler(event) {
  const item = viewer.world.getItemAt(0), tooltip = $('tile-tooltip');
  if (!item || !state.heat || !$('overlay-visible').checked) { tooltip.hidden = true; return; }
  const point = item.viewerElementToImageCoordinates(event.position), size = state.heat.patch_size;
  const candidates = grid.get(`${Math.floor(point.x / size)},${Math.floor(point.y / size)}`) || [];
  const index = candidates.find(i => { const [x,y] = state.heat.coords[i]; return point.x >= x && point.x < x + size && point.y >= y && point.y < y + size && state.heat.percentiles[i] * 100 >= Number($('threshold').value); });
  if (index === undefined) { tooltip.hidden = true; return; }
  const [x,y] = state.heat.coords[index];
  tooltip.textContent = `${(state.heat.percentiles[index] * 100).toFixed(1)}th percentile · Tile ${index}\nAttention ${state.heat.weights[index].toExponential(3)} · (${x}, ${y})`;
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(8, Math.min(event.position.x + 14, viewer.container.clientWidth - tooltip.offsetWidth - 8))}px`;
  tooltip.style.top = `${Math.max(8, Math.min(event.position.y + 14, viewer.container.clientHeight - tooltip.offsetHeight - 8))}px`;
}, exitHandler() { $('tile-tooltip').hidden = true; }}).setTracking(true);
$('search-input').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.offset = 0; searchSlides(); }, 220); });
for (const id of ['organ-filter', 'site-filter']) $(id).addEventListener('change', () => { if (id === 'organ-filter') $('condition-filter').value = ''; state.offset = 0; searchSlides(); });
$('condition-filter').addEventListener('change', () => { state.offset = 0; searchSlides(true); });
$('previous-page').addEventListener('click', () => { state.offset = Math.max(0, state.offset - 40); searchSlides(); });
$('next-page').addEventListener('click', () => { state.offset += 40; searchSlides(); });
$('branch-select').addEventListener('change', () => selectBranch(Number($('branch-select').value)));
$('opacity').addEventListener('input', () => { $('opacity-value').textContent = `${$('opacity').value}%`; queueDraw(); });
$('threshold').addEventListener('input', () => { $('threshold-value').textContent = `${$('threshold').value}th`; $('tile-tooltip').hidden = true; queueDraw(); });
$('overlay-visible').addEventListener('change', () => { $('tile-tooltip').hidden = true; queueDraw(); });
$('export-button').addEventListener('click', () => { if (state.heat) location.href = `/api/slides/${encodeURIComponent(state.key)}/export/${state.branch}.csv`; });
document.addEventListener('keydown', event => {
  if (event.target.closest('input, select, textarea, button, summary') || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key.toLowerCase() === 'h') { $('overlay-visible').checked = !$('overlay-visible').checked; $('tile-tooltip').hidden = true; queueDraw(); }
});
document.addEventListener('attention-slide', event => { if (event.detail !== state.key) loadSlide(event.detail); });
searchSlides(true);
