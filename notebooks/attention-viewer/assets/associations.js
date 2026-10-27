'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const names = {age_months: 'Age (approx. months)', sex: 'Sex', age_group: 'Age group', site: 'Site'};
  const format = value => value === null || value === undefined ? 'Not estimable' : value !== 0 && Math.abs(value) < 0.001 ? Number(value).toPrecision(2) : Number(value).toFixed(3);
  const count = value => Number(value).toLocaleString();
  const percent = value => `${(100 * value).toFixed(1)}%`;
  let data = null, selection = null, controller, generation = 0;
  function el(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
  function clusterName(label) { return label < 0 ? 'Noise / unassigned' : `Cluster ${label + 1}`; }
  function clusterColor(label) { return label < 0 ? '#a1a1aa' : `hsl(${(label * 137.508 + 250) % 360} 58% 43%)`; }
  function clear(message) {
    data = null; $('association-metrics').replaceChildren(); $('association-chart').replaceChildren();
    $('association-status').textContent = message; $('association-notes').textContent = '';
    $('association-color').disabled = $('association-export').disabled = true;
  }
  async function load() {
    controller?.abort(); const current = ++generation;
    if (!selection) { clear('Run clustering to compare properties with cluster membership.'); $('association-panel').setAttribute('aria-busy', 'false'); return; }
    controller = new AbortController(); const signal = controller.signal;
    clear('Computing patient-level associations and bootstrap intervals…');
    $('association-panel').setAttribute('aria-busy', 'true');
    const params = new URLSearchParams({...selection, site: $('association-site').value, include_noise: $('association-noise').checked});
    try {
      for (;;) {
        const response = await fetch(`/api/graph/associations?${params}`, {signal});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
        if (current !== generation) return;
        if (result.status !== 'computing') {
          data = result;
          const site = $('association-site').value;
          $('association-site').replaceChildren(new Option('All sites', ''), ...result.sites.map(value => new Option(value.replaceAll('_', ' '), value)));
          $('association-site').value = site;
          $('association-color').disabled = $('association-export').disabled = false;
          render(); return;
        }
        await new Promise((resolve, reject) => {
          const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
          const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 1000);
          signal.addEventListener('abort', abort, {once:true});
          if (signal.aborted) abort();
        });
      }
    } catch (error) { if (error.name !== 'AbortError' && current === generation) clear(error.message); }
    finally { if (current === generation) $('association-panel').setAttribute('aria-busy', 'false'); }
  }
  function metric(title, value, description, interval) {
    const card = el('div', undefined, 'association-metric');
    card.append(el('div', title, 'metric-label'), el('strong', format(value)), el('p', description));
    if (interval) card.append(el('p', `95% bootstrap interval ${interval.map(v => v.toFixed(3)).join('–')}`, 'metric-interval'));
    return card;
  }
  function render() {
    if (!data) return;
    const key = $('association-property').value, property = data.properties[key], numeric = property.kind === 'numeric';
    const metrics = $('association-metrics'); metrics.replaceChildren();
    metrics.append(metric(numeric ? 'Age η²' : 'Cramér’s V', property.value,
      numeric ? 'Fraction of age variance between cluster means; 0–1.' : 'Property–cluster association; 0–1. Larger means stronger association.', property.ci));
    if (!numeric) metrics.append(metric('Adjusted mutual information', property.ami, 'Chance-adjusted agreement. Around 0: chance; 1: identical partitions.'));
    if (key !== 'site') metrics.append(metric(numeric ? 'Age partial η² · site-adjusted' : 'Within-site information (bits)', property.site_adjusted,
      numeric ? `Fraction of remaining age variance explained by clusters after accounting for site; ${count(property.site_adjusted_n)} patients with known site.` : `Conditional mutual information given site; ${count(property.site_adjusted_n)} patients. 0 means no observed within-site association; not chance-adjusted.`));
    metrics.append(metric('Missingness Cramér’s V', property.missingness_v, `Association of unknown vs known values with clusters. ${count(property.missing)} patients missing ${names[key].toLowerCase()}.`));
    const cohort = data.cohort;
    $('association-status').textContent = `${count(cohort.included)} patients included · ${count(property.known)} known / ${count(property.missing)} unknown ${names[key].toLowerCase()} · ${count(cohort.noise_patients)} noise patients ${data.include_noise ? 'included' : 'excluded'}${data.site ? ` · ${data.site.replaceAll('_', ' ')}` : ' · all sites'}.`;
    $('association-notes').textContent = `Each mapped patient counts once, in their most frequent slide cluster. Ties choose the smallest assigned cluster, with noise last. ${count(cohort.multiple_cluster_patients)} of ${count(cohort.mapped_patients)} mapped patients span multiple clusters; ${count(cohort.tied_patients)} ties. ${count(cohort.unmapped_slides)} unmapped slides excluded. Metrics exclude unknown property values; the chart includes unknown counts. Intervals are omitted when too few valid bootstrap samples are available. Approximate 95% intervals use ${count(data.bootstrap_resamples)} patient bootstrap resamples (seed ${data.seed}). Not estimable means insufficient variation, samples, or separable site effects. Site-adjusted values are exploratory and use known site values. Graph age filters and highlights do not change this analysis; use the analysis site/noise controls. PCA, t-SNE, UMAP and squares share the same metrics.`;
    const chart = $('association-chart'); chart.replaceChildren();
    if (!cohort.included) { chart.append(el('p', 'No patients match this analysis. Choose another site or include noise.', 'list-message')); return; }
    if (numeric) histogram(property); else heatmap(property);
  }
  function showGraph() { $('graph-canvas').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',block:'center'}); }
  function highlight(label) { window.dispatchEvent(new CustomEvent('graph-highlight-cluster', {detail:label})); showGraph(); }
  function rowLabel(label, total, missing) {
    const cell = el('th'); cell.scope = 'row';
    const button = el('button', clusterName(label), 'qc-button'); button.type = 'button'; button.dataset.variant = 'outline';
    button.title = 'Highlight this cluster in the point view'; button.addEventListener('click', () => highlight(label));
    cell.append(button, el('small', `${count(total)} patients · ${count(missing)} unknown`)); return cell;
  }
  function heatmap(property) {
    const chart = $('association-chart');
    chart.append(el('h3', `${names[$('association-property').value]} enrichment by cluster`), el('p', 'Cell = cluster proportion ÷ overall proportion. 1× is expected; indigo is enriched, amber is depleted. Counts remain visible. Select a cluster name to highlight its points.', 'graph-help'));
    const categories = [...property.categories, 'Unknown / missing'];
    const rows = property.distributions.map(row => ({...row, cells:[...row.counts,row.missing]}));
    const totals = categories.map((_, i) => rows.reduce((sum, row) => sum + row.cells[i], 0));
    const n = totals.reduce((a,b) => a+b, 0);
    const table = el('table', undefined, 'association-table');
    const caption = el('caption', 'Patient counts and enrichment relative to the selected cohort. Unknown values are excluded from Cramér’s V and AMI.'); table.append(caption);
    const head = el('thead'), headers = el('tr'); headers.append(el('th','Cluster'));
    categories.forEach((category, i) => { const th=el('th'); th.scope='col'; th.append(el('span', category.replaceAll('_',' ')), el('small', `${count(totals[i])} overall · ${percent(totals[i]/n)}`)); headers.append(th); });
    head.append(headers); table.append(head); const body=el('tbody');
    for (const row of rows) {
      const tr=el('tr'), total=row.cells.reduce((a,b)=>a+b,0); tr.append(rowLabel(row.cluster,total,row.missing));
      row.cells.forEach((value,i) => {
        const cell=el('td'), baseline=totals[i]/n, observed=value/total, enrichment=baseline ? observed/baseline : null;
        const intensity=enrichment === null ? 0 : Math.min(1, Math.abs(Math.log2(Math.max(enrichment,0.25)))/2);
        const token=enrichment >= 1 ? 'var(--primary)' : 'var(--warning)';
        cell.style.background=`color-mix(in oklch, ${token} ${Math.round(intensity*18)}%, var(--background))`;
        cell.append(el('strong', enrichment === null ? '—' : `${enrichment.toFixed(2)}×`), el('small', `${count(value)} / ${count(total)} · ${percent(observed)}`));
        cell.title=`${clusterName(row.cluster)} · ${categories[i]}: ${count(value)} of ${count(total)} patients (${percent(observed)}), versus ${percent(baseline)} overall.`;
        tr.append(cell);
      }); body.append(tr);
    }
    table.append(body); const scroll=el('div',undefined,'association-scroll'); scroll.tabIndex=0; scroll.setAttribute('aria-label','Scrollable enrichment table'); scroll.append(table); chart.append(scroll);
  }
  function histogram(property) {
    const chart=$('association-chart'); chart.append(el('h3','Age distributions by cluster'),el('p','Shared age bins; bar height is the percentage of patients with known age. Gray outlines show the overall cohort distribution. Compare colored bars against those outlines. Select a cluster name to highlight its points.', 'graph-help'));
    const totals=property.categories.map((_,i)=>property.distributions.reduce((sum,row)=>sum+row.counts[i],0));
    const n=totals.reduce((a,b)=>a+b,0), baseline=totals.map(value=>n ? value/n : 0);
    const grid=el('div',undefined,'association-histograms');
    for (const row of property.distributions) {
      const known=row.counts.reduce((a,b)=>a+b,0), card=el('section',undefined,'association-histogram');
      const button=el('button',clusterName(row.cluster),'qc-button'); button.type='button';button.dataset.variant='outline';button.addEventListener('click',()=>highlight(row.cluster));
      card.append(button,el('p',`${count(known)} known age · ${count(row.missing)} unknown`,'graph-help'));
      const bars=el('div',undefined,'histogram-bars'); bars.style.gridTemplateColumns=`repeat(${property.categories.length}, minmax(0, 1fr))`;
      row.counts.forEach((value,i)=>{
        const ratio=known ? value/known : 0, bin=el('div',undefined,'histogram-bin'), plot=el('div',undefined,'histogram-plot');
        plot.setAttribute('role','img');plot.setAttribute('aria-label',`${property.categories[i]}: ${count(value)} patients, ${percent(ratio)}; overall ${percent(baseline[i])}.`);
        const reference=el('i',undefined,'histogram-reference');reference.style.height=`${baseline[i]*100}%`;
        const bar=el('i',undefined,'histogram-bar');bar.style.height=`${ratio*100}%`;bar.style.background=clusterColor(row.cluster);
        plot.append(reference,bar); plot.title=plot.getAttribute('aria-label');
        bin.append(el('strong',percent(ratio)),plot,el('span',property.categories[i]),el('small',`${count(value)} patients`));bars.append(bin);
      });
      card.append(bars);grid.append(card);
    }
    chart.append(grid);
  }
  window.addEventListener('graph-clusters', event => {
    const detail=event.detail;
    selection=detail ? {organ:detail.organ,branch:detail.branch,method:detail.result.method,...detail.result.parameters} : null;
    if (!detail) { $('association-site').replaceChildren(new Option('All sites','')); }
    load();
  });
  $('association-property').addEventListener('change',render);
  $('association-site').addEventListener('change',load);
  $('association-noise').addEventListener('change',load);
  $('association-color').addEventListener('click',()=>{ $('graph-color').value=$('association-property').value; $('graph-color').dispatchEvent(new Event('input')); showGraph(); });
  $('association-export').addEventListener('click',()=>{ if (!data) return; const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=`cluster-associations-${data.provenance.branch}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); });
})();
