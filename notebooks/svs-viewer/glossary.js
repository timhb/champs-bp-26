// Published NLM definitions, matched locally. No report text goes to an external website.
(() => {
  const $ = selector => document.querySelector(selector);
  const panel = $('#reference'), content = $('#referenceContent'), query = $('#referenceQuery');
  const status = $('#referenceStatus'), info = $('#info');
  let metadata = null, requestVersion = 0, searchTimer, linkTimer, linking = false, dirty = false, lastFocus;
  const processed = new WeakMap();

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  async function json(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(response.status === 503
      ? 'The medical glossary is not installed. Run pixi run python notebooks/svs-viewer/build_medical_glossary.py, then restart the viewer.'
      : 'Medical reference is unavailable. Please retry.');
    return response.json();
  }
  function showPanel() {
    panel.hidden = false;
    document.body.classList.add('reference-open');
    window.dispatchEvent(new Event('resize'));
  }
  function sourceLink(label, url) {
    const link = node('a', label);
    link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    return link;
  }
  function furtherReading(term) {
    const p = node('p', 'Further reading: ', 'sub');
    p.append(sourceLink('Search MedlinePlus', 'https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&query=' + encodeURIComponent(term)));
    p.append(document.createTextNode(' · '), sourceLink('NCI medical dictionary', 'https://www.cancer.gov/publications/dictionaries/cancer-terms'));
    return p;
  }
  function histologyExample(example) {
    const figure = node('figure', null, 'histology-example');
    const image = node('img');
    image.alt = example.caption;
    image.loading = 'lazy'; image.referrerPolicy = 'no-referrer';
    const caption = node('figcaption', example.caption + ' ', 'sub');
    caption.append(sourceLink(example.credit + ' · ' + example.license, example.source_url));
    image.onerror = () => image.remove();
    image.src = example.url;
    figure.append(image, caption);
    return figure;
  }
  function displayEntry(entry, wording) {
    content.replaceChildren(node('h3', entry.name));
    if (wording && wording.toLowerCase() !== entry.name.toLowerCase()) content.append(node('p', 'Selected wording: ' + wording, 'sub'));
    content.append(node('p', entry.definition || 'NLM does not supply a standalone definition for this concept. Open the source record for its terminology context.'));
    if (!entry.definition && entry.broader_definition) {
      content.append(node('h4', 'Broader heading: ' + entry.descriptor_name), node('p', entry.broader_definition));
    }
    content.append(sourceLink('Open the original reference ↗', entry.url));
    content.append(node('p', `Source: ${entry.source_name || 'U.S. National Library of Medicine, MeSH ' + (metadata?.year || '')}.` +
      (entry.definition_kind ? ' ' + entry.definition_kind + '.' : ''), 'sub'));
    if (entry.histology_image) {
      content.append(node('h4', 'Histopathology example'), histologyExample(entry.histology_image));
      content.append(node('p', 'Reference image from another case. Appearance depends on tissue, stain and sampling; this does not identify disease in the current slide.', 'sub'));
    }
    content.append(furtherReading(wording || entry.name));
  }
  async function showEntry(uid, wording) {
    const version = ++requestVersion;
    content.replaceChildren(node('p', 'Loading definition…', 'sub'));
    try {
      const entry = await json('/api/glossary/entry/' + encodeURIComponent(uid));
      if (version === requestVersion) displayEntry(entry, wording);
    } catch (error) { if (version === requestVersion) content.replaceChildren(node('p', error.message)); }
  }
  function choices(entries, wording, ambiguous = false) {
    content.replaceChildren(node('p', entries.length
      ? ambiguous ? 'This wording has multiple MeSH meanings. Choose the relevant concept:' : 'Choose a medical term:'
      : 'No glossary entry matched. Try a longer phrase, another spelling, or the references below.'));
    for (const entry of entries) {
      const button = node('button', entry.name, 'btn ref-choice');
      button.type = 'button';
      button.onclick = () => showEntry(entry.id, wording);
      content.append(button);
    }
    content.append(furtherReading(wording));
  }
  async function lookup(term, ids) {
    term = term.trim().slice(0, 120);
    clearTimeout(searchTimer);
    showPanel(); query.value = term;
    const version = ++requestVersion;
    if (!term) { content.replaceChildren(node('p', 'Click an underlined term, or enter a medical word or condition.')); return; }
    content.replaceChildren(node('p', 'Looking up “' + term + '”…', 'sub'));
    try {
      const entries = ids
        ? await Promise.all(ids.map(id => json('/api/glossary/entry/' + encodeURIComponent(id))))
        : (await json('/api/glossary/search?' + new URLSearchParams({q: term}))).items;
      if (version !== requestVersion) return;
      if (entries.length === 1) displayEntry(entries[0], term);
      else choices(entries, term, Boolean(ids));
    } catch (error) { if (version === requestVersion) content.replaceChildren(node('p', error.message), furtherReading(term)); }
  }

  $('#openGlossary').onclick = () => {
    lastFocus = $('#openGlossary');
    const selected = window.getSelection()?.toString().trim();
    if (selected) lookup(selected); else { showPanel(); query.focus(); }
  };
  $('#closeGlossary').onclick = () => {
    ++requestVersion; clearTimeout(searchTimer);
    panel.hidden = true; document.body.classList.remove('reference-open');
    window.dispatchEvent(new Event('resize')); lastFocus?.focus();
  };
  query.addEventListener('input', () => {
    ++requestVersion; clearTimeout(searchTimer);
    searchTimer = setTimeout(() => lookup(query.value), 250);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !panel.hidden) $('#closeGlossary').click();
  });
  info.addEventListener('click', event => {
    const link = event.target.closest('a.medical-term');
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); event.stopPropagation(); lastFocus = link;
    lookup(link.textContent, JSON.parse(link.dataset.medicalIds));
  });

  function scheduleLinks() {
    if (linking) { dirty = true; return; }
    clearTimeout(linkTimer); linkTimer = setTimeout(linkTerms, 100);
  }
  const observer = new MutationObserver(scheduleLinks);
  const observe = () => observer.observe(info, {childList: true, subtree: true, characterData: true});
  async function linkTerms() {
    if (!metadata) return;
    linking = true; dirty = false;
    try {
      const walker = document.createTreeWalker(info, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) {
        const current = walker.currentNode;
        if (!current.nodeValue.trim() || current.nodeValue.length > 30000 || processed.get(current) === current.nodeValue) continue;
        if (current.parentElement.closest('a, button, input, select, textarea, pre, code, script, style')) continue;
        nodes.push({node: current, text: current.nodeValue});
      }
      while (nodes.length) {
        const batch = []; let size = 0;
        while (nodes.length && batch.length < 200 && size + nodes[0].text.length <= 200000) {
          const item = nodes.shift(); batch.push(item); size += item.text.length;
        }
        const result = await json('/api/glossary/matches', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({texts: batch.map(item => item.text)}),
        });
        observer.disconnect();
        try {
          batch.forEach((item, index) => {
            if (!item.node.isConnected || item.node.nodeValue !== item.text) return;
            processed.set(item.node, item.text);
            const matches = result.matches[index];
            if (!matches.length) return;
            // Server offsets count Unicode characters; Array.from handles non-BMP text too.
            const chars = Array.from(item.text), fragment = document.createDocumentFragment();
            let cursor = 0;
            const appendText = text => {
              if (!text) return;
              const child = document.createTextNode(text); processed.set(child, text); fragment.append(child);
            };
            for (const match of matches) {
              appendText(chars.slice(cursor, match.start).join(''));
              const link = node('a', chars.slice(match.start, match.end).join(''), 'medical-term');
              link.href = match.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
              link.dataset.medicalIds = JSON.stringify(match.ids);
              link.title = 'Open NLM medical definition'; fragment.append(link); cursor = match.end;
            }
            appendText(chars.slice(cursor).join('')); item.node.replaceWith(fragment);
          });
        } finally { observe(); }
      }
    } catch (error) { status.textContent = error.message; }
    finally { linking = false; if (dirty) scheduleLinks(); }
  }
  observe();
  json('/api/glossary/status').then(data => {
    metadata = data;
    status.textContent = data.condition_links
      ? `NLM MeSH ${data.year} · automatic links for medical conditions`
      : 'Rebuild the glossary and restart the viewer to enable condition links.';
    scheduleLinks();
  }).catch(error => { status.textContent = error.message; });
})();
