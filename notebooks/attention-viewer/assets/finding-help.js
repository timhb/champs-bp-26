'use strict';
window.findingHelp = (() => {
  const panel = document.createElement('section');
  panel.id = 'finding-help'; panel.className = 'finding-help'; panel.hidden = true;
  panel.setAttribute('role', 'dialog'); panel.setAttribute('aria-labelledby', 'finding-help-title');
  panel.innerHTML = '<button type="button" class="finding-help-close" aria-label="Close finding guide">×</button><h2 id="finding-help-title"></h2><p id="finding-help-summary"></p><h3>What to look for on H&amp;E</h3><p id="finding-help-histology"></p><p id="finding-help-context" class="finding-help-context"></p><div class="finding-help-sources"></div>';
  document.body.append(panel);
  const content = fetch('/assets/findings.json').then(response => {
    if (!response.ok) throw new Error('Finding guide unavailable');
    return response.json();
  }).catch(() => ({}));
  let anchor = null, closeTimer, request = 0, restoringFocus = false;
  function hide(restoreFocus = false) {
    request++; clearTimeout(closeTimer);
    const previous = anchor; anchor = null; panel.hidden = true;
    previous?.querySelector('.finding-help-button')?.setAttribute('aria-expanded', 'false');
    if (restoreFocus && panel.contains(document.activeElement)) {
      restoringFocus = true; previous?.querySelector('.finding-help-button')?.focus(); restoringFocus = false;
    }
  }
  function position() {
    if (!anchor || panel.hidden) return;
    const rect = anchor.getBoundingClientRect(), width = panel.offsetWidth, height = panel.offsetHeight;
    let left = rect.left - width - 10, top = rect.top;
    if (left < 8) {
      left = Math.max(8, Math.min(rect.left, innerWidth - width - 8));
      top = rect.bottom + 8;
      if (top + height > innerHeight - 8) top = rect.top - height - 8;
    }
    panel.style.left = `${Math.max(8, Math.min(left, innerWidth - width - 8))}px`;
    panel.style.top = `${Math.max(8, Math.min(top, innerHeight - height - 8))}px`;
  }
  async function show(row, label) {
    clearTimeout(closeTimer);
    const id = ++request, entries = await content;
    if (id !== request || !row.isConnected) return;
    const entry = entries[label];
    hide(); anchor = row;
    panel.querySelector('h2').textContent = label.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
    panel.querySelector('#finding-help-summary').textContent = entry?.description || 'No description is available for this finding yet.';
    panel.querySelector('#finding-help-histology').textContent = entry?.histology || 'Use the slide and reference report to interpret this label.';
    panel.querySelector('#finding-help-context').textContent = entry?.context || '';
    const sources = panel.querySelector('.finding-help-sources'); sources.replaceChildren();
    if (entry) {
      for (const [url, name] of [[entry.source, entry.source_name], [entry.additional_source, entry.additional_source_name]]) {
        if (!url) continue;
        const link = document.createElement('a'); link.href = url; link.textContent = name;
        link.target = '_blank'; link.rel = 'noopener noreferrer'; sources.append(link);
      }
    }
    row.querySelector('.finding-help-button').setAttribute('aria-expanded', 'true');
    panel.hidden = false; position();
  }
  function leave() {
    clearTimeout(closeTimer);
    closeTimer = setTimeout(() => {
      if (!anchor?.matches(':hover') && !panel.matches(':hover') && !anchor?.contains(document.activeElement) && !panel.contains(document.activeElement)) hide();
    }, 180);
  }
  function attach(button, label) {
    const row = document.createElement('div'); row.className = 'branch-row';
    const help = document.createElement('button'); help.type = 'button'; help.className = 'finding-help-button'; help.textContent = 'i';
    help.setAttribute('aria-label', `About ${label.replaceAll('_', ' ')}: description and histology`);
    help.setAttribute('aria-haspopup', 'dialog'); help.setAttribute('aria-controls', panel.id); help.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-describedby', 'finding-help-summary finding-help-histology finding-help-context');
    row.append(button, help);
    row.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') show(row, label); });
    row.addEventListener('pointerleave', leave);
    row.addEventListener('focusin', () => { if (!restoringFocus) show(row, label); }); row.addEventListener('focusout', leave);
    help.addEventListener('click', () => show(row, label));
    help.addEventListener('keydown', event => { if (event.key === 'ArrowDown' && !panel.hidden) { event.preventDefault(); panel.querySelector('button').focus(); } });
    return row;
  }
  panel.addEventListener('pointerenter', () => clearTimeout(closeTimer));
  panel.addEventListener('pointerleave', leave); panel.addEventListener('focusout', leave);
  panel.querySelector('button').addEventListener('click', () => hide(true));
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !panel.hidden) { event.preventDefault(); hide(true); } });
  document.addEventListener('pointerdown', event => { if (!panel.contains(event.target) && !anchor?.contains(event.target)) hide(); });
  window.addEventListener('resize', position);
  window.addEventListener('scroll', event => {
    if (!anchor || panel.contains(event.target)) return;
    const rect = anchor.getBoundingClientRect();
    if (!anchor.isConnected || rect.bottom < 0 || rect.top > innerHeight) hide();
    else position();
  }, true);
  return {attach, hide};
})();
