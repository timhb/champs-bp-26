'use strict';
window.casePanel = (() => {
  const $ = id => document.getElementById(id);
  const title = value => String(value ?? '').replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
  const present = value => value !== null && value !== undefined && value !== '';
  const text = value => typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
  const el = (tag, className, value) => { const n = document.createElement(tag); if(className) n.className = className; if(value !== undefined) n.textContent = value; return n; };
  const empty = message => el('p', 'patient-empty', message);
  let data = null, currentKey = null, request = 0, controller = null, timer = null, filterTimer;
  const matches = (value, query) => !query || JSON.stringify(value).toLowerCase().includes(query);
  function section(label, count, content, open = false) {
    const details = el('details', 'patient-section'), summary = el('summary');
    summary.append(el('span', '', label));
    if(count !== undefined && count !== null) summary.append(el('span', 'count', String(count)));
    const body = el('div', 'patient-section-body'); body.append(content); details.append(summary, body); details.open = open;
    return details;
  }
  function fields(record, query = '') {
    const list = el('dl', 'patient-fields');
    for(const [key,value] of Object.entries(record || {})) {
      if(!present(value) || !matches([key,value], query)) continue;
      list.append(el('dt', '', title(key)), el('dd', '', text(value)));
    }
    return list.children.length ? list : empty(query ? 'No matching recorded fields.' : 'No recorded fields.');
  }
  function table(rows, columns, query = '') {
    const filtered = (rows || []).filter(row => matches(row, query));
    if(!filtered.length) return empty(query ? 'No matching records.' : 'No records available.');
    const wrapper = el('div', 'patient-table-wrap'), table = el('table', 'patient-table');
    const head = el('thead'), header = el('tr');
    for(const [key,label] of columns) { const cell=el('th', '', label || title(key)); cell.scope='col'; header.append(cell); }
    head.append(header); table.append(head);
    const body=el('tbody');
    for(const row of filtered) {
      const tr=el('tr');
      for(const [key] of columns) {
        const cell=el('td', '', present(row[key]) ? text(row[key]) : '—');
        if(key==='result' && /^positive$/i.test(row[key] || '')) cell.className='patient-positive';
        tr.append(cell);
      }
      body.append(tr);
    }
    table.append(body); wrapper.append(table); return wrapper;
  }
  function age(record) {
    for(const [key,unit] of [['age_years','years'],['age_months','months'],['age_days','days'],['age_hours','hours']]) {
      if(present(record[key]) && Number(record[key]) >= 1) return `${record[key]} ${unit}`;
    }
    return record.age_group_subcat || record.age_group || (record.age_hours === '0' ? '0 hours' : 'Not recorded');
  }
  function renderSummary() {
    const c=data.case,d=c.demographics || {},target=$('patient-summary'); target.replaceChildren(); target.hidden=false;
    const identity=el('div','patient-identity');
    identity.append(el('strong','',`Case ${c.case}`),el('span','',c.deid || 'No clinical ID mapping'));
    target.append(identity);
    const stats=el('div','patient-stats');
    for(const [label,value] of [['Age',age(d)],['Sex',d.sex],['Postmortem interval',present(d.calc_postmortem_hrs)?`${d.calc_postmortem_hrs} h`:null],['Gestational age',present(d.GA_child_wks)?`${d.GA_child_wks} weeks`:null]]) {
      const item=el('div'); item.append(el('span','',label),el('strong','',value || 'Not recorded')); stats.append(item);
    }
    target.append(stats);
    $('case-scope').textContent=`${title(data.context.organ)} · ${data.context.source}`;
  }
  function reportSection(part, source) {
    const content=el('div');
    if(part.entries.length) content.append(table(part.entries,[['site','Organ / site'],['text','Reported finding']]));
    else if(['diagnosis','site_findings','site_review','cpl_slides','microscopy'].includes(part.kind)) content.append(empty('No explicit finding for this organ in this section.'));
    if(part.notes) content.append(section('Report-level notes · organ not assigned',null,el('p','patient-narrative',part.notes),false));
    if(!content.children.length) content.append(empty('Not recorded.'));
    const primary=(source==='SITE' && ['site_review','site_findings'].includes(part.kind)) || (source==='CPL' && part.kind==='cpl_slides');
    const selected=section(part.title,part.entries.length || null,content,primary || part.kind==='diagnosis' && part.entries.length>0);
    if(primary) selected.classList.add('patient-current-source');
    return selected;
  }
  function renderOrgan() {
    const target=$('organ-context-panel'); target.replaceChildren();
    if(!data.case.deid) target.append(empty('No clinical case mapping is available for this WSI.'));
    target.append(section('Specimen adequacy',data.organ_adequacy.length,
      table(data.organ_adequacy,[['section','Source section'],['organ'],['subsite'],['present_count','Present'],['total_count','Total'],['adequate'],['technically_inadequate','Technically inadequate'],['no_target_tissue','No target tissue'],['match_text','Report statement']]),data.organ_adequacy.length>0));
    const reports=el('div');
    data.organ_reports.forEach((report,index)=>{
      const body=el('div');
      const sourceNote=el('p','patient-note',`Sources: ${report.sources.join(', ')}`);body.append(sourceNote);
      if(report.specimen) body.append(section('Received material · case level',null,el('p','patient-narrative',report.specimen)));
      const organKinds=['diagnosis','cpl_slides','site_review','site_findings','microscopy'];
      const parts=report.sections.filter(part=>organKinds.includes(part.kind));
      for(const part of parts) body.append(reportSection(part,data.context.source));
      const other=report.sections.filter(part=>!organKinds.includes(part.kind));
      if(other.length) {
        const notes=el('div'); for(const part of other) notes.append(reportSection(part,data.context.source));
        body.append(section('Case-level report context',other.length,notes));
      }
      reports.append(section(data.organ_reports.length>1?`Pathology report ${index+1}`:'Pathology report',null,body,index===0));
    });
    if(!data.organ_reports.length) reports.append(empty('No linked pathology narrative is available.'));
    target.append(reports);
    const cod=(data.case.decode || []).filter(row=>/^(Immediate COD|Underlying Cause)$/i.test(row.field));
    target.append(section('Cause of death · case level',cod.length,table(cod,[['field'],['value','Code'],['desc','Description'],['group','CHAMPS group']])));
    const tests=el('div');
    tests.append(el('p','patient-note','Blood results and available tests of the selected organ. All specimens are in All patient data.'));
    if(data.related_tac.length) tests.append(table(data.related_tac,[['specimen'],['target','Assay'],['code'],['result']]));
    if(data.related_lab.length) tests.append(table(data.related_lab,[['specimen'],['test'],['result'],['detail']]));
    if(!data.related_tac.length&&!data.related_lab.length) tests.append(empty('No related test results recorded.'));
    target.append(section('Related microbiology · specimen level',data.related_tac.length+data.related_lab.length,tests));
    const slides=el('div','patient-case-slides');
    for(const slide of data.available_slides) {
      const button=el('button','patient-slide'); button.type='button';button.setAttribute('aria-current',String(slide.id===currentKey));
      button.append(el('strong','',slide.slide_id),el('span','',`${title(slide.organ)} · ${title(slide.site)}`));
      button.addEventListener('click',()=>document.dispatchEvent(new CustomEvent('attention-slide',{detail:slide.id})));slides.append(button);
    }
    target.append(section('Available attention slides in this case',data.available_slides.length,slides));
    target.append(el('p','patient-note','Report text is scoped to the selected organ. SITE findings, CPL review of SITE scans, and CPL slides retain their separate sources.'));
  }
  function renderAll() {
    if(!data) return;
    const target=$('all-patient-content'),query=$('patient-search').value.trim().toLowerCase();target.replaceChildren();
    const c=data.case;
    const add=(label,value,content,open=false)=>{ if(!query||matches([label,value],query)) target.append(section(label,Array.isArray(value)?value.length:Object.keys(value || {}).length,content,open||Boolean(query))); };
    const identity={study_id:c.case,champs_deid:c.deid,...(c.slide || {})};
    add('Slide & case identifiers',identity,fields(identity,query));
    add('Demographics',c.demographics,fields(c.demographics,query),true);
    add('Cause of death · DeCoDe',c.decode,table(c.decode,[['field'],['value','Code'],['desc','Description'],['group','CHAMPS group']],query));
    add('TAC · all specimens and results',c.tac,table(c.tac,[['specimen'],['target','Target'],['code','Assay code'],['result']],query));
    add('Laboratory results · all specimens',c.lab,table(c.lab,[['specimen'],['test'],['result'],['detail'],['col','Original field']],query));
    add('Verbal autopsy',c.verbal_autopsy,fields(c.verbal_autopsy,query));
    add('Specimen adequacy · all organs',c.adequacy,table(c.adequacy,[['section'],['organ'],['subsite'],['present_count','Present'],['total_count','Total'],['adequate'],['technically_inadequate'],['no_target_tissue'],['match_text','Statement']],query));
    add('Pigment confirmation',c.pigment,fields(c.pigment,query));
    for(const group of data.reports) {
      if(query&&!matches([group.title,group.records],query)) continue;
      const body=el('div');body.append(el('p','patient-note',`Source: ${group.source || 'Not recorded'}`));
      group.records.forEach((record,index)=>{
        if(matches(record,query)) body.append(section(`Record ${index+1}`,Object.keys(record).length,fields(record,query),Boolean(query)));
      });
      if(!group.records.length)body.append(empty('No linked records.'));
      target.append(section(`${group.title} · original records`,group.records.length,body,Boolean(query)));
    }
    const raw=el('div');
    for(const [name,record] of Object.entries(data.raw_tables)) if(matches([name,record],query)) raw.append(section(title(name),Object.keys(record).length,fields(record,query),Boolean(query)));
    if(raw.children.length)target.append(section('Original coded patient tables',Object.keys(data.raw_tables).length,raw,Boolean(query)));
    if(!target.children.length)target.append(empty('No patient data matches this search.'));
    if(!query) {
      const details=section('Complete linked case payload',null,el('pre','patient-json',''),false);
      details.addEventListener('toggle',()=>{if(details.open)details.querySelector('pre').textContent=JSON.stringify(data,null,2);});
      target.append(details);
    }
  }
  async function poll(key,id) {
    try {
      const response=await fetch(`/api/slides/${encodeURIComponent(key)}/case`,{signal:controller.signal,cache:'no-store'});
      const result=await response.json();if(id!==request)return;
      if(!response.ok)throw new Error(result.error || 'Could not load case data.');
      if(!result.ready) {
        if(result.error)throw new Error(result.error);
        $('patient-status').textContent='Loading patient tables and reports…';
        timer=setTimeout(()=>poll(key,id),1200);return;
      }
      data=result;renderSummary();renderOrgan();renderAll();
      $('patient-status').textContent=result.case.deid?'Read-only linked patient records.':'No clinical ID mapping found; unavailable records are left empty.';
      $('patient-status').dataset.error='false';
    } catch(error) {
      if(error.name==='AbortError'||id!==request)return;
      $('patient-status').textContent=`Patient data unavailable: ${error.message}`;$('patient-status').dataset.error='true';
      const retry=el('button','qc-button','Try again');retry.type='button';retry.addEventListener('click',()=>load(key));
      $('organ-context-panel').replaceChildren(retry);
    }
  }
  function load(key) {
    controller?.abort();clearTimeout(timer);controller=new AbortController();currentKey=key;data=null;const id=++request;
    $('patient-summary').hidden=true;$('patient-summary').replaceChildren();$('case-scope').textContent='Loading case…';
    $('organ-context-panel').replaceChildren();$('all-patient-content').replaceChildren();$('patient-search').value='';
    $('patient-status').dataset.error='false';$('patient-status').textContent='Loading case information…';poll(key,id);
  }
  const tabs=[$('organ-context-tab'),$('all-patient-tab')];
  function activate(index,focus=false) {
    tabs.forEach((tab,i)=>{tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;$(tab.getAttribute('aria-controls')).hidden=i!==index;});
    if(focus)tabs[index].focus();
  }
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>activate(index));
    tab.addEventListener('keydown',event=>{
      if(['ArrowRight','ArrowLeft','Home','End'].includes(event.key)){event.preventDefault();activate(event.key==='Home'?0:event.key==='End'?1:1-index,true);}
    });
  });
  $('patient-search').addEventListener('input',()=>{clearTimeout(filterTimer);filterTimer=setTimeout(renderAll,150);});
  return {load};
})();
