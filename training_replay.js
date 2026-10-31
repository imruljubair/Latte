/* Replay recorded PyTorch training through the existing matrix explorer. */
(() => {
  function targetProbability(logits,target){
    const max=Math.max(...logits),values=logits.map(value=>Math.exp(value-max));
    return values[target]/values.reduce((sum,value)=>sum+value,0);
  }
  function parameterLabel(name){
    const global={'token_embedding.weight':'Token embeddings','position_embedding.weight':'Position embeddings','final_norm.weight':'Final normalization scale','lm_head.weight':'Vocabulary projection'};
    if(global[name])return global[name];
    const layer=name.match(/^blocks\.(\d+)\.(.*)\.weight$/);
    const names={'attention.wq':'Query weights','attention.wk':'Key weights','attention.wv':'Value weights','attention.wo':'Attention output weights','attention_norm':'Attention normalization scale','ffn_norm':'MLP normalization scale','ffn.0':'MLP input weights','ffn.2':'MLP output weights'};
    return layer?`Layer ${Number(layer[1])} · ${names[layer[2]]||layer[2]}`:name;
  }
  if(typeof module!=='undefined')module.exports={targetProbability,parameterLabel};
  if(typeof document==='undefined')return;
  const data=document.getElementById('training-record-data');
  if(!data)return;
  const record=JSON.parse(data.textContent),bundle=JSON.parse(document.getElementById('sequence-generation-data').textContent);
  const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const tab=document.createElement('button');tab.textContent='Training';tab.type='button';
  document.querySelector('.stage-view-tabs').append(tab);
  const panel=document.createElement('section');panel.className='training-replay';panel.hidden=true;
  panel.innerHTML='<header><button data-action="previous">← Previous</button><label>Training record <select class="training-record" aria-label="Training record"></select></label><button data-action="next">Next →</button><label>Sequence <button type="button" data-action="sequence-previous" aria-label="Previous batch sequence">‹</button> <select class="training-sequence"></select> <button type="button" data-action="sequence-next" aria-label="Next batch sequence">›</button></label><button data-action="apply">Apply update</button><span class="training-status" role="status"></span></header><div class="training-body"><iframe title="Training forward pass"></iframe><aside><div class="training-legend" aria-label="Color legend"><span><i style="background:#a5abb3"></i>Before weights</span><span><i style="background:#f6ac37"></i>Updated weights</span><span><i style="background:#5d8fcd"></i>Negative values</span><span><i style="background:#ee6861"></i>Positive values</span><span class="training-correct">✓ Correct</span><span class="training-incorrect">✗ Incorrect</span></div><small class="training-resize-hint">Drag each section’s bottom-right corner to resize.</small><section class="training-corpus-section"><h3 class="training-corpus-heading">Corpus sequence <button type="button" class="training-locate" aria-label="Locate selected corpus sequence" title="Locate selected sequence">⌖</button></h3><p class="training-source"></p><pre class="training-corpus"></pre></section><section class="training-weight-section" hidden><h3>Weight update · whole batch</h3><p>Click a weight in the diagram or choose a parameter.</p><select class="training-parameter"></select><label> Row <input class="training-row" type="number" min="0" value="0"></label><label> Column <input class="training-column" type="number" min="0" value="0"></label><p class="training-parameter-path"></p><pre class="training-inspection"></pre></section></aside></div>';
  document.body.append(panel);
  const style=document.createElement('style');style.textContent='.training-replay{position:fixed;inset:0 0 30px;z-index:190;background:#f4f6fa;color:#405978;font:12px/1.5 system-ui}.training-replay[hidden]{display:none}.training-replay header{height:56px;display:flex;align-items:center;gap:12px;padding:0 16px;border-bottom:1px solid #dce4ef;flex-wrap:wrap}.training-replay button,.training-replay select,.training-replay input{font:inherit;border:1px solid #dce4ef;border-radius:5px;padding:5px;background:white;color:inherit}.training-body{display:flex;min-height:0;height:calc(100% - 57px)}.training-body iframe{flex:1;min-width:0;border:0}.training-body aside{width:350px;flex-shrink:0;overflow-y:auto;overscroll-behavior:contain;box-sizing:border-box;min-height:0;display:flex;flex-direction:column;gap:8px;padding:16px;border-left:1px solid #dce4ef;background:#fff}.training-body pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.8 ui-monospace,monospace}.training-corpus-section,.training-predictions-section{flex:1 1 0;min-height:90px;display:flex;flex-direction:column}.training-predictions-section{min-height:210px}.training-weight-section{flex:0 0 auto}.training-corpus-section>p{margin:0}.training-corpus,.training-predictions{flex:1;min-height:0;margin:6px 0}.training-corpus{font-size:14px;line-height:1.8;max-height:18vh;overflow:auto;overscroll-behavior:contain}.training-predictions{max-height:22vh;overflow:auto;scrollbar-gutter:stable;overscroll-behavior:contain}.training-predictions table{width:100%;border-collapse:separate;border-spacing:0;font-size:10px;line-height:1.5;white-space:nowrap}.training-predictions caption{text-align:left;font-size:11px;color:#70819a;padding:0 0 6px}.training-predictions th,.training-predictions td{padding:5px 4px;border-bottom:1px solid #e6ebf2;text-align:left}.training-predictions thead th{position:sticky;top:0;background:#f4f6fa;font-weight:650;z-index:1}.training-predictions tbody th{font-weight:400;color:#70819a}.training-predictions td:nth-child(2),.training-predictions td:nth-child(3),.training-predictions td:nth-child(4){font-family:ui-monospace,monospace}.training-predictions td:last-child,.training-predictions thead th:last-child{text-align:right;font-variant-numeric:tabular-nums}.training-predictions tbody tr:hover{background:#f8fafc}.training-corpus mark{background:#ffe7a3;padding:2px}.training-parameter{width:100%}.training-body input{width:48px}.training-status{font-size:11px}.training-body h3{margin:12px 0 6px}@media(max-width:800px){.training-body aside{width:250px}.training-replay header{height:90px}.training-body{height:calc(100% - 91px)}}.training-replay button[data-action=apply]{background:#ffdb91;border-color:#dd8a32;color:#75430d;transition:background-color .3s,border-color .3s}.training-replay button[data-action=apply][data-applied=true]{background:#e5e7eb;border-color:#9ca3af;color:#374151}.training-corpus-heading{display:flex;align-items:center;justify-content:space-between}.training-replay button.training-locate{padding:1px 6px;font-size:18px;line-height:1.2}.training-correct{color:#237343;font-weight:650}.training-incorrect{color:#b43b3b;font-weight:650}.training-weight-section h3{font-size:13px;margin:6px 0 3px}.training-weight-section p{font-size:11px;margin:3px 0 6px}.training-weight-section select,.training-weight-section input{padding:3px}.training-weight-section .training-inspection{font-size:11px;line-height:1.5;margin:6px 0 0}.training-legend{display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:10px;flex-shrink:0}.training-legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}.training-resize-hint{font-size:10px;color:#70819a}.training-body aside>section{flex:none;resize:vertical;overflow:auto;min-height:120px;border-bottom:1px solid #dce4ef;padding-bottom:8px;box-sizing:border-box}.training-body .training-corpus-section{height:220px}.training-body .training-predictions-section{height:260px;min-height:120px}.training-body .training-weight-section{height:240px}.training-body .training-corpus,.training-body .training-predictions{max-height:none}.training-parameter-path{overflow-wrap:anywhere;color:#70819a;font:10px/1.3 ui-monospace,monospace!important}';document.head.append(style);
  const frame=panel.querySelector('iframe'),sequence=panel.querySelector('.training-sequence'),parameter=panel.querySelector('.training-parameter');
  const row=panel.querySelector('.training-row'),column=panel.querySelector('.training-column');
  let stepIndex=0,sequenceIndex=0,applied=false,frameReady=false,pendingGrid=null;
  let heatmap=false,lastLocatedKey='';
  const current=()=>record.steps[stepIndex];
  const recordSelect=panel.querySelector('.training-record');
  record.steps.forEach((step,i)=>{
    const option=document.createElement('option');option.value=i;
    option.textContent=`Step ${step.step} · ${i} / ${record.steps.length-1}`;
    recordSelect.append(option);
  });
  function chooseRecord(index){
    stepIndex=index;applied=false;
    sequenceIndex=Math.min(sequenceIndex,current().inputs.length-1);
    render();
  }
  recordSelect.onchange=()=>chooseRecord(Number(recordSelect.value));
  for(const name of Object.keys(current().before)){const option=document.createElement('option');option.value=name;option.textContent=parameterLabel(name);option.title=name;parameter.append(option);}
  function inspect(){
    const step=current(),name=parameter.value,r=Number(row.value),c=Number(column.value);
    const at=values=>Array.isArray(values?.[r])?values[r][c]:c===0?values?.[r]:undefined;
    const before=at(step.before[name]),grad=at(step.gradients[name]),after=at(step.after[name]);
    panel.querySelector('.training-parameter-path').textContent=name;
    row.max=step.before[name].length-1;column.max=Array.isArray(step.before[name][0])?step.before[name][0].length-1:0;
    panel.querySelector('.training-inspection').textContent=before===undefined?'Choose a valid row and column.':`Old weight  ${before.toPrecision(8)}\nGradient    ${grad.toPrecision(8)}\nChange      ${(after-before).toPrecision(8)}\nNew weight  ${after.toPrecision(8)}\nAdamW · ${step.inputs.length} sequences (batch mean)`;
  }
  [parameter,row,column].forEach(input=>input.addEventListener('input',inspect));
  function traceFor(weights){
    const a=record.architecture,step=current(),t={architecture:a,vocabulary:record.vocabulary,global_weights:{},layers:[],training:{targets:step.targets[sequenceIndex].map(i=>record.vocabulary[i]),losses:step.token_losses[sequenceIndex]}};
    for(const name of ['token_embedding','position_embedding','final_norm','lm_head'])t.global_weights[name]=weights[name+'.weight'];
    for(let i=0;i<a.num_layers;i++){
      const prefix=`blocks.${i}.`,w={};
      for(const [key,path] of Object.entries({attention_norm:'attention_norm',ffn_norm:'ffn_norm',wo:'attention.wo',ffn_in:'ffn.0',ffn_out:'ffn.2'}))w[key]=weights[prefix+path+'.weight'];
      for(const letter of ['q','k','v']){const width=letter==='v'?a.value_dim:a.qk_dim,matrix=weights[prefix+`attention.w${letter}.weight`];w[`w${letter}_by_head`]=Array.from({length:a.num_heads},(_,h)=>matrix.slice(h*width,(h+1)*width));}
      t.layers.push({layer:i,weights:w});
    }return t;
  }
  function weightCell(cell,select=true,origin=false){
    const name=cell.dataset.matrixName;let r=Number(cell.dataset.matrixRow)-1,c=Number(cell.dataset.matrixColumn)-1,key;
    const layer=name.match(/layer (\d+)/),head=name.match(/head (\d+)/);
    if(name==='Position embeddings'){key='position_embedding.weight';[r,c]=[c,r];}
    else if(name==='Vocabulary weights')key='lm_head.weight';
    else if(name==='RMS scale · Final hidden')key='final_norm.weight';
    else if(name==='Token embeddings'){key='token_embedding.weight';[r,c]=[current().inputs[sequenceIndex][c],r];}
    else if(layer){const prefix=`blocks.${layer[1]}.`;
      if(/^W[QKV] ·/.test(name)){const letter=name[1].toLowerCase();key=prefix+`attention.w${letter}.weight`;r+=Number(head[1])*(letter==='v'?record.architecture.value_dim:record.architecture.qk_dim);}
      else if(name.startsWith('WO ·'))key=prefix+'attention.wo.weight';
      else if(name.startsWith('W1 ·'))key=prefix+'ffn.0.weight';
      else if(name.startsWith('W2 ·'))key=prefix+'ffn.2.weight';
      else if(name.startsWith('RMS scale · Attention'))key=prefix+'attention_norm.weight';
      else if(name.startsWith('RMS scale · FFN'))key=prefix+'ffn_norm.weight';
    }
    if(key){if(select){parameter.value=key;row.value=r;column.value=c;inspect();}const get=values=>Array.isArray(values[r])?values[r][c]:values[r];
      if(origin)return {before:get(current().before[key]),after:get(current().after[key]),gradient:get(current().gradients[key]),applied,step:current().step,batchSize:current().inputs.length};
      return get(current().before[key])!==get(current().after[key]);}
    return false;
  }
  function highlightUpdates() {
    frame.contentDocument.documentElement.toggleAttribute('data-training-updated',applied);
    const step=current();
    frame.contentWindow.LattePresentation?.setTokenMatches(step.logits[sequenceIndex].map((logits,i)=>
      logits.indexOf(Math.max(...logits))===step.targets[sequenceIndex][i]));
  }
  function updateDiagram(grid) {
    frame.contentDocument.documentElement.toggleAttribute('data-training-updated',applied);
    frame.contentWindow.LittleLlamaWorkbook.updateTrainingGrid([...grid.cells.values()]);
    highlightUpdates();
  }
  function locateSequence(){
    const corpus=panel.querySelector('.training-corpus'),mark=corpus.querySelector('mark');
    if(!mark)return;
    const fragments=[...mark.getClientRects()];
    if(!fragments.length)return;
    const box=corpus.getBoundingClientRect();
    const top=Math.min(...fragments.map(rect=>rect.top));
    const bottom=Math.max(...fragments.map(rect=>rect.bottom));
    corpus.scrollTop += (top+bottom)/2-(box.top+corpus.clientTop+corpus.clientHeight/2);
  }
  panel.querySelector('.training-locate').onclick=locateSequence;
  const usageTools=document.createElement('div');usageTools.className='latte-corpus-usage-tools';
  const usageLabel=document.createElement('label'),usageToggle=document.createElement('input');usageToggle.type='checkbox';
  const usageScale=document.createElement('span');usageScale.className='latte-usage-scale';
  usageLabel.append(usageToggle,document.createTextNode('Corpus heatmap'));usageTools.append(usageLabel,usageScale);
  panel.querySelector('.training-source').after(usageTools);
  usageToggle.onchange=()=>{heatmap=usageToggle.checked;render();};
  function render(){
    const step=current(),tokens=step.inputs[sequenceIndex].map(i=>record.vocabulary[i]),t=traceFor(applied?step.after:step.before);
    recordSelect.value=stepIndex;
    panel.querySelector('[data-action="previous"]').disabled=stepIndex===0;panel.querySelector('[data-action="next"]').disabled=stepIndex===record.steps.length-1;
    panel.querySelector('[data-action="apply"]').textContent=applied?'Show before update':'Apply update';
    panel.querySelector('[data-action="apply"]').dataset.applied=String(applied);
    panel.querySelector('.training-status').textContent=`${applied?'After update · recomputed forward pass':'Before update'} · recorded batch loss ${step.loss.toFixed(5)}`;
    sequence.replaceChildren();step.inputs.forEach((_,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i} / ${step.inputs.length-1}`;sequence.append(option);});sequence.value=sequenceIndex;
    panel.querySelector('[data-action="sequence-previous"]').disabled=sequenceIndex===0;
    panel.querySelector('[data-action="sequence-next"]').disabled=sequenceIndex===step.inputs.length-1;
    const start=step.starts[sequenceIndex],text=Array.from(record.corpus.text),corpus=panel.querySelector('.training-corpus');
    const predictions=step.logits[sequenceIndex].map(logits=>logits.indexOf(Math.max(...logits)));
    const correct=predictions.map((id,i)=>id===step.targets[sequenceIndex][i]);
    const usage=record.corpus_usage,heatmapOn=heatmap&&!!usage;
    const {usageAt,usageColor,usageSegments}=window.LatteCorpusUsage;
    usageToggle.disabled=!usage;usageToggle.checked=heatmapOn;
    usageScale.replaceChildren();
    usageScale.className=usage?'latte-usage-scale':'latte-usage-unavailable';
    if(!usage)usageScale.textContent='Usage wasn’t recorded for this run.';
    else if(heatmapOn){usageScale.append(document.createElement('i'),document.createTextNode(`0–${usage.max_count} uses · entire run · ${(100*usage.used_positions/record.corpus.train_end).toFixed(1)}% covered`));}
    corpus.classList.toggle('corpus-heatmap-enabled',heatmapOn);
    const slice=(from,to)=>{
      if(!heatmapOn)return [document.createTextNode(text.slice(from,to).join(''))];
      return usageSegments(usage,from,to,record.corpus.train_end).map(part=>{
        const span=document.createElement('span');span.className='corpus-usage-span';span.tabIndex=0;span.setAttribute('role','button');span.setAttribute('aria-label',`Show usage for characters ${part.start}–${part.end-1}`);
        Object.assign(span.dataset,{usageStart:String(part.start),usageEnd:String(part.end),usageCount:String(part.count),usageValidation:String(part.validation)});
        span.style.backgroundColor=part.validation?'transparent':usageColor(part.count,usage.max_count);span.textContent=text.slice(part.start,part.end).join('');return span;
      });
    };
    corpus.replaceChildren(...slice(0,start));
    const mark=document.createElement('mark');
    tokens.forEach((token,i)=>{
      const span=document.createElement('span');
      span.className='training-corpus-token';
      if(heatmapOn){Object.assign(span.dataset,{usageStart:String(start+i),usageEnd:String(start+i+1),usageCount:String(usageAt(usage,start+i))});span.style.backgroundColor=usageColor(usageAt(usage,start+i),usage.max_count);}
      span.textContent=token;span.tabIndex=0;span.setAttribute('role','button');
      const predicted=record.vocabulary[predictions[i]],target=record.vocabulary[step.targets[sequenceIndex][i]];
      Object.assign(span.dataset,{corpusPosition:String(i),input:token,prediction:predicted,target,targetProbability:String(targetProbability(step.logits[sequenceIndex][i],step.targets[sequenceIndex][i])),match:String(correct[i])});
      span.setAttribute('aria-label',`Position ${i}: input ${JSON.stringify(token)}, predicted ${JSON.stringify(predicted)}, target ${JSON.stringify(target)}. Recorded before update.`);
      mark.append(span);
    });
    corpus.append(mark,...slice(start+tokens.length,text.length));
    const locateKey=`${stepIndex}:${sequenceIndex}`;
    if(locateKey!==lastLocatedKey){lastLocatedKey=locateKey;requestAnimationFrame(locateSequence);}
    panel.querySelector('.training-source').textContent=`Characters ${start}–${start+tokens.length-1} · batch of ${step.inputs.length}`;
    const grid=new window.LittleLlamaSequenceGrid(t,tokens,{}).build();
    inspect();
    if(frameReady){updateDiagram(grid);return;}
    if(pendingGrid){pendingGrid=grid;return;}
    pendingGrid=grid;
    const css='<style>.sequence-editor,.sampling-chart-toggle,.toolbar-group:has([data-action="sequence"]){display:none!important}.latte-visible-stage .matrix-value{transition:background-color .4s,filter .3s}html:not([data-training-updated]) .latte-visible-stage .matrix-value.learned-parameter{filter:grayscale(1)}</style>';
    const html=bundle.shell.replace(/<html\b/, '<html data-training-replay').replace('__FORWARD_TABLE__',()=>grid.table()).replace('__GENERATION_CONTROLS__','').replace('__SEQUENCE_SCRIPT__','').replace('</head>',css+'</head>');
    frame.onload=()=>{
      frame.contentWindow.LittleLlamaWorkbook.weightOrigin=cell=>weightCell(cell,false,true);
      const doc=frame.contentDocument;doc.querySelectorAll('input:not(.stage-picker input):not(.stage-pick-control input)').forEach(input=>input.disabled=true);doc.querySelectorAll('[contenteditable]').forEach(cell=>cell.contentEditable='false');
      doc.addEventListener('click',event=>{const cell=event.target.closest('[data-matrix-name]');if(cell)weightCell(cell);},true);
      frameReady=true;const latest=pendingGrid;pendingGrid=null;
      if(latest!==grid)updateDiagram(latest);else highlightUpdates();
    };frame.srcdoc=html;inspect();
  }
  const stepsMenu=document.querySelector('.steps-menu');
  function disableStepChooser(disabled){
    if(!stepsMenu)return;
    stepsMenu.open=false;
    stepsMenu.inert=disabled;
    stepsMenu.querySelector('summary').setAttribute('aria-disabled',String(disabled));
    stepsMenu.style.opacity=disabled?'.45':'';
    stepsMenu.title=disabled?'Choose steps is unavailable in Training':'';
    if(disabled){const picker=document.querySelector('.stage-picker');if(picker)picker.hidden=true;}
  }
  tab.onclick=()=>{disableStepChooser(true);panel.hidden=false;document.querySelectorAll('.stage-view-tabs button').forEach(button=>button.classList.toggle('active',button===tab));render();};
  document.querySelector('.stage-view-tabs').addEventListener('click',event=>{if(event.target.closest('button')&&event.target!==tab){disableStepChooser(false);panel.hidden=true;tab.classList.remove('active');}});
  panel.querySelector('[data-action="sequence-previous"]').onclick=()=>{if(sequenceIndex>0){sequenceIndex--;render();}};
  panel.querySelector('[data-action="sequence-next"]').onclick=()=>{if(sequenceIndex<current().inputs.length-1){sequenceIndex++;render();}};
  sequence.onchange=()=>{sequenceIndex=Number(sequence.value);render();};
  panel.querySelector('[data-action="previous"]').onclick=()=>chooseRecord(stepIndex-1);
  panel.querySelector('[data-action="next"]').onclick=()=>chooseRecord(stepIndex+1);
  panel.querySelector('[data-action="apply"]').onclick=()=>{applied=!applied;render();};
})();
