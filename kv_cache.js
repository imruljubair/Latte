/* Cache steps rendered by the same stage, camera and calculation engine as Selected steps. */
(function(root){
  'use strict';
  // Append-only numerical inference. Earlier tokens' K/V and activations stay cached.
  class CacheInference {
    constructor(trace,erf){this.trace=trace;this.erf=erf;this.tokens=[];this.tensors=new Map();}
    run(tokens,settings){
      const t=this.trace,a=t.architecture;
      if(this.tokens.some((token,i)=>tokens[i]!==token)){this.tokens=[];this.tensors.clear();}
      const dot=(x,y)=>x.reduce((sum,v,i)=>sum+v*y[i],0),mul=(w,x)=>w.map(row=>dot(row,x));
      const add=(x,y)=>x.map((v,i)=>v+y[i]),norm=(x,g)=>{const scale=Math.sqrt(x.reduce((sum,v)=>sum+v*v,0)/x.length+1.1920928955078125e-7);return x.map((v,i)=>v*g[i]/scale);};
      const softmax=x=>{const max=Math.max(...x),exp=x.map(v=>Math.exp(v-max)),sum=exp.reduce((s,v)=>s+v,0);return exp.map(v=>v/sum);};
      const append=(name,values)=>{if(!this.tensors.has(name))this.tensors.set(name,values.map(()=>[]));this.tensors.get(name).forEach((row,i)=>row.push(values[i]));return values;};
      const column=(name,i)=>this.tensors.get(name).map(row=>row[i]);
      for(let position=this.tokens.length;position<tokens.length;position++){
        const token=t.global_weights.token_embedding[t.vocabulary.indexOf(tokens[position])],pos=t.global_weights.position_embedding[position];
        append('Token embeddings',token);append('Position embeddings',pos);
        let x=append('X · embedding output',add(token,pos));
        for(const layer of t.layers){
          const w=layer.weights,suffix=`layer ${layer.layer}`,input=append(`Attention input · ${suffix}`,norm(x,w.attention_norm)),heads=[];
          for(let head=0;head<a.num_heads;head++){
            const tag=`${suffix} · head ${head}`,k=append(`K · ${tag}`,mul(w.wk_by_head[head],input)),v=append(`V · ${tag}`,mul(w.wv_by_head[head],input)),q=append(`Q · ${tag}`,mul(w.wq_by_head[head],input));
            const keys=this.tensors.get(`K · ${tag}`),queries=this.tensors.get(`Q · ${tag}`),values=this.tensors.get(`V · ${tag}`);
            this.tensors.set(`Kᵀ · ${tag}`,Array.from({length:position+1},(_,i)=>column(`K · ${tag}`,i)));
            const scores=Array.from({length:position+1},(_,i)=>dot(column(`K · ${tag}`,i),q)/Math.sqrt(a.qk_dim));
            const weights=softmax(scores),h=values.map(row=>dot(row,weights));
            const raw=this.tensors.get(`Scores · ${tag}`)||[];
            raw.forEach((row,i)=>row.push(scores[i]));raw.push(Array.from({length:position+1},(_,i)=>dot(k,queries.map(row=>row[i]))/Math.sqrt(a.qk_dim)));this.tensors.set(`Scores · ${tag}`,raw);
            const attention=this.tensors.get(`A · ${tag}`)||[];
            attention.forEach((row,i)=>row.push(weights[i]));attention.push([...Array(position).fill(0),weights[position]]);this.tensors.set(`A · ${tag}`,attention);
            heads.push(append(`H · ${tag}`,h));
          }
          const merged=append(`Concatenated heads · ${suffix}`,heads.flat());
          const attention=append(`Attention output · ${suffix}`,mul(w.wo,merged));
          const residual=append(`Residual attention · ${suffix}`,add(x,attention));
          const inputFFN=append(`FFN input · ${suffix}`,norm(residual,w.ffn_norm));
          const z=append(`FFN pre-activation · ${suffix}`,mul(w.ffn_in,inputFFN));
          const activated=append(`GELU · ${suffix}`,z.map(v=>v*.5*(1+this.erf(v/Math.sqrt(2)))));
          const output=append(`FFN output · ${suffix}`,mul(w.ffn_out,activated));
          x=append(`Layer output · ${suffix}`,add(residual,output));
        }
        const hidden=append('Final hidden',norm(x,t.global_weights.final_norm)),logits=append('Logits',mul(t.global_weights.lm_head,hidden));
        append('Vocabulary probabilities',softmax(logits));this.tokens.push(tokens[position]);
      }
      const logits=column('Logits',tokens.length-1),max=Math.max(...logits),top=settings.top_k===0?logits.length:Math.min(logits.length,Math.max(1,Math.trunc(settings.top_k)));
      const cutoff=[...logits].sort((x,y)=>y-x)[top-1],eligible=logits.map(v=>v>=cutoff?1:0);
      let probabilities;
      if(settings.temperature===0)probabilities=logits.map((_,i)=>i===logits.indexOf(max)?1:0);
      else {const exp=logits.map((v,i)=>eligible[i]?Math.exp((v-max)/settings.temperature):0),sum=exp.reduce((s,v)=>s+v,0);probabilities=exp.map(v=>v/sum);}
      const step=settings.samplingStep||0,g=t.generation,noise=step===0&&g&&settings.seed===(g.seed||0)?g.sampling_noise:root.LittleLlamaSamplingNoise(settings.seed,step,logits.length);
      const scores=probabilities.map((v,i)=>v/noise[i]);
      for(const [name,values] of [['Top-k eligibility',eligible],['Top-k candidates',logits.map((v,i)=>eligible[i]?v:'—')],['Top-k probabilities',probabilities],['Sampling noise · saved seeded draw',noise],['Sampling score',scores]])this.tensors.set(name,values.map(v=>[v]));
      this.tensors.set('Temperature · Top-k · Seed',[[settings.temperature],[settings.top_k],[settings.seed]]);
      this.tensors.set('Selected vocabulary index · 1-based',[[scores.indexOf(Math.max(...scores))+1]]);
      return this.tensors;
    }
    gridValues(grid){
      const values=new Map([...grid.cells.values()].map(c=>[`${c.r},${c.c}`,c.v]));
      grid.matrixBlocks.forEach(block=>{
        let tensor=this.tensors.get(block.name),offset=0;
        if(block.cacheSource){
          tensor=this.tensors.get(block.cacheSource);offset=block.cacheOffset||0;
          if(!tensor){const src=grid.matrixBlocks.find(b=>b.name===block.cacheSource);if(src)tensor=Array.from({length:src.rows},(_,r)=>Array.from({length:src.cols},(_,c)=>values.get(`${src.row+r},${src.column+c}`)));}
        }
        if(tensor)for(let r=0;r<block.rows;r++)for(let c=0;c<block.cols;c++)values.set(`${block.row+r},${block.column+c}`,tensor[r+(block.cacheRowOffset||0)][c+offset]);
      });return values;
    }
  }

  function buildCacheGrid(trace,tokens,settings,selection={}){
    const Grid=root.LittleLlamaSequenceGrid||(typeof require!=='undefined'?require('./sequence_generation.js').SequenceGrid:null);
    const grid=new Grid(trace,tokens,settings).build(),n=tokens.length,arch=trace.architecture;
    const layer=Number(selection.layer??trace.layers[0].layer),head=Number(selection.head??0);
    const prefill=Math.max(1,Math.min(n,Math.trunc(selection.prefillTokens??trace.input.tokens.length)));
    if(!trace.layers.some(item=>item.layer===layer)||head<0||head>=arch.num_heads)throw Error('Invalid layer/head selection');
    const sources=new Map();
    for(const cell of grid.cells.values())if(typeof cell.v==='string'&&!sources.has(cell.v))sources.set(cell.v,{ref:(r,c)=>`R${cell.r+1+r}C${cell.c+c}`});
    const sourceTag=`layer ${layer} · head ${head}`,source=name=>sources.get(`${name} · ${sourceTag}`);
    const operation=`ATTENTION • KV cache · prefill → decoding • layer ${layer} • head ${head}`;
    grid.stage(operation);const top=grid.row,steps=[];
    const at=(row,name,rows,cols,col,formula)=>{grid.row=row;return grid.matrix(name,rows,cols,{column:col,formula});};
    const alias=(row,name,rows,cols,col,sourceName,offset=0)=>{
      const src=sources.get(sourceName);
      const matrix=at(row,name,rows,cols,col,(r,c)=>'='+src.ref(r,c+offset));
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:sourceName,cacheOffset:offset});return matrix;
    };
    const inputRow=top+4;
    const keyProjectionRow=inputRow+arch.model_dim+1;
    const valueProjectionRow=keyProjectionRow+arch.qk_dim+1;
    const queryProjectionRow=valueProjectionRow+arch.value_dim+1;
    const attentionRow=queryProjectionRow+arch.qk_dim+2;
    const softmaxRow=attentionRow+n+2;
    const valueRow=softmaxRow+n+2;
    grid.put(softmaxRow-1,2,'KV attention spacing');
    grid.put(valueRow-1,2,'KV attention spacing');
    const projections=[['K',arch.qk_dim,keyProjectionRow],['V',arch.value_dim,valueProjectionRow],['Q',arch.qk_dim,queryProjectionRow]];
    const compact=!!selection.compact;
    const prefillColumn=compact?Math.max(arch.model_dim+6,arch.context_length+3-Math.max(prefill,arch.qk_dim)-1):arch.model_dim+6;
    const prefillRight=prefillColumn+Math.max(prefill,arch.qk_dim)+1;
    const weightColumn=prefillRight-arch.model_dim-1;
    const weightBlocks={};
    for(const [letter,dim,row] of projections){
      weightBlocks[letter]=alias(row,`W${letter} · shared projection · ${sourceTag}`,dim,arch.model_dim,weightColumn,`W${letter} · ${sourceTag}`);
    }
    const sharedKey=compact?alias(attentionRow,`Kᵀ · cache combined · ${sourceTag}`,n,arch.qk_dim,prefillRight-arch.qk_dim-1,`Kᵀ · ${sourceTag}`):null;
    const sharedValue=compact?alias(valueRow,`V · cache combined · ${sourceTag}`,arch.value_dim,n,prefillRight-n-1,`V · ${sourceTag}`):null;
    let column=prefillColumn;
    for(let end=prefill;end<=n;end++){
      const decode=end-prefill,cols=decode?1:prefill,offset=decode?end-1:0;
      const label=decode?`Decode ${decode} · token ${end}`:`Prefill · ${prefill} tokens`;
      const tag=`${sourceTag} · ${decode?'decode '+decode:'prefill'}`;
      const operandWidth=compact&&decode?0:Math.max(cols,arch.qk_dim),right=column+operandWidth+1;
      const step={end,decode,column,right,label,tag,projections:{}};steps.push(step);grid.put(top,column,label);
      const x=alias(inputRow,`x · attention input · ${tag}`,arch.model_dim,cols,right,`Attention input · layer ${layer}`,offset);
      for(const [letter,dim,row] of projections){
        const w=weightBlocks[letter];
        step.projections[letter]=at(row,`${letter} · projection · ${tag}`,dim,cols,right,(r,c)=>'='+Array.from({length:arch.model_dim},(_,j)=>`${w.ref(r,j)}*${x.ref(j,c)}`).join('+'));
        Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`${letter} · ${sourceTag}`,cacheOffset:offset});
      }
      const k=step.projections.K,v=step.projections.V,q=step.projections.Q;
      if(compact){step.kt=sharedKey;step.v=sharedValue;}
      else {
      const kt=at(attentionRow+(decode?end-1:0),`Kᵀ · cache ${decode?'append':'prefill'} · ${tag}`,cols,arch.qk_dim,right-arch.qk_dim-1,(r,c)=>'='+k.ref(c,r));
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`Kᵀ · ${sourceTag}`,cacheRowOffset:offset});step.kt=kt;
      step.v=at(valueRow,`V · cache ${decode?'append':'prefill'} · ${tag}`,arch.value_dim,cols,right-cols-1,(r,c)=>'='+v.ref(r,c));
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`V · ${sourceTag}`,cacheOffset:offset});
      }
      const keyRef=(token,dim)=>compact?sharedKey.ref(token,dim):token<prefill?steps[0].kt.ref(token,dim):steps[token-prefill+1].kt.ref(0,dim);
      const valueRef=(dim,token)=>compact?sharedValue.ref(dim,token):token<prefill?steps[0].v.ref(dim,token):steps[token-prefill+1].v.ref(dim,0);
      const score=(token,c)=>'('+Array.from({length:arch.qk_dim},(_,dim)=>`${keyRef(token,dim)}*${q.ref(dim,c)}`).join('+')+`)/SQRT(${arch.qk_dim})`;
      const raw=at(attentionRow,`Scores · raw attention · ${tag}`,end,cols,right,(r,c)=>'='+score(r,c));
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`Scores · ${sourceTag}`,cacheOffset:offset});
      const attention=at(softmaxRow,`A · attention weights · ${tag}`,end,cols,right,(r,c)=>{
        const last=decode?end-1:c;if(r>last)return '=0';
        const scores=Array.from({length:last+1},(_,token)=>raw.ref(token,c)),max=`MAX(${scores.join(',')})`,exp=token=>`EXP(${scores[token]}-${max})`;
        return '='+exp(r)+'/('+scores.map((_,token)=>exp(token)).join('+')+')';
      });
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`A · ${sourceTag}`,cacheOffset:offset});
      at(valueRow,`VA · attention output · ${tag}`,arch.value_dim,cols,right,(r,c)=>'='+Array.from({length:end},(_,token)=>`${valueRef(r,token)}*${attention.ref(token,c)}`).join('+'));
      Object.assign(grid.matrixBlocks.at(-1),{cacheSource:`H · ${sourceTag}`,cacheOffset:offset});
      column=right+cols+1;
    }
    if(compact)for(let token=0;token<n;token++){
      const step=token<prefill?steps[0]:steps[token-prefill+1],c=token<prefill?token:0;
      for(let r=0;r<arch.qk_dim;r++)grid.put(sharedKey.row+token,sharedKey.column+r,0,'='+step.projections.K.ref(r,c));
      for(let r=0;r<arch.value_dim;r++)grid.put(sharedValue.row+r,sharedValue.column+token,0,'='+step.projections.V.ref(r,c));
    }
    for(const [key,cell] of grid.cells)if(cell.v==='Navigate to block')grid.cells.delete(key);
    grid.width=Math.max(grid.width,column);grid.put(1,grid.width+3,'Navigate to block');
    grid.row=valueRow+arch.value_dim+3;
    grid.stages.forEach(([r],i)=>grid.put(r,1,`[${String(i+1).padStart(2,'0')}/${String(grid.stages.length).padStart(2,'0')}]`));
    return {grid,operations:[operation],steps,prefill,top};
  }

  root.LittleLlamaBuildCacheGrid=buildCacheGrid;
  if(typeof module!=='undefined')module.exports={buildCacheGrid,CacheInference};
  if(typeof document==='undefined')return;
  const data=document.getElementById('sequence-generation-data');if(!data)return;
  const bundle=JSON.parse(data.textContent),trace=bundle.trace,workbook=root.LittleLlamaWorkbook;
  if(!workbook)return;
  const tabs=document.querySelector('.stage-view-tabs'),tab=document.createElement('button');
  tab.type='button';tab.textContent='KV Cache';tab.setAttribute('aria-pressed','false');tabs.append(tab);
  const frame=document.createElement('iframe');frame.className='kv-native-view';frame.title='KV cache stages';frame.hidden=true;document.body.append(frame);
  const style=document.createElement('style');style.textContent='.kv-native-view{position:fixed;inset:0 0 30px;width:100%;height:calc(100% - 30px);border:0;z-index:190;background:#fff}.kv-native-view[hidden]{display:none}body.kv-native-active .workbook-toolbar,body.kv-native-active .sampling-chart-toggle{display:none!important}body.kv-native-active .sequence-editor{z-index:200}';document.head.append(style);
  const normalize=x=>({'␣':' ','<space>':' ','\\n':'\n','\\t':'\t'}[x]??x);
  const bar=document.querySelector('.sequence-editor');
  const layerSelect=document.createElement('select'),headSelect=document.createElement('select');
  layerSelect.setAttribute('aria-label','KV cache layer');headSelect.setAttribute('aria-label','KV cache head');
  trace.layers.forEach(item=>layerSelect.add(new Option(`Layer ${item.layer}`,item.layer)));
  for(let h=0;h<trace.architecture.num_heads;h++)headSelect.add(new Option(`Head ${h}`,h));
  const selectors=document.createElement('span');selectors.className='kv-head-selectors';selectors.append(layerSelect,headSelect);selectors.hidden=true;bar.append(selectors);
  const saved=bundle.kvState||{};
  const compactLabel=document.createElement('label'),compactToggle=document.createElement('input');
  compactToggle.type='checkbox';compactToggle.checked=saved.compact??true;compactToggle.setAttribute('aria-label','Compact KV cache view');compactLabel.append(compactToggle,document.createTextNode('Compact'));selectors.append(compactLabel);compactToggle.onchange=()=>{memorySelection=null;changeSelection();};
  layerSelect.value=saved.layer??trace.layers[0].layer;headSelect.value=saved.head??0;
  let prefillTokens=saved.prefillTokens??trace.input.tokens.length;
  let lastKey=null,pendingView=saved.view||null,frameReady=false,prediction=null,dirty=false,layoutTimer=0;
  const deleted=[];
  const inference=new CacheInference(trace,workbook.erf);
  style.textContent+='.kv-head-selectors{display:flex;gap:4px;margin-left:8px}.kv-head-selectors select{font:11px system-ui;color:#56718f;background:#f4f7fb;border:1px solid #dce4ef;border-radius:5px;padding:4px}';
  let memoryPrecision=4,memorySelection=null,memoryWidth=saved.memoryWidth??null,showMemoryValues=saved.showMemoryValues??false;
  function updateMemory(doc,tokens,selection){
    let panel=doc.querySelector('.kv-memory-panel');
    if(!panel){
      const css=doc.createElement('style');css.textContent=`
      .kv-memory-panel{position:fixed;right:10px;top:112px;bottom:48px;width:310px;z-index:105;overflow:auto;background:#f8fafdee;border:1px solid #dce4ef;border-radius:8px;padding:12px;font:11px/1.4 system-ui;color:#465365;box-sizing:border-box;backdrop-filter:blur(8px)}
      .kv-memory-panel h3{font-size:13px;margin:0}.kv-memory-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}.kv-memory-size{border:1px solid #dce4ef;background:#fff;color:#465365;border-radius:4px;padding:3px 7px;font:11px system-ui;cursor:pointer}.kv-memory-panel.kv-memory-expanded{width:min(650px,calc(100vw - 270px))}.kv-memory-panel h4{margin:14px 0 5px}.kv-memory-panel select{font:inherit;margin-left:8px}.kv-memory-summary{margin:8px 0}.kv-memory-array{overflow:auto;padding:2px 0 5px}.kv-memory-table{border-collapse:collapse;width:max-content}.kv-memory-table td,.kv-memory-table th{padding:0;border:0}.kv-memory-table button{border:1px solid #fff8;padding:2px;width:28px;height:16px;font:10px ui-monospace,monospace;cursor:pointer;color:#243447}.kv-values-visible .kv-memory-table button{width:38px;height:20px}.kv-memory-value-toggle{display:flex;align-items:center;gap:5px;margin-top:7px;cursor:pointer}.kv-memory-table th button{background:transparent;font:10px system-ui}.kv-memory-table .kv-byte{white-space:nowrap;font-size:8px;padding:1px;background:transparent;font:8px system-ui;color:#56718f}.kv-memory-table .kv-decode-start{border-left:2px solid #68788a}.kv-memory-table button[aria-pressed=true]{outline:2px solid #238b8b;outline-offset:-2px}.kv-memory-source{outline:2px solid #238b8b;outline-offset:-2px}.kv-memory-links{position:fixed;inset:0;width:100vw;height:100vh;z-index:106;pointer-events:none}.kv-memory-resize{position:fixed;width:10px;z-index:107;cursor:ew-resize;touch-action:none;border:0;background:transparent;padding:0}.kv-memory-resize:hover,.kv-memory-resize:focus{background:#9ab6cc55;outline:none}.kv-memory-note{font-size:10px;color:#718097;margin-top:12px}
      @media(max-width:760px){.kv-memory-panel{width:230px;top:150px}}`;
      doc.head.append(css);panel=doc.createElement('aside');panel.className='kv-memory-panel';panel.setAttribute('aria-label','KV tensor memory');doc.body.append(panel);
      const svg=doc.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('kv-memory-links');doc.body.append(svg);
      const handle=doc.createElement('div');handle.className='kv-memory-resize';handle.tabIndex=0;handle.setAttribute('role','separator');handle.setAttribute('aria-label','Resize memory panel');handle.setAttribute('aria-orientation','vertical');doc.body.append(handle);
      const clampWidth=value=>Math.max(Math.min(240,doc.defaultView.innerWidth-30),Math.min(doc.defaultView.innerWidth-Math.min(260,doc.defaultView.innerWidth*.3),value));
      const resize=value=>{memoryWidth=clampWidth(value);panel.style.width=memoryWidth+'px';handle.setAttribute('aria-valuenow',String(Math.round(memoryWidth)));panel.drawMemoryLinks?.();};
      handle.addEventListener('pointerdown',event=>{
        event.preventDefault();event.stopPropagation();handle.setPointerCapture(event.pointerId);
        const start=event.clientX,width=panel.getBoundingClientRect().width;
        const move=e=>resize(width+start-e.clientX);
        const finish=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',finish);handle.removeEventListener('pointercancel',finish);};
        handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',finish);handle.addEventListener('pointercancel',finish);
      });
      handle.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();event.stopPropagation();resize(panel.getBoundingClientRect().width+(event.key==='ArrowLeft'?20:-20));});
      doc.defaultView.addEventListener('resize',()=>{if(memoryWidth!==null)resize(memoryWidth);else panel.drawMemoryLinks?.();});
      const draw=()=>{
        const panelRect=panel.getBoundingClientRect();handle.style.left=(panelRect.left-5)+'px';handle.style.top=panelRect.top+'px';handle.style.height=panelRect.height+'px';
        svg.replaceChildren();doc.querySelectorAll('.kv-memory-source').forEach(c=>c.classList.remove('kv-memory-source'));
        panel.querySelectorAll('[data-kind]').forEach(b=>b.setAttribute('aria-pressed',String(!!memorySelection&&b.dataset.kind===memorySelection.kind&&Number(b.dataset.token)===memorySelection.token&&(b.dataset.dim==='all'?null:Number(b.dataset.dim))===memorySelection.dim)));
        if(!memorySelection)return;
        const sharedClear=doc.querySelector('.selection-clear');if(sharedClear)sharedClear.hidden=false;
        const {kind,token,dim}=memorySelection,decode=token>=prefillTokens;
        const tag=`layer ${layerSelect.value} · head ${headSelect.value} · ${decode?'decode '+(token-prefillTokens+1):'prefill'}`;
        const compact=!!doc.querySelector('[data-kv-compact="true"]');
        const name=compact?`${kind==='K'?'Kᵀ':'V'} · cache combined · layer ${layerSelect.value} · head ${headSelect.value}`:`${kind==='K'?'Kᵀ':'V'} · cache ${decode?'append':'prefill'} · ${tag}`;
        const points=dim===null?Array.from({length:kind==='K'?trace.architecture.qk_dim:trace.architecture.value_dim},(_,i)=>i):[dim];
        for(const i of points){
          const row=kind==='K'?(compact?token+1:decode?1:token+1):i+1,col=kind==='K'?i+1:(compact?token+1:decode?1:token+1);
          const source=[...doc.querySelectorAll('.stage-card[data-kv-timeline] .matrix-value')].find(c=>c.dataset.matrixName===name&&Number(c.dataset.matrixRow)===row&&Number(c.dataset.matrixColumn)===col);
          const target=panel.querySelector(`[data-kind="${kind}"][data-token="${token}"][data-dim="${i}"]`);
          if(!source||!target)continue;source.classList.add('kv-memory-source');
          const a=source.getBoundingClientRect(),b=target.getBoundingClientRect(),clip=target.closest('.kv-memory-array').getBoundingClientRect();
          if(b.right>clip.right+1||b.left<clip.left-1)continue;
          const path=doc.createElementNS(svg.namespaceURI,'path');const x=a.right,y=a.top+a.height/2,endX=b.left,endY=b.top+b.height/2;
          path.setAttribute('d',`M ${x} ${y} C ${x+40} ${y}, ${endX-40} ${endY}, ${endX} ${endY}`);path.setAttribute('fill','none');path.setAttribute('stroke','#238b8b');path.setAttribute('stroke-width','1.3');path.setAttribute('opacity','.65');svg.append(path);
        }
      };
      const clearMemory=()=>{memorySelection=null;draw();const shared=doc.querySelector('.selection-clear');if(shared&&!doc.querySelector('.math-selected'))shared.hidden=true;};
      panel.clearMemory=clearMemory;
      doc.addEventListener('click',event=>{if(event.target.closest('.selection-clear'))clearMemory();});
      doc.addEventListener('keydown',event=>{if(event.key==='Escape')clearMemory();});
      panel.addEventListener('click',event=>{const button=event.target.closest('[data-kind]');if(!button)return;
        const next={kind:button.dataset.kind,token:Number(button.dataset.token),dim:button.dataset.dim==='all'?null:Number(button.dataset.dim)};
        if(memorySelection&&memorySelection.kind===next.kind&&memorySelection.token===next.token&&memorySelection.dim===next.dim){clearMemory();return;}
        memorySelection=next;draw();
        const sources=[...doc.querySelectorAll('.kv-memory-source')],edge=panel.getBoundingClientRect().left;
        if(sources.some(c=>{const r=c.getBoundingClientRect();return r.left<245||r.right>edge||r.top<110||r.bottom>doc.defaultView.innerHeight-45;})){
          doc.defaultView.LittleLlamaWorkbook.focusCells(sources,{preserveZoom:true,smooth:true});doc.defaultView.requestAnimationFrame(draw);
        }
      });
      panel.addEventListener('change',event=>{if(event.target.matches('select')){memoryPrecision=Number(event.target.value);updateMemory(doc,[...bar.querySelectorAll('.sequence-token')].map(i=>normalize(i.value)),{layer:Number(layerSelect.value),head:Number(headSelect.value)});}});
      panel.addEventListener('scroll',draw,true);doc.defaultView.addEventListener('latte:annotations',draw);panel.drawMemoryLinks=draw;
    }
    const addedToken=tokens.length>Number(panel.dataset.tokenCount||0);
    panel.dataset.tokenCount=tokens.length;
    const scrolls=[...panel.querySelectorAll('.kv-memory-array')].map(el=>el.scrollLeft);
    const a=trace.architecture,n=tokens.length,perToken=(a.qk_dim+a.value_dim)*memoryPrecision,total=n*perToken;
    panel.classList.toggle('kv-values-visible',showMemoryValues);
    panel.style.width=memoryWidth===null?'':memoryWidth+'px';
    panel.replaceChildren();const heading=doc.createElement('h3');heading.textContent='KV tensor memory';
    const header=doc.createElement('div');header.className='kv-memory-header';
    header.append(heading);panel.append(header);
    const controls=doc.createElement('div');controls.textContent=`Layer ${selection.layer} · Head ${selection.head}`;
    const precision=doc.createElement('select');precision.setAttribute('aria-label','KV memory precision');for(const [label,bytes] of [['FP32',4],['FP16 / BF16',2],['INT8',1]]){const option=doc.createElement('option');option.textContent=label;option.value=bytes;precision.append(option);}precision.value=memoryPrecision;controls.append(precision);panel.append(controls);
    const valueLabel=doc.createElement('label');valueLabel.className='kv-memory-value-toggle';const valueToggle=doc.createElement('input');valueToggle.type='checkbox';valueToggle.checked=showMemoryValues;valueToggle.onchange=()=>{showMemoryValues=valueToggle.checked;updateMemory(doc,tokens,selection);};valueLabel.append(valueToggle,doc.createTextNode('Show values'));panel.append(valueLabel);
    const summary=doc.createElement('div');summary.className='kv-memory-summary';summary.textContent=`Selected head: ${total.toLocaleString()} B`;panel.append(summary);

    for(const [kind,dim,base] of [['K',a.qk_dim,[138,219,139]],['V',a.value_dim,[223,157,215]]]){
      const title=doc.createElement('h4');title.textContent=`${kind} · ${n*dim*memoryPrecision} B`;panel.append(title);
      const wrap=doc.createElement('div');wrap.className='kv-memory-array';const table=doc.createElement('table');table.className='kv-memory-table';wrap.append(table);panel.append(wrap);
      const data=inference.tensors.get(`${kind} · layer ${selection.layer} · head ${selection.head}`),max=Math.max(1,...data.flat().map(Math.abs));
      for(let r=-1;r<=dim;r++){
        const tr=doc.createElement('tr');table.append(tr);
        for(let token=0;token<n;token++){
          const td=doc.createElement(r===-1?'th':'td'),button=doc.createElement('button');button.type='button';button.dataset.kind=kind;button.dataset.token=token;button.dataset.dim=r<0||r===dim?'all':r;button.setAttribute('aria-pressed','false');
          if(token>=prefillTokens)button.classList.add('kv-decode-start');
          const label=token<prefillTokens?`Prefill token ${token+1}`:`Decode ${token-prefillTokens+1}`;
          if(r===-1)button.textContent=token<prefillTokens?`P${token+1}`:`D${token-prefillTokens+1}`;
          else if(r===dim){button.textContent=`+${dim*memoryPrecision}\u00a0B`;button.classList.add('kv-byte');}
          else{const value=data[r][token],strength=.12+.88*Math.min(1,Math.abs(value)/max);button.textContent=showMemoryValues?(Math.abs(value)<.005?0:value).toFixed(2):'';button.style.backgroundColor=`rgb(${base.map(c=>Math.round(255+(c-255)*strength)).join(',')})`;}
          button.title=`${kind} · ${label} · ${r>=0&&r<dim?'feature '+(r+1)+' · value '+data[r][token]+' · '+memoryPrecision:dim*memoryPrecision} bytes. Click to connect to diagram.`;
          button.setAttribute('aria-label',button.title);td.append(button);tr.append(td);
        }
      }
      wrap.scrollLeft=addedToken?wrap.scrollWidth:(scrolls[kind==='K'?0:1]||0);
    }
    const note=doc.createElement('div');note.className='kv-memory-note';note.textContent='Click a value or token column to connect it to the diagram.';panel.append(note);panel.drawMemoryLinks();
  }
  function changeSelection(){lastKey=null;pendingView=null;render(true);}
  layerSelect.onchange=headSelect.onchange=changeSelection;
  function view(){const w=frame.contentWindow?.LittleLlamaWorkbook,v=w?.captureView?.();return v?{zoom:v.zoom,pan:v.pan}:null;}
  function render(resetView=false){
    const tokens=[...document.querySelectorAll('.sequence-token')].map(i=>normalize(i.value));
    if(!tokens.length||tokens.some(t=>!trace.vocabulary.includes(t)))return;
    const controls=[...document.querySelectorAll('.generation-control input')];
    const settings={temperature:Number(controls[0]?.value??1),top_k:Number(controls[1]?.value??0),seed:Number(controls[2]?.value??0),samplingStep:root.LittleLlamaSequenceSession.step()};
    prefillTokens=Math.min(prefillTokens,tokens.length);
    const selection={layer:Number(layerSelect.value),head:Number(headSelect.value),prefillTokens,compact:compactToggle.checked};
    const key=JSON.stringify({tokens,settings,selection});if(key===lastKey)return;lastKey=key;
    pendingView=resetView?null:(view()||pendingView);
    const {grid,operations,steps,dividerRow,top}=buildCacheGrid(trace,tokens,settings,selection);
    inference.run(tokens,settings);
    const values=inference.gridValues(grid);
    const selected=[...grid.cells.values()].find(cell=>cell.v==='Selected vocabulary index · 1-based');
    prediction=Number(values.get(`${selected.r+1},${selected.c}`))-1;
    if(!Number.isInteger(prediction)||!trace.vocabulary[prediction])throw Error('The generated token is unavailable.');
    const next=trace.vocabulary[prediction],display=next===' '?'␣':next==='\n'?'<newline>':next==='\t'?'<tab>':next;
    bar.querySelector('.generated-token-output').textContent=`→ Generated: ${display}`;
    const css='<style>.matrix-layout td{grid-row:1}.matrix-layout tr:has(> .matrix-value) > .matrix-label{display:none!important}.matrix-layout td.matrix-value{height:28px!important;min-height:28px!important;align-self:end}.matrix-layout td.kv-appended-key[data-column-index]{margin-top:0!important}.matrix-value[data-kv-color]{color:#243447!important}.stage-view-tabs,.sequence-editor,.sampling-chart-toggle,.toolbar-group:has([data-action="sequence"]){display:none!important}.matrix-layout tr.kv-section-gap{display:grid!important;height:6px!important;min-height:6px!important}.matrix-layout tr.kv-section-gap td{display:none!important}.matrix-layout td.kv-masked-cell{visibility:visible!important;background:transparent!important;color:transparent!important;border-color:transparent!important}.matrix-layout td[data-kv-top]{border-top:1px solid #68788a!important}.matrix-layout td[data-kv-bottom]{border-bottom:1px solid #68788a!important}.matrix-layout td[data-kv-left]{border-left:1px solid #68788a!important}.matrix-layout td[data-kv-right]{border-right:1px solid #68788a!important}.matrix-layout td[data-kv-color]{border-radius:0}.matrix-layout td.kv-cache-boundary[data-kv-color=key]{border-top:2px solid #68788a!important}.matrix-layout td.kv-cache-boundary[data-kv-color=value]{border-left:2px solid #68788a!important}.matrix-layout tr.kv-step-titles{display:grid!important;min-height:32px}.matrix-layout td.kv-step-title{visibility:visible!important;opacity:0;font-size:13px;font-weight:600;color:#56718f}.matrix-layout td.kv-step-title font{font:inherit!important}.matrix-layout td.kv-step-divider{visibility:visible!important;border-left:1px solid #b8c4d6!important}.matrix-layout tr.kv-cache-divider{display:grid!important;min-height:52px;border:0;position:relative}.matrix-layout tr.kv-cache-divider::before{content:"";position:absolute;left:0;right:0;top:32px;height:1px;background:#879bb4;pointer-events:none}.matrix-layout tr.kv-cache-divider td{visibility:visible!important}</style>';
    const decorate=(doc)=>{
      const card=[...doc.querySelectorAll('.stage-card')].find(c=>c.dataset.operation===operations[0]);
      card.dataset.kvTimeline='true';
      card.dataset.kvCompact=String(!!selection.compact);
      const tokenText=token=>token===' '?'␣':token==='\n'?'\\n':token==='\t'?'\\t':token;
      card.dataset.kvSteps=JSON.stringify(steps.map(({tag,label,decode,end})=>({tag,label,hoverTitle:decode?`Position ${end}`:label,hoverDecode:!!decode,hoverTokens:(decode?[tokens[end-1]]:tokens.slice(0,prefillTokens)).map(tokenText)})));
      card.querySelectorAll('.matrix-value').forEach(cell=>{
        const name=cell.dataset.matrixName;
        cell.classList.toggle('kv-appended-key',name.startsWith('Kᵀ · cache append'));
        const block=grid.matrixBlocks.find(block=>block.name===name);
        const row=Number(cell.dataset.matrixRow),col=Number(cell.dataset.matrixColumn);
        const causal=name.startsWith('A ·')&&name.endsWith(' · prefill');
        const visible=!causal||row<=col;
        const edges=[['top',row===1],['bottom',causal?row===col:row===block.rows],['left',causal?row===col:col===1],['right',col===block.cols]];
        for(const [edge,on] of edges){
          if(visible&&on)cell.setAttribute('data-kv-'+edge,'');else cell.removeAttribute('data-kv-'+edge);
        }
        cell.classList.toggle('kv-cache-boundary',!!selection.compact&&name.includes('cache combined')&&(name.startsWith('Kᵀ')?row===prefillTokens+1:col===prefillTokens+1));
        cell.dataset.kvColor=/^W[KQV] ·/.test(name)?'weight':/^Q ·/.test(name)?'query':/^K/.test(name)?'key':/^VA ·/.test(name)?'output':/^V/.test(name)?'value':/^(A ·|Scores ·)/.test(name)?'attention':'output';
      });
      for(const step of steps){
        const title=[...card.querySelectorAll('.matrix-label')].find(cell=>cell.textContent.trim()===step.label);
        if(title){title.classList.add('kv-step-title');title.style.gridColumn=`${step.column} / span ${step.right-step.column+(step.decode?1:step.end)}`;title.closest('tr').classList.add('kv-step-titles');}
      }
      card.querySelectorAll('.matrix-value').forEach(cell=>{
        const name=cell.dataset.matrixName,r=Number(cell.dataset.matrixRow),c=Number(cell.dataset.matrixColumn);
        cell.classList.remove('kv-new-cell');
        if(name.startsWith('A ·')&&name.endsWith(' · prefill')&&r>c)cell.classList.add('kv-masked-cell');
      });
      card.querySelectorAll('.matrix-label').forEach(cell=>{if(cell.textContent==='KV attention spacing')cell.closest('tr').classList.add('kv-section-gap');});
      frame.contentWindow.LittleLlamaWorkbook.refreshColors();
      updateMemory(doc,tokens,selection);
    };
    if(frameReady&&!resetView){
      const doc=frame.contentDocument,w=frame.contentWindow.LittleLlamaWorkbook;
      const animateUpdate=w.updateCacheGrid(grid,values,{top,operation:operations[0]});
      decorate(doc);
      animateUpdate?.();
      frame.contentWindow.dispatchEvent(new frame.contentWindow.CustomEvent('latte:layout',{detail:{preserveCamera:true}}));
      // Measure bracket endpoints again after the cell transitions settle.
      frame.contentWindow.clearTimeout(layoutTimer);
      layoutTimer=frame.contentWindow.setTimeout(()=>frame.contentWindow.dispatchEvent(new frame.contentWindow.CustomEvent('latte:layout',{detail:{preserveCamera:true}})),200);
      pendingView=null;return;
    }
    frameReady=false;
    frame.onload=()=>{
      const doc=frame.contentDocument,w=frame.contentWindow.LittleLlamaWorkbook;
      w.restoreStageSelection({stages:operations,view:'selected'});
      decorate(doc);
      frame.contentWindow.addEventListener('latte:annotations',()=>frame.contentWindow.queueMicrotask(()=>{
        doc.querySelectorAll('.annotation-overlay text').forEach(text=>{
          const label=text.firstChild;
          if(label?.nodeType===3)label.nodeValue=label.nodeValue.replace(/ · (prefill|decode \d+)$/,'').replace('Scores · raw attention','Scores').replace('A · attention weights','A · softmax').replace('VA · attention output','VA');
        });
      }));
      doc.querySelectorAll('[contenteditable]').forEach(cell=>cell.contentEditable='false');
      if(pendingView)w.restoreView({...pendingView,sheet:doc.querySelector('.worksheet.active')});
      else {
        if(selection.compact){const current=w.captureView?.();if(current)w.restoreView({...current,zoom:.55});}
        w.recenter?.();
      }
      pendingView=null;frameReady=true;
    };
    frame.srcdoc=bundle.shell.replace('__FORWARD_TABLE__',()=>grid.table()).replace('__GENERATION_CONTROLS__','').replace('__SEQUENCE_SCRIPT__','').replace('</head>',css+'</head>');
  }
  tab.onclick=()=>{frame.hidden=false;selectors.hidden=false;document.body.classList.add('kv-native-active');document.querySelector('.sequence-editor').classList.remove('hidden');tabs.querySelector('.steps-menu').open=false;tabs.querySelector('.steps-menu').inert=true;tabs.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===tab));tab.setAttribute('aria-pressed','true');render();};
  tabs.addEventListener('click',event=>{const button=event.target.closest('button');if(button&&button!==tab){frame.hidden=true;selectors.hidden=true;document.body.classList.remove('kv-native-active');tabs.querySelector('.steps-menu').inert=false;tab.classList.remove('active');tab.setAttribute('aria-pressed','false');if(dirty)root.LittleLlamaSequenceSession.refresh(button.textContent);}});
  document.addEventListener('latte:recalculated',()=>{if(!frame.hidden)render();});
  function replaceTokens(tokens){
    const inputs=[...bar.querySelectorAll('.sequence-token')];
    while(inputs.length>tokens.length)inputs.pop().remove();
    tokens.forEach((token,i)=>{
      let input=inputs[i];
      if(!input){input=document.createElement('input');input.className='sequence-token';input.setAttribute('aria-label',`Sequence token ${i+1}`);bar.insertBefore(input,bar.querySelector('.generated-token-output'));input.onfocus=()=>input.select();input.onclick=()=>input.select();inputs.push(input);}
      input.value=token===' '?'␣':token==='\n'?'\\n':token==='\t'?'\\t':token;
    });
    dirty=true;lastKey=null;render();root.LittleLlamaSequenceSession.controls();
    bar.querySelector('[aria-label="Undo last token deletion"]').disabled=!deleted.length;
  }
  bar.addEventListener('input',event=>{if(!frame.hidden&&event.target.matches('.sequence-token')){lastKey=null;render();}});
  bar.addEventListener('keydown',event=>{
    if(frame.hidden||!event.target.matches('.sequence-token')||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    const inputs=[...bar.querySelectorAll('.sequence-token')],index=inputs.indexOf(event.target),target=event.key==='Home'?0:event.key==='End'?inputs.length-1:index+(event.key==='ArrowLeft'?-1:1);
    if(inputs[target]){event.preventDefault();event.stopPropagation();inputs[target].focus();inputs[target].select();}
  },true);
  root.LittleLlamaKVCache={
    active:()=>!frame.hidden,
    generate(){
      try{
        if(!frameReady)return;
        const tokens=[...bar.querySelectorAll('.sequence-token')].map(input=>normalize(input.value));
        if(tokens.length>=trace.architecture.context_length)return;
        render();if(!tokens.every(token=>trace.vocabulary.includes(token)))throw Error('Enter valid vocabulary tokens first.');
        const next=trace.vocabulary[prediction];deleted.length=0;
        root.LittleLlamaSequenceSession.setStep(root.LittleLlamaSequenceSession.step()+1);
        replaceTokens([...tokens,next]);
      }catch(error){root.LittleLlamaSequenceSession.status(error.message);}
    },
    remove(){const tokens=[...bar.querySelectorAll('.sequence-token')].map(input=>normalize(input.value));if(tokens.length<2)return;deleted.push({tokens,step:root.LittleLlamaSequenceSession.step()});root.LittleLlamaSequenceSession.setStep(Math.max(0,root.LittleLlamaSequenceSession.step()-1));replaceTokens(tokens.slice(0,-1));},
    undo(){const previous=deleted.pop();if(!previous)return;root.LittleLlamaSequenceSession.setStep(previous.step);replaceTokens(previous.tokens);},
    state:()=>({active:!frame.hidden,view:view()||pendingView,layer:Number(layerSelect.value),head:Number(headSelect.value),prefillTokens,memoryWidth,showMemoryValues,compact:compactToggle.checked}),
  };
  if(bundle.kvState?.active)tab.click();
})(typeof globalThis!=='undefined'?globalThis:this);
