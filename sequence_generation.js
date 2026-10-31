/* Grow the existing formula explorer by one token, retaining its original UI. */
(function(root) {
  'use strict';
  const esc = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const label = x => x==='\n'?'\\n':x==='\t'?'\\t':x;
  const matrixKey = cell => `${cell.dataset.matrixName}|${cell.dataset.matrixRow}|${cell.dataset.matrixColumn}`;
  function anchoredPan(pan, zoom, before, after) {
    return {x:pan.x+(before.x-after.x)/zoom,y:pan.y+(before.y-after.y)/zoom};
  }
  // Mulberry32: advance a deterministic stream, then transform uniform draws
  // into independent Exp(1) values for this generation step.
  function samplingNoise(seed, step, count) {
    let state=seed>>>0;
    const random=()=>{
      state=(state+0x6D2B79F5)>>>0;
      let value=Math.imul(state^(state>>>15),1|state);
      value^=value+Math.imul(value^(value>>>7),61|value);
      return ((value^(value>>>14))>>>0)+.5;
    };
    for(let i=0;i<step*count;i++)random();
    return Array.from({length:count},()=>-Math.log(random()/4294967296));
  }
  class SequenceGrid {
    constructor(trace,tokens,settings) {
      this.trace=trace;this.tokens=tokens;this.settings=settings;this.n=tokens.length;this.row=3;this.cells=new Map();this.stages=[];this.matrixBlocks=[];
      const a=trace.architecture;
      this.right=Math.max(a.model_dim,a.qk_dim,a.value_dim,a.ffn_dim,a.num_heads*a.value_dim,this.n)+3;
      this.width=this.right+Math.max(this.n,a.model_dim,5);
      this.put(1,1,'Latte');
    }
    put(r,c,v,f=null,w=false) {this.cells.set(`${r},${c}`,{r,c,v,f,w});}
    stage(title) {this.stages.push([this.row,title]);this.put(this.row,2,title);this.row++;}
    matrix(name,rows,cols,{column=this.right,values=null,formula=null,weight=false,tokens=false}={}) {
      this.put(this.row++,column,name);
      if(tokens) {this.tokens.forEach((t,j)=>{this.put(this.row,column+j,label(t));if(this.trace.training)this.cells.get(`${this.row},${column+j}`).tokenHeader=true;});this.row++;}
      const block={row:this.row,column,rows,cols,ref:(i,j)=>`R${block.row+i}C${column+j}`};
      this.matrixBlocks.push({...block,name,weight});
      for(let i=0;i<rows;i++)for(let j=0;j<cols;j++)this.put(this.row+i,column+j,values?values[i][j]:0,formula?formula(i,j):null,weight);
      this.row+=rows+1;return block;
    }
    pair(name,weight,result,rows,formula) {
      const values=Array.isArray(weight[0])?weight:weight.map(v=>[v]), start=this.row;
      const left=this.matrix(name,values.length,values[0].length,{values,weight:true,column:this.right-1-values[0].length}),end=this.row;
      this.row=start;const right=this.matrix(result,rows,this.n,{formula:(i,j)=>formula(left,i,j)});this.row=Math.max(end,this.row);return right;
    }
    mul(w,x,name,result) {return this.pair(name,w,result,w.length,(left,i,j)=>'='+w[0].map((_,k)=>`${left.ref(i,k)}*${x.ref(k,j)}`).join('+'));}
    norm(x,g,title,name) {
      this.stage(title);return this.pair('RMS scale · '+name,g,name,g.length,(w,i,j)=>`=${x.ref(i,j)}*${w.ref(i,0)}/SQRT((`+g.map((_,k)=>`${x.ref(k,j)}^2`).join('+')+`)/${g.length}+1.1920928955078125e-07)`);
    }
    add(x,y,title,name) {this.stage(title);return this.matrix(name,x.rows,this.n,{formula:(i,j)=>`=${x.ref(i,j)}+${y.ref(i,j)}`});}
    build() {
      const t=this.trace,n=this.n;
      this.stage('EMBEDDING • Token + position embedding');
      const token=this.matrix('Token embeddings',t.architecture.model_dim,n,{tokens:true,weight:true,formula:(i,j)=>{
        if(t.training)return '='+t.global_weights.token_embedding[t.vocabulary.indexOf(this.tokens[j])][i];
        let value='NA()';for(let k=t.vocabulary.length-1;k>=0;k--)value=`IF(EXACT(R5C,Embeddings!R${k+6}C1),Embeddings!R${k+6}C${i+2},${value})`;return '='+value;
      }});
      const positions=this.matrix('Position embeddings',token.rows,n,{weight:true,values:Array.from({length:token.rows},(_,i)=>Array.from({length:n},(_,j)=>t.global_weights.position_embedding[j][i]))});
      let x=this.matrix('X · embedding output',token.rows,n,{formula:(i,j)=>`=${token.ref(i,j)}+${positions.ref(i,j)}`});
      t.layers.forEach(layer=>{
        const w=layer.weights,suffix=`layer ${layer.layer}`;
        const norm=this.norm(x,w.attention_norm,`ATTENTION • RMSNorm • ${suffix}`,`Attention input · ${suffix}`),heads=[];
        for(let head=0;head<t.architecture.num_heads;head++) {
          const tag=`${suffix} · head ${head}`,p={};
          for(const [letter,name] of [['k','Key'],['v','Value'],['q','Query']]) {
            this.stage(`ATTENTION • ${name} projection • ${suffix} • head ${head}`);
            p[letter]=this.mul(w[`w${letter}_by_head`][head],norm,`W${letter.toUpperCase()} · ${tag}`,`${letter.toUpperCase()} · ${tag}`);
          }
          const {k,q,v}=p;
          this.stage(`ATTENTION • Scores KᵀQ / √dₖ • ${suffix} • head ${head}`);
          let start=this.row;
          const kt=this.matrix(`Kᵀ · ${tag}`,n,k.rows,{column:this.right-1-k.rows,formula:(i,j)=>'='+k.ref(j,i)}),end=this.row;
          this.row=start;
          const scores=this.matrix(`Scores · ${tag}`,n,n,{formula:(i,j)=>'=('+Array.from({length:q.rows},(_,a)=>`${kt.ref(i,a)}*${q.ref(a,j)}`).join('+')+`)/SQRT(${q.rows})`});
          this.row=Math.max(end,this.row);
          this.stage(`ATTENTION • Causal mask + softmax • ${suffix} • head ${head}`);
          const a=this.matrix(`A · ${tag}`,n,n,{formula:(i,j)=>{
            if(i>j)return '=0';const max=`MAX(${scores.ref(0,j)}:${scores.ref(j,j)})`,exp=b=>`EXP(${scores.ref(b,j)}-${max})`;
            return '='+exp(i)+'/('+Array.from({length:j+1},(_,b)=>exp(b)).join('+')+')';
          }});
          this.stage(`ATTENTION • V · A (weighted value aggregation) • ${suffix} • head ${head}`);
          start=this.row;
          const left=this.matrix(`V · ${tag}`,v.rows,n,{column:this.right-1-n,formula:(i,j)=>'='+v.ref(i,j)}),vEnd=this.row;
          this.row=start;
          heads.push(this.matrix(`H · ${tag}`,v.rows,n,{formula:(i,j)=>'='+Array.from({length:n},(_,b)=>`${left.ref(i,b)}*${a.ref(b,j)}`).join('+')}));this.row=Math.max(vEnd,this.row);
        }
        this.stage(`ATTENTION • Concatenate heads • ${suffix}`);
        const merged=this.matrix(`Concatenated heads · ${suffix}`,heads.length*heads[0].rows,n,{formula:(i,j)=>'='+heads[Math.floor(i/heads[0].rows)].ref(i%heads[0].rows,j)});
        this.stage(`ATTENTION • Output projection WO • ${suffix}`);
        const output=this.mul(w.wo,merged,`WO · ${suffix}`,`Attention output · ${suffix}`);
        const residual=this.add(x,output,`RESIDUAL • Add attention output • ${suffix}`,`Residual attention · ${suffix}`);
        const ffn=this.norm(residual,w.ffn_norm,`MLP • RMSNorm • ${suffix}`,`FFN input · ${suffix}`);
        this.stage(`MLP • Input projection • ${suffix}`);
        const z=this.mul(w.ffn_in,ffn,`W1 · ${suffix}`,`FFN pre-activation · ${suffix}`);
        this.stage(`MLP • GELU activation • ${suffix}`);
        const gelu=this.matrix(`GELU · ${suffix}`,z.rows,n,{formula:(i,j)=>`=0.5*${z.ref(i,j)}*(1+ERF(${z.ref(i,j)}/SQRT(2)))`});
        this.stage(`MLP • Output projection • ${suffix}`);
        const result=this.mul(w.ffn_out,gelu,`W2 · ${suffix}`,`FFN output · ${suffix}`);
        x=this.add(residual,result,`RESIDUAL • Add MLP output • ${suffix}`,`Layer output · ${suffix}`);
      });
      const final=this.norm(x,t.global_weights.final_norm,'OUTPUT • Final RMSNorm','Final hidden');
      this.stage('OUTPUT • Vocabulary projection');
      const logits=this.mul(t.global_weights.lm_head,final,'Vocabulary weights','Logits');
      this.stage('OUTPUT • Vocabulary softmax');
      const outputProbabilities=this.matrix('Vocabulary probabilities',t.vocabulary.length,n,{formula:(i,j)=>{
        const max=`MAX(${logits.ref(0,j)}:${logits.ref(t.vocabulary.length-1,j)})`,exp=a=>`EXP(${logits.ref(a,j)}-${max})`;
        return '='+exp(i)+'/('+t.vocabulary.map((_,a)=>exp(a)).join('+')+')';
      }});
      if(t.training) {
        this.matrix('Predicted next character',1,n,{formula:(i,j)=>{
          const range=`${outputProbabilities.ref(0,j)}:${outputProbabilities.ref(t.vocabulary.length-1,j)}`;
          const winner=`MATCH(MAX(${range}),${range},0)`;
          let result='"?"';
          for(let k=t.vocabulary.length-1;k>=0;k--){
            const character=label(t.vocabulary[k])===' '?'␣':label(t.vocabulary[k]);
            result=`IF(${winner}=${k+1},"${character.replaceAll('"','""')}",${result})`;
          }
          return '='+result;
        }});
        this.stage('TRAINING • Targets and recorded loss');
        this.matrix('Correct next character',1,n,{values:[t.training.targets.map(label)]});
        const loss=this.matrix('Recorded loss before update',1,n,{values:[t.training.losses]});
        for(let j=0;j<n;j++)this.cells.get(`${loss.row},${loss.column+j}`).recordedTargetRow=t.vocabulary.indexOf(t.training.targets[j])+1;
        this.stages.forEach(([r],i)=>this.put(r,1,`[${String(i).padStart(2,'0')}/${String(this.stages.length).padStart(2,'0')}]`));
        this.put(1,this.width+3,'Navigate to block');return this;
      }
      this.stage('GENERATION • Sampling settings');
      const edge=Math.max(this.right+n-1,10),s=this.settings;
      this.width=Math.max(this.width,edge+3);
      const settings=this.matrix('Temperature · Top-k · Seed',3,1,{column:edge,values:[[s.temperature],[s.top_k],[s.seed]]});
      this.controls={temperature:settings.ref(0,0),top_k:settings.ref(1,0),seed:settings.ref(2,0)};
      this.stage('GENERATION • Seeded top-k sampling');
      const full=`${logits.ref(0,n-1)}:${logits.ref(t.vocabulary.length-1,n-1)}`,max=`MAX(${full})`,temp=settings.ref(0,0),top=settings.ref(1,0);
      const cutoff=`LARGE(${full},IF(${top}=0,${t.vocabulary.length},MIN(${t.vocabulary.length},MAX(1,${top}))))`,exp=a=>`EXP((${logits.ref(a,n-1)}-${max})/${temp})`;
      const start=this.row;
      this.matrix('Top-k eligibility',t.vocabulary.length,1,{column:edge-8,formula:i=>`=IF(${logits.ref(i,n-1)}>=${cutoff},1,0)`});
      this.row=start;
      this.matrix('Top-k candidates',t.vocabulary.length,1,{column:edge-6,formula:i=>`=IF(${logits.ref(i,n-1)}>=${cutoff},${logits.ref(i,n-1)},"—")`});
      this.row=start;
      const probabilities=this.matrix('Top-k probabilities',t.vocabulary.length,1,{column:edge-4,formula:i=>`=IF(${temp}=0,IF(MATCH(${max},${full},0)=${i+1},1,0),IF(${logits.ref(i,n-1)}>=${cutoff},${exp(i)}/(`+t.vocabulary.map((_,a)=>`IF(${logits.ref(a,n-1)}>=${cutoff},${exp(a)},0)`).join('+')+'),0))'}),end=this.row;
      this.row=start;
      const g=t.generation,step=s.samplingStep??0;
      const draws=step===0&&g&&s.seed===(g.seed||0)?g.sampling_noise:samplingNoise(s.seed,step,t.vocabulary.length);
      const noise=this.matrix('Sampling noise · saved seeded draw',t.vocabulary.length,1,{column:edge-2,weight:true,values:draws.map(value=>[value]),formula:i=>{
        // Keep the initial export formulas for compatibility. The live runtime
        // replaces them with the current stream draw when the seed changes.
        if(step>0)return '='+draws[i];
        const fresh=`-LN(MAX(0.0000001,MIN(0.9999999,MOD(ABS(SIN(${settings.ref(2,0)}*12.9898+${i+1}*78.233)*43758.5453),1))))`;
        return g?`=IF(${settings.ref(2,0)}=${g.seed||0},${g.sampling_noise[i]},${fresh})`:'='+draws[i];
      }});
      this.row=start;
      const scores=this.matrix('Sampling score',t.vocabulary.length,1,{column:edge,formula:i=>`=${probabilities.ref(i,0)}/${noise.ref(i,0)}`});this.row=Math.max(end,this.row);
      this.stage('GENERATION • Generated token');
      const range=`${scores.ref(0,0)}:${scores.ref(t.vocabulary.length-1,0)}`;
      this.matrix('Selected vocabulary index · 1-based',1,1,{column:edge,formula:()=>`=MATCH(MAX(${range}),${range},0)`});
      this.stages.forEach(([r],i)=>this.put(r,1,`[${String(i).padStart(2,'0')}/${String(this.stages.length).padStart(2,'0')}]`));this.put(1,this.width+3,'Navigate to block');return this;
    }
    table() {
      const rows=new Map();this.cells.forEach(c=>{if(!rows.has(c.r))rows.set(c.r,[]);rows.get(c.r).push(c);});
      let html=`<table style="table-layout:fixed;width:${(this.width+3)*44}px"><colgroup span="${this.width+3}" width="44"></colgroup>`;
      for(let r=1;r<this.row;r++)html+='<tr>'+(rows.get(r)||[]).sort((a,b)=>a.c-b.c).map(c=>{
        const numeric=typeof c.v==='number',data=numeric?{'1':3,'3':c.v}:{'1':2,'2':c.v};
        return `<td${c.tokenHeader?' class="live-input-token" data-training-character="true" align="center"':''}${c.recordedTargetRow?` data-recorded-target-row="${c.recordedTargetRow}"`:''} data-grid-column="${c.c}" data-sheets-value="${esc(JSON.stringify(data))}"${numeric?` align="center" sdval="${c.v}" bgcolor="${c.w?'#fff6b9':'#f4f7fb'}"`:''}${c.f?` data-sheets-formula="${esc(c.f)}"`:''}><font>${esc(c.v)}</font></td>`;
      }).join('')+'</tr>';return html+'</table>';
    }
  }
  root.LittleLlamaSequenceGrid=SequenceGrid;root.LittleLlamaSamplingNoise=samplingNoise;
  if(typeof module!=='undefined')module.exports={SequenceGrid,anchoredPan,samplingNoise};
  if(typeof document==='undefined')return;
  const data=document.getElementById('sequence-generation-data');if(!data)return;
  const bundle=JSON.parse(data.textContent),trace=bundle.trace,bar=document.querySelector('.sequence-editor'),workbook=root.LittleLlamaWorkbook;
  if(!bar||!workbook)return;
  workbook.setTrainingCorpus(trace.training_corpus);
  workbook.setSamplingChartOpen(!!bundle.chartOpen);
  if(bundle.stageSelection)workbook.restoreStageSelection(bundle.stageSelection);
  const button=document.createElement('button');button.type='button';button.className='toolbar-button';button.textContent='＋';button.setAttribute('aria-label','Generate next token');bar.append(button);
  const removeButton=document.createElement('button');removeButton.type='button';removeButton.className='toolbar-button';removeButton.textContent='⌫';removeButton.setAttribute('aria-label','Delete last token');bar.append(removeButton);
  const undoButton=document.createElement('button');undoButton.type='button';undoButton.className='toolbar-button';undoButton.textContent='↶';undoButton.title='Restore the last deleted token';undoButton.setAttribute('aria-label','Undo last token deletion');bar.append(undoButton);
  bundle.deletionHistory ||= [];
  const message=document.createElement('span');message.setAttribute('role','status');bar.append(message);
  const normalize=x=>({'␣':' ','<space>':' ','\\n':'\n','\\t':'\t'}[x]??x);
  const inputs=()=>[...document.querySelectorAll('.sequence-token')];
  const selected=()=>document.querySelector('[data-matrix-name="Selected vocabulary index · 1-based"]');
  let deleting=false;
  function update() {
    const full=inputs().length>=trace.architecture.context_length;
    const invalid=inputs().some(i=>!trace.vocabulary.includes(normalize(i.value)));
    button.disabled=full||invalid||!selected();button.title=full?'Model context limit reached':invalid?'Enter valid vocabulary tokens':'Append the currently generated token and calculate the next prediction';
    removeButton.disabled=inputs().length<=1;
    removeButton.title=inputs().length<=1?'Keep at least one input token':'Remove the final token';
    undoButton.disabled=!bundle.deletionHistory.length;
    if(deleting)button.disabled=removeButton.disabled=undoButton.disabled=true;
    message.textContent=full?'Context limit reached':'';
  }
  let appliedSeed=Number(document.querySelector('.generation-control:last-child input')?.value??trace.generation?.seed??0);
  function syncNoise() {
    const controls=[...document.querySelectorAll('.generation-control input')];
    const seed=Number(controls[2]?.value??trace.generation?.seed??0);
    if(seed===appliedSeed)return;
    appliedSeed=seed;
    const step=bundle.samplingStep||0,g=trace.generation;
    const draws=step===0&&g&&seed===(g.seed||0)?g.sampling_noise:samplingNoise(seed,step,trace.vocabulary.length);
    workbook.setSamplingNoise(draws);workbook.recalculate(true);
  }
  document.addEventListener('latte:recalculated',()=>{syncNoise();update();});bar.addEventListener('input',update);update();
  function snapshot() {
    const result={};document.querySelectorAll('.matrix-value').forEach(cell=>{
      const key=matrixKey(cell);
      result[key]=cell.textContent;
    });return result;
  }
  function rebuild(tokens,settings) {
    bundle.kvState=root.LittleLlamaKVCache?.state();
    bundle.stageSelection=workbook.stageSelection();
    bundle.chartOpen=document.querySelector('.sampling-chart-card')?.hidden===false;
    if(!tokens.length||tokens.some(token=>!trace.vocabulary.includes(token)))throw Error('Enter valid vocabulary tokens first.');
    settings.samplingStep=bundle.samplingStep||0;
      const grid=new SequenceGrid(trace,tokens,settings).build();
      bundle.previous=snapshot();
      const view=workbook.captureView?.();
      bundle.view={sheetId:view?.sheet?.id||'sheet-0',zoom:view?.zoom??.25,pan:view?.pan||{x:0,y:0},scrollX:window.scrollX,scrollY:window.scrollY};
      // Keep the viewed calculation in place even when earlier matrices grow.
      let nearest=null, distance=Infinity;
      document.querySelectorAll('.worksheet.active .matrix-value').forEach(cell=>{
        if(cell.style.display==='none'||cell.closest('.stage-card')?.style.display==='none')return;
        // Anchor deletion to a surviving cell, never the highlighted cell
        // that is about to disappear from the rebuilt grid.
        if(bundle.deletionTransition&&cell.getAnimations().length)return;
        const rect=cell.getBoundingClientRect();
        if(rect.right<0||rect.left>innerWidth||rect.bottom<0||rect.top>innerHeight)return;
        const x=(rect.left+rect.right)/2,y=(rect.top+rect.bottom)/2;
        const d=(x-innerWidth/2)**2+(y-innerHeight/2)**2;
        if(d<distance){distance=d;nearest={key:matrixKey(cell),x,y};}
      });
      bundle.view.anchor=nearest;
      const payload=JSON.stringify(bundle).replaceAll('<','\\u003c');
      const trainingScript=bundle.trainingRecord?`<script id="training-record-data" type="application/json">${JSON.stringify(bundle.trainingRecord).replaceAll('<','\\u003c')}</script><script>${bundle.trainingRuntime.replaceAll('</script','<\\/script')}</script>`:'';
      const kvScript=bundle.kvRuntime?`<script>${bundle.kvRuntime.replaceAll('</script','<\\/script')}</script>`:'';
      const script=`<script id="sequence-generation-data" type="application/json">${payload}</script><script>${bundle.runtime.replaceAll('</script','<\\/script')}</script>${trainingScript}${kvScript}`;
      const html=bundle.shell.replace('__FORWARD_TABLE__',()=>grid.table()).replace('__GENERATION_CONTROLS__',()=>`<script id="generation-controls" type="application/json">${JSON.stringify(grid.controls)}</script>`).replace('__SEQUENCE_SCRIPT__',()=>script);
      workbook.disposeCamera?.();
      workbook.disposeAnnotations?.();
      document.open();document.write(html);document.close();
  }
  function currentSettings() {
    const controls=[...document.querySelectorAll('.generation-control input')];
    return {temperature:Number(controls[0]?.value??1),top_k:Number(controls[1]?.value??0),seed:Number(controls[2]?.value??0)};
  }
  removeButton.onclick=async()=>{
    if(root.LittleLlamaKVCache?.active()){root.LittleLlamaKVCache.remove();return;}
    if(deleting)return;
    const animations=[];
    const locked=[];
    try {
      const tokens=inputs().map(i=>normalize(i.value));
      if(tokens.length<=1)return;
      const remaining=tokens.slice(0,-1);
      if(remaining.some(token=>!trace.vocabulary.includes(token)))throw Error('Enter valid vocabulary tokens first.');
      const settings=currentSettings();
      deleting=true;update();
      document.querySelectorAll('.sequence-token,.generation-control input').forEach(input=>{if(!input.disabled){input.disabled=true;locked.push(input);}});
      const n=tokens.length;
      const sequenceColumns=/^(Token embeddings|Position embeddings|X ·|Attention input ·|[KVQ] ·|Scores ·|A ·|H ·|Concatenated heads ·|Attention output ·|Residual attention ·|FFN input ·|FFN pre-activation ·|GELU ·|FFN output ·|Layer output ·|Final hidden|Logits|Vocabulary probabilities)/;
      const duration=matchMedia('(prefers-reduced-motion: reduce)').matches?450:800;
      document.querySelectorAll('.matrix-value').forEach(cell=>{
        const name=cell.dataset.matrixName||'',row=Number(cell.dataset.matrixRow),column=Number(cell.dataset.matrixColumn);
        const removedColumn=sequenceColumns.test(name)&&column===n;
        const removedRow=/^(Kᵀ ·|Scores ·|A ·)/.test(name)&&row===n;
        if(!removedColumn&&!removedRow)return;
        if(cell.style.display==='none'||cell.closest('.stage-card')?.style.display==='none')return;
        animations.push(cell.animate([{boxShadow:'inset 0 0 0 2px #000'},{boxShadow:'inset 0 0 0 2px #000'}],{duration,fill:'forwards'}));
      });
      message.textContent='Removing highlighted token rows and columns…';
      if(animations.length)await Promise.all(animations.map(animation=>animation.finished.catch(()=>{})));
      bundle.deletionHistory.push({tokens,settings,step:bundle.samplingStep||0});
      bundle.samplingStep=Math.max(0,(bundle.samplingStep||0)-1);
      bundle.deletionTransition=true;
      rebuild(remaining,settings);
    } catch(error){message.textContent=error.message;}
    finally{animations.forEach(animation=>animation.cancel());locked.forEach(input=>input.disabled=false);deleting=false;update();}
  };
  undoButton.onclick=()=>{
    if(root.LittleLlamaKVCache?.active()){root.LittleLlamaKVCache.undo();return;}
    try {
      const previous=bundle.deletionHistory.at(-1);if(!previous)return;
      if(previous.tokens.some(token=>!trace.vocabulary.includes(token)))throw Error('Cannot restore a sequence containing invalid tokens.');
      bundle.deletionHistory.pop();bundle.samplingStep=previous.step;
      rebuild(previous.tokens,previous.settings);
    } catch(error){message.textContent=error.message;}
  };
  button.onclick=()=>{
    if(root.LittleLlamaKVCache?.active()){root.LittleLlamaKVCache.generate();return;}
    try {
      // Commit a pending input edit before reading the displayed prediction.
      workbook.commitSequenceInputs();workbook.recalculate();
      const tokens=inputs().map(i=>normalize(i.value));
      if(tokens.length>=trace.architecture.context_length)return;
      if(tokens.some(t=>!trace.vocabulary.includes(t)))throw Error('Enter valid vocabulary tokens first.');
      const cell=selected(),address=cell.dataset.cellAddress.match(/!([A-Z]+)(\d+)$/);
      let column=0;for(const c of address[1])column=column*26+c.charCodeAt(0)-64;
      const index=workbook.value('Full Forward Pass',Number(address[2]),column)-1;
      if(!Number.isInteger(index)||!trace.vocabulary[index])throw Error('The generated token is unavailable.');
      tokens.push(trace.vocabulary[index]);
      const controls=[...document.querySelectorAll('.generation-control input')];
      const settings={temperature:Number(controls[0]?.value??1),top_k:Number(controls[1]?.value??0),seed:Number(controls[2]?.value??0)};
      bundle.samplingStep=(bundle.samplingStep||0)+1;
      bundle.deletionHistory=[];
      settings.samplingStep=bundle.samplingStep;
      rebuild(tokens,settings);
    } catch(error) {message.textContent=error.message;}
  };
  root.LittleLlamaSequenceSession={
    settings:currentSettings,
    step:()=>bundle.samplingStep||0,
    setStep:step=>{bundle.samplingStep=step;appliedSeed=currentSettings().seed;},
    refresh:tab=>{bundle.resumeTab=tab;rebuild(inputs().map(i=>normalize(i.value)),currentSettings());},
    controls:update,
    status:text=>{message.textContent=text;},
  };
  if(bundle.previous) {
    if(!bundle.deletionTransition)document.querySelectorAll('.matrix-value').forEach(cell=>{
      const key=matrixKey(cell);
      if(bundle.previous[key]!==cell.textContent)cell.animate([{boxShadow:'inset 0 0 0 2px #000'},{boxShadow:'inset 0 0 0 2px #000',offset:.8},{boxShadow:'none'}],{duration:3500});
    });
    const saved=bundle.view;
    // Force layout and restore in the same script task, before any camera
    // callback or paint can expose the default startup position.
    (()=>{
      const sheet=document.getElementById(saved.sheetId);
      window.scrollTo({left:0,top:0,behavior:'instant'});
      const pan={x:saved.pan.x-saved.scrollX/saved.zoom,y:saved.pan.y-saved.scrollY/saved.zoom};
      const view={sheet,zoom:saved.zoom,pan};
      workbook.restoreView(view);
      if(saved.anchor) {
        const anchor=[...sheet.querySelectorAll('.matrix-value')].find(cell=>matrixKey(cell)===saved.anchor.key&&cell.style.display!=='none');
        if(anchor) {
          const rect=anchor.getBoundingClientRect();
          view.pan=anchoredPan(pan,saved.zoom,saved.anchor,{x:(rect.left+rect.right)/2,y:(rect.top+rect.bottom)/2});
          workbook.restoreView(view);
        }
      }
    })();
    delete bundle.previous;
    delete bundle.deletionTransition;
  }
  if(bundle.show){
    const applySingleView=()=>{
      const tabs=document.querySelector('.stage-view-tabs');
      const name={forward:'Full model',training:'Training',kvcache:'KV Cache'}[bundle.show];
      const selected=[...tabs.querySelectorAll('button')].find(button=>button.textContent===name);
      selected?.click();
      tabs.querySelectorAll('button').forEach(button=>{if(button!==selected)button.remove();});
      const chooser=tabs.querySelector('.steps-menu');if(chooser)chooser.hidden=true;
      if(bundle.show!=='forward'){
        document.querySelectorAll('.worksheet,.layer-overview,.sampling-chart-toggle,.annotation-overlay,.bracket-fade,.bracket-fade-top').forEach(element=>element.style.display='none');
        if(bundle.show==='training')document.querySelector('.sequence-editor').style.display='none';
      }
    };
    if(document.readyState==='complete')applySingleView();
    else window.addEventListener('load',applySingleView,{once:true});
  }
  if(bundle.resumeTab){const name=bundle.resumeTab;delete bundle.resumeTab;requestAnimationFrame(()=>[...document.querySelectorAll('.stage-view-tabs button')].find(b=>b.textContent===name)?.click());}
})(typeof globalThis!=='undefined'?globalThis:this);
