(() => {
  'use strict';
  const footerInset=document.documentElement.hasAttribute('data-training-replay')?0:30;
  const namespace = 'http://www.w3.org/2000/svg';
  const fade = document.createElement('div');
  fade.className = 'bracket-fade';
  fade.setAttribute('aria-hidden','true');
  document.body.append(fade);
  const topFade=document.createElement('div');
  topFade.className='bracket-fade-top';topFade.setAttribute('aria-hidden','true');document.body.append(topFade);
  const rightFade = document.createElement('div');
  rightFade.className = 'bracket-fade bracket-fade-right';
  rightFade.setAttribute('aria-hidden','true');
  document.body.append(rightFade);
  const overlay = document.createElementNS(namespace, 'svg');
  overlay.classList.add('annotation-overlay');
  overlay.setAttribute('aria-label', 'Layer and operation hierarchy');
  document.body.append(overlay);
  // User-space coordinates keep the clip independent of the labels' bounds.
  const nameDefs=document.createElementNS(namespace,'defs');
  const nameClip=document.createElementNS(namespace,'clipPath');
  nameClip.id='latte-matrix-names-clip';nameClip.setAttribute('clipPathUnits','userSpaceOnUse');
  const nameClipRect=document.createElementNS(namespace,'rect');
  nameClip.append(nameClipRect);nameDefs.append(nameClip);overlay.append(nameDefs);
  const matrixNames=document.createElementNS(namespace,'g');
  matrixNames.setAttribute('clip-path','url(#latte-matrix-names-clip)');
  matrixNames.classList.add('matrix-name-layer');
  overlay.append(matrixNames);
  const stepOverlay=document.createElementNS(namespace,'svg');
  stepOverlay.classList.add('annotation-overlay','step-annotation-overlay');
  stepOverlay.setAttribute('aria-label','Prefill and decoding steps');
  document.body.append(stepOverlay);
  document.body.classList.add('annotated-explorer');
  const footer = document.createElement('footer');
  footer.className = 'explorer-footer';
  footer.setAttribute('aria-hidden', 'true');
  document.body.append(footer);
  // Keep parameter sheets in memory for formula evaluation, but expose only
  // the forward-pass explorer in this copy.
  document.querySelectorAll('.sheet-tab:not([data-sheet="sheet-0"])').forEach(tab => tab.remove());
  document.querySelector('.sheet-tab[data-sheet="sheet-0"]')?.click();
  const cards = [...document.querySelectorAll('.stage-card')];
  // Index matrix elements once. During camera movement, read just the first
  // and last cell of matrices in visible cards, not every cell in the model.
  const matrixIndex = new Map(cards.map(card => {
    const matrices = new Map();
    card.querySelectorAll('.matrix-value[data-matrix-name]').forEach(cell => {
      if (cell.style.display === 'none') return;
        const name = cell.dataset.matrixName;
      if (!matrices.has(name)) matrices.set(name,{first:cell,last:cell});
      else matrices.get(name).last = cell;
    });
    return [card,matrices];
  }));
  let hoveredCard = null;
  const bands = [];

  function addRuns(items, level, classify) {
    let previous = null;
    items.forEach(card => {
      const label = classify(card.dataset.operation || '');
      if (!label) { previous = null; return; }
      if (!previous || previous.label !== label) {
        previous = {label, level, cards: []};
        bands.push(previous);
      }
      previous.cards.push(card);
    });
  }

  const groups = [...document.querySelectorAll('.layer-group')];
  groups.forEach(group => {
    const items = cards.filter(card => card.closest('.layer-group') === group);
    const layer = items[0]?.dataset.operation.match(/layer\s+\d+/i)?.[0];
    const label = layer ? layer.replace(/^layer/i, 'Layer') :
      (items[0]?.dataset.category === 'embedding' ? 'Embedding' : 'Output');
    bands.push({label, level: 0, cards: items});
    if (!layer) return;
    addRuns(items, 1, operation => {
      if (/^ATTENTION/.test(operation)) return 'Attention';
      if (/^MLP/.test(operation)) return 'MLP';
      if (/^RESIDUAL/.test(operation)) return 'Residual';
      return null;
    });
    addRuns(items, 2, operation => operation.match(/head\s+\d+/i)?.[0].replace(/^head/i, 'Head') || null);
  });

  const groupPicks=new Map();
  const pickLayer=document.createElement('div');pickLayer.className='stage-group-picks';document.body.append(pickLayer);
  const pickStyle=document.createElement('style');
  pickStyle.textContent='.stage-group-picks{position:fixed;inset:0;pointer-events:none;z-index:62}.stage-group-pick-control{position:fixed;pointer-events:auto;display:flex;align-items:center;justify-content:center;width:15px;height:15px;cursor:pointer}.stage-group-pick-control[hidden]{display:none}.stage-group-pick-control input{width:13px;height:13px;margin:0;accent-color:#a27342;cursor:pointer}';
  document.head.append(pickStyle);
  function groupPick(band,index){
    let control=groupPicks.get(index);
    if(control)return control;
    control=document.createElement('label');control.className='stage-group-pick-control';
    const checkbox=document.createElement('input');checkbox.type='checkbox';
    const path=[...bands.filter(parent=>parent.level<band.level&&band.cards.every(card=>parent.cards.includes(card)))
      .sort((a,b)=>a.level-b.level).map(parent=>parent.label),band.label].join(' / ');
    checkbox.setAttribute('aria-label',`Select ${path} on diagram`);control.title=`Select ${path}`;
    control.dataset.stageBand=String(index);control.append(checkbox);pickLayer.append(control);groupPicks.set(index,control);
    control.addEventListener('pointerdown',event=>event.stopPropagation());
    control.addEventListener('click',event=>event.stopPropagation());
    checkbox.onchange=()=>window.LittleLlamaWorkbook.setStageGroupSelection(band.cards,checkbox.checked);
    return control;
  }

  const nodePools = new Map();
  const poolOffsets = new Map();
  const usedNodes = new Set();
  const nameContext=document.createElement('canvas').getContext('2d');
  const nameWidths=new Map();
  let geometry = null;
  let elementRects = new WeakMap();
  let visibleCards = new Set();
  window.LittleLlamaWorkbook.visibleStageCards=()=>visibleCards;
  let presentationViewport=null;
  window.LittleLlamaWorkbook.presentationViewport=()=>presentationViewport;
  function measureGeometry(active, zoom, origin) {
    elementRects = new WeakMap();
    const local = rect => ({left:(rect.left-origin.left)/zoom,right:(rect.right-origin.left)/zoom,
      top:(rect.top-origin.top)/zoom,bottom:(rect.bottom-origin.top)/zoom});
    const cardRects = new Map();
    const matrices = new Map();
    cards.forEach(card => {
      cardRects.set(card,local(card.getBoundingClientRect()));
      const bounds = new Map();
      matrixIndex.get(card).forEach(({first,last},name) => {
        const a=first.getBoundingClientRect(),b=last.getBoundingClientRect();
        bounds.set(name,local({left:a.left,right:b.right,top:a.top,bottom:b.bottom}));
      });
      matrices.set(card,bounds);
    });
    geometry = {active,cardRects,matrices};
  }
  function make(tag, attributes, text, semanticKey) {
    const offset = poolOffsets.get(tag) || 0;
    poolOffsets.set(tag,offset+1);
    const key=tag+':'+(semanticKey||
      (attributes['data-matrix-label']?'matrix:'+attributes['data-matrix-label']:
       attributes['data-operation-index']!==undefined?'operation:'+attributes['data-operation-index']:'slot:'+offset));
    let element = nodePools.get(key);
    if (!element) {
      element = document.createElementNS(namespace, tag);
      nodePools.set(key,element);
      overlay.append(element);
    }
    usedNodes.add(key);
    const name=attributes['data-matrix-label'];
    if(name&&window.LattePresentation?.connectedNames().has(name))attributes.class=(attributes.class||'')+' latte-name-connected';
    const parent=name||attributes['data-name-background']?matrixNames:/kv-step-bracket|hierarchy-bracket-label/.test(attributes.class||'')?stepOverlay:overlay;
    if(element.parentNode!==parent)parent.append(element);
    [...element.attributes].forEach(attribute => {
      if (attribute.name!=='style'&&!(attribute.name in attributes)) element.removeAttribute(attribute.name);
    });
    if(element.style.display==='none')element.style.display = '';
    element.onclick = element.onkeydown = null;
    Object.entries(attributes).forEach(([name,value]) => {if(element.getAttribute(name)!==String(value))element.setAttribute(name,value);});
    if (text !== undefined) {
      if (element.firstChild?.nodeType === 3) {
        if (element.firstChild.nodeValue !== text) element.firstChild.nodeValue = text;
      } else element.prepend(document.createTextNode(text));
    }
    return element;
  }

  function draw() {
    poolOffsets.clear();
    usedNodes.clear();
    const active = document.querySelector('#sheet-0.active');
    if (!active) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const nameFont=getComputedStyle(document.documentElement).getPropertyValue('--matrix-title-font').trim()||'"Architects Daughter",sans-serif';
    let rightEdgeActive = false;
    const zoom = window.LittleLlamaWorkbook.cameraZoom?.() || Number(document.documentElement.style.getPropertyValue('--workbook-zoom') || 1);
    const origin = active.getBoundingClientRect();
    if (!geometry || geometry.active !== active) measureGeometry(active,zoom,origin);
    const screen = rect => ({left:origin.left+rect.left*zoom,right:origin.left+rect.right*zoom,
      top:origin.top+rect.top*zoom,bottom:origin.top+rect.bottom*zoom});
    const frameMatrices = new Map();
    cards.forEach(card => {
      if (card.style.display === 'none') return;
      const box = screen(geometry.cardRects.get(card));
      if (box.bottom < 76 || box.top > height-44) return;
      const matrices = new Map();
      geometry.matrices.get(card).forEach((rect,name) => matrices.set(name,screen(rect)));
      frameMatrices.set(card,matrices);
    });
    const nextVisible=new Set(frameMatrices.keys());
    for(const card of visibleCards)if(!nextVisible.has(card))card.classList.remove('latte-visible-stage');
    for(const card of nextVisible)if(!visibleCards.has(card))card.classList.add('latte-visible-stage');
    visibleCards=nextVisible;
    overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
    stepOverlay.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const visibleTop = width < 760 ? 84 : 44;
    const picking=window.LittleLlamaWorkbook.stagePicking?.()||false;
    groupPicks.forEach(control=>control.hidden=true);
    const rail = picking?(width<760?[20,105,190]:[28,126,224]):(width < 760 ? [12,66,120] : [28,102,176]);
    const level=Math.max(0,...bands.filter(band=>band.cards.some(card=>frameMatrices.has(card))).map(band=>band.level));
    // Like KeVin's scene clip, keep matrix ink on the content side of the
    // innermost visible bracket, with a short feathered edge.
    presentationViewport={left:rail[level]+8,top:width<760?116:76,right:width,bottom:height-footerInset};
    for(const [key,value] of Object.entries({x:presentationViewport.left,y:presentationViewport.top,
      width:width-presentationViewport.left,height:height-footerInset-presentationViewport.top}))nameClipRect.setAttribute(key,value);
    bands.forEach((band,bandIndex) => {
      const visible = band.cards.filter(card=>card.style.display!=='none'&&!card.hidden);
      if (!visible.length) return;
      const first = screen(geometry.cardRects.get(visible[0]));
      const last = screen(geometry.cardRects.get(visible.at(-1)));
      let top = first.top + 1,bottom = last.bottom - 1;
      const cacheTimeline=visible.every(card=>card.dataset.kvTimeline==='true');
      if(cacheTimeline){
        let rects=visible.flatMap(card=>[...(frameMatrices.get(card)||[])]);
        if(band.level===1&&band.label==='Attention')rects=rects.filter(([name])=>!name.includes(' · cache ')&&!/^(A ·|VA ·|Scores ·)/.test(name));
        if(rects.length){top=Math.min(...rects.map(([,rect])=>rect.top));bottom=Math.max(...rects.map(([,rect])=>rect.bottom));}
      }
      if (bottom < 90 || top > height - 44) return;
      const color = band.level === 2 ? '#b16a25'
        : band.level === 0 ? '#8060ad'
        : band.label === 'Attention' ? '#367daf'
        : band.label === 'MLP' ? '#328775' : '#697b94';
      const railLevel=cacheTimeline?(band.level===1?2:band.level===2?1:0):band.level;
      const x = rail[railLevel];
      make('path', {d:`M ${x+7} ${top} H ${x} V ${bottom} H ${x+7}`, fill:'none', stroke:color, 'stroke-width':band.level === 0 ? 1.5 : 1, opacity:.7,'data-bracket-label':cacheTimeline&&band.level===1&&band.label==='Attention'?'Projections':band.label},undefined,`band:${bandIndex}`);
      const labelY = Math.min(height-55,Math.max(98,top+13));
      if(labelY>bottom+7)return;
      const label=make('text',{x:x+10,y:labelY,fill:color,'font-size':12,'font-weight':700,class:'hierarchy-bracket-label','data-stage-band-label':band.label},cacheTimeline&&band.label==='Attention'?'Projections':band.label,`band:${bandIndex}`);
      const title=label.querySelector('title')||document.createElementNS(namespace,'title');
      if(title.textContent!==band.label)title.textContent=band.label;
      if(title.parentNode!==label)label.append(title);
      label.style.pointerEvents=picking?'auto':'';label.style.cursor=picking?'pointer':'';
      label.onclick=picking?()=>{const current=window.LittleLlamaWorkbook.stageGroupSelection(band.cards);window.LittleLlamaWorkbook.setStageGroupSelection(band.cards,!current.checked);}:null;
      if(picking){
        const control=groupPick(band,bandIndex),checkbox=control.querySelector('input');
        const state=window.LittleLlamaWorkbook.stageGroupSelection(band.cards);
        control.hidden=false;control.style.left=`${Math.max(2,x-17)}px`;control.style.top=`${labelY-12}px`;
        checkbox.checked=state.checked;checkbox.indeterminate=state.indeterminate;
      }

    });

    frameMatrices.forEach((matrices,card)=>{
      if(card.dataset.kvTimeline!=='true')return;
      const attentionRects=[...matrices].filter(([name])=>name.includes(' · cache ')||/^(Scores ·|A ·|VA ·)/.test(name)).map(([,rect])=>rect);
      if(attentionRects.length){
        const top=Math.min(...attentionRects.map(rect=>rect.top)),bottom=Math.max(...attentionRects.map(rect=>rect.bottom));
        if(bottom>=90&&top<=height-44){
          const x=rail[2],color='#367daf';
          make('path',{d:`M ${x+7} ${top} H ${x} V ${bottom} H ${x+7}`,fill:'none',stroke:color,'stroke-width':1,opacity:.7,'data-bracket-label':'Attention'});
          const y=Math.min(height-55,Math.max(98,top+13));
          if(y<=bottom+7)make('text',{x:x+10,y,fill:color,'font-size':12,'font-weight':700,class:'hierarchy-bracket-label'},'Attention');
        }
      }
      const steps=JSON.parse(card.dataset.kvSteps||'[]');
      const groupDecodes=zoom<.55&&steps.length>2,decodeBounds=[];
      steps.forEach(({tag,label,details,hoverTitle,hoverTokens,hoverDecode},index)=>{
        const rects=[...matrices].filter(([name])=>name.endsWith(` · ${tag}`)).map(([,rect])=>rect);
        if(!rects.length)return;
        const left=Math.min(...rects.map(rect=>rect.left)),right=Math.max(...rects.map(rect=>rect.right));
        const naturalTop=Math.min(...rects.map(rect=>rect.top))-32;
        const bottom=Math.max(...rects.map(rect=>rect.bottom));
        if(right<250||left>width||bottom<76||naturalTop>height-44)return;
        // Keep step identification visible while its matrices remain on screen.
        const top=visibleTop+24;
        const color=index===0?'#328775':'#56718f';
        const bracket=make('path',{d:`M ${left} ${top+7} V ${top} H ${right} V ${top+7}`,fill:'none',stroke:color,'stroke-width':1,'pointer-events':'stroke',class:'kv-step-bracket','data-bracket-label':label});
        bracket.dataset.stepDetails=hoverTitle||details||label;
        bracket.dataset.stepTokens=JSON.stringify(hoverTokens||[]);
        bracket.dataset.stepDecode=String(!!hoverDecode);
        if(groupDecodes&&index>0){decodeBounds.push({left,right,index});return;}
        const labelX=(Math.max(250,left)+Math.min(width-16,right))/2;
        const shortLabel=index===0?'Prefill':`Decode ${index}`;
        const heading=make('text',{x:labelX,y:top-8,'text-anchor':'middle',fill:color,'font-size':11,'font-weight':400,'pointer-events':'auto',class:'kv-step-bracket-label'},shortLabel);
        heading.querySelector('title')?.remove();
        heading.dataset.stepDetails=hoverTitle||details||label;
        heading.dataset.stepTokens=JSON.stringify(hoverTokens||[]);
        heading.dataset.stepDecode=String(!!hoverDecode);
        const visibleWidth=Math.min(width-16,right)-Math.max(250,left);
        if(index===0&&visibleWidth>150){
          const count=label.match(/(\d+ tokens)/)?.[1];
          if(count)make('text',{x:labelX,y:top+21,'text-anchor':'middle',fill:color,'font-size':9,'font-weight':400,class:'kv-step-bracket-label'},count);
        }
      });
      if(decodeBounds.length){
        const first=decodeBounds[0],last=decodeBounds.at(-1);
        const left=Math.max(250,first.left),right=Math.min(width-16,last.right);
        const label=first.index===last.index?`Decode ${first.index}`:`Decode ${first.index}–${last.index}`;
        make('text',{x:(left+right)/2,y:visibleTop+16,'text-anchor':'middle',fill:'#56718f','font-size':11,'font-weight':400,class:'kv-step-bracket-label kv-decode-group-label'},label);
      }
    });

    document.body.classList.toggle('annotation-detail', zoom >= .55);
    cards.forEach((card,index) => {
      if (card.dataset.kvTimeline==='true'||!frameMatrices.has(card)) return;
      const rects = [...frameMatrices.get(card).values()];
      if (!rects.length) return;
      const naturalX = Math.max(...rects.map(rect => rect.right))+14;
      const top = Math.min(...rects.map(rect => rect.top));
      const bottom = Math.max(...rects.map(rect => rect.bottom));
      const operation = (card.dataset.operation || '').split(' • ').slice(1)
        .filter(part => !/^layer\s|^head\s/i.test(part)).join(' · ');
      const shortNames = {'Token + position embedding':'Token + position',
        'V · A (weighted value aggregation)':'Value aggregation',
        'Scores KᵀQ / √dₖ':'Attention scores',
        'Causal mask + softmax':'Masked softmax',
        'Vocabulary head + softmax':'Vocabulary probabilities'};
      const label = shortNames[operation] || operation;
      // Follow the matrix until the bracket or its label reaches the edge.
      // Only then switch to an inward-facing, viewport-clamped label.
      const labelWidth = Math.min(300,label.length*6+20);
      const pinned = naturalX+labelWidth > width-16;
      const x = pinned ? width-16 : naturalX;
      if (!pinned && x < (width < 760 ? 185 : 250)) return;
      rightEdgeActive ||= pinned;
      make('path',{d:`M ${x-7} ${top} H ${x} V ${bottom} H ${x-7}`,fill:'none',stroke:'#879bb4','stroke-width':1,opacity:.8},undefined,`operation:${index}`);
      const y = Math.max(100,Math.min(height-57,(top+bottom)/2+3));
      const text = make('text',{x:pinned ? x-12 : x+9,y,'text-anchor':pinned ? 'end' : 'start',fill:'#56718f','font-size':10,'font-weight':550,
        class:'operation-annotation','data-operation-index':index,role:'button',tabindex:0,
        'aria-label':`Zoom into ${operation}`},label);
      const activate = event => {
        event.preventDefault();
        event.stopPropagation();
        const target = rects[Math.floor(rects.length/2)];
        window.LittleLlamaWorkbook.zoomToPoint?.(Math.max(1,zoom),{
          x:Math.max(260,Math.min(width-24,(target.left+target.right)/2)),
          y:Math.max(100,Math.min(height-60,(target.top+target.bottom)/2)),
        });
      };
      text.onclick = activate;
      text.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') activate(event); };
      const title = text.querySelector('title') || document.createElementNS(namespace,'title');
      if(title.textContent!==operation)title.textContent = operation;
      if(title.parentNode!==text)text.append(title);
    });
    frameMatrices.forEach((matrices,card) => {
      matrices.forEach((rect,name) => {
        if (rect.bottom < 100 || rect.top > height-44) return;
        const samplingHeaders={
          'Top-k eligibility':['Logits','Full scores; ✓ marks top-k candidates and × marks excluded tokens.'],
          'Top-k candidates':['Top-k','Retained candidates only.'],
          'Top-k probabilities':['Temperature','Probabilities after temperature adjustment.'],
          'Sampling noise · saved seeded draw':['Random noise','Seeded exponential random values E, refreshed at each generation step.'],
          'Sampling score':['Randomized','Sampling score p / E; the largest score selects the next token.'],
        };
        if(samplingHeaders[name]) {
          const [label,description]=samplingHeaders[name];
          const text=make('text',{x:(rect.left+rect.right)/2,y:rect.top-22,'text-anchor':'middle',fill:'#637791','font-size':10,'pointer-events':'auto'},label);
          const title=text.querySelector('title')||document.createElementNS(namespace,'title');
          title.textContent=description;text.append(title);
          return;
        }
        const kind=window.LatteMath.matrixKind(name);
        const baseSymbol=window.LatteMath?.matrixSymbol(name)||name;
        const symbol=baseSymbol;
        const {x:nameX,y:nameY,size,labelHeight,labelScale}=window.LatteMath.matrixNamePosition(rect,name,zoom);
        const font=`400 ${size}px ${nameFont}`,measureKey=font+':'+symbol;
        if(!nameWidths.has(measureKey)){
          if(nameContext)nameContext.font=font;
          nameWidths.set(measureKey,nameContext?nameContext.measureText(symbol).width:size*symbol.length*.65);
        }
        const labelWidth=Math.max(labelHeight,nameWidths.get(measureKey)+8*labelScale),halfWidth=labelWidth/2;
        if(nameX+halfWidth<=presentationViewport.left||nameX-halfWidth>=presentationViewport.right)return;
        make('rect',{x:nameX-labelWidth/2,y:nameY-labelHeight/2,width:labelWidth,height:labelHeight,rx:3*labelScale,
          class:'latte-matrix-name-bg','data-name-background':name,'pointer-events':'auto'},undefined,`name-background:${name}`);
        const text = make('text',{x:nameX,y:nameY,
          'text-anchor':'middle','dominant-baseline':'central',fill:'#465365','font-size':size,
          'font-weight':400,'data-math-kind':kind,'data-matrix-label':name,class:'latte-matrix-name',
          'pointer-events':'auto',role:'button',tabindex:0,
          'aria-label':`${name} · matrix equation (Math mode) or focus matrix`},symbol);
        text.style.fontSize=size+'px';
        const title = text.querySelector('title') || document.createElementNS(namespace,'title');
        if(title.textContent!==name)title.textContent = name;
        if(title.parentNode!==text)text.append(title);
      });
    });
    for(const [key,element] of nodePools)if(!usedNodes.has(key)&&element.style.display!=='none')element.style.display='none';
    const rightDisplay=rightEdgeActive?'':'none';
    if(rightFade.style.display!==rightDisplay)rightFade.style.display=rightDisplay;
    // Publish a complete label frame, not the half-built state above.
    window.dispatchEvent(new Event('latte:annotations'));
  }
  const stepTooltip=document.createElement('div');
  stepTooltip.className='kv-step-tooltip';stepTooltip.hidden=true;
  stepTooltip.setAttribute('role','tooltip');document.body.append(stepTooltip);
  let stepTooltipTimer=0;
  const hideStepTooltip=()=>{clearTimeout(stepTooltipTimer);stepTooltip.hidden=true;};
  stepOverlay.addEventListener('pointerover',event=>{
    const label=event.target.closest('[data-step-details]');if(!label)return;
    hideStepTooltip();
    stepTooltipTimer=setTimeout(()=>{
      if(!label.isConnected||label.style.display==='none')return;
      stepTooltip.classList.toggle('kv-step-tooltip-inline',label.dataset.stepDecode==='true');
      stepTooltip.replaceChildren();
      const heading=document.createElement('div');heading.textContent=label.dataset.stepDetails;stepTooltip.append(heading);
      const tokens=document.createElement('div');tokens.className='kv-step-token-list';
      for(const token of JSON.parse(label.dataset.stepTokens||'[]')){
        const box=document.createElement('span');box.className='kv-step-token';box.textContent=token;tokens.append(box);
      }
      stepTooltip.append(tokens);
      stepTooltip.hidden=false;
      const rect=label.getBoundingClientRect();
      const box=stepTooltip.getBoundingClientRect();
      stepTooltip.style.left=`${Math.max(8,Math.min(window.innerWidth-box.width-8,(rect.left+rect.right-box.width)/2))}px`;
      stepTooltip.style.top=`${rect.bottom+10}px`;
    },450);
  });
  stepOverlay.addEventListener('pointerout',hideStepTooltip);
  window.addEventListener('blur',hideStepTooltip);
  let frame = 0;
  let disposed = false;
  window.LittleLlamaWorkbook.disposeAnnotations = () => { disposed = true; cancelAnimationFrame(frame); frame = 0; };
  const schedule = () => { if (!disposed && !frame) frame = requestAnimationFrame(() => { frame=0; if (!disposed) draw(); }); };
  document.addEventListener('pointermove',event => {
    const annotation = event.target.closest?.('[data-operation-index]');
    const card = annotation ? cards[Number(annotation.dataset.operationIndex)] : event.target.closest?.('.stage-card');
    if (card === hoveredCard) return;
    hoveredCard = card || null;
    schedule();
  });
  document.addEventListener('pointerleave',() => { hoveredCard = null; schedule(); });
  window.addEventListener('scroll',schedule,{passive:true});
  const invalidate = () => { geometry=null; schedule(); };
  window.addEventListener('latte:layout', () => {
    cards.forEach(card => {
      const matrices = new Map();
      card.querySelectorAll('.matrix-value[data-matrix-name]').forEach(cell => {
        if (cell.style.display === 'none') return;
        const name = cell.dataset.matrixName;
        if (!matrices.has(name)) matrices.set(name, {first:cell,last:cell});
        else matrices.get(name).last = cell;
      });
      matrixIndex.set(card, matrices);
    });
    invalidate();
  });
  window.addEventListener('latte:stage-selection',schedule);
  window.addEventListener('resize',invalidate);
  window.addEventListener('latte:camera',() => {
    cancelAnimationFrame(frame); frame=0; if (!disposed) draw();
  });
  const observer = new MutationObserver(invalidate);
  document.querySelectorAll('.worksheet').forEach(section => observer.observe(section,{attributes:true,attributeFilter:['class']}));
  const sizes = new ResizeObserver(invalidate);
  cards.forEach(card => sizes.observe(card));
  window.LittleLlamaWorkbook.screenRect = element => {
    const active = document.querySelector('#sheet-0.active');
    if (!active || !active.contains(element)) return element.getBoundingClientRect();
    const zoom = window.LittleLlamaWorkbook.cameraZoom?.() || Number(document.documentElement.style.getPropertyValue('--workbook-zoom') || 1);
    const origin = active.getBoundingClientRect();
    if (!geometry || geometry.active !== active) measureGeometry(active,zoom,origin);
    let rect = elementRects.get(element);
    if (!rect) {
      const measured = element.getBoundingClientRect();
      rect = {left:(measured.left-origin.left)/zoom,top:(measured.top-origin.top)/zoom,
        width:measured.width/zoom,height:measured.height/zoom};
      elementRects.set(element,rect);
    }
    return {left:origin.left+rect.left*zoom,top:origin.top+rect.top*zoom,
      width:rect.width*zoom,height:rect.height*zoom};
  };
  document.fonts.ready.then(()=>{nameWidths.clear();schedule();});
  schedule();
})();
