/* Presentation only: Latte's own tensors and formulas, in KeVin's visual language. */
(() => {
  'use strict';
  const workbook=window.LittleLlamaWorkbook;
  if(!workbook)return;
  const make=(tag,cls,content)=>{const node=document.createElement(tag);node.className=cls;if(content!==undefined)node.textContent=content;return node;};
  const svg=(tag,attrs)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attrs||{}))node.setAttribute(key,value);return node;};
  // Renderer-owned SVG hosts: retain nodes and GPU filter resources on pan.
  function renderSVG(parent,nodes){
    window.d3.select(parent).selectAll(function(){return [...this.children];})
      .data(nodes,node=>node.key).join(enter=>enter.append(node=>svg(node.tag))).order()
      .each(function(node){
        const attrs=node.attrs||{};
        for(const attribute of [...this.attributes])if(!(attribute.name in attrs))this.removeAttribute(attribute.name);
        for(const [name,value] of Object.entries(attrs))if(this.getAttribute(name)!==String(value))this.setAttribute(name,value);
        if(node.text!==undefined&&this.textContent!==node.text)this.textContent=node.text;
        if(node.children)renderSVG(this,node.children);
      });
  }
  const state={math:false,numbers:true,selected:null,description:null,groups:[],badges:[],panel:null,wire:null,matrix:null,related:[]};
  let mathHost=null,highlightTimer=0,highlighted=[],traceLabel=null;
  function endHighlight(){
    clearTimeout(highlightTimer);
    traceLabel=null;
    highlighted.forEach(element=>element.classList.remove('latte-math-destination'));
    highlighted=[];
    overlay.querySelectorAll('.latte-tracing-ribbon').forEach(element=>element.classList.remove('latte-tracing-ribbon'));
  }
  function navigate(elements,label){
    endHighlight();
    const targets=elements.filter(element=>element.isConnected);
    const hidden=new Set(targets.map(element=>element.closest('.stage-card')).filter(card=>card?.style.display==='none'));
    hidden.forEach(card=>workbook.addSelectedStep?.(card.dataset.operation));
    if(!targets.length)return;
    traceLabel=label;
    workbook.focusCells?.(targets,{preserveZoom:true,smooth:true});
    highlighted=targets;
    targets.forEach(element=>element.classList.add('latte-math-destination'));
    overlay.querySelectorAll('[data-operand]').forEach(element=>element.classList.toggle('latte-tracing-ribbon',element.dataset.operand===label||Boolean(state.matrix&&label===window.LatteMath.matrixSymbol(state.matrix.name))));
    state.panel?.querySelectorAll('[data-math-variable]').forEach(element=>element.classList.toggle('latte-active-variable',element.dataset.mathVariable===label));
    highlightTimer=setTimeout(endHighlight,1300);
  }
  const overlay=make('div','latte-math-overlay');document.body.append(overlay);
  const clearButton=make('button','latte-clear-selection','Clear selection');clearButton.hidden=true;clearButton.type='button';clearButton.title='Clear selection (Esc)';clearButton.onclick=clear;
  document.body.append(clearButton);
  const outlines=svg('svg',{'class':'latte-matrix-outlines','aria-hidden':'true'});document.body.append(outlines);
  let matrices=[];
  let hoveredMatrix=null;
  function hoverMatrix(name){
    const next=state.math?matrices.find(matrix=>matrix.name===name)||null:null;
    if(next===hoveredMatrix)return;
    hoveredMatrix?.cells.forEach(cell=>cell.classList.remove('latte-math-hover'));
    hoveredMatrix=next;
    hoveredMatrix?.cells.forEach(cell=>cell.classList.add('latte-math-hover'));
    schedule();
  }
  function clipToContent(node){
    const r=workbook.presentationViewport?.();
    if(!r)return;
    node.style.clipPath=`inset(${r.top}px ${Math.max(0,innerWidth-r.right)}px ${Math.max(0,innerHeight-r.bottom)}px ${r.left}px)`;
    node.style.maskImage=`linear-gradient(to right,transparent ${r.left}px,black ${r.left+18}px),linear-gradient(to bottom,transparent ${r.top}px,black ${r.top+14}px)`;
    node.style.maskComposite='intersect';
  }
  function indexMatrices(){
    matrices=[];
    document.querySelectorAll('.stage-card').forEach(card=>{
      const groups=new Map();
      card.querySelectorAll('.matrix-value[data-matrix-name]').forEach(cell=>{
        if(cell.style.display==='none')return;
        const name=cell.dataset.matrixName;
        if(!groups.has(name))groups.set(name,[]);
        groups.get(name).push(cell);
        cell.title=`${name} · row ${Number(cell.dataset.matrixRow)-1}, column ${Number(cell.dataset.matrixColumn)-1} · ${cell.getAttribute('sdval')||cell.textContent}`;
      });
      for(const [name,cells] of groups)matrices.push({name,cells});
    });
  }
  function bounds(elements){
    const rects=elements.filter(element=>element.isConnected&&element.getClientRects().length&&element.style.display!=='none').map(element=>element.getBoundingClientRect());
    return rects.length?{left:Math.min(...rects.map(r=>r.left)),right:Math.max(...rects.map(r=>r.right)),top:Math.min(...rects.map(r=>r.top)),bottom:Math.max(...rects.map(r=>r.bottom))}:null;
  }
  function drawOutlines(){
    clipToContent(outlines);
    const nodes=[];
    for(const matrix of matrices){
      const r=bounds([matrix.cells[0],matrix.cells.at(-1)]);
      if(!r)continue;
      nodes.push({key:matrix.name,tag:'rect',attrs:{x:r.left,y:r.top,width:r.right-r.left,height:r.bottom-r.top,
        display:r.bottom<60||r.top>innerHeight-30?'none':'inline',
        class:[state.matrix?.name===matrix.name?'latte-selected-matrix':'',hoveredMatrix?.name===matrix.name?'latte-hovered-matrix':''].filter(Boolean).join(' ')}});
    }
    renderSVG(outlines,nodes);
  }
  function clear(){
    endHighlight();state.panel?.remove();
    if(mathHost)mathHost.hidden=true;
    state.selected=state.description=state.matrix=state.panel=state.wire=null;
    state.groups=[];state.badges=[];state.related=[];overlay.replaceChildren();clearButton.hidden=true;
    document.body.classList.remove('latte-math-selection','latte-name-selection');
    document.querySelectorAll('.latte-name-connected').forEach(node=>node.classList.remove('latte-name-connected'));
    document.querySelectorAll('.latte-math-focus').forEach(node=>node.classList.remove('latte-math-focus'));
    workbook.clearPresentationSelection?.();
    schedule();
  }
  function forwardWheel(event){
    event.preventDefault();
    const target=document.querySelector('.worksheet.active .stage-content')||document.body;
    target.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,clientX:event.clientX,clientY:event.clientY,deltaX:event.deltaX,deltaY:event.deltaY,deltaMode:event.deltaMode,ctrlKey:event.ctrlKey,metaKey:event.metaKey,shiftKey:event.shiftKey}));
  }
  function showCell(element){
    const description=workbook.presentationCell?.(element);
    if(!description)return;
    clear();state.selected=element;state.description=description;
    if(description.weight){
      show(window.LatteMath.weightCellOrigin(description.name,workbook.weightOrigin?.(element),description.value),
        [{label:'a',sources:[{element}],result:true}]);
      return;
    }
    const explanation=window.LatteMath.explainCell(description);
    show(explanation.markup,[{label:'a',sources:[{element}],result:true},...explanation.groups]);
  }
  function showMatrix(name){
    const matrix=matrices.find(item=>item.name===name);if(!matrix)return;
    if(state.matrix?.name===name){clear();schedule();return;}
    clear();state.matrix=matrix;state.selected=matrix.cells[0];
    const dependencies=matrix.cells[0].classList.contains('learned-parameter')?[]:workbook.presentationMatrix(matrix.cells);
    state.related=dependencies.map(dependency=>matrices.find(item=>item.name===dependency.name)).filter(Boolean);
    document.body.classList.add('latte-name-selection');
    for(const related of [matrix,...state.related])related.cells.forEach(element=>element.classList.add('latte-math-focus'));
    // Matrix names use named-matrix connections, never scalar a/b/c badges.
    show(window.LatteMath.matrixFormula(name,state.related.map(matrix=>matrix.name)),[]);
    schedule();
  }
  function positionMatrixNames(){
    const labels=[...document.querySelectorAll('[data-matrix-label]')];
    const connected=new Set([state.matrix.name,...state.related.map(matrix=>matrix.name)]);
    labels.forEach(label=>label.classList.toggle('latte-name-connected',connected.has(label.dataset.matrixLabel)));
    // Annotation nodes are pooled. Hidden nodes retain their old names, but
    // their zero-sized rectangles must never replace a live label's bounds.
    const boxes=new Map(labels.map(label=>[label.dataset.matrixLabel,label.getBoundingClientRect()])
      .filter(([,rect])=>rect.width>0&&rect.height>0));
    // Labels are culled outside the bracket/viewport. Their logical matrices
    // still exist: anchor offscreen links to their current label position rather
    // than dropping a dependency or using a hidden pooled label at (0,0).
    const labelBox=matrix=>{
      if(boxes.has(matrix.name))return boxes.get(matrix.name);
      const r=bounds([matrix.cells[0],matrix.cells.at(-1)]);
      if(!r)return null;
      const {x,y}=window.LatteMath.matrixNamePosition(r,matrix.name,workbook.cameraZoom?.()||.25);
      return {left:x,right:x,top:y,bottom:y,width:0,height:0};
    };
    const source=labelBox(state.matrix);
    if(!source){renderSVG(state.wire,[]);return;}
    const nodes=[];
    const rectangle=r=>({x:r.left,y:r.top,w:r.width,h:r.height});
    for(const related of state.related){
      const target=labelBox(related);if(!target)continue;
      const obstacles=[...boxes].filter(([name])=>name!==state.matrix.name&&name!==related.name)
        .map(([,r])=>({x:r.left-5,y:r.top-5,w:r.width+10,h:r.height+10}));
      const d=window.LatteRibbons.nameConnection(rectangle(source),rectangle(target),obstacles);
      const symbol=window.LatteMath.matrixSymbol(related.name);
      const tracing=traceLabel===symbol||traceLabel===window.LatteMath.matrixSymbol(state.matrix.name);
      if(d)nodes.push({key:related.name,tag:'path',attrs:{d,class:'latte-math-name-link'+(tracing?' latte-tracing-ribbon':''),'data-operand':symbol,'data-source':state.matrix.name,'data-target':related.name}});
    }
    const p=state.panel.getBoundingClientRect(),center={x:(source.left+source.right)/2,y:(source.top+source.bottom)/2};
    if(!mathHost&&!state.panel.hidden)nodes.push({key:'card',tag:'path',attrs:{d:`M${center.x},${center.y} L${Math.max(p.left,Math.min(center.x,p.right))},${Math.max(p.top,Math.min(center.y,p.bottom))}`,class:'latte-card-link'}});
    renderSVG(state.wire,nodes);
  }
  function show(markup,groups){
    state.groups=groups;
    document.body.classList.add('latte-math-selection');
    for(const group of groups)group.sources.forEach(source=>source.element.classList.add('latte-math-focus'));
    const panel=make('div','latte-math-panel');panel.setAttribute('role','region');panel.setAttribute('aria-label',state.matrix?'Matrix calculation':'Selected cell calculation');
    panel.innerHTML=markup;
    const dismiss=make('div','latte-math-dismiss');
    const close=make('button','latte-math-close','×');close.type='button';
    close.setAttribute('aria-label','Close calculation');close.title='Close calculation (Esc)';
    close.onclick=clear;dismiss.append(close);panel.prepend(dismiss);
    const destinations=state.matrix?[state.matrix,...state.related].map(matrix=>({
      label:window.LatteMath.matrixSymbol(matrix.name),kind:window.LatteMath.matrixKind(matrix.name),name:matrix.name,elements:matrix.cells,
    })):groups.map(group=>({label:group.label,kind:group.sources.length>1?'vector':'scalar',name:group.name||'Selected result',elements:group.sources.map(source=>source.element)}));
    const normalize=value=>value.toLowerCase().replace(/[₀-₉]/g,digit=>String('₀₁₂₃₄₅₆₇₈₉'.indexOf(digit)));
    const locate=label=>destinations.find(destination=>normalize(destination.label)===normalize(label));
    for(const identifier of panel.querySelectorAll('mi')){
      let token=identifier.closest('munder[data-math-kind="vector"]')||identifier,label=identifier.textContent;
      const subscript=identifier.closest('msub');
      if(subscript){
        if(!subscript.firstElementChild.contains(identifier))continue;
        const index=subscript.lastElementChild.textContent;
        const indexed=locate(label+index);
        if(indexed){token=subscript;label=indexed.label;}
      }
      const destination=locate(label);
      if(!destination)continue;
      token.dataset.mathVariable=destination.label;token.setAttribute('role','link');token.setAttribute('tabindex','0');
      token.setAttribute('aria-label',`Locate ${destination.label}: ${destination.name}`);
    }
    if(destinations.some(destination=>destination.elements.some(element=>element.closest('.stage-card')?.style.display==='none'))){
      panel.append(make('p','latte-hidden-input-note','Some inputs are in hidden blocks. Select a variable below to show its block.'));
    }
    const actions=make('div','latte-math-actions');
    for(const destination of destinations){
      const button=make('button','latte-variable-button',destination.label);
      if(destination.kind==='vector')button.classList.add('latte-vector-underline');
      button.type='button';button.dataset.mathVariable=destination.label;button.dataset.mathKind=destination.kind;button.title=destination.name;actions.append(button);
    }
    const back=make('button','latte-math-return',state.matrix?'Back to result':'Back to result a');back.type='button';
    back.onclick=()=>navigate(state.matrix?state.matrix.cells:[state.selected],state.matrix?destinations[0].label:'a');
    panel.append(actions,back);
    panel.addEventListener('click',event=>{
      const token=event.target.closest?.('[data-math-variable]'),destination=token&&locate(token.dataset.mathVariable);
      if(destination)navigate(destination.elements,destination.label);
    });
    panel.addEventListener('keydown',event=>{
      if(event.key==='Escape'){event.preventDefault();clear();return;}
      const token=event.target.closest?.('[data-math-variable]');
      if(token&&token.tagName.toLowerCase()!=='button'&&['Enter',' '].includes(event.key)){
        event.preventDefault();const destination=locate(token.dataset.mathVariable);if(destination)navigate(destination.elements,destination.label);
      }
    });
    const wire=svg('svg',{'class':'latte-math-wire','aria-hidden':'true'});overlay.append(wire);
    (mathHost?.querySelector('.latte-math-content')||overlay).append(panel);
    if(mathHost){mathHost.hidden=false;panel.classList.add('latte-math-docked');}
    state.panel=panel;state.wire=wire;
    state.badges=groups.map(group=>{
      const badge=make('button','latte-math-badge');badge.type='button';
      const arrow=svg('svg',{viewBox:'0 0 14 14','aria-hidden':'true'});arrow.append(svg('path',{d:'M1 7H12M8 3L12 7L8 11'}));
      badge.dataset.mathKind=group.sources.length>1?'vector':'scalar';
      badge.append(arrow,make('span',group.sources.length>1?'latte-vector-underline':'',group.label));
      badge.setAttribute('aria-label',`${group.label}: ${group.name||'selected output'}`);
      badge.onclick=()=>navigate(group.sources.map(source=>source.element),group.label);
      badge.addEventListener('wheel',forwardWheel,{passive:false});overlay.append(badge);return badge;
    });
    clearButton.hidden=false;position();
  }
  function position(){
    if(!state.panel||!state.selected)return;
    const hiddenNote=state.panel.querySelector('.latte-hidden-input-note');
    if(hiddenNote)hiddenNote.hidden=state.matrix
      ? !state.related.some(matrix=>matrix.cells.some(element=>element.closest('.stage-card')?.style.display==='none'))
      : !state.groups.some(group=>group.sources.some(source=>source.element.closest('.stage-card')?.style.display==='none'));
    clipToContent(state.wire);
    const anchor=state.matrix?bounds(state.matrix.cells):state.selected.getBoundingClientRect();
    if(!anchor)return;
    const content=workbook.presentationViewport?.()||{left:0,top:40,right:innerWidth,bottom:innerHeight-28};
    const inView=r=>r.bottom>content.top&&r.top<content.bottom&&r.right>content.left&&r.left<content.right;
    const visible=inView(anchor);
    // Selection lives in workbook space. Its output may leave the viewport
    // while the user follows an operand or the connection between them.
    state.panel.hidden=mathHost?false:!visible;state.wire.style.display='';
    const panel=state.panel;
    const w=panel.offsetWidth||300,h=panel.offsetHeight||100;
    let left=anchor.right+18,top=(anchor.top+anchor.bottom-h)/2;
    if(left+w>innerWidth-12){left=anchor.left;top=anchor.bottom+38;}
    if(top+h>innerHeight-36)top=anchor.top-h-38;
    left=Math.max(12,Math.min(innerWidth-w-12,left));top=Math.max(48,Math.min(innerHeight-h-36,top));
    if(!mathHost){panel.style.left=left+'px';panel.style.top=top+'px';panel.style.maxHeight=Math.max(60,innerHeight-96)+'px';}
    if(state.matrix){positionMatrixNames();return;}
    const definitions=[{key:'shadow',tag:'filter',attrs:{id:'latte-tape-shadow',x:'-30%',y:'-30%',width:'160%',height:'180%','color-interpolation-filters':'sRGB'},
      children:[{key:'drop-shadow',tag:'feDropShadow',attrs:{dx:0,dy:5,stdDeviation:3,'flood-color':'#243655','flood-opacity':.16}}]}];
    const nodes=[{key:'defs',tag:'defs',children:definitions}];
    const rectangle=r=>({x:r.left,y:r.top,w:r.right-r.left,h:r.bottom-r.top});
    const matrixFor=element=>matrices.find(matrix=>matrix.cells.includes(element));
    const bodyFor=element=>{const matrix=matrixFor(element);return matrix&&bounds([matrix.cells[0],matrix.cells.at(-1)]);};
    const sourceBody=bodyFor(state.selected)||anchor;
    const sourceColor=window.LatteMath.roleColors[window.LatteMath.matrixRole(state.selected.dataset.matrixName||'',state.description?.weight)];
    const badgeBounds={left:content.left+8,top:content.top+8,right:content.right-8,bottom:content.bottom-8};
    for(let i=0;i<state.groups.length;i++){
      const group=state.groups[i],badge=state.badges[i],r=bounds(group.sources.map(source=>source.element));
      if(!r){badge.hidden=true;continue;}
      let anchors=null,lift=0,point=null,rowMargin=null;
      if(!group.result){
        const operand=group.sources[0].element,targetBody=bodyFor(operand)||r;
        anchors=window.LatteRibbons.connectionAnchors(rectangle(sourceBody),rectangle(targetBody),rectangle(anchor),rectangle(r));
        if(group.sources.length===1)anchors=window.LatteRibbons.scalarConnection(anchors);
        if(group.specific){
          // A separate narrow tape identifies the numerator cell inside the
          // score vector; bow outward to keep the vector tape distinguishable.
          const width=window.LatteRibbons.scalarConnection({sourceWidth:anchor.right-anchor.left,targetWidth:r.right-r.left}).sourceWidth;
          const above=r.bottom<=anchor.top;
          anchors={start:{x:anchor.right-width/2,y:above?anchor.top:anchor.bottom},
            end:{x:r.right-width/2,y:above?r.bottom:r.top},
            sourceWidth:width,targetWidth:width,vertical:true};
        }
        const {start,end,vertical}=anchors;
        lift=group.specific?120:Math.max(24,Math.min(72,Math.hypot(end.x-start.x,end.y-start.y)*.14));
        const singleRow=group.sources.every(source=>source.row===group.sources[0].row);
        // Like KeVin, prefer the row's left margin while it fits. Otherwise
        // dock at the operand end, or follow the visible ribbon toward it.
        rowMargin=!vertical&&singleRow&&r.left-42>=badgeBounds.left
          ? window.LatteRibbons.placeOperandEndMarker({x:r.left-50,y:end.y},{x:r.left,y:end.y},false,badgeBounds):null;
        point=rowMargin||(!group.specific&&window.LatteRibbons.placeOperandEndMarker(start,end,vertical,badgeBounds))
          ||window.LatteRibbons.placeRibbonIndicator(start,end,vertical,lift,badgeBounds);
      }
      const angle=point?.angle||0;
      const horizontal=Math.abs(Math.sin(angle*Math.PI/180))<.65;
      badge.classList.toggle('horizontal',horizontal);
      badge.classList.toggle('vertical',!horizontal);
      badge.classList.toggle('upward',Math.sin(angle*Math.PI/180)<-.65);
      badge.classList.toggle('leftward',Boolean(anchors&&!anchors.vertical&&!rowMargin&&anchors.end.x<anchors.start.x));
      badge.dataset.angle=String(angle);
      badge.querySelector('svg').style.transform=`rotate(${angle}deg)`;
      const x=point?point.x-19:r.left-42;
      const y=point?point.y-11:(r.top+r.bottom-22)/2;
      badge.style.left=Math.max(badgeBounds.left,Math.min(badgeBounds.right-38,x))+'px';
      badge.style.top=Math.max(badgeBounds.top,Math.min(badgeBounds.bottom-32,y))+'px';
      badge.hidden=group.result?!visible:!point&&!inView(r);
      // One contour per logical row/column, including the selected result.
      nodes.push({key:`range:${group.label}`,tag:'path',attrs:{d:`M${r.left} ${r.top} H${r.right} V${r.bottom} H${r.left} Z`,class:'latte-range-outline'+(traceLabel===group.label?' latte-tracing-ribbon':''),'data-operand':group.label}});
      if(group.result){
        if(visible&&!mathHost)nodes.push({key:'card',tag:'path',attrs:{d:`M${r.right} ${(r.top+r.bottom)/2} L${left} ${Math.max(top+12,Math.min(top+h-12,(r.top+r.bottom)/2))}`,'class':'latte-card-link'}});
      }else{
        // Use KeVin's matrix-edge anchors and lifted, full-width tape geometry.
        const operand=group.sources[0].element;
        const {start,end,vertical,sourceWidth,targetWidth}=anchors;
        const id=`latte-ribbon-${i}`;
        const targetColor=window.LatteMath.roleColors[window.LatteMath.matrixRole(operand.dataset.matrixName||'',operand.classList.contains('learned-parameter'))];
        const shine=sourceColor.map((value,index)=>Math.round((value+targetColor[index]+255)/3));
        definitions.push({key:`gradient:${group.label}`,tag:'linearGradient',attrs:{id,gradientUnits:'userSpaceOnUse',x1:start.x,y1:start.y,x2:end.x,y2:end.y},
          children:[[0,sourceColor,.18],[.45,shine,.24],[1,targetColor,.18]].map(([offset,color,opacity])=>({key:String(offset),tag:'stop',attrs:{offset,'stop-color':`rgb(${color.join(',')})`,'stop-opacity':opacity}}))});
        nodes.push({key:`ribbon:${group.label}`,tag:'path',attrs:{d:window.LatteRibbons.ribbonConnection(start,end,sourceWidth,targetWidth,vertical,lift),class:'latte-ribbon'+(traceLabel===group.label?' latte-tracing-ribbon':''),filter:'url(#latte-tape-shadow)',style:`fill:url(#${id})`,'data-operand':group.label,'data-kind':group.sources.length===1?'scalar':'vector','data-orientation':vertical?'vertical':'horizontal','data-source-width':sourceWidth,'data-target-width':targetWidth}});
      }
    }
    renderSVG(state.wire,nodes);
  }
  function setMathMode(enabled){state.math=Boolean(enabled);document.body.classList.toggle('latte-math-enabled',state.math);if(!state.math){hoverMatrix(null);clear();}}
  function setNumbers(enabled){state.numbers=Boolean(enabled);document.body.classList.toggle('latte-hide-numbers',!state.numbers);}
  function setTokenMatches(correct){
    const groups=[
      [...document.querySelectorAll('.live-input-token[data-training-character]')],
      [...document.querySelectorAll('.training-output-character')],
    ];
    groups.forEach((tokens,group)=>tokens.forEach((token,index)=>{
      const match=group===0?(index===0?undefined:correct[index-1]):correct[index];
      token.dataset.tokenMatch=match===undefined?'neutral':String(match);
      token.classList.toggle('latte-token-start',index===0);
      token.classList.toggle('latte-token-end',index===tokens.length-1);
    }));
  }
  window.LattePresentation={setTokenMatches,setMathMode,setNumbers,clear,setMathHost:host=>{
    mathHost=host;
    if(state.panel){host.querySelector('.latte-math-content').append(state.panel);state.panel.classList.add('latte-math-docked');host.hidden=false;}
  },connectedNames:()=>new Set(state.matrix?[state.matrix.name,...state.related.map(matrix=>matrix.name)]:[])};
  for(const type of ['pointerover','focusin'])document.addEventListener(type,event=>{
    const label=event.target.closest?.('[data-matrix-label],[data-name-background]');
    if(label)hoverMatrix(label.dataset.matrixLabel||label.dataset.nameBackground);
  });
  for(const type of ['pointerout','focusout'])document.addEventListener(type,event=>{
    const label=event.target.closest?.('[data-matrix-label],[data-name-background]');
    if(label&&!label.contains(event.relatedTarget))hoverMatrix(null);
  });
  document.addEventListener('click',event=>{
    const element=event.target.closest?.('.matrix-value');
    if(element){
      // The existing capture handler handles overview-to-detail zoom first.
      event.preventDefault();event.stopPropagation();
      if(!state.math){clear();return;}
      if(state.selected===element)clear();else showCell(element);
      return;
    }
    const label=event.target.closest?.('[data-matrix-label],[data-name-background],[data-operation-index]');
    if(label){
      const card=[...document.querySelectorAll('.stage-card')][Number(label.dataset.operationIndex)];
      const name=label.dataset.matrixLabel||label.dataset.nameBackground||matrices.filter(item=>item.cells[0].closest('.stage-card')===card).at(-1)?.name;
      if(name){
        event.preventDefault();event.stopPropagation();
        if(state.math)showMatrix(name);
        else {
          const matrix=matrices.find(item=>item.name===name),r=matrix&&bounds(matrix.cells);
          if(r)workbook.zoomToPoint?.(1,{x:(r.left+r.right)/2,y:(r.top+r.bottom)/2});
        }
      }
    }
  },true);
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      const hadSelection=Boolean(state.selected);
      clear();
      // Clicking a cell focuses its stage region. Escape switches the browser
      // to keyboard focus styling, so dismiss that pointer-origin focus along
      // with the selection rather than outlining the whole stage.
      if(hadSelection&&document.activeElement?.matches('.stage-content'))document.activeElement.blur();
    }
    const label=event.target.closest?.('[data-matrix-label]');
    if(label&&['Enter',' '].includes(event.key)){event.preventDefault();event.stopPropagation();label.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));}
  },true);
  let frame=0;
  function render(){drawOutlines();position();}
  function schedule(){if(!frame)frame=requestAnimationFrame(()=>{frame=0;render();});}
  // Annotations publish only after their full camera update. Complete the
  // overlays in that same task, before the browser can paint a partial state.
  window.addEventListener('latte:annotations',()=>{cancelAnimationFrame(frame);frame=0;render();});
  window.addEventListener('latte:layout',()=>{indexMatrices();schedule();});
  document.addEventListener('latte:recalculated',()=>{
    // Replay changes values in place; matrix identities and topology persist.
    if(state.matrix){
      const connected=new Set([state.matrix.name,...state.related.map(matrix=>matrix.name)]);
      matrices.filter(matrix=>connected.has(matrix.name)).forEach(matrix=>matrix.cells.forEach(element=>element.classList.add('latte-math-focus')));
    }
    if(state.math&&state.selected&&state.selected.isConnected&&!state.matrix)showCell(state.selected);
    schedule();
  });
  indexMatrices();schedule();document.fonts.ready.then(schedule);

  function installCorpusTooltip(panel){
    const corpus=panel.querySelector('.training-corpus');
    if(!corpus)return;
    const tip=make('div','latte-corpus-tooltip');tip.id='latte-corpus-tooltip';tip.setAttribute('role','tooltip');tip.hidden=true;tip.style.pointerEvents='auto';document.body.append(tip);
    const frame=panel.querySelector('iframe[title="Training forward pass"]');
    const wire=svg('svg',{'class':'latte-corpus-connector','aria-hidden':'true'});wire.hidden=true;
    Object.assign(wire.style,{position:'fixed',inset:'0',width:'100%',height:'100%',pointerEvents:'none',zIndex:'240',overflow:'visible',display:'none'});
    const defs=svg('defs'),marker=svg('marker',{id:'latte-corpus-arrow',viewBox:'0 0 6 6',refX:'6',refY:'3',markerWidth:'6',markerHeight:'6',markerUnits:'userSpaceOnUse',orient:'auto',overflow:'visible'});
    marker.append(svg('path',{d:'M0 0L6 3L0 6Z',fill:'#a77b4f','fill-opacity':'.75'}));defs.append(marker);
    const mask=svg('mask',{id:'latte-corpus-wire-mask',maskUnits:'userSpaceOnUse'}),maskArea=svg('rect',{fill:'white'}),toolbarMask=svg('rect',{fill:'black',rx:'10'});mask.append(maskArea,toolbarMask);defs.append(mask);
    const path=svg('path',{'data-corpus-link':'true',fill:'none',stroke:'#a77b4f','stroke-width':'1.2','stroke-opacity':'.75','marker-end':'url(#latte-corpus-arrow)',mask:'url(#latte-corpus-wire-mask)'});wire.append(defs,path);document.body.append(wire);
    const linkStyle=make('style','','.training-corpus-token.latte-corpus-selected-token{box-shadow:0 0 0 1px #a77b4f;border-radius:2px}.latte-corpus-tooltip .latte-corpus-go-input{display:block;margin-top:6px;padding:3px 8px;border:1px solid #dcc6ad;border-radius:4px;background:#fff9ef;color:#88623e;font:11px system-ui;cursor:pointer;pointer-events:auto}.latte-corpus-go-input[hidden]{display:none!important}');document.head.append(linkStyle);
    let active=null,pinned=false,target=null,goInput=null,wireFrame=0;
    function clearLink(){
      active?.classList.remove('latte-corpus-selected-token');target?.classList.remove('latte-corpus-input-target');target=null;
      wire.hidden=true;wire.style.display='none';cancelAnimationFrame(wireFrame);wireFrame=0;
    }
    function inputTarget(){
      if(!active||active.dataset.corpusPosition===undefined)return null;
      return frame?.contentDocument?.querySelectorAll('.stage-card .live-input-token[data-training-character="true"]')[Number(active.dataset.corpusPosition)]||null;
    }
    function drawLink(){
      wireFrame=0;
      if(!active||active.dataset.corpusPosition===undefined)return;
      const next=inputTarget();
      if(target!==next){target?.classList.remove('latte-corpus-input-target');target=next;}
      const book=frame?.contentWindow?.LittleLlamaWorkbook;
      if(target){target.classList.add('latte-corpus-input-target');target.style.setProperty('--latte-link-outline',`${1.5/(book?.cameraZoom?.()||1)}px`);}
      const box=target?.getBoundingClientRect(),viewport=book?.presentationViewport?.()||{left:0,top:0,right:frame?.clientWidth||0,bottom:frame?.clientHeight||0};
      const visible=box&&box.width>0&&box.height>0&&box.left>=viewport.left&&box.right<=viewport.right&&box.top>=viewport.top&&box.bottom<=viewport.bottom;
      if(goInput)goInput.hidden=!!visible;
      wire.hidden=!visible;wire.style.display=visible?'block':'none';
      if(!visible)return;
      const source=active.getBoundingClientRect(),outer=frame.getBoundingClientRect();
      const sx=source.left-3,sy=(source.top+source.bottom)/2,tx=outer.left+(box.left+box.right)/2,ty=outer.top+box.top-4;
      maskArea.setAttribute('width',innerWidth);maskArea.setAttribute('height',innerHeight);
      const toolbar=frame.contentDocument.querySelector('.workbook-toolbar')?.getBoundingClientRect();
      for(const [name,value] of Object.entries(toolbar?{x:outer.left+toolbar.left-2,y:outer.top+toolbar.top-2,width:toolbar.width+4,height:toolbar.height+4}:{width:0,height:0}))toolbarMask.setAttribute(name,value);
      const bend=Math.max(35,(sx-tx)*.45);
      // Finish with a straight stem so the long curve cannot kink directly
      // behind the arrowhead, particularly when the input is far to the left.
      path.setAttribute('d',`M ${sx} ${sy} C ${sx-bend} ${sy}, ${tx} ${ty-70}, ${tx} ${ty-14} L ${tx} ${ty}`);
      wire.dataset.tokenPosition=active.dataset.corpusPosition;
    }
    function scheduleLink(){if(!wireFrame)wireFrame=requestAnimationFrame(drawLink);}
    function bindFrame(){
      const doc=frame?.contentDocument,win=frame?.contentWindow;if(!doc||!win)return;
      if(!doc.getElementById('latte-corpus-input-style')){
        const style=doc.createElement('style');style.id='latte-corpus-input-style';
        style.textContent='.matrix-layout td.live-input-token.latte-corpus-input-target{outline:none!important;box-shadow:inset 0 0 0 var(--latte-link-outline,2px) #a77b4f!important}';doc.head.append(style);
      }
      ['latte:camera','latte:layout','latte:annotations','resize'].forEach(name=>win.addEventListener(name,scheduleLink));
      doc.addEventListener('pointerdown',event=>{
        // Navigation moves the linked cell; it does not change the selected token.
        if(event.target.closest?.('.workbook-toolbar'))return;
        hide();
      });
      doc.addEventListener('keydown',event=>{if(event.key==='Escape')hide();});
      scheduleLink();
    }
    frame?.addEventListener('load',bindFrame);bindFrame();
    const display=value=>value===' '?'␣':value==='\n'?'↵':value==='\t'?'⇥':value;
    const tokenAt=target=>target?.closest?.('.training-corpus-token[data-corpus-position],.corpus-usage-span');
    function hide(){clearLink();active?.removeAttribute('aria-describedby');active=null;pinned=false;tip.hidden=true;goInput=null;}
    function show(token,includeUsage=false){
      if(active!==token){clearLink();active?.removeAttribute('aria-describedby');}
      active=token;token.setAttribute('aria-describedby',tip.id);
      const data=token.dataset;
      const selected=data.corpusPosition!==undefined;
      const range=Number(data.usageEnd)-Number(data.usageStart)===1?`Character ${data.usageStart}`:`Characters ${data.usageStart}–${Number(data.usageEnd)-1}`;
      const heading=make('small','',selected?`Position ${data.corpusPosition} · recorded before update`:range);
      const line=make('div','');
      if(selected)line.append('Input ',make('code','',display(data.input)),' → Predicted ',make('code',data.match==='true'?'training-correct':'training-incorrect',display(data.prediction)),
        ' · Target ',make('code','',display(data.target)),data.match==='true'?' ✓':' ✗');
      tip.replaceChildren(heading);
      if(selected){
        if(data.targetProbability!==undefined)line.append(` · P(target) ${(100*Number(data.targetProbability)).toFixed(1)}%`);
        tip.append(line);
      }
      if(includeUsage&&data.usageCount!==undefined){
        const count=Number(data.usageCount);
        tip.append(make('div','latte-usage-tooltip',data.usageValidation==='true'?'Validation text · never used for training.':
          `${selected?`Corpus position ${data.usageStart} · `:''}Used ${count} ${count===1?'time':'times'} as input across the entire training run.`));
      }
      goInput=null;
      if(selected){
        active.classList.add('latte-corpus-selected-token');
        goInput=make('button','latte-corpus-go-input','Go to input');goInput.type='button';goInput.hidden=true;
        goInput.onclick=()=>{
          const cell=inputTarget(),book=frame?.contentWindow?.LittleLlamaWorkbook;if(!cell||!book)return;
          const card=cell.closest('.stage-card');
          if(card&&frame.contentWindow.getComputedStyle(card).display==='none')book.addSelectedStep(card.dataset.operation);
          book.focusPresentationElements([cell],{smooth:true});scheduleLink();
        };
        tip.append(goInput);drawLink();
      }else clearLink();
      tip.hidden=false;
      const sidebar=corpus.closest('aside')?.getBoundingClientRect()||corpus.getBoundingClientRect();
      const left=Math.max(8,sidebar.left+8),right=Math.min(innerWidth-8,sidebar.right-8);
      const top=Math.max(8,sidebar.top+8),bottom=Math.min(innerHeight-8,sidebar.bottom-8);
      Object.assign(tip.style,{maxWidth:`${Math.max(1,right-left)}px`,whiteSpace:'normal',overflowWrap:'anywhere',maxHeight:`${Math.max(1,bottom-top)}px`,overflowY:'auto'});
      const r=token.getBoundingClientRect(),box=tip.getBoundingClientRect();
      tip.style.left=`${Math.max(left,Math.min(right-box.width,r.left+r.width/2-box.width/2))}px`;
      const below=r.bottom+8;
      tip.style.top=`${Math.max(top,Math.min(bottom-box.height,below+box.height<=bottom?below:r.top-box.height-8))}px`;
    }
    corpus.addEventListener('click',event=>{const token=tokenAt(event.target);if(!token)return;if(active===token&&pinned)hide();else{show(token,true);pinned=true;}});
    corpus.addEventListener('keydown',event=>{if(!['Enter',' '].includes(event.key))return;const token=tokenAt(event.target);if(token){event.preventDefault();token.click();}});
    document.addEventListener('pointerdown',event=>{if(!tokenAt(event.target)&&!tip.contains(event.target))hide();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape')hide();});
    panel.addEventListener('scroll',hide,true);
    window.addEventListener('resize',hide);
    new MutationObserver(hide).observe(corpus,{subtree:true,childList:true,characterData:true});
  }

  function enhance(panel){
    if(panel.dataset.lattePresentation)return;
    panel.dataset.lattePresentation='true';
    const sidebar=panel.querySelector('.training-body > aside');
    if(sidebar){
      sidebar.id='latte-corpus-sidebar';
      const toggle=make('button','latte-sidebar-toggle');toggle.type='button';
      toggle.setAttribute('aria-controls',sidebar.id);
      toggle.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg><span>Hide panel</span>';
      sidebar.prepend(toggle);
      const style=make('style','','html:root .training-body>aside.latte-sidebar-collapsed{width:34px;padding:0;overflow:hidden}html:root .training-body>aside.latte-sidebar-collapsed>:not(.latte-sidebar-toggle){display:none!important}html:root .training-replay .latte-sidebar-toggle{display:flex;align-items:center;justify-content:center;gap:4px;flex:none;margin-left:auto;padding:3px 5px;font-size:10px;background:transparent}html:root .training-replay .latte-sidebar-collapsed .latte-sidebar-toggle{width:100%;height:36px;border:0;border-radius:0;margin:0} .latte-sidebar-collapsed .latte-sidebar-toggle span{display:none}.latte-sidebar-collapsed .latte-sidebar-toggle svg{transform:rotate(180deg)}');document.head.append(style);
      const mobile=matchMedia('(max-width:760px)');let chosen=false;
      function setCollapsed(collapsed){
        sidebar.classList.toggle('latte-sidebar-collapsed',collapsed);
        toggle.setAttribute('aria-expanded',String(!collapsed));
        toggle.setAttribute('aria-label',collapsed?'Show corpus panel':'Hide corpus panel');
        toggle.title=collapsed?'Show corpus panel':'Hide corpus panel';
        requestAnimationFrame(()=>{window.dispatchEvent(new Event('resize'));sidebar.parentElement.querySelector('iframe')?.contentWindow?.dispatchEvent(new Event('resize'));});
      }
      toggle.onclick=()=>{chosen=true;setCollapsed(!sidebar.classList.contains('latte-sidebar-collapsed'));};
      mobile.addEventListener('change',()=>{if(!chosen)setCollapsed(mobile.matches);});
      setCollapsed(mobile.matches);
    }
    installCorpusTooltip(panel);
    let chrome=panel.querySelector('.latte-chrome-host');
    if(!chrome){chrome=make('div','latte-chrome-host');panel.prepend(chrome);}
    chrome.innerHTML='<header class="latte-topbar"><div class="latte-brand"><span class="latte-brand-mark" aria-hidden="true"></span><div><strong>One Small Latte</strong><small>Language-model Architecture, Training &amp; Tuning Explorer</small></div></div><span class="latte-local-badge">ON YOUR MACHINE · TRAINING EXPLORER</span></header><footer class="latte-footer"><span>Real training · recorded steps · AdamW</span><span>FULL PRECISION TENSORS · LOCAL PYTORCH</span></footer>';
    const header=panel.querySelector(':scope > header');
    let view=panel.querySelector('.latte-view-host');
    if(!view){view=make('div','latte-view-host');header.append(view);}
    const controls=make('div','latte-view-controls');
    for(const [name,checked] of [['Numbers',true],['Math',false]]){
      const label=make('label','check'),input=document.createElement('input');input.type='checkbox';input.checked=checked;input.id=`latte-${name.toLowerCase()}`;
      input.addEventListener('change',sync);label.append(input,document.createTextNode(name));controls.append(label);
    }
    view.append(controls);
    let optimizer=panel.querySelector('.latte-optimizer-math-host');
    if(!optimizer){optimizer=make('div','latte-optimizer-math-host');panel.querySelector('.training-parameter').before(optimizer);}
    optimizer.innerHTML=window.LatteMath.adamWFormula();
    optimizer.setAttribute('aria-label','AdamW update equations');
    optimizer.title='Symbolic optimizer equations; the actual recorded weight, gradient, and update remain below.';
    const legend=panel.querySelector('.training-legend');
    if(legend){legend.innerHTML='<span><i style="background:#a5abb3"></i>Before weights</span><span><i style="background:#f2a368"></i>Updated weights</span><span><i style="background:#8adb8b"></i>Keys</span><span><i style="background:#df9dd7"></i>Values</span><span><i style="background:#8fa7d8"></i>Queries</span><span><i style="background:#00b5e2"></i>Attention</span><span><i style="background:#f1d753"></i>Input/output</span><span><i style="background:#e0a18b"></i>Recorded loss</span>';}
    if(legend)chrome.querySelector('.latte-footer').replaceChildren(legend);
    panel.querySelector('.training-resize-hint')?.remove();
    const iframe=panel.querySelector('iframe');
    const mathSection=make('section','latte-math-section');mathSection.hidden=true;
    mathSection.innerHTML='<h3>Selected calculation</h3><p>Click a variable to trace it in the diagram.</p><div class="latte-math-content"></div>';
    const sidebarToggle=panel.querySelector('.latte-sidebar-toggle');
    if(sidebarToggle)sidebarToggle.after(mathSection);
    else panel.querySelector('.training-body > aside').prepend(mathSection);
    document.addEventListener('keydown',event=>{if(event.key==='Escape')iframe.contentWindow?.LattePresentation?.clear();});
    function sync(){const math=controls.querySelector('#latte-math').checked;panel.classList.toggle('latte-training-math-enabled',math);const api=iframe.contentWindow?.LattePresentation;if(api){api.setMathHost(mathSection);api.setMathMode(math);api.setNumbers(controls.querySelector('#latte-numbers').checked);}}
    iframe.addEventListener('load',sync);sync();
  }
  if(!document.documentElement.hasAttribute('data-training-replay')){
    const panel=document.querySelector('.training-replay');
    if(panel)enhance(panel);
    else {
      const observer=new MutationObserver(()=>{const panel=document.querySelector('.training-replay');if(panel){enhance(panel);observer.disconnect();}});
      observer.observe(document.body,{childList:true,subtree:true});
    }
  }
})();
