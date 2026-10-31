(() => {
  'use strict';
  const footerInset=document.documentElement.hasAttribute('data-training-replay')?0:30;
  const workbook = window.LittleLlamaWorkbook;
  if (!workbook) return;
  const originalSetZoom = workbook.setZoom;
  let frame = 0;
  let disposed = false;
  let targetZoom = .25;
  let lastPoint = null;
  let cameraZoom = Number(document.documentElement.style.getPropertyValue('--workbook-zoom') || .25);
  const currentZoom = () => cameraZoom;
  const panOffsets = new WeakMap();
  function applyCamera(sheet,dx=0,dy=0) {
    const pan = panOffsets.get(sheet) || {x:0,y:0};
    pan.x += dx; pan.y += dy;
    panOffsets.set(sheet,pan);
    sheet.style.transform = `scale(${cameraZoom}) translate(${pan.x}px,${pan.y}px)`;
  }
  const syncZoomCursor = () => document.body.classList.toggle('can-zoom-in',currentZoom() < 1 || !!frame);
  const activeSheet = () => document.querySelector('.worksheet.active');
  const clampZoom = value => Math.max(.15, Math.min(2, value));
  const contentBounds = new WeakMap();
  function boundsFor(sheet) {
    if (contentBounds.has(sheet)) return contentBounds.get(sheet);
    const origin = sheet.getBoundingClientRect();
    const scale = currentZoom();
    const endpoints = new Map();
    sheet.querySelectorAll('.matrix-value[data-matrix-name]').forEach(cell => {
      if (cell.style.display === 'none' || cell.closest('.stage-card')?.style.display === 'none') return;
      const name=cell.dataset.matrixName;
      if (!endpoints.has(name)) endpoints.set(name,{first:cell,last:cell});
      else endpoints.get(name).last=cell;
    });
    const rects = [...endpoints.values()].flatMap(({first,last}) =>
      [first.getBoundingClientRect(),last.getBoundingClientRect()]);
    if (!rects.length) return null;
    const bounds = {
      left:(Math.min(...rects.map(rect => rect.left))-origin.left)/scale,
      right:(Math.max(...rects.map(rect => rect.right))-origin.left)/scale,
      top:(Math.min(...rects.map(rect => rect.top))-origin.top)/scale,
      bottom:(Math.max(...rects.map(rect => rect.bottom))-origin.top)/scale,
    };
    contentBounds.set(sheet,bounds);
    return bounds;
  }
  function constrainCamera() {
    if (disposed) return;
    const sheet = activeSheet();
    if (!sheet) return;
    const bounds = boundsFor(sheet);
    if (!bounds) return;
    const scale = currentZoom();
    const origin = sheet.getBoundingClientRect();
    const left = window.innerWidth < 760 ? 90 : 255;
    const right = window.innerWidth-16;
    const top = window.innerWidth < 760 ? 136 : 90;
    const bottom = window.innerHeight-footerInset-16;
    const visibleX = Math.min(96,Math.max(20,(right-left)/2),(bounds.right-bounds.left)*scale);
    const x = Math.max(left+visibleX-bounds.right*scale,
      Math.min(right-visibleX-bounds.left*scale,origin.left));
    // Give both ends half a viewport of breathing room: either endpoint
    // can reach the center instead of sticking to the viewport edge.
    const centerY = (top + bottom) / 2;
    const startY = centerY - bounds.top * scale;
    // The generated-token badge extends below the final numeric cell.
    // Measure it live because its label size can change with zoom or output.
    const output = sheet.querySelector('.generated-token-bottom');
    const contentBottom = Math.max(bounds.bottom * scale,
      output && output.closest('.stage-card')?.style.display!=='none' ? output.getBoundingClientRect().bottom - origin.top : 0);
    const endY = Math.min(startY, centerY - contentBottom);
    const y = Math.max(endY, Math.min(startY, origin.top));
    sheet.style.position = 'relative';
    applyCamera(sheet,(x-origin.left)/scale,(y-origin.top)/scale);
  }
  function onCanvas(event, allowStageCheckbox = false) {
    return event.clientY >= (window.innerWidth < 760 ? 126 : 76) && event.clientY < window.innerHeight-footerInset &&
      !event.target.closest('.training-replay,.stage-picker,.stage-group-pick-control,.hierarchy-bracket-label,.stage-view-tabs,.sampling-chart-card,.workbook-toolbar,.selection-clear,.input-matrix-card,.sequence-editor,.generation-settings,.math-panel,.sheet-tabs,textarea,[contenteditable="true"]') &&
      (!event.target.closest('input') || (allowStageCheckbox && !!event.target.closest('.stage-pick-control')));
  }
  function panBy(dx,dy) {
    const sheet = activeSheet();
    if (!sheet) return;
    cancelAnimationFrame(frame);
    frame = 0;
    targetZoom = currentZoom();
    syncZoomCursor();
    sheet.style.position = 'relative';
    applyCamera(sheet,dx/currentZoom(),dy/currentZoom());
    constrainCamera();
    lastPoint = null;
    window.dispatchEvent(new Event('latte:camera'));
  }
  let panFrame = 0;
  let pendingX = 0;
  let pendingY = 0;
  function flushPan() {
    cancelAnimationFrame(panFrame);
    panFrame = 0;
    const dx = pendingX, dy = pendingY;
    pendingX = pendingY = 0;
    if (dx || dy) panBy(dx,dy);
  }
  function queuePan(dx,dy) {
    pendingX += dx;
    pendingY += dy;
    if (!panFrame) panFrame = requestAnimationFrame(flushPan);
  }
  let panMode = false;
  let spaceHeld = false;
  let gesture = null;
  let suppressClick = false;
  const panButton = document.createElement('button');
  panButton.type = 'button';
  panButton.className = 'toolbar-button pan-button';
  panButton.textContent = 'Pan';
  panButton.title = 'Drag to pan. You can also hold Space or use the middle mouse button.';
  panButton.setAttribute('aria-pressed','false');
  document.querySelector('.workbook-toolbar').prepend(panButton);
  panButton.addEventListener('click',() => {
    panMode = !panMode;
    panButton.setAttribute('aria-pressed',String(panMode));
    document.body.classList.toggle('pan-enabled',panMode);
  });
  const recenterButton = document.createElement('button');
  recenterButton.type = 'button';
  recenterButton.className = 'toolbar-button';
  recenterButton.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5"/><path d="M8 12h8M12 8v8"/></svg>';
  recenterButton.setAttribute('aria-label', 'Recenter diagram');
  recenterButton.title = 'Align result matrices with the left brackets';
  panButton.after(recenterButton);
  function recenter() {
    cancelAnimationFrame(boundsFrame); boundsFrame = 0;
    cancelAnimationFrame(frame);
    cancelAnimationFrame(panFrame);
    frame = panFrame = 0;
    pendingX = pendingY = 0;
    targetZoom = currentZoom();
    lastPoint = null;
    const sheet = activeSheet();
    if (!sheet) return;
    const bounds = boundsFor(sheet);
    if (!bounds) return;
    const scale = currentZoom();
    const origin = sheet.getBoundingClientRect();
    const left = window.innerWidth < 760 ? 90 : 255;
    const top = window.innerWidth < 760 ? 136 : 90;
    // Token embeddings share the result-matrix column; auxiliary weights
    // extend farther left and must not determine the reset alignment.
    const resultCell = [...sheet.querySelectorAll('.matrix-value[data-matrix-name="Token embeddings"]')].find(cell=>cell.closest('.stage-card')?.style.display!=='none');
    const resultLeft = resultCell
      ? (resultCell.getBoundingClientRect().left - origin.left) / scale
      : bounds.left;
    const x = left - resultLeft * scale;
    const y = top - bounds.top * scale;
    applyCamera(sheet, (x - origin.left) / scale, (y - origin.top) / scale);
    originalSetZoom(targetZoom);
    constrainCamera();
    syncZoomCursor();
    window.dispatchEvent(new Event('latte:camera'));
  }
  recenterButton.addEventListener('click', recenter);
  document.addEventListener('keydown',event => {
    if (event.code !== 'Space' || event.target.closest('input,textarea,button,[contenteditable="true"]')) return;
    spaceHeld = true;
    document.body.classList.add('pan-enabled');
    event.preventDefault();
  });
  document.addEventListener('keyup',event => {
    if (event.code !== 'Space') return;
    spaceHeld = false;
    document.body.classList.toggle('pan-enabled',panMode);
  });
  document.addEventListener('pointerdown',event => {
    if (!onCanvas(event) || !(panMode || spaceHeld || event.button === 1)) return;
    if (event.target.closest('input,[contenteditable="true"]')) return;
    cancelAnimationFrame(frame);frame=0;
    gesture = {id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};
    document.body.classList.add('is-panning');
    document.documentElement.setPointerCapture(event.pointerId);
    event.preventDefault();
  },true);
  document.addEventListener('pointermove',event => {
    if (!gesture || gesture.id !== event.pointerId) return;
    if (!gesture.moved && Math.hypot(event.clientX-gesture.startX,event.clientY-gesture.startY) < 4) return;
    gesture.moved = true;
    queuePan(event.clientX-gesture.x,event.clientY-gesture.y);
    gesture.x = event.clientX;
    gesture.y = event.clientY;
  });
  const endPan = () => {
    if (gesture) {
      suppressClick = gesture.moved;
      if (document.documentElement.hasPointerCapture(gesture.id)) document.documentElement.releasePointerCapture(gesture.id);
    }
    gesture = null;
    document.body.classList.remove('is-panning');
  };
  document.addEventListener('pointerup',endPan);
  document.addEventListener('pointercancel',endPan);
  window.addEventListener('blur',() => { endPan(); spaceHeld=false; document.body.classList.toggle('pan-enabled',panMode); });

  function animateZoom(next, point) {
    flushPan();
    cancelAnimationFrame(frame);
    const sheet = activeSheet();
    if (!sheet) return;
    const startZoom = currentZoom();
    targetZoom = Math.round(clampZoom(next)*20)/20;
    const anchor = point || lastPoint || {x:window.innerWidth / 2,y:window.innerHeight / 2};
    lastPoint = anchor;
    const initialRect = sheet.getBoundingClientRect();
    const local = {x:(anchor.x-initialRect.left)/startZoom,y:(anchor.y-initialRect.top)/startZoom};
    const start = performance.now();
    const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 340;
    function update(time) {
      if (disposed) return;
      const progress = duration ? Math.min(1,(time-start)/duration) : 1;
      const eased = progress < .5 ? 4*progress**3 : 1-(-2*progress+2)**3/2;
      const scale = startZoom + (targetZoom-startZoom)*eased;
      cameraZoom = scale;
      applyCamera(sheet);
      syncZoomCursor();
      document.body.classList.toggle('color-overview',scale <= .3);
      document.querySelector('.zoom-label').textContent = `${Math.round(scale*100)}%${scale <= .3 ? ' · colors' : ''}`;
      const rect = sheet.getBoundingClientRect();
      // Move the transformed camera directly, keeping the anchor under the
      // pointer without introducing a second, native scrolling camera.
      const dx = anchor.x-(rect.left+local.x*scale);
      const dy = anchor.y-(rect.top+local.y*scale);
      sheet.style.position = 'relative';
      applyCamera(sheet,dx/scale,dy/scale);
      constrainCamera();
      window.dispatchEvent(new Event('latte:camera'));
      if (progress < 1) frame = requestAnimationFrame(update);
      else {
        originalSetZoom(targetZoom);
        frame = 0;
        syncZoomCursor();
      }
    }
    frame = requestAnimationFrame(update);
  }

  document.addEventListener('click',event => {
    if (suppressClick) {
      suppressClick = false;
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (['zoom-in','zoom-out','zoom-reset'].includes(action)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const next = action === 'zoom-reset' ? .25 : targetZoom * (action === 'zoom-in' ? 1.35 : 1/1.35);
      // Preserve the last inspected point, not the toolbar button position.
      animateZoom(next);
      return;
    }
    if (event.target.closest('.stage-content') && !event.target.closest('button,.stage-pick-control')) {
      lastPoint = {x:event.clientX,y:event.clientY};
      if (currentZoom() < 1 || frame) {
        event.preventDefault();
        event.stopImmediatePropagation();
        animateZoom(1,lastPoint);
      }
    }
  },true);
  document.addEventListener('wheel',event => {
    if (event.target.closest('.training-replay,.training-corpus-panel,.sampling-chart-card,.stage-picker,.stage-view-tabs,.kv-memory-panel')) return;
    const overInferenceInput = !!event.target.closest('.sequence-editor,.live-input-token');
    const overSelectionCard=!!event.target.closest('.input-matrix-card');
    if (!overInferenceInput && !overSelectionCard && !onCanvas(event, true)) return;
    if (!(event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      const factor = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
      queuePan(-(event.shiftKey ? event.deltaX || event.deltaY : event.deltaX)*factor,event.shiftKey ? 0 : -event.deltaY*factor);
      return;
    }
    event.preventDefault();
    animateZoom(targetZoom*Math.exp(-Math.max(-100,Math.min(100,event.deltaY))*.004),{x:event.clientX,y:event.clientY});
  },{passive:false});
  let boundsFrame = 0;
  const scheduleBounds = () => {
    if (disposed) return;
    cancelAnimationFrame(boundsFrame);
    boundsFrame = requestAnimationFrame(constrainCamera);
  };
  window.addEventListener('scroll',scheduleBounds,{passive:true});
  window.addEventListener('resize',scheduleBounds);
  window.addEventListener('latte:layout', event => {
    const sheet = activeSheet();
    if (sheet) contentBounds.delete(sheet);
    if(!event.detail?.preserveCamera)scheduleBounds();
  });
  workbook.constrainCamera = constrainCamera;
  workbook.cameraZoom = currentZoom;
  workbook.recenter = recenter;
  const returnButton = document.createElement('button');
  returnButton.type = 'button'; returnButton.className = 'toolbar-button';
  returnButton.textContent = 'Diagram'; returnButton.hidden = true;
  document.querySelector('.workbook-toolbar').append(returnButton);
  const activateSheet = sheet => {
    document.querySelectorAll('.worksheet').forEach(section => {
      section.classList.toggle('active', section === sheet);
      section.style.removeProperty('display');
    });
    if (sheet.id !== 'sheet-0') sheet.style.setProperty('display', 'block', 'important');
    returnButton.hidden = sheet.id === 'sheet-0';
  };
  returnButton.addEventListener('click', () => {
    activateSheet(document.querySelector('#sheet-0')); recenter();
  });
  workbook.disposeCamera = () => {
    disposed = true;
    cancelAnimationFrame(frame); cancelAnimationFrame(boundsFrame); cancelAnimationFrame(panFrame);
    frame = boundsFrame = panFrame = 0; pendingX = pendingY = 0;
  };
  workbook.captureView = () => {
    const sheet = activeSheet();
    return {sheet, zoom:currentZoom(), pan:{...(panOffsets.get(sheet) || {x:0,y:0})}};
  };
  workbook.restoreView = view => {
    if (!view?.sheet?.isConnected) return;
    cancelAnimationFrame(frame); frame = 0;
    cancelAnimationFrame(boundsFrame); boundsFrame = 0;
    cancelAnimationFrame(panFrame); panFrame = 0; pendingX = pendingY = 0;
    activateSheet(view.sheet);
    cameraZoom = targetZoom = view.zoom; lastPoint = null;
    panOffsets.set(view.sheet, {...view.pan});
    originalSetZoom(view.zoom); applyCamera(view.sheet); syncZoomCursor();
    window.dispatchEvent(new Event('latte:camera'));
  };
  workbook.focusCells = (elements, {preserveZoom = false, smooth = false} = {}) => {
    const sheet = elements[0]?.closest('.worksheet');
    if (!sheet) return;
    cancelAnimationFrame(frame); frame = 0;
    cancelAnimationFrame(panFrame); panFrame = 0; pendingX = pendingY = 0;
    activateSheet(sheet);
    cameraZoom = targetZoom = preserveZoom ? currentZoom() : 1; lastPoint = null;
    originalSetZoom(cameraZoom); applyCamera(sheet); syncZoomCursor();
    const startPan={...(panOffsets.get(sheet)||{x:0,y:0})};
    const rects = elements.filter(element => element.closest('.worksheet') === sheet).map(element => element.getBoundingClientRect());
    const left = Math.min(...rects.map(rect => rect.left));
    const right = Math.max(...rects.map(rect => rect.right));
    const top = Math.min(...rects.map(rect => rect.top));
    const bottom = Math.max(...rects.map(rect => rect.bottom));
    const viewport=workbook.presentationViewport?.()||{left:255,right:innerWidth,top:76,bottom:innerHeight-footerInset};
    applyCamera(sheet, ((viewport.left+viewport.right)/2 - (left + right) / 2) / cameraZoom,
      ((viewport.top+viewport.bottom)/2 - (top + bottom) / 2) / cameraZoom);
    constrainCamera();
    if(smooth&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
      const destination={...panOffsets.get(sheet)},started=performance.now();
      panOffsets.set(sheet,startPan);applyCamera(sheet);
      const move=now=>{
        const t=Math.min(1,(now-started)/300),ease=1-Math.pow(1-t,3);
        panOffsets.set(sheet,{x:startPan.x+(destination.x-startPan.x)*ease,y:startPan.y+(destination.y-startPan.y)*ease});applyCamera(sheet);
        window.dispatchEvent(new Event('latte:camera'));
        frame=t<1?requestAnimationFrame(move):0;
      };
      frame=requestAnimationFrame(move);
    }
    window.dispatchEvent(new Event('latte:camera'));
  };
  const jumpToStage = edge => {
    const sheet = activeSheet();
    if (!sheet) return;
    const cards = [...sheet.querySelectorAll('.stage-card')].filter(card =>
      card.style.display !== 'none' && card.getClientRects().length &&
      !card.closest('.layer-group')?.matches(':not([open])'));
    const card = edge === 'top' ? cards[0] : cards[cards.length - 1];
    if (!card) return;
    const values = [...card.querySelectorAll('.matrix-value')].filter(cell => cell.getClientRects().length);
    if (!values.length) return;
    if(card.dataset.kvTimeline==='true'){
      const candidates=values.filter(cell=>edge==='top'?cell.dataset.matrixName.startsWith('x · attention input ·'):cell.dataset.matrixName.startsWith('VA · attention output ·'));
      const name=(edge==='top'?candidates[0]:candidates.at(-1))?.dataset.matrixName;
      if(name)workbook.focusCells(candidates.filter(cell=>cell.dataset.matrixName===name),{preserveZoom:true,smooth:true});
      return;
    }
    // Start at the input matrix; end at the last visible output stage.
    const targets = edge === 'top'
      ? values.filter(cell => cell.dataset.matrixName === values[0].dataset.matrixName)
      : values;
    workbook.focusCells(targets, {preserveZoom: true, smooth: true});
  };
  const jumpGroup = document.createElement('span');
  jumpGroup.className = 'toolbar-group';
  for (const [edge, text] of [['top', '↑ Top'], ['bottom', '↓ Bottom']]) {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'toolbar-button';
    button.textContent = text; button.title = `Go to ${edge} stage`;
    button.dataset.action = `jump-${edge}`;
    button.setAttribute('aria-label', button.title);
    button.addEventListener('click', () => jumpToStage(edge));
    jumpGroup.append(button);
  }
  document.querySelector('.zoom-label')?.parentElement.after(jumpGroup);
  if(document.documentElement.hasAttribute('data-training-replay')){
    const toolbar=document.querySelector('.workbook-toolbar');
    const icon=(button,label,path)=>{
      button.setAttribute('aria-label',label);button.title=label;
      button.innerHTML=`<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg>`;
      return button;
    };
    const plus=icon(toolbar.querySelector('[data-action="zoom-in"]'),'Zoom in','M5 12h14M12 5v14');
    const minus=icon(toolbar.querySelector('[data-action="zoom-out"]'),'Zoom out','M5 12h14');
    icon(recenterButton,'Center diagram','M12 2v4m0 12v4M2 12h4m12 0h4M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0');
    const top=icon(jumpGroup.querySelector('[data-action="jump-top"]'),'Go to input','M5 4h14M12 20V8m-5 5 5-5 5 5');
    const bottom=icon(jumpGroup.querySelector('[data-action="jump-bottom"]'),'Go to output','M5 20h14M12 4v12m-5-5 5 5 5-5');
    const overview=icon(toolbar.querySelector('[data-action="zoom-reset"]'),'Overview','M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6');
    icon(panButton,'Pan diagram','M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3');
    icon(returnButton,'Return to diagram','M9 5l-7 7 7 7M2 12h20');
    const label=toolbar.querySelector('.zoom-label');label.setAttribute('aria-label','Zoom level');
    const separator=document.createElement('span');separator.className='latte-camera-separator';separator.setAttribute('aria-hidden','true');
    toolbar.setAttribute('role','navigation');toolbar.setAttribute('aria-label','Diagram navigation');
    toolbar.replaceChildren(plus,minus,separator,recenterButton,top,bottom,overview,panButton,returnButton,label);
  }

  workbook.setZoom = next => animateZoom(next);
  workbook.zoomToPoint = animateZoom;
  workbook.focusPresentationElements = (elements, {smooth = false} = {}) => {
    if(smooth){
      workbook.focusCells(elements,{preserveZoom:true,smooth:true});
      return;
    }
    const rects=elements.filter(element=>element.isConnected).map(element=>element.getBoundingClientRect());
    if(!rects.length)return;
    const left=Math.min(...rects.map(rect=>rect.left)),right=Math.max(...rects.map(rect=>rect.right));
    const top=Math.min(...rects.map(rect=>rect.top)),bottom=Math.max(...rects.map(rect=>rect.bottom));
    panBy((Math.min(innerWidth-24,Math.max(260,innerWidth/2))-(left+right)/2),
      (Math.max(100,(innerHeight-footerInset)/2)-(top+bottom)/2));
  };
  syncZoomCursor();
})();
