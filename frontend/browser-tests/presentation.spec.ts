import { test, expect } from './fixtures';

test('Escape clears a clicked cell without leaving a stage focus border',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="X · embedding output"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  await expect(page.locator('.latte-math-panel')).toBeVisible();
  expect(await cell.evaluate(element=>element.ownerDocument.activeElement===element.closest('.stage-content'))).toBe(true);
  await page.keyboard.press('Escape');
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
  await expect(frame.locator('.latte-math-focus')).toHaveCount(0);
  const cleared=await cell.evaluate(element=>{
    const stage=element.closest('.stage-content')!;
    return {focused:stage.matches(':focus-visible'),outline:getComputedStyle(stage).outlineStyle};
  });
  expect(cleared.focused).toBe(false);
  expect(cleared.outline).toBe('none');
  // Keyboard navigation still gets its normal focus indicator.
  await cell.evaluate(element=>(element.closest('.stage-content') as HTMLElement).focus());
  expect(await cell.evaluate(element=>element.closest('.stage-content')!.matches(':focus-visible'))).toBe(true);
});

test('detail names sit immediately above indices without overlapping the preceding matrix',async({page})=>{
  await page.goto('/');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const name=frame.locator('[data-matrix-label="WQ · layer 0 · head 0"]');
  await expect(name).toBeVisible();
  for(const zoom of [.35,.55,1,2,.25]){
    await name.evaluate((element,zoom)=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(zoom),zoom);
    await page.waitForTimeout(450);
    await name.evaluate(element=>{
      const doc=element.ownerDocument;
      (doc.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([...doc.querySelectorAll('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"]')]);
    });
    const position=await name.evaluate(element=>{
      const doc=element.ownerDocument;
      const plate=doc.querySelector('[data-name-background="WQ · layer 0 · head 0"]')!.getBoundingClientRect();
      const cells=[...doc.querySelectorAll('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"]')].map(cell=>cell.getBoundingClientRect());
      const first=doc.querySelector('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"]')!;
      const previous=[...first.closest('.stage-card')!.previousElementSibling!.querySelectorAll('.matrix-value')].map(cell=>cell.getBoundingClientRect());
      return {plateTop:plate.top,plateBottom:plate.bottom,centerX:(plate.left+plate.right)/2,centerY:(plate.top+plate.bottom)/2,
        previousBottom:Math.max(...previous.map(r=>r.bottom)),
        matrixTop:Math.min(...cells.map(r=>r.top)),matrixBottom:Math.max(...cells.map(r=>r.bottom)),
        matrixX:(Math.min(...cells.map(r=>r.left))+Math.max(...cells.map(r=>r.right)))/2};
    });
    expect(position.centerX).toBeCloseTo(position.matrixX,1);
    if(zoom>.3){
      const gap=position.matrixTop-12*zoom-position.plateBottom;
      expect(gap).toBeGreaterThan(0);
      expect(gap).toBeLessThan(1.5);
      expect(position.plateTop).toBeGreaterThan(position.previousBottom);
    }
    else expect(position.centerY).toBeCloseTo((position.matrixTop+position.matrixBottom)/2,1);
  }
});

test('Math name hover previews a matrix with a translucent KeVin label plate',async({page})=>{
  await page.goto('/');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const name=frame.locator('[data-matrix-label="Q · layer 0 · head 0"]');
  await name.hover();
  await expect(frame.locator('.latte-math-hover')).toHaveCount(0);
  await page.mouse.move(10,10);
  const ink=await name.evaluate(element=>({fill:Number(getComputedStyle(element).fillOpacity),
    stroke:getComputedStyle(element).stroke}));
  expect(ink.fill).toBe(1);
  expect(ink.stroke).toBe('none');
  const plate=frame.locator('[data-name-background="Q · layer 0 · head 0"]');
  const square=await plate.evaluate(element=>{
    const rect=element.getBoundingClientRect(),name=element.nextElementSibling!.getBoundingClientRect();
    return {fill:getComputedStyle(element).fill,radius:element.getAttribute('rx'),
      centered:Math.abs((rect.left+rect.right-name.left-name.right)/2)<1,
      padding:rect.width-name.width,height:rect.height};
  });
  expect(square.fill).toMatch(/rgba\(.+, 0\./);
  expect(square.radius).toBe('3');
  expect(square.centered).toBe(true);
  expect(square.padding).toBeGreaterThan(5);
  expect(square.height).toBeLessThan(24);
  await page.locator('#latte-math').check();
  await name.hover();
  await expect(frame.locator('.latte-math-hover').first()).toHaveAttribute('data-matrix-name','Q · layer 0 · head 0');
  await expect(frame.locator('.latte-hovered-matrix')).toHaveCount(1);
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
  const value=await frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first().textContent();
  await frame.locator('[data-name-background="V · layer 0 · head 0"]').click({position:{x:4,y:4}});
  await name.hover();
  expect(await frame.locator('.latte-math-hover').first().evaluate(element=>getComputedStyle(element).opacity)).toBe('0.65');
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  expect(await frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first().textContent()).toBe(value);
  await page.mouse.move(10,10);
  await expect(frame.locator('.latte-math-hover')).toHaveCount(0);
  await expect(frame.locator('.latte-hovered-matrix')).toHaveCount(0);
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
});

test('matrix links retain the input connection when its name is offscreen',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="V · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(2));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>{
    const doc=element.ownerDocument;
    (doc.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([...doc.querySelectorAll('.matrix-value[data-matrix-name="V · layer 0 · head 0"]')]);
  });
  await frame.locator('[data-matrix-label="V · layer 0 · head 0"]').click();
  const link=frame.locator('.latte-math-name-link[data-target="Attention input · layer 0"]');
  await expect(link).toHaveCount(1);
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  const node=await link.elementHandle();
  const offscreen=await link.evaluate(element=>{
    const path=element as SVGPathElement,doc=element.ownerDocument;
    const cells=[...doc.querySelectorAll('.matrix-value[data-matrix-name="Attention input · layer 0"]')];
    const first=cells[0].getBoundingClientRect(),last=cells.at(-1)!.getBoundingClientRect();
    const end=path.getPointAtLength(path.getTotalLength());
    return {y:first.top-34.8,endY:end.y,endX:end.x,expectedX:(first.left+last.right)/2};
  });
  expect(offscreen.y).toBeLessThan(76);
  expect(offscreen.endY).toBeCloseTo(offscreen.y,1);
  expect(offscreen.endX).toBeCloseTo(offscreen.expectedX,1);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(.25));
  await page.waitForTimeout(450);
  await expect(frame.locator('[data-matrix-label="Attention input · layer 0"]')).toBeVisible();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  expect(await node!.evaluate(element=>element.isConnected)).toBe(true);
});

test('crossing the bracket hides matrix names and docks b to the row right',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  const badge=frame.locator('.latte-math-badge[aria-label^="b:"]');
  await expect(badge).toHaveAttribute('data-angle','0');
  const offset=await cell.evaluate(element=>{
    const doc=element.ownerDocument,win=doc.defaultView as any;
    const row=doc.querySelector('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"]')!.getBoundingClientRect();
    return row.left-(win.LittleLlamaWorkbook.presentationViewport().left-140);
  });
  await cell.evaluate((element,deltaX)=>element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
    bubbles:true,cancelable:true,clientX:600,clientY:400,deltaX,
  })),offset);
  await expect.poll(()=>badge.getAttribute('data-angle')).toBe('180');
  const state=await badge.evaluate(element=>{
    const doc=element.ownerDocument,win=doc.defaultView as any;
    const row=[...doc.querySelectorAll<HTMLElement>('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"][data-matrix-row="1"]')].map(cell=>cell.getBoundingClientRect());
    const name=doc.querySelector<HTMLElement>('[data-matrix-label="WQ · layer 0 · head 0"]')!;
    return {left:element.getBoundingClientRect().left,right:Math.max(...row.map(r=>r.right)),
      margin:win.LittleLlamaWorkbook.presentationViewport().left,nameHidden:getComputedStyle(name).display==='none',
      layerClip:Number(doc.querySelector('#latte-matrix-names-clip rect')!.getAttribute('x'))};
  });
  expect(state.left).toBeGreaterThan(state.right);
  expect(state.nameHidden).toBe(true);
  expect(state.layerClip).toBe(state.margin);
  await cell.evaluate((element,deltaX)=>element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
    bubbles:true,cancelable:true,clientX:600,clientY:400,deltaX:-deltaX,
  })),offset);
  await expect(badge).toHaveAttribute('data-angle','0');
  await expect(frame.locator('[data-matrix-label="WQ · layer 0 · head 0"]')).toBeVisible();
});

test('operand ink follows bracket margins and unrelated content stays readable',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  await expect(frame.locator('.latte-ribbon')).toHaveCount(2);
  for(const width of [1440,760]){
    await page.setViewportSize({width,height:1000});
    await page.waitForTimeout(150);
    const result=await frame.locator('body').evaluate(body=>{
      const doc=body.ownerDocument,win=doc.defaultView as any;
      const margin=win.LittleLlamaWorkbook.presentationViewport();
      const wire=doc.querySelector<SVGElement>('.latte-math-wire')!;
      const unrelated=doc.querySelector('.matrix-value[data-matrix-name="K · layer 0 · head 0"]')!;
      const names=[...doc.querySelectorAll('[data-matrix-label]')].filter(node=>node.getBoundingClientRect().width>0);
      return {margin,clip:wire.style.clipPath,mask:wire.style.maskImage,composite:wire.style.maskComposite,
        matrixOpacity:Number(getComputedStyle(unrelated).opacity),
        nameOpacity:Math.min(...names.map(node=>Number(getComputedStyle(node).opacity)))};
    });
    expect(result.margin.left).toBeGreaterThan(100);
    expect(result.clip).toContain(`${result.margin.left}px`);
    expect(result.mask).toContain(`${result.margin.left+18}px`);
    expect(result.composite).toBe('intersect');
    expect(result.matrixOpacity).toBe(.4);
    expect(result.nameOpacity).toBe(.6);
  }
});

test('replay keeps offscreen stages current without repainting them until revealed',async({page})=>{
  await page.goto('/');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  await expect(frame.locator('.matrix-value').first()).toBeVisible();
  await frame.locator('body').evaluate(body=>{
    const win=body.ownerDocument.defaultView as any;
    const card=body.querySelector('.matrix-value[data-matrix-name="Logits"]')!.closest('.stage-card')!;
    win.offscreenMutations=0;
    win.paintObserver=new MutationObserver(records=>win.offscreenMutations+=records.length);
    win.paintObserver.observe(card,{attributes:true,characterData:true,childList:true,subtree:true});
  });
  await page.locator('[data-action="next"]').click();
  await expect(page.locator('.training-record')).toHaveValue('1');
  await page.locator('[data-action="sequence-next"]').click();
  await expect(page.locator('.training-sequence')).toHaveValue('1');
  const state=await frame.locator('body').evaluate(body=>{
    const win=body.ownerDocument.defaultView as any;
    win.paintObserver.disconnect();
    const data=win.LittleLlamaWorkbook.matrixData();
    const logits=data.filter((cell:any)=>cell.name==='Logits').map((cell:any)=>({row:cell.row,column:cell.column,value:cell.value}));
    const candidate=data.find((cell:any)=>cell.name==='Logits'&&
      (Math.abs(cell.value)<.005?0:cell.value).toFixed(2)!==cell.element.querySelector('font').textContent);
    if(!candidate)throw Error('Expected an updated offscreen logit');
    win.pendingLogit=candidate;
    return {mutations:win.offscreenMutations,logits};
  });
  expect(state.mutations).toBe(0);
  const recorded=await (await page.request.get(`/api/runs/${process.env.LATTE_TEST_RUN||'default'}/steps/1`)).json();
  for(const cell of state.logits)expect(cell.value).toBeCloseTo(recorded.logits[1][cell.column][cell.row],4);
  await frame.locator('body').evaluate(body=>{
    const win=body.ownerDocument.defaultView as any;
    win.LittleLlamaWorkbook.focusPresentationElements([win.pendingLogit.element]);
  });
  const painted=await frame.locator('body').evaluate(body=>{
    const cell=(body.ownerDocument.defaultView as any).pendingLogit;
    return {actual:cell.element.querySelector('font').textContent,expected:(Math.abs(cell.value)<.005?0:cell.value).toFixed(2)};
  });
  expect(painted.actual).toBe(painted.expected);
});

test('camera movement leaves inactive diagram pickers untouched',async({page})=>{
  await page.goto('/');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  await expect(frame.locator('.matrix-value').first()).toBeVisible({timeout:60000});
  const result=await frame.locator('body').evaluate(async body=>{
    const doc=body.ownerDocument,win=doc.defaultView!;
    const controls=[...doc.querySelectorAll<HTMLElement>('.stage-pick-control')];
    let mutations=0;
    const observer=new MutationObserver(records=>mutations+=records.length);
    controls.forEach(control=>observer.observe(control,{attributes:true}));
    for(let i=0;i<6;i++){
      doc.querySelector('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
        bubbles:true,cancelable:true,clientX:600,clientY:400,deltaY:i<3?18:-18,
      }));
      await new Promise(resolve=>win.requestAnimationFrame(()=>win.requestAnimationFrame(resolve)));
    }
    observer.disconnect();
    const hidden=controls.every(control=>control.hidden);
    const picker=[...doc.querySelectorAll<HTMLButtonElement>('.steps-menu button')].find(button=>button.textContent==='Pick on diagram')!;
    picker.click();
    const active=controls.filter(control=>!control.hidden&&control.style.visibility==='visible').length;
    picker.click();
    return {controls:controls.length,hidden,mutations,active};
  });
  expect(result.controls).toBeGreaterThan(0);
  expect(result.hidden).toBe(true);
  expect(result.mutations).toBe(0);
  expect(result.active).toBeGreaterThan(0);
});

for(const mode of ['matrix','cell'] as const){
  test(`scroll retains ${mode} overlays and selection on every camera update`,async({page})=>{
    const errors:string[]=[];
    page.on('pageerror',error=>errors.push(String(error)));
    await page.goto('/');
    await page.locator('#latte-math').check();
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    if(mode==='matrix')await frame.locator('[data-matrix-label="V · layer 0 · head 0"]:visible').click();
    else {
      const cell=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
      await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
      await expect.poll(()=>cell.evaluate(element=>element.getBoundingClientRect().width)).toBeGreaterThan(40);
      await page.waitForTimeout(450);
      await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
      await cell.click();
      await page.waitForTimeout(450);
      await expect(frame.locator('.latte-ribbon')).toHaveCount(2);
    }
    await expect(page.locator('.latte-math-panel')).toBeVisible();
    const stats=await frame.locator('body').evaluate(async(body,mode)=>{
      const doc=body.ownerDocument,win=doc.defaultView!;
      const hosts=[doc.querySelector('.latte-matrix-outlines')!,doc.querySelector('.latte-math-wire')!];
      const originalNodes=hosts.flatMap(host=>[...host.querySelectorAll('*')]);
      const label=doc.querySelector('[data-matrix-label="V · layer 0 · head 0"]')!;
      let removed=0,cameraEvents=0,selectionLost=0;
      const observer=new MutationObserver(records=>{for(const record of records)removed+=record.removedNodes.length;});
      hosts.forEach(host=>observer.observe(host,{childList:true,subtree:true}));
      const onCamera=()=>{
        cameraEvents++;
        if(mode==='matrix'&&!label.classList.contains('latte-name-connected'))selectionLost++;
      };
      win.addEventListener('latte:camera',onCamera);
      for(let i=0;i<30;i++){
        doc.querySelector('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
          bubbles:true,cancelable:true,clientX:600,clientY:400,deltaY:i<15?2:-2,
        }));
        await new Promise(resolve=>win.requestAnimationFrame(()=>win.requestAnimationFrame(resolve)));
      }
      observer.disconnect();win.removeEventListener('latte:camera',onCamera);
      return {cameraEvents,selectionLost,removed,retained:originalNodes.every(node=>node.isConnected)};
    },mode);
    expect(stats.cameraEvents).toBe(30);
    expect(stats.selectionLost).toBe(0);
    expect(stats.removed).toBe(0);
    expect(stats.retained).toBe(true);
    expect(errors).toEqual([]);
  });
}

test('Math mode explains actual training operands and clears without changing the run', async ({ page }) => {
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(String(error)));
  await page.goto('/');
  await expect(page.locator('.latte-brand strong')).toHaveText('One Small Latte');
  await expect(page.locator('#latte-math')).not.toBeChecked();
  await expect(page.locator('#latte-numbers')).toBeChecked();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const query=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
  await expect(query).toBeVisible();
  await page.waitForTimeout(500);
  await query.click();
  await page.waitForTimeout(450);
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
  const original=await query.textContent();
  await page.locator('#latte-math').check();
  await query.click();
  await expect(page.locator('.latte-math-panel')).toBeVisible();
  await expect(page.locator('.latte-math-panel math')).toContainText('a=b_·c_');
  for(const [letter,kind] of [['a','scalar'],['b','vector'],['c','vector']]){
    const badge=frame.locator(`.latte-math-badge[aria-label^="${letter}:"]`);
    await expect(badge).toHaveAttribute('data-math-kind',kind);
    await expect(badge.locator('span')).toHaveText(letter);
    if(kind==='vector')await expect(badge.locator('span')).toHaveCSS('text-decoration-line','underline');
    const face=await badge.locator('span').evaluate(element=>({weight:getComputedStyle(element).fontWeight,style:getComputedStyle(element).fontStyle}));
    expect(face).toEqual({weight:'400',style:'normal'});
    const token=page.locator(`.latte-math-panel math [data-math-variable="${letter}"]`).first();
    await expect(token).toHaveCSS('font-weight','400');
    await expect(token).toHaveCSS('font-style','normal');
    await expect(page.locator(`.latte-variable-button[data-math-variable="${letter}"]`)).toHaveCSS('font-weight','400');
  }
  await expect(frame.locator('.latte-math-badge')).toHaveCount(3);
  await expect(frame.locator('.latte-range-outline')).toHaveCount(3);
  await expect(frame.locator('.latte-ribbon')).toHaveCount(2);
  await expect(frame.locator('.latte-ribbon[data-orientation="vertical"]')).toHaveCount(1);
  await expect(frame.locator('.latte-ribbon[data-orientation="horizontal"]')).toHaveCount(1);
  expect(await frame.locator('.latte-math-focus').evaluateAll(elements=>elements.every(element=>getComputedStyle(element).boxShadow==='none'))).toBe(true);
  const cellWidth=await query.evaluate(element=>element.getBoundingClientRect().width);
  expect(Number(await frame.locator('.latte-ribbon[data-orientation="vertical"]').getAttribute('data-source-width'))).toBeCloseTo(cellWidth,1);
  const weightBadge=frame.locator('.latte-math-badge[aria-label^="b:"]');
  await query.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await expect(weightBadge).toHaveAttribute('data-angle','0');
  await query.evaluate(element=>element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
    bubbles:true,cancelable:true,clientX:300,clientY:400,deltaX:220,deltaY:0,
  })));
  await expect.poll(async()=>Math.cos(Number(await weightBadge.getAttribute('data-angle'))*Math.PI/180)).toBeLessThan(0);
  await query.evaluate(element=>element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
    bubbles:true,cancelable:true,clientX:300,clientY:400,deltaX:-220,deltaY:0,
  })));
  await expect(weightBadge).toHaveAttribute('data-angle','0');
  await expect(page.locator('.latte-math-panel math')).toHaveAttribute('xmlns','http://www.w3.org/1998/Math/MathML');
  expect(await query.textContent()).toBe(original);
  await expect(page.locator('.latte-optimizer-math-host math')).toHaveCount(5);
  await expect(page.locator('.training-inspection')).toContainText('Gradient');
  await page.locator('[data-action="apply"]').click();
  await expect(page.locator('.training-status')).toContainText('After update');
  await expect(page.locator('.latte-math-panel')).toBeVisible();
  await page.locator('#latte-math').uncheck();
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
  await expect(page.locator('.latte-optimizer-math-host')).toBeHidden();
  await page.locator('#latte-numbers').uncheck();
  await expect(frame.locator('body')).toHaveClass(/latte-hide-numbers/);
  await page.locator('#latte-numbers').check();
  await expect(frame.locator('body')).not.toHaveClass(/latte-hide-numbers/);
  await expect(page.locator('.training-record')).toHaveValue('0');
  await expect(page.locator('body')).not.toContainText('KV memory');
  expect(errors).toEqual([]);
});

test('matrix titles show Latte equations and Escape dismisses the card', async ({ page }) => {
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const query=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]').first();
  await query.click();
  await page.waitForTimeout(450);
  await frame.locator('[data-matrix-label="Q · layer 0 · head 0"]').click();
  await expect(page.getByRole('region',{name:'Matrix calculation'})).toBeVisible();
  await expect(page.locator('.latte-math-panel')).toContainText('Q');
  await expect(frame.locator('[data-matrix-label="Q · layer 0 · head 0"]:visible')).toHaveCSS('font-weight','400');
  await expect(page.locator('.latte-math-panel math mi').filter({hasText:/^Q$/}).first()).toHaveCSS('font-weight','400');
  await expect(page.locator('.latte-variable-button[data-math-variable="Q"]')).toHaveCSS('font-weight','400');
  await expect(page.locator('.latte-math-panel')).not.toContainText('RoPE');
  await page.keyboard.press('Escape');
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
});

test('matrix variables sit at matrix centers and open equations directly from overview', async ({ page }) => {
  await page.goto('/');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const name=frame.locator('[data-matrix-label="Q · layer 0 · head 0"]');
  await expect(name).toBeVisible();
  expect(await name.evaluate(element=>element.firstChild?.textContent)).toBe('Q');
  const center=await name.evaluate(element=>{
    const cells=[...element.ownerDocument.querySelectorAll<HTMLElement>('.matrix-value[data-matrix-name="Q · layer 0 · head 0"]')];
    const first=cells[0].getBoundingClientRect(),last=cells.at(-1)!.getBoundingClientRect();
    return {x:Number(element.getAttribute('x')),y:Number(element.getAttribute('y')),
      expectedX:(first.left+last.right)/2,expectedY:(first.top+last.bottom)/2};
  });
  expect(center.x).toBeCloseTo(center.expectedX,1);
  expect(center.y).toBeCloseTo(center.expectedY,1);
  await page.locator('#latte-math').check();
  await name.click();
  await expect(page.getByRole('region',{name:'Matrix calculation'})).toBeVisible();
  await expect(page.locator('.latte-math-panel')).toContainText('Q');
  await expect(page.locator('.latte-math-panel')).toContainText('W');
  await expect(page.locator('.latte-math-panel')).toContainText('X');
  await expect(frame.locator('.latte-math-badge')).toHaveCount(0);
  await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  await frame.locator('.latte-clear-selection').click();
  await expect(page.locator('.latte-math-panel')).toHaveCount(0);
});

test('clicking V connects its named weight and input matrices without scalar labels',async({page})=>{
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(String(error)));
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  await frame.locator('[data-matrix-label="V · layer 0 · head 0"]').click();
  await expect(page.getByRole('region',{name:'Matrix calculation'})).toBeVisible();
  await expect(frame.locator('.latte-math-badge')).toHaveCount(0);
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  const targets=await frame.locator('.latte-math-name-link').evaluateAll(paths=>paths.map(path=>path.getAttribute('data-target')).sort());
  expect(targets).toEqual(['Attention input · layer 0','WV · layer 0 · head 0'].sort());
  await expect(frame.locator('[data-matrix-label="WV · layer 0 · head 0"]')).toHaveClass(/latte-name-connected/);
  await expect(frame.locator('[data-matrix-label="Attention input · layer 0"]')).toHaveClass(/latte-name-connected/);
  await expect(frame.locator('[data-matrix-label="Q · layer 0 · head 0"]')).not.toHaveClass(/latte-name-connected/);
  await page.locator('[data-action="apply"]').click();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  await expect(frame.locator('.latte-math-badge')).toHaveCount(0);
  await frame.locator('.latte-clear-selection').click();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(0);
  await expect(frame.locator('.latte-name-connected')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('output math connectors stay attached to live labels after camera movement',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const probabilities=frame.locator('.matrix-value[data-matrix-name="Vocabulary probabilities"]');
  await probabilities.first().waitFor();
  await page.waitForTimeout(450);
  await probabilities.first().evaluate(element=>{
    const cells=[...element.ownerDocument.querySelectorAll('.matrix-value[data-matrix-name="Vocabulary probabilities"]')];
    (element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements(cells);
  });
  const name=frame.locator('[data-matrix-label="Vocabulary probabilities"]:visible');
  await name.click();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(1);
  for(const deltaY of [80,-80,120,-120]){
    await probabilities.first().evaluate((element,deltaY)=>element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
      bubbles:true,cancelable:true,clientX:650,clientY:400,deltaY,
    })),deltaY);
    await page.waitForTimeout(100);
    const anchors=await name.evaluate(label=>{
      const doc=label.ownerDocument,source=label.getBoundingClientRect();
      const target=[...doc.querySelectorAll('[data-matrix-label="Logits"]')].map(node=>node.getBoundingClientRect()).find(rect=>rect.width>0&&rect.height>0)!;
      const link=doc.querySelector<SVGPathElement>('.latte-math-name-link')!;
      const start=link.getPointAtLength(0),end=link.getPointAtLength(link.getTotalLength());
      const card=doc.querySelector('.latte-card-link');
      return {
        startsAtSource:start.x>=source.left-1&&start.x<=source.right+1&&start.y>=source.top-1&&start.y<=source.bottom+1,
        endsAtLogits:end.x>=target.left-1&&end.x<=target.right+1&&end.y>=target.top-1&&end.y<=target.bottom+1,
        floatingCard:!!card,
      };
    });
    expect(anchors.startsAtSource).toBe(true);
    expect(anchors.endsAtLogits).toBe(true);
    expect(anchors.floatingCard).toBe(false);
  }
});

test('softmax has a separate ribbon and label for the specific numerator score',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="A · layer 0 · head 0"][data-matrix-row="5"][data-matrix-column="8"]');
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  await expect(frame.locator('.latte-math-badge').filter({hasText:'b₄'})).toBeVisible();
  await expect(frame.locator('.latte-ribbon[data-operand="b"]')).toHaveCount(1);
  const ribbon=frame.locator('.latte-ribbon[data-operand="b₄"]');
  await expect(ribbon).toHaveCount(1);
  const score=frame.locator('.matrix-value[data-matrix-name="Scores · layer 0 · head 0"][data-matrix-row="5"][data-matrix-column="8"]');
  const target=await score.evaluate(element=>{
    const r=element.getBoundingClientRect();
    return {x:r.x,y:r.y,width:r.width,height:r.height};
  });
  const path=await ribbon.getAttribute('d');
  const endpoint=path!.match(/ C[^C]+? ([\d.-]+),([\d.-]+) L/)!;
  expect(Number(endpoint[1])).toBeGreaterThan(target!.x);
  expect(Number(endpoint[1])).toBeLessThan(target!.x+target!.width);
  expect(Number(endpoint[2])).toBeCloseTo(target!.y+target!.height,1);
  await expect(page.locator('.latte-math-panel mn').first()).toHaveText('4');
  await page.keyboard.press('Escape');
  await expect(ribbon).toHaveCount(0);
});

test('all scalar operations use narrow tapes while matrix products retain full vector tapes',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  for(const name of ['X · embedding output','Residual attention · layer 0','GELU · layer 0','Attention input · layer 0','Q · layer 0 · head 0']){
    const cell=frame.locator(`.matrix-value[data-matrix-name="${name}"]`).first();
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
    await page.waitForTimeout(350);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await cell.click();
    await expect(page.locator('.latte-math-panel')).toBeVisible();
    const widths=await frame.locator('.latte-ribbon').evaluateAll(elements=>elements.map(element=>({
      scalar:element.getAttribute('data-kind')==='scalar',
      width:Number(element.getAttribute('data-source-width')),
    })));
    expect(widths.length).toBeGreaterThan(0);
    for(const ribbon of widths){
      if(ribbon.scalar){expect(ribbon.width).toBeGreaterThan(0);expect(ribbon.width).toBeLessThanOrEqual(12);}
      else expect(ribbon.width).toBeGreaterThan(12);
    }
    if(name==='Q · layer 0 · head 0')expect(widths.every(ribbon=>!ribbon.scalar)).toBe(true);
    else expect(widths.some(ribbon=>ribbon.scalar)).toBe(true);
    if(name==='Attention input · layer 0'){
      expect(widths.some(ribbon=>!ribbon.scalar)).toBe(true);
      expect(widths).toHaveLength(3);
      await expect(page.locator('.latte-math-panel')).toContainText('mean');
      await expect(page.locator('.latte-math-panel')).toContainText('ε');
      await expect(frame.locator('.latte-math-badge').filter({hasText:/[bc]₀/})).toHaveCount(1);
      await expect(frame.locator('.latte-math-badge').filter({hasText:/[bc][₁₂₃]/})).toHaveCount(0);
    }
    await page.keyboard.press('Escape');
  }
});

test('tracing scalar and vector operands keeps ribbons when the output scrolls away',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  for(const name of ['Q · layer 0 · head 0','GELU · layer 0','Attention input · layer 0','A · layer 0 · head 0']){
    const cell=frame.locator(`.matrix-value[data-matrix-name="${name}"]`).first();
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(2));
    await page.waitForTimeout(350);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await cell.click();
    const count=await frame.locator('.latte-ribbon').count();
    expect(count).toBeGreaterThan(0);
    const traced=await cell.evaluate(async element=>{
      const doc=element.ownerDocument,win=doc.defaultView as any,book=win.LittleLlamaWorkbook;
      const groups=win.LatteMath.explainCell(book.presentationCell(element)).groups;
      const center=(el:Element)=>{const r=el.getBoundingClientRect();return (r.top+r.bottom)/2;};
      const outputY=center(element);
      groups.sort((a:any,b:any)=>Math.abs(center(b.sources[0].element)-outputY)-Math.abs(center(a.sources[0].element)-outputY));
      const group=groups[0],operand=group.sources[0].element;
      const ribbon=doc.querySelector(`.latte-ribbon[data-operand="${group.label}"]`);
      book.focusPresentationElements([operand]);
      await new Promise(resolve=>win.requestAnimationFrame(()=>win.requestAnimationFrame(resolve)));
      const r=element.getBoundingClientRect(),viewport=book.presentationViewport();
      if(r.bottom>viewport.top&&r.top<viewport.bottom){
        const above=center(operand)<center(element);
        const delta=above?r.top-viewport.bottom-40:r.bottom-viewport.top+40;
        element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{
          bubbles:true,cancelable:true,clientX:600,clientY:400,deltaY:delta,
        }));
        await new Promise(resolve=>win.requestAnimationFrame(()=>win.requestAnimationFrame(resolve)));
      }
      const out=element.getBoundingClientRect(),input=operand.getBoundingClientRect();
      return {label:group.label,retained:ribbon?.isConnected,outputOff:out.top>=viewport.bottom||out.bottom<=viewport.top,
        inputVisible:input.top<viewport.bottom&&input.bottom>viewport.top};
    });
    expect(traced.outputOff,name).toBe(true);
    expect(traced.inputVisible,name).toBe(true);
    expect(traced.retained,name).toBe(true);
    await expect(frame.locator('.latte-math-wire')).toBeVisible();
    await expect(frame.locator(`.latte-math-badge[aria-label^="${traced.label}:"]`)).toBeVisible();
    await expect(frame.locator('.latte-ribbon')).toHaveCount(count);
    await expect(page.locator('.latte-math-panel')).toBeVisible();
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await expect(page.locator('.latte-math-panel')).toBeVisible();
    await expect(frame.locator('.latte-ribbon')).toHaveCount(count);
    await page.keyboard.press('Escape');
    await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
  }
});

test('matrix-name connections remain visible while tracing an offscreen dependency',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="V · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(2));
  await page.waitForTimeout(350);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await frame.locator('[data-matrix-label="V · layer 0 · head 0"]:visible').click();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  const input=frame.locator('.matrix-value[data-matrix-name="Attention input · layer 0"]').first();
  await input.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await expect(page.locator('.latte-math-panel')).toBeVisible();
  await expect(frame.locator('.latte-math-wire')).toBeVisible();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await expect(page.locator('.latte-math-panel')).toBeVisible();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
});

test('tracing a horizontal weight row keeps its ribbon when the result leaves the right edge',async({page})=>{
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="Q · layer 0 · head 0"][data-matrix-row="2"][data-matrix-column="8"]');
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(2));
  await page.waitForTimeout(350);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  const weight=frame.locator('.matrix-value[data-matrix-name="WQ · layer 0 · head 0"][data-matrix-row="2"]').first();
  await weight.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  expect(await cell.evaluate(element=>element.getBoundingClientRect().left>=element.ownerDocument.defaultView!.innerWidth)).toBe(true);
  await expect(frame.locator('.latte-math-wire')).toBeVisible();
  await expect(frame.locator('.latte-ribbon[data-operand="b"]')).toHaveCount(1);
  await expect(frame.locator('.latte-math-badge[aria-label^="b:"]')).toBeVisible();
  await expect(page.locator('.latte-math-panel')).toBeVisible();
});

for(const width of [1440,760]){
  test(`sidebar equation traces variables without losing context at ${width}px`,async({page})=>{
    await page.setViewportSize({width,height:1000});
    await page.goto('/');
    await page.locator('#latte-math').check();
    const frame=page.frameLocator('iframe[title="Training forward pass"]');
    const cell=frame.locator('.matrix-value[data-matrix-name="A · layer 0 · head 0"][data-matrix-row="5"][data-matrix-column="8"]');
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
    await page.waitForTimeout(350);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await cell.click();
    const panel=page.locator('.latte-math-panel');
    await expect(panel).toBeVisible();
    await expect(frame.locator('.latte-math-panel')).toHaveCount(0);
    const fixed=await panel.boundingBox();
    const sidebar=await page.locator('.training-body > aside').boundingBox();
    expect(fixed!.x).toBeGreaterThan(sidebar!.x);
    expect(fixed!.x+fixed!.width).toBeLessThanOrEqual(sidebar!.x+sidebar!.width);
    const numerator=panel.locator('math [data-math-variable="b₄"]');
    await expect(numerator).toHaveCount(1);
    await numerator.click();
    const score=frame.locator('.matrix-value[data-matrix-name="Scores · layer 0 · head 0"][data-matrix-row="5"][data-matrix-column="8"]');
    await expect(score).toHaveClass(/latte-math-destination/);
    await page.waitForTimeout(400);
    const zoom=await score.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.cameraZoom());
    expect(zoom).toBe(1);
    expect(await panel.boundingBox()).toEqual(fixed);
    const scoreVisible=await score.evaluate(element=>{
      const r=element.getBoundingClientRect(),v=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.presentationViewport();
      return r.left>=v.left&&r.right<=v.right&&r.top>=v.top&&r.bottom<=v.bottom;
    });
    expect(scoreVisible).toBe(true);
    await panel.getByRole('button',{name:'Back to result a',exact:true}).click();
    await expect(cell).toHaveClass(/latte-math-destination/);
    await page.waitForTimeout(400);
    expect(await panel.boundingBox()).toEqual(fixed);
    await panel.locator('math [data-math-variable="b"]').first().focus();
    await page.keyboard.press('Enter');
    await expect(frame.locator('.latte-ribbon[data-operand="b"]')).toHaveClass(/latte-tracing-ribbon/);
    await panel.getByRole('button',{name:'Close calculation',exact:true}).click();
    await expect(panel).toHaveCount(0);
    await expect(page.locator('.latte-math-section')).toBeHidden();
    await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
  });
}

test('named matrix links navigate smoothly and manual scrolling interrupts the camera',async({page})=>{
  const errors:string[]=[];
  page.on('pageerror',error=>errors.push(String(error)));
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const cell=frame.locator('.matrix-value[data-matrix-name="V · layer 0 · head 0"]').first();
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(350);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await frame.locator('[data-matrix-label="V · layer 0 · head 0"]:visible').click();
  const equation=page.locator('.latte-math-panel');
  const input=equation.locator('math [data-math-variable="Xₙ"]');
  await expect(input).toHaveCount(1);
  await input.click();
  await page.waitForTimeout(70);
  await cell.evaluate(async element=>{
    const win=element.ownerDocument.defaultView!;
    element.closest('.stage-content')!.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,clientX:600,clientY:400,deltaY:50}));
    await new Promise(resolve=>win.requestAnimationFrame(()=>win.requestAnimationFrame(resolve)));
  });
  const interrupted=await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.captureView().pan);
  await page.waitForTimeout(400);
  expect(await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.captureView().pan)).toEqual(interrupted);
  await expect(equation).toBeVisible();
  await expect(frame.locator('.latte-math-name-link')).toHaveCount(2);
  await expect(frame.locator('.latte-name-connected')).not.toHaveCount(0);
  await equation.getByRole('button',{name:'Back to result',exact:true}).click();
  await page.waitForTimeout(400);
  await expect(cell).toHaveClass(/latte-math-destination/);
  await page.locator('[data-action="next"]').click();
  await expect(page.locator('.training-record')).toHaveValue('1');
  await expect(equation).toBeVisible();
  await expect(equation.locator('math [data-math-variable="Xₙ"]')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('weight selections explain their origin without downstream links',async({page})=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(String(error)));
  await page.goto('/');
  await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const names=['WQ · layer 0 · head 0','WK · layer 0 · head 0','WV · layer 0 · head 0','WO · layer 0',
    'W1 · layer 0','W2 · layer 0','Token embeddings','Position embeddings','Vocabulary weights',
    'RMS scale · Attention input · layer 0','RMS scale · FFN input · layer 0','RMS scale · Final hidden'];
  for(const name of names){
    const cell=frame.locator(`.matrix-value[data-matrix-name="${name}"]`).first();
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
    await page.waitForTimeout(350);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await frame.locator(`[data-matrix-label="${name}"]:visible`).click();
    const panel=page.locator('.latte-math-panel');
    await expect(panel).toContainText('learned parameter');
    await expect(panel).toContainText(name.startsWith('RMS scale')?'Initialized to ones':'Initialized randomly');
    await expect(frame.locator('.latte-math-name-link')).toHaveCount(0);
    await expect(panel.locator('.latte-variable-button')).toHaveCount(1);
    await page.keyboard.press('Escape');
    await cell.click();
    await expect(panel).toContainText('Parameter entering training step 1.');
    await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
    await expect(frame.locator('.latte-math-badge')).toHaveCount(1);
    const before=await cell.evaluate(element=>{
      const book=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook;
      return {origin:book.weightOrigin(element),value:book.presentationCell(element).value};
    });
    expect(before.origin.before,name).toBeCloseTo(before.value,7);
    await page.locator('[data-action="apply"]').click();
    await expect(panel).toContainText('Updated by AdamW at training step 1');
    await expect(panel).toContainText('batch of 4 sequences');
    await expect(panel.locator('details')).not.toHaveAttribute('open');
    const after=await cell.evaluate(element=>{
      const book=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook;
      return {origin:book.weightOrigin(element),value:book.presentationCell(element).value};
    });
    expect(after.origin.applied,name).toBe(true);
    expect(after.origin.after,name).toBeCloseTo(after.value,7);
    expect(after.origin.before,name).toBe(before.origin.before);
    await expect(frame.locator('.latte-ribbon')).toHaveCount(0);
    await expect(page.locator('.training-weight-section')).toBeHidden();
    await page.locator('[data-action="apply"]').click();
    await expect(panel).toContainText('Parameter entering training step 1.');
    await page.keyboard.press('Escape');
  }
  expect(errors).toEqual([]);
});

test('corpus prediction tooltip opens only on click or keyboard activation',async({page})=>{
  await page.goto('/');
  await page.locator('.training-locate').click();
  const token=page.locator('[data-corpus-position="0"]');
  const tip=page.getByRole('tooltip');
  const values=await token.evaluate(element=>({...((element as HTMLElement).dataset)}));
  const visible=(value:string)=>value===' '?'␣':value==='\n'?'↵':value==='\t'?'⇥':value;
  await token.hover();
  await expect(tip).toBeHidden();
  await token.focus();
  await expect(tip).toBeHidden();
  await page.keyboard.press('Enter');
  await expect(tip).toContainText(`Input ${visible(values.input!)} → Predicted ${visible(values.prediction!)} · Target ${visible(values.target!)}`);
  await expect(tip).toContainText('recorded before update');
  await expect(tip).toContainText(`P(target) ${(100*Number(values.targetProbability)).toFixed(1)}%`);
  await expect(page.locator('.training-predictions-section')).toHaveCount(0);
  await page.mouse.move(0,0);await expect(tip).toBeVisible();
  await token.click();await expect(tip).toBeHidden();
  await token.click();await expect(tip).toBeVisible();
  await page.keyboard.press('Escape');await expect(tip).toBeHidden();
  await page.locator('.training-locate').focus();await token.focus();await expect(tip).toBeHidden();
  await page.keyboard.press('Space');await expect(tip).toBeVisible();
  await page.locator('[data-action="sequence-next"]').click();await expect(tip).toBeHidden();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  await expect(frame.locator('.explorer-footer')).toBeHidden();
  const viewport=await frame.locator('body').evaluate(element=>{
    const win=element.ownerDocument.defaultView as any;
    return {bottom:win.LittleLlamaWorkbook.presentationViewport().bottom,height:win.innerHeight};
  });
  expect(viewport.bottom).toBe(viewport.height);
});

test('predicted character selections connect to the corresponding probability column',async({page})=>{
  await page.goto('/');await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  for(const column of [1,7]){
    const cell=frame.locator(`.training-output-character[data-matrix-column="${column}"]`);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
    await page.waitForTimeout(450);
    await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
    await cell.click();
    const dependencies=await cell.evaluate(element=>{
      const book=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook;
      return book.presentationCell(element).sources.map((source:any)=>({name:source.name,column:source.column}));
    });
    expect(dependencies.length).toBeGreaterThan(1);
    expect(dependencies.every((source:any)=>source.name==='Vocabulary probabilities'&&source.column===column-1)).toBe(true);
    await expect(frame.locator('.latte-ribbon')).toHaveCount(1);
    await expect(page.locator('.latte-math-panel math')).toContainText('argmax');
    await expect(page.locator('.latte-variable-button[data-math-variable="b"]')).toBeVisible();
    await page.keyboard.press('Escape');
  }
});

test('recorded loss connects to its target probability and preserves before-update provenance',async({page})=>{
  await page.goto('/');await page.locator('#latte-math').check();
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const loss=frame.locator('.matrix-value[data-matrix-name="Recorded loss before update"][data-matrix-column="7"]');
  await loss.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await loss.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await loss.click();
  const read=()=>loss.evaluate(element=>{
    const book=(element.ownerDocument.defaultView as any).LittleLlamaWorkbook;
    const result=book.presentationCell(element),source=result.sources[0];
    return {value:result.value,recorded:result.recordedLoss,count:result.sources.length,row:source.row,column:source.column,
      expectedRow:Number((element as HTMLElement).dataset.recordedTargetRow)-1,p:book.presentationCell(source.element).value};
  });
  const before=await read();
  expect(before.count).toBe(1);expect(before.row).toBe(before.expectedRow);expect(before.column).toBe(6);
  expect(-Math.log(before.p)).toBeCloseTo(before.value,5);
  await expect(frame.locator('.latte-ribbon')).toHaveCount(1);
  await page.locator('[data-action="apply"]').click();
  await expect(page.locator('.latte-math-panel')).toContainText('the connected cell now shows the updated probability');
  const after=await read();expect(after.value).toBe(before.value);expect(after.recorded.updated).toBe(true);
  await page.locator('[data-action="sequence-next"]').click();
  await expect.poll(async()=>{const result=await read();return result.row===result.expectedRow;}).toBe(true);
});

test('visible matrix and token indices start at zero and operand subscripts bind the same cells',async({page})=>{
  await page.goto('/');
  await expect(page.locator('.training-predictions-section')).toHaveCount(0);
  await expect(page.locator('[data-corpus-position="7"]')).toHaveCount(1);
  await expect(page.locator('[data-corpus-position="0"]')).toHaveCount(1);
  await expect(page.locator('.training-sequence option').first()).toHaveText('0 / 3');
  const frame=page.frameLocator('iframe[title="Training forward pass"]');
  const first=frame.locator('.matrix-value[data-matrix-name="Token embeddings"][data-matrix-row="1"][data-matrix-column="1"]');
  await expect(first).toHaveAttribute('data-column-index','0');
  await expect(first).toHaveAttribute('data-row-index','0');
  const allIndices=await frame.locator('.matrix-value').evaluateAll(elements=>elements.every(element=>{
    const d=(element as HTMLElement).dataset;
    return (d.columnIndex===undefined||Number(d.columnIndex)===Number(d.matrixColumn)-1)
      &&(d.rowIndex===undefined||Number(d.rowIndex)===Number(d.matrixRow)-1);
  }));
  expect(allIndices).toBe(true);
  await page.locator('#latte-math').check();
  const cell=frame.locator('.matrix-value[data-matrix-name="A · layer 0 · head 0"][data-matrix-row="1"][data-matrix-column="8"]');
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.setZoom(1));
  await page.waitForTimeout(450);
  await cell.evaluate(element=>(element.ownerDocument.defaultView as any).LittleLlamaWorkbook.focusPresentationElements([element]));
  await cell.click();
  await expect(page.locator('.latte-math-panel math [data-math-variable="b₀"]')).toHaveCount(1);
  await page.locator('.latte-math-panel math [data-math-variable="b₀"]').click();
  await expect(frame.locator('.matrix-value[data-matrix-name="Scores · layer 0 · head 0"][data-matrix-row="1"][data-matrix-column="8"]')).toHaveClass(/latte-math-destination/);
});
