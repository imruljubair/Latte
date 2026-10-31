  function renderWorkedCalculation(cell) {
    const palette = ['#b77922','#378a80','#8a5ab4','#567fb4','#b75e73'];
    const colors = new Map();
    const sources = new Map();
    const colorFor = source => {
      const name = source?.element.dataset.matrixName || 'Constants';
      if (!colors.has(name)) colors.set(name,palette[colors.size % palette.length]);
      return colors.get(name);
    };
    const make = (tag,className,text) => {
      const element = document.createElement(tag);
      element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    };
    const locate = source => {
      document.querySelectorAll('.worked-input-highlight').forEach(element => {
        element.classList.remove('worked-input-highlight');
        element.style.removeProperty('--input-accent');
      });
      if (!source) return;
      const block = blocks.find(item => item.rows.includes(source.element.closest('tr')));
      if (block) { block.group.open=true;setBlockCollapsed(block,false); }
      source.element.style.setProperty('--input-accent',colorFor(source));
      source.element.classList.add('worked-input-highlight');
    };
    const chip = (label,value,source) => {
      const element = make(source ? 'button':'span','worked-chip');
      element.append(make('span','worked-coordinate',label),make('strong','',plainValue(value)));
      if (source) {
        element.type='button';
        element.style.setProperty('--input-accent',colorFor(source));
        element.addEventListener('pointerenter',()=>locate(source));
        element.addEventListener('focus',()=>locate(source));
        element.addEventListener('click',()=>{
          locate(source);
          if(source.element.offsetParent!==null) source.element.scrollIntoView({behavior:'smooth',block:'center',inline:'center'});
        });
      }
      return element;
    };
    const unwrap = node => {
      while(node?.type==='call' && node.name==='IF') {
        const condition = evaluate(node.args[0],cell);
        if(isError(condition)) break;
        node = node.args[condition ? 1 : 2] || {type:'literal',value:false};
      }
      return node;
    };
    const reference = node => {
      const address=resolveReference(node,cell);
      const source=cells.get(key(address.sheet,address.row,address.column));
      if(source) sources.set(key(source.sheet,source.row,source.column),source);
      return {label:gridLabel(source),value:getCellValue(source),source};
    };
    const root=unwrap(cell.ast);
    const equation=mathPanel.querySelector('.math-equation');
    equation.textContent=mathematicalFormula(cell);
    mathPanel.querySelector('.math-address').textContent=gridLabel(cell);
    equation.previousElementSibling.textContent='Mathematical operation';
    const work=mathPanel.querySelector('.math-substitution');
    work.onpointerleave=()=>locate(null);
    work.previousElementSibling.textContent='Worked calculation';
    work.replaceChildren();
    const flattenSum = node => node?.type==='binary' && node.operator==='+'
      ? [...flattenSum(node.left),...flattenSum(node.right)] : [node];
    const terms=flattenSum(root);
    const isProduct=node=>node?.type==='binary' && node.operator==='*';
    if(terms.length>1 && terms.every(isProduct)) {
      const table=make('table','worked-products');
      const header=make('tr','');
      ['Input A','Input B','Product'].forEach(title=>header.append(make('th','',title)));
      const thead=make('thead','');thead.append(header);table.append(thead);
      const body=make('tbody','');
      const operand=node=>node.type==='reference'?reference(node):
        {label:node.type==='literal'?'Constant':'Expression',value:evaluate(node,cell)};
      terms.forEach(term=>{
        const row=make('tr','');
        const a=operand(term.left),b=operand(term.right);
        [a,b].forEach(item=>{const td=make('td','');td.append(chip(item.label,item.value,item.source));row.append(td);});
        row.append(make('td','worked-product',plainValue(evaluate(term,cell))));
        row.addEventListener('pointerenter',()=>{
          locate(a.source);
          if(b.source){b.source.element.style.setProperty('--input-accent',colorFor(b.source));b.source.element.classList.add('worked-input-highlight');}
        });
        body.append(row);
      });
      table.append(body);work.append(table);
      work.append(make('div','worked-total',`Sum of ${terms.length} products → ${plainValue(getCellValue(cell))}`));
    } else {
      let count=0;
      const names={SQRT:'Square root',EXP:'Exponentiate',ERF:'Error function',LN:'Natural logarithm',MAX:'Maximum',MIN:'Minimum',COUNT:'Count',ROWS:'Row count',LARGE:'Ranked value',MATCH:'Locate matching value',EXACT:'Compare',ABS:'Absolute value',SIN:'Sine',MOD:'Remainder'};
      const operators={'+':'Add','-':'Subtract','*':'Multiply','/':'Divide','^':'Power','=':'Compare','<':'Compare','>':'Compare','<=':'Compare','>=':'Compare','<>':'Compare','&':'Join'};
      function visit(raw) {
        const node=unwrap(raw);
        if(!node) return {label:'Empty',value:false};
        if(node.type==='reference') return reference(node);
        if(node.type==='literal') return {label:'Constant',value:node.value};
        if(node.type==='range') {
          const first=resolveReference(node.start,cell),last=resolveReference(node.end,cell,first.sheet);
          const value=evaluate(node,cell);
          const firstCell=cells.get(key(first.sheet,first.row,first.column));
          const lastCell=cells.get(key(last.sheet,last.row,last.column));
          const details=make('details','worked-range');
          details.append(make('summary','',`${gridLabel(firstCell)} … ${gridLabel(lastCell)} · ${value.values.length} inputs`));
          const list=make('div','worked-range-values');
          const dr=first.row<=last.row?1:-1,dc=first.column<=last.column?1:-1;
          for(let row=first.row;;row+=dr){
            for(let column=first.column;;column+=dc){
              const source=cells.get(key(first.sheet,row,column));
              if(source){sources.set(key(source.sheet,row,column),source);list.append(chip(gridLabel(source),getCellValue(source),source));}
              if(column===last.column)break;
            }
            if(row===last.row)break;
          }
          details.append(list);work.append(details);
          return {label:`Range (${value.values.length} values)`,value};
        }
        const children=node.type==='binary'?[node.left,node.right]:node.type==='unary'?[node.value]:node.args||[];
        const args=children.map(visit);
        const value=evaluate(node,cell);
        const index=++count;
        const title=node.type==='binary'?operators[node.operator]||node.operator:node.type==='unary'?'Negate':names[node.name]||node.name.toLowerCase();
        const step=make('section','worked-step');
        step.append(make('div','worked-step-title',`${index}. ${title}`));
        const inputs=make('div','worked-step-inputs');
        args.forEach((arg,i)=>{
          if(i)inputs.append(make('span','worked-operator',node.type==='binary'?({'*':'×','/':'÷'}[node.operator]||node.operator):','));
          inputs.append(chip(arg.label,arg.value,arg.source));
        });
        step.append(inputs,make('div','worked-step-result',`→ ${plainValue(value)}`));
        work.append(step);
        return {label:`Step ${index}`,value};
      }
      const result=visit(root);
      if(count===0)work.append(chip(result.label,result.value,result.source));
      work.append(make('div','worked-total',`Result → ${plainValue(getCellValue(cell))}`));
    }
    const legend=mathPanel.querySelector('.math-grid-inputs');
    legend.previousElementSibling.textContent='Input matrices';
    legend.replaceChildren(...[...colors].map(([name,color])=>{
      const item=make('div','worked-legend',name);
      item.style.setProperty('--input-accent',color);
      return item;
    }));
  }
