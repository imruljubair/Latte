/* Latte equations: derived from Latte's formula tree, independent of presentation. */
(function (root) {
  'use strict';
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const mi = (s,kind=/^[A-Z]/.test(s)?'matrix':'scalar') => kind==='vector'
    ? `<munder accentunder="true" data-math-kind="vector"><mi mathvariant="normal">${esc(s)}</mi><mo stretchy="true">_</mo></munder>`
    : `<mi mathvariant="normal">${esc(s)}</mi>`, mn = s => {
    const scientific=String(s).match(/^(.+)[eE]([+-]?\d+)$/);
    return scientific?`<mrow><mn>${esc(scientific[1])}</mn><mo>×</mo><msup><mn>10</mn><mn>${Number(scientific[2])}</mn></msup></mrow>`:`<mn>${esc(s)}</mn>`;
  };
  const mo = s => `<mo>${esc(s)}</mo>`, text = s => `<mtext>${esc(s)}</mtext>`;
  const row = (...s) => `<mrow>${s.join('')}</mrow>`;
  const frac = (a,b) => `<mfrac>${row(a)}${row(b)}</mfrac>`;
  const sqrt = a => `<msqrt>${a}</msqrt>`, power = (a,b) => `<msup>${row(a)}${row(b)}</msup>`;
  const sub = (a,b) => `<msub>${row(a)}${row(b)}</msub>`;
  const vector = a => mi(a,'vector');
  const call = (name,...args) => row(mi(name,'operator'),mo('('),row(...args),mo(')'));
  const equation = (a,b) => `<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">${row(a,mo('='),b)}</math>`;
  const sum = (...items) => row(...items.flatMap((item,i) => i ? [mo('+'),item] : [item]));
  const roleColors = {weight:[242,163,104],key:[138,219,139],value:[223,157,215],query:[143,167,216],attention:[0,181,226],output:[241,215,83],loss:[224,161,139]};
  function matrixRole(name, weight = false) {
    if (weight || /^(W[QKVO12] ·|Vocabulary weights|RMS scale|Token embeddings|Position embeddings)/.test(name)) return 'weight';
    if (/^K(?:ᵀ)? ·/.test(name)) return 'key';
    if (/^V ·/.test(name)) return 'value';
    if (/^Q ·/.test(name)) return 'query';
    if (/^(A ·|Scores ·|Vocabulary probabilities)/.test(name)) return 'attention';
    if (/loss/i.test(name)) return 'loss';
    return 'output';
  }
  function matrixSymbol(name){
    const label=name.replace(/\s*[•·]\s*layer\s+\d+/ig,'').replace(/\s*[•·]\s*head\s+\d+/ig,'').replace(/\s*·\s*matrix\s+\d+$/i,'');
    const symbols={
      'Token embeddings':'E','Position embeddings':'P','X · embedding output':'X',
      'Attention input':'Xₙ','FFN input':'Xₙ','Final hidden':'Xₙ',
      'WQ':'Wq','WK':'Wk','WV':'Wv','WO':'Wo','W1':'W₀','W2':'W₁',
      'Q':'Q','K':'K','Kᵀ':'Kᵀ','V':'V','Scores':'S','A':'A','H':'H',
      'Concatenated heads':'Hcat','Attention output':'O','Residual attention':'R',
      'FFN pre-activation':'Z','GELU':'G','FFN output':'F','Layer output':'X′',
      'Vocabulary weights':'Wvocab','Logits':'L','Vocabulary probabilities':'P',
      'Predicted next character':'ŷ','Correct next character':'y','Recorded loss before update':'ℓ',
    };
    return /^RMS scale/.test(label)?'γ':symbols[label]||label;
  }
  function matrixNamePosition(rect,name,zoom){
    const x=(rect.left+rect.right)/2;
    const symbol=matrixSymbol(name);
    const labelScale=zoom>.3?Math.min(1,zoom):1;
    const baseSize=Math.min(13,Math.max(9,(rect.right-rect.left-4)/Math.max(1,symbol.length*.65)));
    const size=zoom>.3?Math.min(baseSize,(25.5*zoom-4*labelScale)/1.2):baseSize;
    // Scale the whole plate at lower numbered zooms so the header fits the
    // inter-matrix gap, then dock its bottom immediately above the indices.
    const labelHeight=size*1.2+4*labelScale;
    const y=zoom>.3?rect.top-12*zoom-.5*zoom-labelHeight/2:(rect.top+rect.bottom)/2;
    return {x,y,size,labelHeight,labelScale};
  }
  function matrixKind(name){
    return /^(RMS scale|Predicted next character|Correct next character|Recorded loss)/.test(name)?'vector':'matrix';
  }
  function matrixFormula(name, operands = []) {
    if(matrixRole(name)==='weight')return weightMatrixOrigin(name);
    const product = (a,b) => row(a,mo('·'),b);
    const inputName=operands.find(operand=>!/^W[QKVO12] ·|^RMS scale|^Vocabulary weights/.test(operand));
    const input=inputName?mi(matrixSymbol(inputName),matrixKind(inputName)):mi('X');
    if(/ · matrix \d+$/.test(name)&&operands.length===1)return equation(mi(matrixSymbol(name),matrixKind(name)),input);
    const symbol = name.match(/^W?([QKV])(?: ·|$)/)?.[1];
    if (symbol) return equation(mi(symbol),product(sub(mi('W'),mi(symbol.toLowerCase(),'operator')),input));
    if (/^Kᵀ/.test(name)) return equation(power(mi('K'),mi('T','operator')),power(operands.length?input:row(sub(mi('W'),mi('k','operator')),mi('X')),mi('T','operator')));
    if (/Scores|Attention scores/.test(name)) return equation(mi('S'),frac(product(power(mi('K'),mi('T','operator')),mi('Q')),sqrt(sub(mi('d'),mi('k')))));
    if (/^A ·|Masked softmax/.test(name)) return equation(mi('A'),call('softmax',sum(mi('S'),mi('M'))));
    if (/Vocabulary probabilities|Vocabulary softmax/.test(name)) return equation(mi('P'),call('softmax',mi('L')));
    if (/^(H ·)|Value aggregation/.test(name)) return equation(mi('H'),product(mi('V'),mi('A')));
    if (/^(WO ·|Attention output)/.test(name)) return equation(mi('Y'),product(sub(mi('W'),mi('o','operator')),mi('H')));
    if (/^(W1 ·|FFN pre-activation)/.test(name)) return equation(mi('Z'),product(sub(mi('W'),mn(0)),mi('X')));
    if (/^(W2 ·|FFN output)/.test(name)) return equation(mi('Y'),product(sub(mi('W'),mn(1)),call('GELU',mi('Z'))));
    if (/GELU/.test(name)) return equation(call('GELU',mi('x')),row(frac(mi('x'),mn(2)),mo('·'),mo('['),mn(1),mo('+'),call('erf',frac(mi('x'),sqrt(mn(2)))),mo(']')));
    if (/^Logits|Vocabulary weights/.test(name)) return equation(mi('L'),product(sub(mi('W'),text('vocab')),mi('X')));
    if (/embedding output|Token \+ position/.test(name)) return equation(mi('X'),sum(mi('E'),mi('P')));
    if (/Residual|Layer output/.test(name)) return equation(mi('Y'),sum(mi('X'),mi('R')));
    if (/norm|Attention input|FFN input|Final hidden/i.test(name)) return equation(sub(mi('y'),mi('i')),frac(row(sub(mi('γ'),mi('i')),mo('·'),sub(mi('x'),mi('i'))),sqrt(row(frac(mn(1),mi('d')),mo('∑'),power(sub(mi('x'),mi('j')),mn(2)),mo('+'),mi('ε')))));
    if (/Recorded loss/.test(name)) return equation(power(sub(mi('ℓ'),mi('t')),text('before')),row(mo('−'),call('ln',power(sub(mi('p'),row(text('target'),mo(','),mi('t'))),text('before')))));
    if (/Token embeddings/.test(name)) return equation(mi('X'),sub(mi('E'),text('tokens')));
    if (/Position embeddings/.test(name)) return equation(mi('P'),text('learned position embeddings'));
    if (/Concatenated/.test(name)) return equation(mi('H'),call('concat',sub(mi('H'),mn(0)),mo(','),mi('…'),mo(','),sub(mi('H'),row(mi('h'),mo('−'),mn(1)))));
    if (/Predicted next/.test(name)) return equation(text('prediction'),call('argmax',vector('p')));
    if (/Correct next/.test(name)) return equation(sub(mi('y'),mi('t')),sub(text('corpus'),row(mi('t'),mo('+'),mn(1))));
    return equation(mi('a'),text('learned parameter'));
  }
  function weightMatrixOrigin(name){
    const initialization=/^RMS scale/.test(name)?'Initialized to ones':'Initialized randomly';
    return equation(mi(matrixSymbol(name),matrixKind(name)),text('learned parameter'))
      +`<p class="latte-weight-origin">${initialization}; learned through AdamW updates during training.</p>`;
  }
  function weightCellOrigin(name,origin,value){
    const number=value=>mn(Number(value).toPrecision(8));
    if(!origin||!Number.isFinite(origin.before)||!Number.isFinite(origin.after)){
      return (Number.isFinite(value)?equation(mi('a'),number(value)):'')
        +'<p class="latte-weight-origin">Learned parameter. Training-update history is not recorded in this view.</p>';
    }
    const step=esc(origin.step),delta=origin.after-origin.before;
    if(!origin.applied)return equation(mi('a'),number(origin.before))
      +`<p class="latte-weight-origin">Parameter entering training step ${step}.</p>`;
    const gradient=Number.isFinite(origin.gradient)?`<p>Batch-mean gradient: ${esc(Number(origin.gradient).toPrecision(8))}</p>`:'';
    return equation(mi('a'),sum(sub(mi('w'),text('before')),mi('Δw')))
      +equation(number(origin.after),row(number(origin.before),mo('+'),mo('('),number(delta),mo(')')))
      +`<p class="latte-weight-origin">Updated by AdamW at training step ${step}, using a batch of ${esc(origin.batchSize)} sequences.</p>`
      +`<details class="latte-weight-details"><summary>How AdamW produced Δw</summary>${gradient}<p>AdamW combines running gradient moments, bias correction, and weight decay. These are the update rules:</p>${adamWFormula()}</details>`;
  }
  function explainCell(description) {
    const groups = [], byName = new Map(), byId = new Map();
    for (const source of description.sources) {
      if (!byName.has(source.name)) {
        const group = {name:source.name,label:String.fromCharCode(98+groups.length),sources:[]};
        groups.push(group); byName.set(source.name,group);
      }
      const group = byName.get(source.name);
      const binding = {group,index:group.sources.length};
      group.sources.push(source); byId.set(source.id,binding);
    }
    if(description.recordedLoss && groups.length===1){
      const b=groups[0].label,record=description.recordedLoss;
      const note=record.updated?'Saved before update; the connected cell now shows the updated probability.':'Target-token probability, recorded before update.';
      return {markup:equation(mi('a'),row(mo('−'),call('ln',mi(b))))
        +equation(power(mi(b),text('before')),mn(record.probability.toPrecision(6)))
        +`<p class="latte-weight-origin">${note}</p>`,groups,learned:false};
    }
    if(description.name==='Predicted next character' && groups.length===1){
      return {markup:equation(mi('a'),call('argmax',vector(groups[0].label))),groups,learned:false};
    }
    const indexedSources=new Set();
    const indexedRef = id => {
      const binding=byId.get(id),source=binding.group.sources[binding.index];
      const column=binding.group.sources.every(item=>item.column===source.column);
      const coordinate=column?source.row:source.column;
      const index=Number.isInteger(coordinate)?coordinate:binding.index;
      const b=binding.group.label;
      if(!indexedSources.has(id)){
        const label=b+String(index).replace(/\d/g,digit=>'₀₁₂₃₄₅₆₇₈₉'[Number(digit)]);
        groups.push({name:source.name,label,sources:[source],specific:true});
        indexedSources.add(id);
      }
      return sub(mi(b),mn(index));
    };
    const ref = id => {
      const binding = byId.get(id);
      if (!binding) return text('input');
      return binding.group.sources.length === 1 ? mi(binding.group.label)
        : indexedRef(id);
    };
    const serialize = node => {
      if (!node) return text('parameter');
      if (node.type === 'reference') return ref(node.id);
      if (node.type === 'range') {
        const group = byId.get(node.ids[0])?.group;
        return group ? vector(group.label) : text('range');
      }
      if (node.type === 'literal') return typeof node.value === 'number' ? mn(node.value) : text(node.value);
      if (node.type === 'unary') return row(mo(node.operator),serialize(node.value));
      if (node.type === 'binary') {
        const a=serialize(node.left),b=serialize(node.right);
        if(node.operator==='/')return frac(a,b);
        if(node.operator==='^')return power(a,b);
        return row(mo('('),a,mo({'*':'·','<>':'≠','&':'∥'}[node.operator]||node.operator),b,mo(')'));
      }
      const args=node.args.map(serialize);
      if(node.name==='SQRT')return sqrt(args[0]);
      if(node.name==='EXP')return power(mi('e'),args[0]);
      if(node.name==='LN')return call('ln',args[0]);
      return call(node.name.toLowerCase(),...args.flatMap((arg,i)=>i?[mo(','),arg]:[arg]));
    };
    const flatten = node => node?.type==='binary' && node.operator==='+' ? [...flatten(node.left),...flatten(node.right)] : [node];
    let expression, node=description.node;
    const numerator = node?.type==='binary'&&node.operator==='/'?node.left:node;
    const terms=flatten(numerator);
    const isDot = terms.length>0 && terms.every(term=>term?.type==='binary'&&term.operator==='*'&&term.left.type==='reference'&&term.right.type==='reference');
    const rmsScale=groups.find(group=>/^RMS scale · /.test(group.name));
    if(rmsScale && node?.type==='binary' && node.operator==='/' && numerator?.operator==='*'
      && numerator.left?.type==='reference' && numerator.right?.type==='reference'){
      // RMSNorm's expanded spreadsheet sum describes a vector reduction,
      // not a separate scalar dependency for every squared component.
      const inputId=[numerator.left.id,numerator.right.id].find(id=>byId.get(id)?.group!==rmsScale);
      const scaleId=[numerator.left.id,numerator.right.id].find(id=>byId.get(id)?.group===rmsScale);
      const inputGroup=byId.get(inputId)?.group;
      if(inputGroup && scaleId){
        const rms=sqrt(sum(call('mean',power(vector(inputGroup.label),mn(2))),mi('ε')));
        expression=frac(row(ref(inputId),mo('·'),ref(scaleId)),rms);
      }
    } else if (isDot && groups.length===2) {
      expression=row(groups[0].sources.length>1?vector(groups[0].label):mi(groups[0].label),mo('·'),groups[1].sources.length>1?vector(groups[1].label):mi(groups[1].label));
      if(node!==numerator)expression=frac(expression,serialize(node.right));
    } else if (/^(A ·|Vocabulary probabilities)/.test(description.name) && groups.length===1) {
      const b=groups[0].label, numeratorSource=node?.left?.args?.[0]?.left;
      if(numeratorSource?.type==='reference') {
        // Matrix indices are zero based. AST traversal visits the numerator
        // first, so its encounter index is not its position in the column.
        const shifted = value => row(value,mo('−'),call('max',vector(b)));
        expression=frac(power(mi('e'),shifted(indexedRef(numeratorSource.id))),
          row(sub(mo('∑'),mi('j')),power(mi('e'),shifted(sub(mi(b),mi('j'))))));
      }
    }
    if(!expression && node)expression=serialize(node);
    const learned=description.weight && !groups.length;
    const markup=learned || !node || /Predicted next character/.test(description.name) ? matrixFormula(description.name) : equation(mi('a'),expression);
    return {markup,groups,learned};
  }
  function adamWFormula(){
    const previous=letter=>sub(mi(letter),row(mi('t'),mo('−'),mn(1)));
    const now=letter=>sub(mi(letter),mi('t'));
    const beta=i=>sub(mi('β'),mn(i));
    const moment=(letter,i,gradient)=>equation(now(letter),sum(row(beta(i),mo('·'),previous(letter)),row(mo('('),mn(1),mo('−'),beta(i),mo(')'),mo('·'),gradient)));
    const corrected=letter=>`<mover>${now(letter)}<mo>^</mo></mover>`;
    return moment('m',1,now('g'))+moment('v',2,power(now('g'),mn(2)))
      +equation(corrected('m'),frac(now('m'),row(mn(1),mo('−'),power(beta(1),mi('t')))))
      +equation(corrected('v'),frac(now('v'),row(mn(1),mo('−'),power(beta(2),mi('t')))))
      +equation(now('w'),row(mo('('),mn(1),mo('−'),mi('η'),mo('·'),mi('λ'),mo(')'),mo('·'),previous('w'),mo('−'),mi('η'),mo('·'),frac(corrected('m'),sum(sqrt(corrected('v')),mi('ε')))));
  }
  const api={matrixRole,roleColors,matrixSymbol,matrixKind,matrixNamePosition,matrixFormula,weightMatrixOrigin,weightCellOrigin,explainCell,adamWFormula};
  root.LatteMath=api;
  if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
