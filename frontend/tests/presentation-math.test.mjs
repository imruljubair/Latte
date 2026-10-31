import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require=createRequire(import.meta.url);
const {explainCell,matrixFormula,matrixRole,matrixSymbol,adamWFormula}=require('../../presentation_math.js');
const ref=id=>({type:'reference',id});
const product=(left,right)=>({type:'binary',operator:'*',left:ref(left),right:ref(right)});

test('dot products bind the actual weight row and input column without numerical outputs',()=>{
  const result=explainCell({name:'Q · layer 0 · head 0',sources:[
    {id:'w0',name:'WQ · layer 0 · head 0'},{id:'x0',name:'Attention input · layer 0'},
    {id:'w1',name:'WQ · layer 0 · head 0'},{id:'x1',name:'Attention input · layer 0'}],
    node:{type:'binary',operator:'+',left:product('w0','x0'),right:product('w1','x1')}});
  assert.equal(result.groups.length,2);
  assert.deepEqual(result.groups[0].sources.map(source=>source.id),['w0','w1']);
  assert.deepEqual(result.groups[1].sources.map(source=>source.id),['x0','x1']);
  assert.match(result.markup,/<mi mathvariant="normal">b<\/mi>/);
  assert.match(result.markup,/<mo>·<\/mo>/);
  assert.equal((result.markup.match(/<mo>\|<\/mo>/g)||[]).length,0);
  assert.doesNotMatch(result.markup,/RoPE|cache|sin|cos|bias/);
});

test('score explanation preserves scaling by the actual head dimension',()=>{
  const result=explainCell({name:'Scores · layer 0 · head 0',sources:[
    {id:'k0',name:'Kᵀ · layer 0 · head 0'},{id:'q0',name:'Q · layer 0 · head 0'}],
    node:{type:'binary',operator:'/',left:product('k0','q0'),right:{type:'call',name:'SQRT',args:[{type:'literal',value:3}]}}});
  assert.match(result.markup,/<mfrac>/);
  assert.match(result.markup,/<msqrt><mn>3<\/mn><\/msqrt>/);
});

test('matrix formulas use Latte projections without imported model operations',()=>{
  for(const name of ['K · layer 0 · head 0','Q · layer 0 · head 0','V · layer 0 · head 0','W1 · layer 0','W2 · layer 0']) {
    assert.doesNotMatch(matrixFormula(name),/RoPE|cache|sin|cos|bias/);
  }
  assert.match(matrixFormula('Vocabulary probabilities'),/>L<\/mi>/);
  assert.match(matrixFormula('A · layer 0 · head 0'),/>M<\/mi>/);
  assert.match(matrixFormula('Recorded loss before update'),/before/);
});

test('AdamW explanation includes both moments, bias correction and decoupled decay',()=>{
  const math=adamWFormula();
  assert.equal((math.match(/<math /g)||[]).length,5);
  assert.match(math,/>λ<\/mi>/);
  assert.match(math,/>β<\/mi>/);
  assert.match(math,/<mover>/);
  assert.match(math,/<msqrt>/);
});

test('role colors distinguish training tensors without changing data',()=>{
  assert.equal(matrixRole('WQ · layer 0 · head 0'),'weight');
  assert.equal(matrixRole('Kᵀ · layer 0 · head 0'),'key');
  assert.equal(matrixRole('Q · layer 0 · head 0'),'query');
  assert.equal(matrixRole('V · layer 0 · head 0'),'value');
  assert.equal(matrixRole('Recorded loss before update'),'loss');
  assert.equal(matrixSymbol('Q · layer 0 · head 0'),'Q');
  assert.equal(matrixSymbol('V · layer 0 · head 0 · matrix 2'),'V');
  assert.equal(matrixSymbol('Vocabulary weights'),'Wvocab');
});

test('softmax identifies the numerator cell by its visible row, independently of reference order',()=>{
  const sources=[4,0,1,2,3,5,6,7].map(row=>({id:`s${row}`,name:'Scores · layer 0 · head 0',row,column:7}));
  const result=explainCell({name:'A · layer 0 · head 0',sources,node:{type:'binary',operator:'/',
    left:{type:'call',name:'EXP',args:[{type:'binary',operator:'-',left:ref('s4'),right:{type:'literal',value:0}}]},
    right:{type:'literal',value:1}}});
  assert.match(result.markup,/<msub><mrow><mi mathvariant="normal">b<\/mi><\/mrow><mrow><mn>4<\/mn>/);
  assert.equal(result.groups.length,2);
  assert.equal(result.groups[0].sources.length,8);
  assert.equal(result.groups[1].label,'b₄');
  assert.equal(result.groups[1].specific,true);
  assert.deepEqual(result.groups[1].sources,[sources[0]]);
});

test('an explicit scalar within a normalization vector gets one indexed connection',()=>{
  const sources=[2,0,1].map(row=>({id:`x${row}`,name:'Attention input',row,column:4}));
  const result=explainCell({name:'FFN input',sources,node:{type:'binary',operator:'/',left:ref('x2'),
    right:{type:'call',name:'SUM',args:[{type:'range',ids:['x0','x1','x2']},ref('x2')]}}});
  assert.equal(result.groups.length,2);
  assert.equal(result.groups[0].sources.length,3);
  assert.equal(result.groups[1].label,'b₂');
  assert.deepEqual(result.groups[1].sources,[sources[0]]);
  assert.equal(result.groups[1].specific,true);
});

test('RMSNorm summarizes the reduction and connects only its selected scalar',()=>{
  const binary=(operator,left,right)=>({type:'binary',operator,left,right});
  const literal=value=>({type:'literal',value});
  const sources=[{id:'x1',name:'X · embedding output',row:1,column:7},
    {id:'g',name:'RMS scale · Attention input · layer 0',row:1,column:0},
    {id:'x0',name:'X · embedding output',row:0,column:7},
    {id:'x2',name:'X · embedding output',row:2,column:7}];
  const squares=[0,1,2].map(i=>binary('^',ref(`x${i}`),literal(2)));
  const result=explainCell({name:'Attention input · layer 0',sources,node:binary('/',product('x1','g'),
    {type:'call',name:'SQRT',args:[binary('+',binary('/',binary('+',binary('+',squares[0],squares[1]),squares[2]),literal(3)),literal(1.1920928955078125e-7))]})});
  assert.match(result.markup,/>mean<\/mi>/);
  assert.match(result.markup,/>ε<\/mi>/);
  assert.match(result.markup,/<msqrt>/);
  assert.equal(result.groups.length,3);
  assert.deepEqual(result.groups.filter(group=>group.specific).map(group=>group.label),['b₁']);
  assert.deepEqual(result.groups[2].sources,[sources[0]]);
  assert.doesNotMatch(result.markup,/1.1920928955078125|<mn>3<\/mn>/);
});

test('weight matrices explain initialization rather than their forward contribution',()=>{
  for(const name of ['WQ · layer 0 · head 0','WK · layer 0 · head 0','WV · layer 0 · head 0',
    'WO · layer 0','W1 · layer 0','W2 · layer 0','Token embeddings','Position embeddings','Vocabulary weights']){
    const markup=matrixFormula(name);
    assert.match(markup,/learned parameter/);
    assert.match(markup,/Initialized randomly/);
    assert.doesNotMatch(markup,/<mo>·<\/mo>|softmax|GELU|lookup/);
  }
  for(const name of ['RMS scale · Attention input · layer 0','RMS scale · FFN input · layer 0','RMS scale · Final hidden']){
    assert.match(matrixFormula(name),/Initialized to ones/);
    assert.doesNotMatch(matrixFormula(name),/randomly/);
  }
});

test('weight cell origins use only recorded updates and keep AdamW details optional',()=>{
  const {weightCellOrigin}=require('../../presentation_math.js');
  const origin={before:.25,after:.2,gradient:.04,step:1000,batchSize:4,applied:false};
  const before=weightCellOrigin('WQ · layer 0 · head 0',origin);
  assert.match(before,/0.25000000/);
  assert.match(before,/Parameter entering training step 1000/);
  assert.doesNotMatch(before,/Δw|<details|0.20000000/);
  const after=weightCellOrigin('WQ · layer 0 · head 0',{...origin,applied:true});
  assert.match(after,/0.20000000/);
  assert.match(after,/-0.050000000/);
  assert.match(after,/Batch-mean gradient: 0.040000000/);
  assert.match(after,/<details class="latte-weight-details">/);
  assert.doesNotMatch(after,/<details[^>]*\bopen/);
  assert.match(after,/batch of 4 sequences/);
});

test('notation distinguishes plain scalars, underlined vectors, and uppercase matrices without bars',()=>{
  const vector=explainCell({name:'Q · layer 0 · head 0',sources:[
    {id:'w0',name:'WQ'},{id:'x0',name:'Input'},{id:'w1',name:'WQ'},{id:'x1',name:'Input'}],
    node:{type:'binary',operator:'+',left:product('w0','x0'),right:product('w1','x1')}}).markup;
  assert.match(vector,/<mi mathvariant="normal">a<\/mi>/);
  assert.match(vector,/<mi mathvariant="normal">b<\/mi>/);
  assert.match(vector,/<mi mathvariant="normal">c<\/mi>/);
  assert.doesNotMatch(vector,/<mo>\|<\/mo>/);
  const matrix=matrixFormula('Q · layer 0 · head 0',['WQ · layer 0 · head 0','Attention input · layer 0']);
  assert.match(matrix,/<mi mathvariant="normal">Q<\/mi>/);
  assert.match(matrix,/<mi mathvariant="normal">W<\/mi>/);
  assert.match(matrix,/<mi mathvariant="normal">q<\/mi>/);
  assert.match(matrix,/<mi mathvariant="normal">Xₙ<\/mi>/);
  assert.match(matrixFormula('Vocabulary probabilities'),/<mi mathvariant="normal">softmax<\/mi>/);
});

test('vector underlines preserve scalar component notation and matrix letters',()=>{
  const result=explainCell({name:'Q · layer 0 · head 0',sources:[
    {id:'w0',name:'WQ'},{id:'x0',name:'Input'},{id:'w1',name:'WQ'},{id:'x1',name:'Input'}],
    node:{type:'binary',operator:'+',left:product('w0','x0'),right:product('w1','x1')}});
  assert.equal((result.markup.match(/<munder accentunder="true" data-math-kind="vector">/g)||[]).length,2);
  assert.equal((result.markup.match(/<mo stretchy="true">_<\/mo>/g)||[]).length,2);
  assert.match(result.markup,/<mi mathvariant="normal">a<\/mi>/);
  assert.match(matrixFormula('Q · layer 0 · head 0'),/<mi mathvariant="normal">Q<\/mi>/);
});

test('character prediction binds argmax to its probability-column connector',()=>{
  const sources=[0,1,2].map(row=>({id:`p${row}`,name:'Vocabulary probabilities',row,column:6}));
  const result=explainCell({name:'Predicted next character',sources,
    node:{type:'call',name:'MATCH',args:[]}});
  assert.equal(result.groups.length,1);
  assert.deepEqual(result.groups[0].sources,sources);
  assert.equal(result.groups[0].label,'b');
  assert.match(result.markup,/argmax/);
  assert.match(result.markup,/<mi mathvariant="normal">a<\/mi>/);
  assert.match(result.markup,/<munder[^>]+data-math-kind="vector"><mi mathvariant="normal">b<\/mi>/);
});
