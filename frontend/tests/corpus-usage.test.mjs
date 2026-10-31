import test from 'node:test';
import assert from 'node:assert/strict';
import { usageAt, usageSegments, usageColor } from '../src/corpus-usage.mjs';

const usage={ranges:[[0,2,1],[2,3,3],[3,5,2],[8,10,1]],max_count:3};
test('usage lookup keeps zero-based positions and skipped positions distinct',()=>{
  assert.deepEqual(Array.from({length:12},(_,i)=>usageAt(usage,i)),[1,1,3,2,2,0,0,0,1,1,0,0]);
  assert.equal(usageAt(undefined,0),0);
});
test('heatmap segments preserve every character and explicitly leave validation unused',()=>{
  const segments=usageSegments(usage,1,12,10);
  assert.deepEqual(segments.map(p=>[p.start,p.end,p.count,p.validation]),[
    [1,2,1,false],[2,3,3,false],[3,5,2,false],[5,8,0,false],[8,10,1,false],[10,12,0,true],
  ]);
  assert.equal(segments.reduce((sum,p)=>sum+p.end-p.start,0),11);
  assert.deepEqual(usageSegments(undefined,0,8,5).map(p=>[p.start,p.end,p.count,p.validation]),[[0,5,0,false],[5,8,0,true]]);
  assert.equal(usageSegments(usage,4,4,10).length,0);
});
test('frequency scales linearly from cool blue through green to warm red',()=>{
  assert.equal(usageColor(0,10),'hsla(210.0, 70%, 65%, 0.45)');
  assert.equal(usageColor(5,10),'hsla(105.0, 70%, 65%, 0.45)');
  assert.equal(usageColor(10,10),'hsla(0.0, 70%, 65%, 0.45)');
  assert.equal(usageColor(11,10),usageColor(10,10));
  assert.ok(!usageColor(0,0).includes('NaN'));
});
