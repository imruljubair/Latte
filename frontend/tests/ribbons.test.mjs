import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require=createRequire(import.meta.url);
const {connectionAnchors,ribbonConnection,placeOperandEndMarker,placeRibbonIndicator}=require('../../ribbon_geometry.js');

test('a row operand connects horizontally at full cell height',()=>{
  const anchors=connectionAnchors({x:800,y:350,w:320,h:200},{x:100,y:350,w:600,h:200},
    {x:800,y:400,w:40,h:20},{x:100,y:400,w:600,h:20});
  assert.equal(anchors.vertical,false);
  assert.deepEqual(anchors.start,{x:800,y:410});
  assert.deepEqual(anchors.end,{x:700,y:410});
  assert.equal(anchors.sourceWidth,20);
  assert.equal(anchors.targetWidth,20);
  assert.match(ribbonConnection(anchors.start,anchors.end,20,20,false,24),/^M800,400 C/);
});

test('an input column above the output connects bottom-to-top at full cell width',()=>{
  const anchors=connectionAnchors({x:800,y:350,w:320,h:200},{x:800,y:40,w:320,h:200},
    {x:1080,y:400,w:40,h:20},{x:840,y:40,w:40,h:200});
  assert.equal(anchors.vertical,true);
  assert.deepEqual(anchors.start,{x:1100,y:400});
  assert.deepEqual(anchors.end,{x:860,y:240});
  assert.equal(anchors.sourceWidth,40);
  assert.equal(anchors.targetWidth,40);
  const path=ribbonConnection(anchors.start,anchors.end,40,40,true,30);
  assert.match(path,/^M1080,400 C/);
  assert.match(path,/840,240 L880,240/);
  assert.doesNotMatch(path,/NaN|Infinity/);
});

test('operand arrows face the input as its left margin moves off screen',()=>{
  const bounds={left:8,top:48,right:1000,bottom:800};
  const margin=placeOperandEndMarker({x:150,y:400},{x:200,y:400},false,bounds);
  assert.equal(margin.angle,0);
  const dock=placeOperandEndMarker({x:400,y:400},{x:200,y:400},false,bounds);
  assert.equal(dock.angle,180);
  assert.equal(placeOperandEndMarker({x:400,y:400},{x:-100,y:400},false,bounds),null);
  const ribbon=placeRibbonIndicator({x:400,y:400},{x:-100,y:400},false,48,bounds);
  assert.ok(ribbon.x>=bounds.left);
  assert.ok(Math.cos(ribbon.angle*Math.PI/180)<0);
});

test('scalar tapes are narrow in either direction and scale down with cells',()=>{
  const {scalarConnection}=require('../../ribbon_geometry.js');
  for(const vertical of [true,false]){
    const original={start:{x:10,y:20},end:{x:80,y:90},vertical,sourceWidth:60,targetWidth:40};
    const narrow=scalarConnection(original);
    assert.equal(narrow.sourceWidth,8.8);
    assert.equal(narrow.targetWidth,8.8);
    assert.deepEqual(narrow.start,original.start);
    assert.deepEqual(narrow.end,original.end);
    assert.equal(narrow.vertical,vertical);
  }
  assert.equal(scalarConnection({sourceWidth:200,targetWidth:200}).sourceWidth,12);
  assert.equal(scalarConnection({sourceWidth:10,targetWidth:10}).sourceWidth,2.2);
});

test('long offscreen connections retain an indicator near either end',()=>{
  const bounds={left:8,top:48,right:1000,bottom:800};
  for(const [start,end] of [[{x:400,y:1000},{x:400,y:-20000}],
    [{x:400,y:20000},{x:400,y:-100}]]){
    const point=placeRibbonIndicator(start,end,true,48,bounds);
    assert.ok(point);
    assert.ok(point.y>=bounds.top&&point.y<=bounds.bottom);
  }
});
