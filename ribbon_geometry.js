/* Ribbon geometry shared with KeVin; no model or cache behavior. */
(function(root){
'use strict';
  function routeConnection(start,end,obstacles){
    const unique=values=>[...new Set(values.map(x=>Math.round(x*100)/100))].sort((a,b)=>a-b);
    const xs=unique([start.x,end.x,...obstacles.flatMap(r=>[r.x-8,r.x+r.w+8])]),ys=unique([start.y,end.y,...obstacles.flatMap(r=>[r.y-8,r.y+r.h+8])]);
    const nearest=(a,v)=>a.reduce((best,x,i)=>Math.abs(x-v)<Math.abs(a[best]-v)?i:best,0),key=(x,y)=>y*xs.length+x;
    const sx=nearest(xs,start.x),sy=nearest(ys,start.y),ex=nearest(xs,end.x),ey=nearest(ys,end.y),origin=key(sx,sy),goal=key(ex,ey);
    const cost=new Map([[origin,0]]),parents=new Map(),closed=new Set(),open=[{x:sx,y:sy,id:origin,f:0}];
    function blocked(a,b){return obstacles.some(r=>a.x===b.x?a.x>r.x+.1&&a.x<r.x+r.w-.1&&Math.max(a.y,b.y)>r.y+.1&&Math.min(a.y,b.y)<r.y+r.h-.1:a.y>r.y+.1&&a.y<r.y+r.h-.1&&Math.max(a.x,b.x)>r.x+.1&&Math.min(a.x,b.x)<r.x+r.w-.1);}
    while(open.length&&closed.size<20000){
      open.sort((a,b)=>b.f-a.f);const node=open.pop();if(closed.has(node.id))continue;
      if(node.id===goal){const points=[];for(let id=goal;id!==undefined;id=parents.get(id))points.push({x:xs[id%xs.length],y:ys[Math.floor(id/xs.length)]});return points.reverse();}
      closed.add(node.id);
      for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]]){
        const x=node.x+dx,y=node.y+dy;if(x<0||y<0||x>=xs.length||y>=ys.length)continue;
        const id=key(x,y);if(closed.has(id))continue;
        const a={x:xs[node.x],y:ys[node.y]},b={x:xs[x],y:ys[y]};if(blocked(a,b))continue;
        const next=cost.get(node.id)+Math.abs(a.x-b.x)+Math.abs(a.y-b.y);if(next>=(cost.get(id)??Infinity))continue;
        cost.set(id,next);parents.set(id,node.id);open.push({x,y,id,f:next+Math.abs(b.x-xs[ex])+Math.abs(b.y-ys[ey])});
      }
    }return null;
  }
  function curvedConnection(points,obstacles=[],radius=24){
    if(!points?.length)return '';
    const line=[];
    for(const p of points){
      if(line.length&&line.at(-1).x===p.x&&line.at(-1).y===p.y)continue;
      if(line.length>1){const a=line.at(-2),b=line.at(-1);if((b.x-a.x)*(p.y-b.y)===(b.y-a.y)*(p.x-b.x))line.pop();}
      line.push(p);
    }
    let d=`M${line[0].x},${line[0].y}`;
    for(let i=1;i<line.length-1;i++){
      const a=line[i-1],b=line[i],c=line[i+1],ab=Math.hypot(b.x-a.x,b.y-a.y),bc=Math.hypot(c.x-b.x,c.y-b.y);
      let r=Math.min(radius,ab/2,bc/2),entry,exit;
      for(let attempt=0;attempt<8;attempt++){
        entry={x:b.x+(a.x-b.x)*r/ab,y:b.y+(a.y-b.y)*r/ab};
        exit={x:b.x+(c.x-b.x)*r/bc,y:b.y+(c.y-b.y)*r/bc};
        const collision=Array.from({length:25},(_,j)=>j/24).some(t=>{
          const x=(1-t)**2*entry.x+2*(1-t)*t*b.x+t*t*exit.x,y=(1-t)**2*entry.y+2*(1-t)*t*b.y+t*t*exit.y;
          return obstacles.some(o=>x>o.x+.1&&x<o.x+o.w-.1&&y>o.y+.1&&y<o.y+o.h-.1);
        });
        if(!collision)break;
        r/=2;
      }
      d+=` L${entry.x},${entry.y} Q${b.x},${b.y} ${exit.x},${exit.y}`;
    }
    if(line.length>1)d+=` L${line.at(-1).x},${line.at(-1).y}`;
    return d;
  }

  function connectionAnchors(source,target,selected,range){
    const above=target.y+target.h<=source.y,below=target.y>=source.y+source.h;
    if(above||below)return {
      start:{x:selected.x+selected.w/2,y:above?selected.y:selected.y+selected.h},
      end:{x:range.x+range.w/2,y:above?range.y+range.h:range.y},
      sourceWidth:selected.w,targetWidth:range.w,vertical:true
    };
    const toLeft=target.x+target.w/2<source.x+source.w/2;
    return {
      start:{x:toLeft?selected.x:selected.x+selected.w,y:selected.y+selected.h/2},
      end:{x:toLeft?range.x+range.w:range.x,y:range.y+range.h/2},
      sourceWidth:selected.h,targetWidth:range.h,vertical:false
    };
  }
  function scalarConnection(anchors){
    const width=Math.min(12,anchors.sourceWidth*.22,anchors.targetWidth*.22);
    return {...anchors,sourceWidth:width,targetWidth:width};
  }
  function ribbonConnection(start,end,startWidth,endWidth,vertical=false,lift=0){
    const a=startWidth/2,b=endWidth/2;
    if(lift){
      if(vertical){
        const dy=(end.y-start.y)*.38;
        return `M${start.x-a},${start.y} C${start.x-a+lift},${start.y+dy} ${end.x-b+lift},${end.y-dy} ${end.x-b},${end.y} L${end.x+b},${end.y} C${end.x+b+lift},${end.y-dy} ${start.x+a+lift},${start.y+dy} ${start.x+a},${start.y} Z`;
      }
      const dx=(end.x-start.x)*.38;
      return `M${start.x},${start.y-a} C${start.x+dx},${start.y-a-lift} ${end.x-dx},${end.y-b-lift} ${end.x},${end.y-b} L${end.x},${end.y+b} C${end.x-dx},${end.y+b-lift} ${start.x+dx},${start.y+a-lift} ${start.x},${start.y+a} Z`;
    }
    if(vertical){
      const mid=(start.y+end.y)/2;
      return `M${start.x-a},${start.y} C${start.x-a},${mid} ${end.x-b},${mid} ${end.x-b},${end.y} L${end.x+b},${end.y} C${end.x+b},${mid} ${start.x+a},${mid} ${start.x+a},${start.y} Z`;
    }
    const mid=(start.x+end.x)/2;
    return `M${start.x},${start.y-a} C${mid},${start.y-a} ${mid},${end.y-b} ${end.x},${end.y-b} L${end.x},${end.y+b} C${mid},${end.y+b} ${mid},${start.y+a} ${start.x},${start.y+a} Z`;
  }

function ribbonSample(start,end,vertical,lift,t){
  const delta=(vertical?end.y-start.y:end.x-start.x)*.38;
  const c1=vertical?{x:start.x+lift,y:start.y+delta}:{x:start.x+delta,y:start.y-lift};
  const c2=vertical?{x:end.x+lift,y:end.y-delta}:{x:end.x-delta,y:end.y-lift};
  const u=1-t;
  const x=u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x;
  const y=u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y;
  const dx=3*u*u*(c1.x-start.x)+6*u*t*(c2.x-c1.x)+3*t*t*(end.x-c2.x);
  const dy=3*u*u*(c1.y-start.y)+6*u*t*(c2.y-c1.y)+3*t*t*(end.y-c2.y);
  return {x,y,angle:Math.atan2(dy,dx)*180/Math.PI};
}
function placeRibbonIndicator(start,end,vertical,lift,bounds){
  let best=null,bestScore=Infinity;
  // Long connections may cross the viewport with only their first or last
  // few percent visible. Do not lose the tracing marker near either end.
  for(let n=1;n<=99;n++){
    const t=n/100,p=ribbonSample(start,end,vertical,lift,t);
    if(p.x<bounds.left+30||p.x>bounds.right-30||p.y<bounds.top+22||p.y>bounds.bottom-36)continue;
    // Prefer a point near the operand, leaving a small gap before its matrix.
    const distance=Math.hypot(p.x-end.x,p.y-end.y),score=Math.abs(distance-28)+(1-t)*5;
    if(score<bestScore){bestScore=score;best={...p,t};}
  }
  return best;
}

// Dock to the operand edge whenever that edge is visible. The ribbon is only
// a fallback when the operand itself has left the viewport.
function placeOperandEndMarker(start,end,vertical,bounds){
  const point=vertical
    ? {x:end.x,y:end.y+(start.y>end.y?13:-13),angle:start.y>end.y?-90:90}
    : {x:end.x+(start.x>end.x?24:-24),y:end.y,angle:start.x>end.x?180:0};
  if(point.x-20<bounds.left||point.x+20>bounds.right||point.y-12<bounds.top||point.y+12>bounds.bottom)return null;
  return {...point,docked:true};
}


function nameConnection(source,target,obstacles=[]){
 const {start,end,vertical}=connectionAnchors(source,target,source,target);
 const dx=end.x-start.x,dy=end.y-start.y,steps=Math.max(80,Math.ceil(Math.hypot(dx,dy)/3));
 for(const offset of [24,-24,48,-48,80,-80,120,-120,180,-180,260,-260,384,-384]){
  const c1=vertical?{x:start.x+offset,y:start.y+dy/3}:{x:start.x+dx/3,y:start.y+offset};
  const c2=vertical?{x:end.x+offset,y:end.y-dy/3}:{x:end.x-dx/3,y:end.y+offset};
  let blocked=false;
  for(let i=1;i<steps&&!blocked;i++){
   const t=i/steps,u=1-t,x=u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x,y=u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y;
   blocked=obstacles.some(r=>x>r.x&&x<r.x+r.w&&y>r.y&&y<r.y+r.h);
  }
  if(!blocked)return `M${start.x},${start.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${end.x},${end.y}`;
 }
 const points=routeConnection(start,end,obstacles);
 return points?curvedConnection(points,obstacles,32):'';
}
const api={connectionAnchors,scalarConnection,ribbonConnection,ribbonSample,placeOperandEndMarker,placeRibbonIndicator,nameConnection};
root.LatteRibbons=api;
if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
