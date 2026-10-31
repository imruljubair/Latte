/** @param {{ranges:number[][]}|undefined} usage @param {number} position */
export function usageAt(usage, position) {
  const ranges=usage?.ranges || [];
  let left=0,right=ranges.length-1;
  while(left<=right){
    const middle=(left+right)>>1,[start,end,count]=ranges[middle];
    if(position<start)right=middle-1;
    else if(position>=end)left=middle+1;
    else return count;
  }
  return 0;
}
/** @param {number} count @param {number} maximum */
export function usageColor(count, maximum) {
  const fraction = Math.max(0, Math.min(1, count / Math.max(1, maximum)));
  return `hsla(${(210 * (1 - fraction)).toFixed(1)}, 70%, 65%, 0.45)`;
}
/** Constant-count spans avoid creating a DOM node for every corpus character.
 * @param {{ranges:number[][]}|undefined} usage @param {number} start
 * @param {number} end @param {number} trainEnd
 */
export function usageSegments(usage, start, end, trainEnd) {
  const result=[];
  const append=(a,b,count)=>{
    if(b<=a)return;
    if(a<trainEnd&&b>trainEnd){append(a,trainEnd,count);append(trainEnd,b,0);return;}
    result.push({start:a,end:b,count:a>=trainEnd?0:count,validation:a>=trainEnd});
  };
  let cursor=start;
  for(const [a,b,count] of usage?.ranges || []){
    if(b<=start)continue;
    if(a>=end)break;
    append(cursor,Math.min(a,end),0);
    append(Math.max(start,a),Math.min(end,b),count);
    cursor=Math.min(end,b);
  }
  append(cursor,end,0);
  return result;
}
