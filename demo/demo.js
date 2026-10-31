/* Label the recorded edition without changing the local workspace layout. */
(() => {
  function enhance(){
    const badge=document.querySelector('.training-replay .latte-local-badge');if(!badge)return false;
    badge.replaceChildren(document.createTextNode('RECORDED DEMO · '));
    const link=document.createElement('a');link.href='run-locally.html';link.textContent='Run locally →';badge.append(link);return true;
  }
  if(!enhance()){
    const observer=new MutationObserver(()=>{if(enhance())observer.disconnect();});
    observer.observe(document.body,{childList:true,subtree:true});
  }
})();
