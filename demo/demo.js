/* Public recorded edition only. The local application has no demo restrictions. */
(() => {
  const nav=document.createElement('nav');nav.className='demo-nav';nav.setAttribute('aria-label','Demo navigation');
  for(const [label,url] of [['Explorer','./'],['Training','training.html'],['Run locally','run-locally.html']]){
    const link=document.createElement('a');link.textContent=label;link.href=url;
    if(label==='Explorer')link.setAttribute('aria-current','page');nav.append(link);
  }
  function enhance(){
    const host=document.querySelector('.training-replay .latte-topbar');if(!host)return false;
    host.querySelector('.latte-local-badge')?.remove();host.append(nav);
    const note=document.createElement('span');note.className='demo-recorded-note';note.textContent='Recorded demo · fixed corpus and model';
    document.querySelector('.training-replay>header')?.append(note);
    document.documentElement.setAttribute('data-latte-demo','');return true;
  }
  if(!enhance()){
    const observer=new MutationObserver(()=>{if(enhance())observer.disconnect();});
    observer.observe(document.body,{childList:true,subtree:true});
  }
})();
