import { useEffect, useRef } from 'react';

export const isDemo = document.documentElement.hasAttribute('data-latte-demo');
export const explorerHref = (id: string) => isDemo ? './' : `/?run=${encodeURIComponent(id)}`;
export function showLocalOnly() { window.dispatchEvent(new Event('latte:local-only')); }
let saved: Promise<any> | undefined;
export async function demoResponse(path: string, options?: RequestInit): Promise<Response> {
  if (options?.method && options.method !== 'GET') {
    showLocalOnly();
    return new Response(JSON.stringify({detail:'Run Latte locally to edit or train.'}), {status:403});
  }
  const data = await (saved ||= fetch(new URL('data/workspace.json', document.baseURI)).then(response => {
    if (!response.ok) throw new Error('Could not load the recorded demo. Please reload.');
    return response.json();
  }));
  const endpoint = path.replace(/^\/api\/runs\/(?:default|demo)/, '');
  let value: unknown;
  if (path === '/api/runs') value = [data.run];
  else if (endpoint === '/metadata') value = data.metadata;
  else if (endpoint === '/shell') return new Response(data.shell);
  else if (endpoint === '/status') value = data.run;
  else if (endpoint === '/corpus') value = {text:data.metadata.corpus.text};
  else if (endpoint === '/settings') value = data.settings;
  else if (/^\/steps\/\d+$/.test(endpoint)) value = data.steps[Number(endpoint.split('/').pop())];
  if (value === undefined) return new Response('Recorded demo endpoint not found', {status:404});
  return new Response(JSON.stringify(value), {headers:{'Content-Type':'application/json'}});
}

export function LocalOnlyNotice() {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const open = () => { if (!dialog.current?.open) dialog.current?.showModal(); };
    window.addEventListener('latte:local-only', open);
    return () => window.removeEventListener('latte:local-only', open);
  }, []);
  if (!isDemo) return null;
  return <dialog ref={dialog} className="latte-local-only-dialog" aria-labelledby="latte-local-only-title">
    <h2 id="latte-local-only-title">Run Latte locally</h2>
    <p>This online demo explores a recorded training run. Run Latte on your computer to change the corpus or settings, start training, and manage your runs.</p>
    <div><button type="button" onClick={() => dialog.current?.close()}>Keep exploring</button><a href="run-locally.html">Run locally →</a></div>
  </dialog>;
}
