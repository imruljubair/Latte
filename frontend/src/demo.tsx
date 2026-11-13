import { useEffect, useRef } from 'react';

export const isDemo = document.documentElement.hasAttribute('data-latte-demo');
export const explorerHref = (id: string) => `${isDemo ? './' : '/'}?run=${encodeURIComponent(id)}`;
export function showLocalOnly() { window.dispatchEvent(new Event('latte:local-only')); }

type DemoEntry = { id: string; file: string };
type DemoWorkspace = {
  metadata: { corpus: { text: string } };
  shell: string;
  run: unknown;
  settings: unknown;
  steps: unknown[];
};

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(new URL(`data/${path}`, document.baseURI));
  if (!response.ok) throw new Error('Could not load the recorded example. Please reload.');
  return response.json();
}

let catalog: Promise<DemoEntry[]> | undefined;
const recordings = new Map<string, Promise<DemoWorkspace>>();

export async function demoResponse(path: string, options?: RequestInit): Promise<Response> {
  if (options?.method && options.method !== 'GET') {
    showLocalOnly();
    return Response.json({ detail: 'Run Latte locally to edit settings or train a custom configuration.' }, { status: 403 });
  }

  const entries = await (catalog ||= loadJson<DemoEntry[]>('runs.json').catch(error => {
    catalog = undefined;
    throw error;
  }));
  if (path === '/api/runs') return Response.json(entries);

  const route = path.match(/^\/api\/runs\/([^/]+)(\/.*)$/);
  const requested = route && decodeURIComponent(route[1]);
  const entry = requested === 'default' ? entries[0] : entries.find(item => item.id === requested);
  if (!entry || !route) return new Response('Recorded example not found', { status: 404 });

  let recording = recordings.get(entry.id);
  if (!recording) {
    recording = loadJson<DemoWorkspace>(entry.file).catch(error => {
      recordings.delete(entry.id);
      throw error;
    });
    recordings.set(entry.id, recording);
  }
  const data = await recording;
  const endpoint = route[2];
  let value: unknown;
  if (endpoint === '/metadata') value = data.metadata;
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
    <p>Choose an example to replay its recorded training online. Install Latte to edit the corpus, change settings, or train a custom configuration.</p>
    <div><button type="button" onClick={() => dialog.current?.close()}>Keep exploring</button><a href="run-locally.html">Run locally →</a></div>
  </dialog>;
}
