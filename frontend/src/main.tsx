import { useLayoutEffect } from 'react';
import { revealApp } from './startup';
import { isDemo, demoResponse } from './demo';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { createStepCache } from './step-cache.mjs';
import type { Metadata, Step } from './types';
import './training-replay.css';
import './brand.css';
document.title = 'One Small Latte';

const root = createRoot(document.getElementById('latte-root')!);
const run = new URLSearchParams(location.search).get('run') || 'default';
const base = `/api/runs/${encodeURIComponent(run)}`;

async function response(path: string) {
  const result = await (isDemo ? demoResponse(path) : fetch(path));
  if (!result.ok) throw new Error(`${result.status}: ${await result.text()}`);
  return result;
}

const cache = createStepCache((index: number) => response(`${base}/steps/${index}`).then(value => value.json() as Promise<Step>));

async function start() {
  const [meta, shell, first]: [Metadata, string, Step] = await Promise.all([
    response(`${base}/metadata`).then(value => value.json()),
    response(`${base}/shell`).then(value => value.text()),
    cache.get(0),
  ]);
  // Retain the existing bottom toolbar and exact diagram assets.
  const tabs = document.querySelector('.stage-view-tabs')!;
  let tab = [...tabs.querySelectorAll('button')].find(button => button.textContent === 'Training');
  if (!tab) { tab = document.createElement('button'); tab.textContent = 'Training'; tab.type = 'button'; tabs.append(tab); }
  tab.classList.add('active');
  const menu = document.querySelector<HTMLElement>('.steps-menu');
  if (menu) { menu.hidden = true; menu.inert = true; }
  root.render(<App meta={meta} shell={shell} first={first} getStep={index => cache.get(index)} />);
}

function StartupError({ error }: { error: unknown }) {
  useLayoutEffect(revealApp, []);
  return <p role="alert" style={{position:'fixed',inset:0,margin:0,padding:32,background:'#f7f2eb',zIndex:1000}}>Could not open Latte: {String(error)}</p>;
}
start().catch(error => root.render(<StartupError error={error} />));
