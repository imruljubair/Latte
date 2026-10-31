import { useEffect, useRef, useState } from 'react';
import './training-studio.css';
import { isDemo, demoResponse, explorerHref, showLocalOnly } from './demo';
import type { Metadata } from './types';

type Model = { model_dim: number; qk_dim: number; value_dim: number; num_heads: number; num_layers: number; ffn_dim: number; context_length: number };
type Settings = { steps: number; batch_size: number; learning_rate: number; weight_decay: number; seed: number; train_split: number; record_every: number };
type Run = { id: string; name: string; created_at?: string; state: string; explorer_ready: boolean; log: string; progress?: { phase?: string; step?: number; total_steps?: number; loss?: number | null; recorded?: number; history_averaged?: boolean; history_sparse?: boolean; validation_loss?: number | null; validation_history?: { step: number; loss: number }[]; history?: { step: number; loss: number }[] } };
const active = (run: Run | null) => !!run && ['running', 'stopping'].includes(run.state);
const states: Record<string, string> = { running: 'Training', complete: 'Complete', failed: 'Failed', stopped: 'Stopped', stopping: 'Stopping…', incomplete: 'Interrupted' };
async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await (isDemo ? demoResponse(path, options) : fetch(path, options));
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = Array.isArray(body.detail) ? body.detail.map((item: { msg: string }) => item.msg).join(' · ') : body.detail || message; } catch { /* retain status */ }
    throw new Error(message);
  }
  return response.json();
}
function remember(id: string) { try { localStorage.setItem('latte-training-run', id); } catch { /* training still works without storage */ } }
function NumberField({ label, value, change, min = 1, max, step = 1 }: { label: string; value: number; change: (n: number) => void; min?: number; max?: number; step?: number | 'any' }) {
  return <label>{label}<input type="number" readOnly={isDemo} required min={min} max={max} step={step} value={Number.isFinite(value) ? value : ''} onChange={event => change(event.target.valueAsNumber)} /></label>;
}
type LossPoint = { step: number; loss: number };
function LossChart({ history, validation, averaged, sparse, totalSteps }: { history: LossPoint[]; validation: LossPoint[]; averaged?: boolean; sparse?: boolean; totalSteps: number }) {
  const all = [...history, ...validation];
  const min = all.length ? Math.min(...all.map(p => p.loss)) : 0, max = all.length ? Math.max(...all.map(p => p.loss)) : 1;
  const start = 0, end = Math.max(1, totalSteps);
  const points = (series: LossPoint[]) => series.map(p => `${44 + 492 * (p.step - start) / Math.max(1, end - start)},${15 + 100 * (max - p.loss) / Math.max(.01, max - min)}`).join(' ');
  return <><div className="latte-loss-legend"><span>Training</span><span>Validation</span></div>
    <svg className="latte-loss-chart" viewBox="0 0 560 150" role="img" aria-label={`Training and validation loss from step ${start} to ${end}`}>
    {[15, 65, 115].map(y => <path key={y} d={`M44 ${y}H536`} stroke="#e2d4c5" strokeDasharray="3 4" />)}
    {[{data:history,color:'#a56b35',name:'training'},{data:validation,color:'#538b9b',name:'validation'}].map(series => <g key={series.name} data-loss-series={series.name}>
      <polyline points={points(series.data)} fill="none" stroke={series.color} strokeDasharray={series.name === 'validation' ? '5 4' : undefined} strokeWidth="1.1" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      {series.data.length === 1 && <circle cx={points(series.data).split(',')[0]} cy={points(series.data).split(',')[1]} r="2" fill={series.color} />}
    </g>)}
    {all.length > 0 ? <><text x="2" y="19">{max.toFixed(3)}</text><text x="2" y="119">{min.toFixed(3)}</text></> : <text x="290" y="65" textAnchor="middle">Waiting for training measurements…</text>}
    {[.25,.5,.75].map(fraction => <text key={fraction} x={44 + 492 * fraction} y="146" textAnchor="middle">{Math.round(end * fraction)}</text>)}
    <text x="44" y="146">Step {start}</text><text x="536" y="146" textAnchor="end">Step {end}</text>
  </svg>{averaged && <p className="latte-form-hint">Training curve averages nearby steps. Latest training loss shows the current batch.</p>}{sparse && <p className="latte-form-hint">Earlier training points use saved recording steps.</p>}{history[0]?.step > 1 && <p className="latte-form-hint">Early training history is unavailable for this run.</p>}{history.length > 0 && !validation.length && <p className="latte-form-hint">This run has no validation measurements. New runs record held-out validation loss.</p>}</>;
}
function readDraft(): { model?: Model; settings?: Settings; name?: string; corpus?: string } {
  if (isDemo) return {};
  try { return JSON.parse(localStorage.getItem('latte-training-draft') || '{}') || {}; } catch { return {}; }
}

export function TrainingStudio({ meta }: { meta: Metadata }) {
  const a = meta.architecture;
  const [draft] = useState(readDraft);
  const [model, setModel] = useState<Model>(draft.model || { model_dim: a.model_dim, qk_dim: a.qk_dim, value_dim: a.value_dim, num_heads: a.num_heads, num_layers: a.num_layers, ffn_dim: a.ffn_dim, context_length: a.context_length });
  const [settings, setSettings] = useState<Settings>(draft.settings || { steps: 100, batch_size: 4, learning_rate: .003, weight_decay: 0, seed: 1337, train_split: .9, record_every: 10 });
  const [name, setName] = useState(draft.name || ''), [corpus, setCorpus] = useState(draft.corpus || '');
  const [run, setRun] = useState<Run | null>(null), [runs, setRuns] = useState<Run[]>([]);
  const generation = useRef(0);
  const monitor = useRef<HTMLElement>(null);
  const [deleting, setDeleting] = useState('');
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState(''), [starting, setStarting] = useState(false), [stopping, setStopping] = useState(false);
  useEffect(() => {
    if (isDemo) return;
    try { localStorage.setItem('latte-training-draft', JSON.stringify({model, settings, name, corpus})); } catch { /* keep the in-memory draft if storage is full */ }
  }, [model, settings, name, corpus]);
  const refresh = () => api<Run[]>('/api/runs').then(setRuns);
  useEffect(() => {
    let disposed = false;
    const requestGeneration = generation.current;
    async function restoreLatest() {
      try {
        const items = await api<Run[]>('/api/runs');
        if (disposed) return;
        setRuns(items);
        const latest = items.find(item => active(item)) || [...items]
          .filter(item => item.created_at)
          .sort((x,y) => y.created_at!.localeCompare(x.created_at!))[0] || (isDemo ? items[0] : undefined);
        if (latest) {
          const [selected, selectedCorpus, selectedSettings] = await Promise.all([
            api<Run>(`/api/runs/${latest.id}/status`),
            api<{text:string}>(`/api/runs/${latest.id}/corpus`),
            api<{model:Model;training:Settings}>(`/api/runs/${latest.id}/settings`),
          ]);
          if (disposed || generation.current !== requestGeneration) return;
          setRun(selected);setCorpus(selectedCorpus.text);setModel(selectedSettings.model);
          setSettings(selectedSettings.training);setName(selected.name);
        }
      } catch (reason) {
        if (!disposed && generation.current === requestGeneration) setError(`Could not restore the latest run: ${(reason as Error).message}`);
      } finally { if (!disposed) setRestoring(false); }
    }
    restoreLatest();
    return () => { disposed = true; };
  }, []);
  useEffect(() => {
    if (!run || !active(run)) return;
    const requestGeneration = generation.current;
    let disposed = false, timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await api<Run>(`/api/runs/${run!.id}/status`);
        if (disposed || generation.current !== requestGeneration) return;
        setRun(next); setError('');
        if (active(next)) timer = setTimeout(poll, 750);
        else { setStopping(false); refresh().catch(() => {}); }
      } catch (reason) {
        if (!disposed && generation.current === requestGeneration) { setError(`Progress connection interrupted: ${(reason as Error).message}. Retrying…`); timer = setTimeout(poll, 2000); }
      }
    }
    poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [run?.id, run?.state]);
  const characters = Array.from(corpus).length;
  const trainingChars = Math.floor(characters * settings.train_split);
  const corpusValid = Math.min(trainingChars, characters - trainingChars) >= model.context_length + 2;
  const busy = restoring || starting || active(run);
  const progress = run?.progress, total = progress?.total_steps || progress?.step || settings.steps, current = progress?.step || 0;
  const phase = restoring ? 'Loading latest run…' : starting ? 'Starting a new run…' : run?.state === 'running' && progress?.phase === 'exporting' ? 'Preparing explorer…' : run ? states[run.state] || run.state : 'Ready when you are';
  const modelField = (key: keyof Model, label: string, max: number) => <NumberField key={key} label={label} value={model[key]} max={max} change={value => setModel(previous => ({ ...previous, [key]: value }))} />;
  const settingField = (key: keyof Settings, label: string, min = 1, step: number | 'any' = 1, max?: number) => <NumberField key={key} label={label} value={settings[key]} min={min} max={max} step={step} change={value => setSettings(previous => ({ ...previous, [key]: value }))} />;
  function focusProgress() {
    monitor.current?.focus({ preventScroll: true });
    monitor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  async function start(event: React.FormEvent) {
    event.preventDefault();
    if (isDemo) { showLocalOnly(); return; }
    const requestGeneration = ++generation.current;
    setError(''); setStarting(true); setStopping(false); setRun(null);
    focusProgress();
    try {
      const next = await api<Run>('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, corpus, model, training: settings }) });
      if (generation.current !== requestGeneration) return;
      remember(next.id); setRun(next); refresh().catch(() => {});
      focusProgress();
    } catch (reason) { setError((reason as Error).message); }
    finally { setStarting(false); }
  }
  async function selectRun(item: Run) {
    const requestGeneration = ++generation.current;
    setError('');
    try {
      const [selected, selectedCorpus, selectedSettings] = await Promise.all([
        api<Run>(`/api/runs/${item.id}/status`),
        api<{text: string}>(`/api/runs/${item.id}/corpus`),
        api<{model: Model; training: Settings}>(`/api/runs/${item.id}/settings`),
      ]);
      if (generation.current !== requestGeneration) return;
      remember(item.id); setRun(selected); setCorpus(selectedCorpus.text);
      setModel(selectedSettings.model); setSettings(selectedSettings.training); setName(selected.name);
      focusProgress();
    } catch (reason) { setError((reason as Error).message); }
  }
  async function deleteRun(item: Run) {
    if (isDemo) { showLocalOnly(); return; }
    setDeleting(item.id); setError('');
    try {
      await api(`/api/runs/${item.id}`, {method:'DELETE'});
      if (run?.id === item.id) { ++generation.current; setRun(null); remember(''); }
      setRuns(previous => previous.filter(candidate => candidate.id !== item.id));
      if (new URL(location.href).searchParams.get('run') === item.id) location.assign('/?tab=training');
    } catch (reason) { setError((reason as Error).message); }
    finally { setDeleting(''); }
  }
  async function stop() {
    if (!run) return;
    setStopping(true); setError('');
    try { setRun(await api<Run>(`/api/runs/${run.id}/stop`, { method: 'POST' })); }
    catch (reason) { setError((reason as Error).message); setStopping(false); }
  }
  return <main className="latte-training-studio" id="latte-training-panel" role="tabpanel" aria-labelledby="latte-training-tab" aria-busy={restoring}>
    <div className="latte-studio-intro"><div><span className="latte-studio-eyebrow">A model, made by you</span><h1>Train your next little model.</h1><p>Bring your own text. Watch it learn. Explore the calculations.</p></div><span className="latte-studio-local">Local PyTorch · AdamW · character tokens</span></div>
    {error && <p className="latte-studio-error" role="alert">{error}</p>}
    <div className="latte-studio-columns">
      <form className="latte-training-form" onSubmit={start}
        onPointerDownCapture={event => { if (isDemo && (event.target as HTMLElement).closest('input,textarea,button')) { event.preventDefault(); showLocalOnly(); } }}
        onKeyDownCapture={event => { if (isDemo && (event.target as HTMLElement).matches('input,textarea') && (event.key.length === 1 || ['Backspace','Delete','ArrowUp','ArrowDown'].includes(event.key)) && !event.ctrlKey && !event.metaKey) { event.preventDefault(); showLocalOnly(); } }}
        onPasteCapture={event => { if (isDemo) { event.preventDefault(); showLocalOnly(); } }}>
        <fieldset disabled={busy}>
          <section className="latte-studio-card"><div className="latte-card-heading"><h2>1. Your corpus</h2><button type="button" onClick={() => isDemo ? showLocalOnly() : setCorpus(meta.corpus.text)}>Use current corpus</button></div>
            <label>Run name <span className="latte-optional">optional</span><input readOnly={isDemo} value={name} maxLength={80} placeholder="A small story experiment" onChange={event => setName(event.target.value)} /></label>
            <label>Training text<textarea readOnly={isDemo} aria-label="Training text" required value={corpus} maxLength={1000000} onChange={event => setCorpus(event.target.value)} placeholder="Paste stories, notes, or any text you want the model to learn…" rows={8} /></label>
            <div className="latte-corpus-stats"><span>{characters.toLocaleString()} characters</span><span>{new Set(Array.from(corpus)).size} unique tokens</span><span>{Math.round(settings.train_split * 100)}% training / {Math.round((1 - settings.train_split) * 100)}% validation</span></div>
            {characters > 0 && !corpusValid && <p className="latte-form-hint">Add more text or reduce context length. Both splits need at least {model.context_length + 2} characters.</p>}
          </section>
          <section className="latte-studio-card latte-parameter-card"><h2>2. Model architecture</h2><div className="latte-field-grid">
            {modelField('model_dim', 'Embedding dimension', 2048)}{modelField('num_layers', 'Layers', 64)}{modelField('num_heads', 'Attention heads', 64)}{modelField('qk_dim', 'Query / key dimension per head', 2048)}{modelField('value_dim', 'Value dimension per head', 2048)}{modelField('ffn_dim', 'Feed-forward dimension', 8192)}{modelField('context_length', 'Context length', 2048)}
          </div><p className="latte-form-hint">Starts with the architecture you’re exploring. Dropout stays at 0 for reproducible replay.</p></section>
          <section className="latte-studio-card latte-parameter-card"><h2>3. Training settings</h2><div className="latte-field-grid">
            {settingField('steps', 'Training steps', 1, 1, 100000)}{settingField('batch_size', 'Batch size', 1, 1, 256)}{settingField('learning_rate', 'Learning rate', .00000001, 'any', 1)}{settingField('record_every', 'Record every N steps', 1, 1, 100000)}
          </div><details className="latte-advanced-settings"><summary>More settings</summary><div className="latte-field-grid">
            {settingField('weight_decay', 'Weight decay', 0, 'any', 1)}{settingField('seed', 'Random seed', 0, 1, 4294967295)}{settingField('train_split', 'Training split', .01, .01, .99)}
          </div></details><p className="latte-form-hint">The first and final steps are always recorded. Recording less often makes replay files smaller.</p></section>
        </fieldset>
        <button className="latte-start-training" type="submit" disabled={busy || !corpusValid}>{starting ? 'Starting…' : active(run) ? 'Training in progress' : 'Start training'}</button>
      </form>
      <aside className="latte-training-monitor"><section ref={monitor} tabIndex={-1} className="latte-studio-card latte-progress-card" aria-label="Training progress">
        <div className="latte-card-heading"><h2>Training progress</h2><span className={`latte-run-state ${run?.state || ''}`} role="status">{phase}</span></div>
        <p className="latte-monitor-name">{run?.name || 'Your next experiment'}</p>
        <progress max={total} value={current} aria-label="Training step progress" />
        <div className="latte-progress-stats"><div><strong>{current} <small>/ {total}</small></strong><span>optimizer steps</span></div><div><strong>{progress?.loss == null ? '—' : progress.loss.toFixed(4)}</strong><span>latest training loss</span></div><div><strong>{progress?.validation_loss == null ? '—' : progress.validation_loss.toFixed(4)}</strong><span>latest validation loss</span></div><div><strong>{progress?.recorded || 0}</strong><span>recorded steps</span></div></div>
        <LossChart key={run?.id || 'new-run'} totalSteps={total} history={progress?.history || []} validation={progress?.validation_history || []} averaged={progress?.history_averaged} sparse={progress?.history_sparse} />
        {run?.state === 'running' && progress?.phase === 'exporting' && <p className="latte-form-hint">Training is finished. Saving the checkpoint and preparing the interactive explorer.</p>}
        <div className="latte-monitor-actions">{active(run) && <button type="button" disabled={stopping || run?.state === 'stopping'} onClick={stop}>{stopping || run?.state === 'stopping' ? 'Stopping…' : 'Stop training'}</button>}{run?.explorer_ready && <a className="latte-open-explorer" href={explorerHref(run.id)}>Open in explorer →</a>}</div>
        {run?.state === 'failed' && <p className="latte-studio-error" role="alert">Training failed. See the run log below, adjust your settings, and try again.</p>}
        {run && <details className="latte-run-log"><summary>Run log</summary><pre>{run.log || 'Waiting for the training process…'}</pre></details>}
      </section>
      <section className="latte-studio-card latte-recent-runs"><h2>Recent runs</h2><p className="latte-form-hint">Select a run to load its corpus, settings, progress, and log.</p>{runs.length ? <ul className="latte-run-list">{[...runs].sort((x, y) => (y.created_at || '').localeCompare(x.created_at || '')).map(item => <li key={item.id}><button type="button" aria-pressed={run?.id === item.id} disabled={restoring || starting || (active(run) && run?.id !== item.id)} onClick={() => selectRun(item)}><span>{item.name}</span><small>{states[item.state] || item.state}{run?.id === item.id ? ' · Viewing' : ' · View →'}</small></button>{item.explorer_ready && <a href={explorerHref(item.id)} aria-label={`Explore ${item.name}`}>Explore →</a>}{item.id !== 'default' && <button className="latte-delete-run" type="button" disabled={restoring || !!deleting || starting || active(item)} aria-label={`Delete ${item.name}`} onClick={() => deleteRun(item)}>{deleting === item.id ? 'Deleting…' : 'Delete'}</button>}</li>)}</ul> : <p className="latte-form-hint">Completed experiments will appear here.</p>}</section>
      </aside>
    </div>
  </main>;
}
