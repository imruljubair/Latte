import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { maxIndex } from 'd3-array';
import { revealApp } from './startup';
import { Diagram } from './Diagram';
import { TrainingStudio } from './TrainingStudio';
import { usageAt, usageColor, usageSegments } from './corpus-usage.mjs';
import { parameterLabel, targetProbability } from './replay';
import type { Metadata, Step, Tensor } from './types';

interface Props {
  meta: Metadata; shell: string; first: Step;
  getStep: (index: number) => Promise<Step>;
}

export function App({ meta, shell, first, getStep }: Props) {
  const [activeTab, setActiveTab] = useState(new URLSearchParams(location.search).get('tab') === 'training' ? 'training' : 'explorer');
  useLayoutEffect(() => { if (activeTab === 'training') document.fonts.ready.then(revealApp); }, [activeTab]);
  const [step, setStep] = useState(first), [index, setIndex] = useState(0);
  const [sequence, setSequence] = useState(0), [applied, setApplied] = useState(false);
  const [heatmap, setHeatmap] = useState(false);
  const [loading, setLoading] = useState(false), [error, setError] = useState('');
  const [selection, setSelection] = useState({ key: Object.keys(first.before)[0], row: 0, column: 0 });
  const requestId = useRef(0);
  const corpus = useRef<HTMLPreElement>(null);
  const tokens = step.inputs[sequence].map(i => meta.vocabulary[i]);
  const start = step.starts[sequence], text = Array.from(meta.corpus.text);
  const correct = step.logits[sequence].map((logits, i) => maxIndex(logits) === step.targets[sequence][i]);

  const usage = meta.corpus_usage;
  const heatmapOn = heatmap && !!usage;
  const usageSlice = (from: number, to: number) => heatmapOn
    ? usageSegments(usage, from, to, meta.corpus.train_end).map(part => <span key={part.start} className="corpus-usage-span" role="button" tabIndex={0} aria-label={`Show usage for characters ${part.start}–${part.end - 1}`}
        data-usage-start={part.start} data-usage-end={part.end} data-usage-count={part.count}
        data-usage-validation={String(part.validation)}
        style={{ backgroundColor: part.validation ? 'transparent' : usageColor(part.count, usage!.max_count) }}>{text.slice(part.start, part.end).join('')}</span>)
    : text.slice(from, to).join('');

  function chooseTab(tab: string) {
    setActiveTab(tab);
    const url = new URL(location.href);
    if (tab === 'training') url.searchParams.set('tab', 'training');
    else url.searchParams.delete('tab');
    history.replaceState(null, '', url);
  }

  async function chooseRecord(next: number) {
    const id = ++requestId.current;
    setLoading(true); setError('');
    try {
      const value = await getStep(next);
      if (id !== requestId.current) return;
      setStep(value); setIndex(next); setApplied(false);
      setSequence(previous => Math.min(previous, value.inputs.length - 1));
    } catch (reason) {
      if (id === requestId.current) setError(`Could not load recorded step: ${String(reason)}`);
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }

  function locateSequence() {
    const element = corpus.current!, mark = element.querySelector('mark');
    if (!mark) return;
    const fragments = [...mark.getClientRects()];
    if (!fragments.length) return;
    const box = element.getBoundingClientRect();
    const top = Math.min(...fragments.map(rect => rect.top)), bottom = Math.max(...fragments.map(rect => rect.bottom));
    element.scrollTop += (top + bottom) / 2 - (box.top + element.clientTop + element.clientHeight / 2);
  }

  useEffect(() => {
    const id = requestAnimationFrame(locateSequence);
    return () => cancelAnimationFrame(id);
  }, [step, sequence]);

  const before = step.before[selection.key];
  const at = (tensor: Tensor) => {
    const row = tensor?.[selection.row];
    return Array.isArray(row) ? row[selection.column] : selection.column === 0 ? row : undefined;
  };
  const oldValue = at(before), gradient = at(step.gradients[selection.key]), newValue = at(step.after[selection.key]);
  const inspection = oldValue === undefined || gradient === undefined || newValue === undefined
    ? 'Choose a valid row and column.'
    : `Old weight  ${oldValue.toPrecision(8)}\nGradient    ${gradient.toPrecision(8)}\nChange      ${(newValue - oldValue).toPrecision(8)}\nNew weight  ${newValue.toPrecision(8)}\nAdamW · ${step.inputs.length} sequences (batch mean)`;

  return <section className={`training-replay${activeTab === 'training' ? ' latte-training-screen' : ''}`}>
    <div className="latte-chrome-host" />
    <nav className="latte-app-tabs" role="tablist" aria-label="Latte workspace">
      <button id="latte-explorer-tab" type="button" role="tab" aria-selected={activeTab === 'explorer'} onClick={() => chooseTab('explorer')}>Explorer</button>
      <button id="latte-training-tab" type="button" role="tab" aria-selected={activeTab === 'training'} aria-controls="latte-training-panel" onClick={() => chooseTab('training')}>Training</button>
    </nav>
    <div hidden={activeTab !== 'training'}><TrainingStudio meta={meta} /></div>
    <header hidden={activeTab !== 'explorer'}>
      <button data-action="previous" disabled={loading || index === 0} onClick={() => chooseRecord(index - 1)}>← Previous</button>
      <label>Training record <select className="training-record" aria-label="Training record" value={index} onChange={event => chooseRecord(Number(event.target.value))}>
        {meta.steps.map((item, i) => <option key={i} value={i}>Step {item.step} · {i} / {meta.steps.length - 1}</option>)}
      </select></label>
      <button data-action="next" disabled={loading || index === meta.steps.length - 1} onClick={() => chooseRecord(index + 1)}>Next →</button>
      <label>Sequence <button type="button" data-action="sequence-previous" aria-label="Previous batch sequence" disabled={sequence === 0} onClick={() => setSequence(sequence - 1)}>‹</button>{' '}
        <select className="training-sequence" value={sequence} onChange={event => setSequence(Number(event.target.value))}>{step.inputs.map((_, i) => <option key={i} value={i}>{i} / {step.inputs.length - 1}</option>)}</select>{' '}
        <button type="button" data-action="sequence-next" aria-label="Next batch sequence" disabled={sequence === step.inputs.length - 1} onClick={() => setSequence(sequence + 1)}>›</button>
      </label>
      <button data-action="apply" data-applied={String(applied)} onClick={() => setApplied(!applied)}>{applied ? 'Show before update' : 'Apply update'}</button>
      <span className="training-status" role={error ? 'alert' : 'status'}>{error || `${applied ? 'After update · recomputed forward pass' : 'Before update'} · recorded batch loss ${step.loss.toFixed(5)}`}</span>
      <div className="latte-view-host" />
    </header>
    <div className="training-body" hidden={activeTab !== 'explorer'}>
      <Diagram meta={meta} step={step} sequence={sequence} applied={applied} shell={shell} onSelect={setSelection} />
      <aside>
        <div className="training-legend" aria-label="Color legend"><span><i style={{ background: '#a5abb3' }} />Before weights</span><span><i style={{ background: '#f6ac37' }} />Updated weights</span><span><i style={{ background: '#5d8fcd' }} />Negative values</span><span><i style={{ background: '#ee6861' }} />Positive values</span><span className="training-correct">✓ Correct</span><span className="training-incorrect">✗ Incorrect</span></div>
        <small className="training-resize-hint">Drag each section’s bottom-right corner to resize.</small>
        <section className="training-corpus-section">
          <h3 className="training-corpus-heading">Corpus sequence <button type="button" className="training-locate" aria-label="Locate selected corpus sequence" title="Locate selected sequence" onClick={locateSequence}>⌖</button></h3>
          <p className="training-source">{`Characters ${start}–${start + tokens.length - 1} · batch of ${step.inputs.length}`}</p>
          <div className="latte-corpus-usage-tools"><label><input type="checkbox" checked={heatmapOn} disabled={!usage} onChange={event => setHeatmap(event.target.checked)} />Corpus heatmap</label>
            {usage ? heatmapOn && <span className="latte-usage-scale"><i />0–{usage.max_count} uses · entire run · {(100 * usage.used_positions / meta.corpus.train_end).toFixed(1)}% covered</span>
              : <span className="latte-usage-unavailable">Usage wasn’t recorded for this run.</span>}
          </div>
          <pre className={`training-corpus${heatmapOn ? ' corpus-heatmap-enabled' : ''}`} ref={corpus}>{usageSlice(0, start)}<mark>{tokens.map((token, i) => {
            const predicted = meta.vocabulary[maxIndex(step.logits[sequence][i])];
            const target = meta.vocabulary[step.targets[sequence][i]];
            return <span key={i} className="training-corpus-token" tabIndex={0} role="button"
              data-usage-start={heatmapOn ? start + i : undefined} data-usage-end={heatmapOn ? start + i + 1 : undefined}
              data-usage-count={heatmapOn ? usageAt(usage, start + i) : undefined}
              style={heatmapOn ? { backgroundColor: usageColor(usageAt(usage, start + i), usage!.max_count) } : undefined}
              data-corpus-position={i} data-input={token} data-prediction={predicted} data-target={target} data-target-probability={targetProbability(step.logits[sequence][i], step.targets[sequence][i])} data-match={String(correct[i])}
              aria-label={`Position ${i}: input ${JSON.stringify(token)}, predicted ${JSON.stringify(predicted)}, target ${JSON.stringify(target)}. Recorded before update.`}>{token}</span>;
          })}</mark>{usageSlice(start + tokens.length, text.length)}</pre>
        </section>
        {/* Temporarily disabled: keep weight inspection available to restore. */}
        <section className="training-weight-section" hidden>
          <h3>Weight update · whole batch</h3><p>Click a weight in the diagram or choose a parameter.</p>
          <div className="latte-optimizer-math-host" />
          <select className="training-parameter" value={selection.key} onChange={event => setSelection({ ...selection, key: event.target.value })}>{Object.keys(step.before).map(name => <option key={name} value={name} title={name}>{parameterLabel(name)}</option>)}</select>
          <label> Row <input className="training-row" type="number" min="0" max={before.length - 1} value={selection.row} onChange={event => setSelection({ ...selection, row: Number(event.target.value) })} /></label>
          <label> Column <input className="training-column" type="number" min="0" max={Array.isArray(before[0]) ? before[0].length - 1 : 0} value={selection.column} onChange={event => setSelection({ ...selection, column: Number(event.target.value) })} /></label>
          <p className="training-parameter-path">{selection.key}</p><pre className="training-inspection">{inspection}</pre>
        </section>
      </aside>
    </div>
  </section>;
}
