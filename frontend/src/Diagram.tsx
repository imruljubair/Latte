import { maxIndex } from 'd3-array';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { revealApp } from './startup';
import { selectedWeight, traceFor } from './replay';
import type { Grid, Metadata, Step } from './types';

const diagramStyle = '<style>.sequence-editor,.sampling-chart-toggle,.toolbar-group:has([data-action="sequence"]){display:none!important}.latte-visible-stage .matrix-value{transition:background-color .4s,filter .3s}html:not([data-training-updated]) .latte-visible-stage .matrix-value.learned-parameter{filter:grayscale(1)}</style>';

interface Props {
  meta: Metadata; step: Step; sequence: number; applied: boolean; shell: string;
  onSelect: (selection: { key: string; row: number; column: number }) => void;
}

export function Diagram(props: Props) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [visible, setVisible] = useState(false);
  useLayoutEffect(() => { if (visible) revealApp(); }, [visible]);
  const latest = useRef(props);
  latest.current = props;
  const ready = useRef(false);
  const pending = useRef<Grid | null>(null);

  useEffect(() => {
    const iframe = frame.current!;
    const { meta, step, sequence, applied, shell } = props;
    const tokens = step.inputs[sequence].map(i => meta.vocabulary[i]);
    const grid = new window.LittleLlamaSequenceGrid(traceFor(meta, step, sequence, applied ? step.after : step.before), tokens, {}).build();
    pending.current = grid;
    const syncTokenMatches = () => {
      const current = latest.current;
      (iframe.contentWindow as any)?.LattePresentation?.setTokenMatches(
        current.step.logits[current.sequence].map((logits, i) => maxIndex(logits) === current.step.targets[current.sequence][i]));
    };
    if (ready.current) {
      iframe.contentDocument!.documentElement.toggleAttribute('data-training-updated', applied);
      iframe.contentWindow!.LittleLlamaWorkbook.updateTrainingGrid([...grid.cells.values()]);
      syncTokenMatches();
      return;
    }
    if (iframe.hasAttribute('srcdoc')) return;
    iframe.onload = () => {
      const doc = iframe.contentDocument!;
      iframe.contentWindow!.LittleLlamaWorkbook.weightOrigin = cell => {
        const current = latest.current;
        const selection = selectedWeight(cell, current.meta, current.step, current.sequence);
        if (!selection) return;
        const at = (weights: Step['before']) => {
          const row = weights[selection.key]?.[selection.row];
          return Array.isArray(row) ? row[selection.column] : selection.column === 0 ? row : undefined;
        };
        return { before: at(current.step.before), after: at(current.step.after), gradient: at(current.step.gradients),
          applied: current.applied, step: current.step.step, batchSize: current.step.inputs.length };
      };
      doc.querySelectorAll<HTMLInputElement>('input:not(.stage-picker input):not(.stage-pick-control input)').forEach(input => input.disabled = true);
      doc.querySelectorAll<HTMLElement>('[contenteditable]').forEach(cell => cell.contentEditable = 'false');
      doc.addEventListener('click', event => {
        const cell = (event.target as HTMLElement).closest<HTMLElement>('[data-matrix-name]');
        if (!cell) return;
        const current = latest.current;
        const selection = selectedWeight(cell, current.meta, current.step, current.sequence);
        if (selection) current.onSelect(selection);
      }, true);
      ready.current = true;
      doc.documentElement.toggleAttribute('data-training-updated', latest.current.applied);
      if (pending.current !== grid) iframe.contentWindow!.LittleLlamaWorkbook.updateTrainingGrid([...pending.current!.cells.values()]);
      syncTokenMatches();
      Promise.all([document.fonts.ready, doc.fonts.ready]).then(() => {
        requestAnimationFrame(() => requestAnimationFrame(() => setVisible(true)));
      });
    };
    iframe.srcdoc = shell.replace(/<html\b/, '<html data-training-replay')
      .replace('__FORWARD_TABLE__', () => grid.table()).replace('__GENERATION_CONTROLS__', '')
      .replace('__SEQUENCE_SCRIPT__', '').replace('</head>', diagramStyle + '</head>');
  }, [props.meta, props.step, props.sequence, props.applied, props.shell]);

  return <iframe ref={frame} title="Training forward pass" style={{visibility: visible ? 'visible' : 'hidden'}} />;
}
