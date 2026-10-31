import { max, sum } from 'd3-array';
import type { Metadata, Step, Tensor, Weights } from './types';

export function targetProbability(logits: number[], target: number) {
  const largest = max(logits)!;
  const values = logits.map(value => Math.exp(value - largest));
  return values[target] / sum(values);
}

export function parameterLabel(name: string) {
  const globals: Record<string, string> = {
    'token_embedding.weight': 'Token embeddings', 'position_embedding.weight': 'Position embeddings',
    'final_norm.weight': 'Final normalization scale', 'lm_head.weight': 'Vocabulary projection',
  };
  if (globals[name]) return globals[name];
  const layer = name.match(/^blocks\.(\d+)\.(.*)\.weight$/);
  const names: Record<string, string> = {
    'attention.wq': 'Query weights', 'attention.wk': 'Key weights', 'attention.wv': 'Value weights',
    'attention.wo': 'Attention output weights', attention_norm: 'Attention normalization scale',
    ffn_norm: 'MLP normalization scale', 'ffn.0': 'MLP input weights', 'ffn.2': 'MLP output weights',
  };
  return layer ? `Layer ${Number(layer[1])} · ${names[layer[2]] || layer[2]}` : name;
}

export function traceFor(meta: Metadata, step: Step, sequence: number, weights: Weights) {
  const architecture = meta.architecture;
  const global_weights: Weights = {};
  for (const name of ['token_embedding', 'position_embedding', 'final_norm', 'lm_head']) {
    global_weights[name] = weights[name + '.weight'];
  }
  const layers = Array.from({ length: architecture.num_layers }, (_, i) => {
    const prefix = `blocks.${i}.`;
    const w: Record<string, Tensor | number[][][]> = {};
    for (const [key, path] of Object.entries({ attention_norm: 'attention_norm', ffn_norm: 'ffn_norm', wo: 'attention.wo', ffn_in: 'ffn.0', ffn_out: 'ffn.2' })) {
      w[key] = weights[prefix + path + '.weight'];
    }
    for (const letter of ['q', 'k', 'v']) {
      const width = letter === 'v' ? architecture.value_dim : architecture.qk_dim;
      const matrix = weights[prefix + `attention.w${letter}.weight`] as number[][];
      w[`w${letter}_by_head`] = Array.from({ length: architecture.num_heads }, (_, h) => matrix.slice(h * width, (h + 1) * width));
    }
    return { layer: i, weights: w };
  });
  return { architecture, vocabulary: meta.vocabulary, global_weights, layers,
    training: { targets: step.targets[sequence].map(i => meta.vocabulary[i]), losses: step.token_losses[sequence] } };
}

export function selectedWeight(cell: HTMLElement, meta: Metadata, step: Step, sequence: number) {
  const name = cell.dataset.matrixName || '';
  let r = Number(cell.dataset.matrixRow) - 1, c = Number(cell.dataset.matrixColumn) - 1;
  let key: string | undefined;
  const layer = name.match(/layer (\d+)/), head = name.match(/head (\d+)/);
  if (name === 'Position embeddings') { key = 'position_embedding.weight'; [r, c] = [c, r]; }
  else if (name === 'Vocabulary weights') key = 'lm_head.weight';
  else if (name === 'RMS scale · Final hidden') key = 'final_norm.weight';
  else if (name === 'Token embeddings') { key = 'token_embedding.weight'; [r, c] = [step.inputs[sequence][c], r]; }
  else if (layer) {
    const prefix = `blocks.${layer[1]}.`;
    if (/^W[QKV] ·/.test(name) && head) {
      const letter = name[1].toLowerCase(); key = prefix + `attention.w${letter}.weight`;
      r += Number(head[1]) * (letter === 'v' ? meta.architecture.value_dim : meta.architecture.qk_dim);
    } else if (name.startsWith('WO ·')) key = prefix + 'attention.wo.weight';
    else if (name.startsWith('W1 ·')) key = prefix + 'ffn.0.weight';
    else if (name.startsWith('W2 ·')) key = prefix + 'ffn.2.weight';
    else if (name.startsWith('RMS scale · Attention')) key = prefix + 'attention_norm.weight';
    else if (name.startsWith('RMS scale · FFN')) key = prefix + 'ffn_norm.weight';
  }
  return key ? { key, row: r, column: c } : undefined;
}
