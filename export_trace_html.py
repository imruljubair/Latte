"""Export a standalone interactive explorer directly from a trace; no Excel."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path

import numpy as np

from create_annotated_copy import create_annotated_copy
from corpus_usage import with_corpus_usage
from export_html import build_workbook_html

EPSILON = 1.1920928955078125e-7


@dataclass
class Matrix:
    values: np.ndarray
    row: int
    column: int

    def ref(self, row: int, column: int) -> str:
        return f'R{self.row + row}C{self.column + column}'


def token_label(token):
    return {'\n': '\\n', '\t': '\\t'}.get(token, token)


class TraceGrid:
    def __init__(self, trace):
        self.trace = trace
        architecture = trace['architecture']
        token_count = len(trace['input']['tokens'])
        if not 0 < token_count <= architecture['context_length']:
            raise ValueError('Trace input must fit the model context length')
        if len(trace['layers']) != architecture['num_layers']:
            raise ValueError('Trace layer count differs from its architecture')
        self.right = max(architecture['model_dim'], architecture['qk_dim'],
                         architecture['value_dim'], architecture['ffn_dim'],
                         architecture['num_heads'] * architecture['value_dim'], token_count) + 3
        self.width = self.right + max(len(trace['input']['tokens']), architecture['model_dim'])
        self.cells = {}
        self.row = 3
        self.stages = []
        self.token_count = len(trace['input']['tokens'])
        self.put(1, 1, 'Latte')

    def put(self, row, column, value, formula=None, weight=False):
        self.cells[row, column] = (value, formula, weight)

    def stage(self, title):
        self.stages.append((self.row, title))
        self.put(self.row, 1, f'[{len(self.stages)-1:02d}/00]')
        self.put(self.row, 2, title)
        self.row += 1

    def matrix(self, name, values, column=None, formulas=None, weight=False, tokens=False):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        column = self.right if column is None else column
        self.put(self.row, column, name)
        self.row += 1
        if tokens:
            for index, token in enumerate(self.trace['input']['tokens']):
                self.put(self.row, column + index, token_label(token))
            self.row += 1
        block = Matrix(values, self.row, column)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                self.put(self.row + row, column + col, float(values[row, col]),
                         formulas(row, col) if formulas else None, weight)
        self.row += values.shape[0] + 1
        return block

    def pair(self, weight_name, weight, result_name, result, formula):
        start = self.row
        values = np.asarray(weight)
        columns = values.shape[1] if values.ndim == 2 else 1
        # Match the established layout: all left matrices end one empty
        # column before the shared activation column, regardless of width.
        left = self.matrix(weight_name, weight, column=self.right - 1 - columns, weight=True)
        end = self.row
        self.row = start
        right = self.matrix(result_name, result, formulas=lambda i, j: formula(left, i, j))
        self.row = max(end, self.row)
        return right

    def multiply(self, weight, input_block, name, result_name):
        weight = np.asarray(weight, dtype=float)
        return self.pair(name, weight, result_name, weight @ input_block.values,
                         lambda left, i, j: '=' + '+'.join(
                             f'{left.ref(i,k)}*{input_block.ref(k,j)}' for k in range(weight.shape[1])))

    def norm(self, input_block, gamma, title, name):
        self.stage(title)
        gamma = np.asarray(gamma)
        values = gamma[:, None] * input_block.values / np.sqrt(np.mean(input_block.values**2, axis=0) + EPSILON)
        return self.pair('RMS scale · ' + name, gamma, name, values,
                         lambda weight, i, j: f'={input_block.ref(i,j)}*{weight.ref(i,0)}/SQRT(('
                         + '+'.join(f'{input_block.ref(k,j)}^2' for k in range(len(gamma)))
                         + f')/{len(gamma)}+{EPSILON})')

    def add(self, left, right, title, name):
        self.stage(title)
        return self.matrix(name, left.values + right.values,
                           formulas=lambda i,j: f'={left.ref(i,j)}+{right.ref(i,j)}')

    def build(self):
        t = self.trace
        self.stage('EMBEDDING • Token + position embedding')
        token_values = np.asarray(t['embeddings']['selected_token_embeddings']).T
        positions = np.asarray(t['embeddings']['selected_position_embeddings']).T
        def lookup(i,j):
            result = 'NA()'
            for index in reversed(range(len(t['vocabulary']))):
                result = f'IF(EXACT(R5C,Embeddings!R{index+6}C1),Embeddings!R{index+6}C{i+2},{result})'
            return '=' + result
        tokens = self.matrix('Token embeddings', token_values, formulas=lookup, tokens=True, weight=True)
        position = self.matrix('Position embeddings', positions, weight=True)
        x = self.matrix('X · embedding output', token_values + positions,
                        formulas=lambda i,j: f'={tokens.ref(i,j)}+{position.ref(i,j)}')
        for layer in t['layers']:
            index = layer['layer']
            w = layer['weights']
            suffix = f'layer {index}'
            norm = self.norm(x, w['attention_norm'], f'ATTENTION • RMSNorm • {suffix}', f'Attention input · {suffix}')
            heads = []
            for head in range(t['architecture']['num_heads']):
                tag = f'{suffix} · head {head}'
                results = {}
                for letter, label in [('k','Key'),('v','Value'),('q','Query')]:
                    self.stage(f'ATTENTION • {label} projection • {suffix} • head {head}')
                    results[letter] = self.multiply(w[f'w{letter}_by_head'][head], norm, f'W{letter.upper()} · {tag}', f'{letter.upper()} · {tag}')
                k, q, v = results['k'], results['q'], results['v']
                self.stage(f'ATTENTION • Scores KᵀQ / √dₖ • {suffix} • head {head}')
                start = self.row
                kt = self.matrix(f'Kᵀ · {tag}', k.values.T, column=self.right - 1 - k.values.shape[0],
                                 formulas=lambda i,j: '=' + k.ref(j,i))
                end = self.row
                self.row = start
                scores_values = k.values.T @ q.values / math.sqrt(q.values.shape[0])
                scores = self.matrix(f'Scores · {tag}', scores_values,
                    formulas=lambda i,j: '=(' + '+'.join(f'{kt.ref(i,a)}*{q.ref(a,j)}' for a in range(q.values.shape[0])) + f')/SQRT({q.values.shape[0]})')
                self.row = max(end, self.row)
                self.stage(f'ATTENTION • Causal mask + softmax • {suffix} • head {head}')
                shifted = scores_values.copy()
                shifted[np.tril_indices(self.token_count, -1)] = -np.inf
                attention = np.exp(shifted - shifted.max(axis=0))
                attention /= attention.sum(axis=0)
                def softmax(i,j):
                    if i > j:
                        return '=0'
                    maximum = f'MAX({scores.ref(0,j)}:{scores.ref(j,j)})'
                    exp = lambda a: f'EXP({scores.ref(a,j)}-{maximum})'
                    return '=' + exp(i) + '/(' + '+'.join(exp(a) for a in range(j+1)) + ')'
                a = self.matrix(f'A · {tag}', attention, formulas=softmax)
                self.stage(f'ATTENTION • V · A (weighted value aggregation) • {suffix} • head {head}')
                start = self.row
                aggregation_v = self.matrix(f'V · {tag}', v.values,
                    column=self.right - 1 - v.values.shape[1],
                    formulas=lambda i,j: '=' + v.ref(i,j))
                end = self.row
                self.row = start
                h = self.matrix(f'H · {tag}', v.values @ a.values,
                    formulas=lambda i,j: '=' + '+'.join(f'{aggregation_v.ref(i,b)}*{a.ref(b,j)}' for b in range(self.token_count)))
                self.row = max(end, self.row)
                heads.append(h)
            self.stage(f'ATTENTION • Concatenate heads • {suffix}')
            concat = np.concatenate([h.values for h in heads], axis=0)
            head_rows = heads[0].values.shape[0]
            merged = self.matrix(f'Concatenated heads · {suffix}', concat,
                formulas=lambda i,j: '=' + heads[i//head_rows].ref(i%head_rows,j))
            self.stage(f'ATTENTION • Output projection WO • {suffix}')
            output = self.multiply(w['wo'], merged, f'WO · {suffix}', f'Attention output · {suffix}')
            residual = self.add(x, output, f'RESIDUAL • Add attention output • {suffix}', f'Residual attention · {suffix}')
            norm = self.norm(residual, w['ffn_norm'], f'MLP • RMSNorm • {suffix}', f'FFN input · {suffix}')
            self.stage(f'MLP • Input projection • {suffix}')
            z = self.multiply(w['ffn_in'], norm, f'W1 · {suffix}', f'FFN pre-activation · {suffix}')
            self.stage(f'MLP • GELU activation • {suffix}')
            gelu = z.values * .5 * (1 + np.vectorize(math.erf)(z.values / math.sqrt(2)))
            activated = self.matrix(f'GELU · {suffix}', gelu,
                formulas=lambda i,j: f'=0.5*{z.ref(i,j)}*(1+ERF({z.ref(i,j)}/SQRT(2)))')
            self.stage(f'MLP • Output projection • {suffix}')
            ffn = self.multiply(w['ffn_out'], activated, f'W2 · {suffix}', f'FFN output · {suffix}')
            x = self.add(residual, ffn, f'RESIDUAL • Add MLP output • {suffix}', f'Layer output · {suffix}')
        final = self.norm(x, t['global_weights']['final_norm'], 'OUTPUT • Final RMSNorm', 'Final hidden')
        self.stage('OUTPUT • Vocabulary projection')
        logits = self.multiply(t['global_weights']['lm_head'], final, 'Vocabulary weights', 'Logits')
        self.stage('OUTPUT • Vocabulary softmax')
        shifted = logits.values - logits.values.max(axis=0)
        p = np.exp(shifted); p /= p.sum(axis=0)
        def probability(i,j):
            maximum = f'MAX({logits.ref(0,j)}:{logits.ref(len(p)-1,j)})'
            exp = lambda a: f'EXP({logits.ref(a,j)}-{maximum})'
            return '=' + exp(i) + '/(' + '+'.join(exp(a) for a in range(len(p))) + ')'
        self.matrix('Vocabulary probabilities', p, formulas=probability)
        generation = t.get('generation')
        if generation:
            self.width = max(self.width,self.right+5)
            self.stage('GENERATION • Sampling settings')
            output_edge = max(self.right + self.token_count - 1, 10)
            self.width = max(self.width, output_edge + 3)
            settings = self.matrix('Temperature · Top-k · Seed',[[generation['temperature']],[generation['top_k']],[generation.get('seed',0)]],column=output_edge)
            self.generation_controls = {'temperature':settings.ref(0,0),'top_k':settings.ref(1,0),'seed':settings.ref(2,0)}
            self.stage('GENERATION • Seeded top-k sampling')
            final_column = self.token_count - 1
            temperature = float(generation['temperature'])
            count = min(int(generation['top_k']) or len(p), len(p))
            vector = logits.values[:, final_column]
            threshold = np.sort(vector)[-count]
            probabilities = np.zeros(len(p))
            if temperature == 0:
                probabilities[np.argmax(vector)] = 1
            else:
                values = np.where(vector >= threshold, np.exp((vector-vector.max())/temperature), 0)
                probabilities = values/values.sum()
            full_range=f'{logits.ref(0,final_column)}:{logits.ref(len(p)-1,final_column)}'
            def sample_probability(i,j):
                logit=logits.ref(i,final_column)
                maximum=f'MAX({full_range})'
                temp = settings.ref(0,0)
                top_k = settings.ref(1,0)
                cutoff=f'LARGE({full_range},IF({top_k}=0,{len(p)},MIN({len(p)},MAX(1,{top_k}))))'
                exp=lambda a: f'EXP(({logits.ref(a,final_column)}-{maximum})/{temp})'
                denominator='+'.join(f'IF({logits.ref(a,final_column)}>={cutoff},{exp(a)},0)' for a in range(len(p)))
                return f'=IF({temp}=0,IF(MATCH({maximum},{full_range},0)={i+1},1,0),IF({logit}>={cutoff},{exp(i)}/({denominator}),0))'
            start = self.row
            cutoff=f'LARGE({full_range},IF({settings.ref(1,0)}=0,{len(p)},MIN({len(p)},MAX(1,{settings.ref(1,0)}))))'
            eligible = vector >= threshold
            self.matrix('Top-k eligibility',eligible.astype(float),column=output_edge-8,
                formulas=lambda i,j:f'=IF({logits.ref(i,final_column)}>={cutoff},1,0)')
            self.row = start
            self.matrix('Top-k candidates',np.where(eligible,vector,0),column=output_edge-6,
                formulas=lambda i,j:f'=IF({logits.ref(i,final_column)}>={cutoff},{logits.ref(i,final_column)},"—")')
            self.row = start
            sampling=self.matrix('Top-k probabilities',probabilities,column=output_edge-4,formulas=sample_probability)
            end = self.row
            self.row = start
            noise=self.matrix('Sampling noise · saved seeded draw',generation['sampling_noise'],column=output_edge-2,weight=True,
                formulas=lambda i,j:f'=IF({settings.ref(2,0)}={generation.get("seed",0)},{float(generation["sampling_noise"][i])},-LN(MAX(0.0000001,MIN(0.9999999,MOD(ABS(SIN({settings.ref(2,0)}*12.9898+{i+1}*78.233)*43758.5453),1)))))')
            end = max(end,self.row)
            self.row = start
            scores=self.matrix('Sampling score',probabilities/np.asarray(generation['sampling_noise']),column=output_edge,
                formulas=lambda i,j:f'={sampling.ref(i,0)}/{noise.ref(i,0)}')
            self.row = max(end,self.row)
            score_range=f'{scores.ref(0,0)}:{scores.ref(len(p)-1,0)}'
            self.stage('GENERATION • Generated token')
            self.matrix('Selected vocabulary index · 1-based',[[int(np.argmax(scores.values))+1]],column=output_edge,
                formulas=lambda i,j:f'=MATCH(MAX({score_range}),{score_range},0)')
        self.final_hidden = final.values.T
        self.logits = logits.values.T
        self.probabilities = p.T
        for row, _ in self.stages:
            old = self.cells[row, 1]
            self.put(row, 1, old[0].replace('/00]', f'/{len(self.stages):02d}]'))
        self.put(1, self.width+3, 'Navigate to block')
        return self

    def document(self):
        def cell_html(value, formula=None, weight=False):
            if isinstance(value, (int,float)):
                number = float(value)
                intensity = min(1, abs(number) / 2)
                neutral = (255,246,185) if weight else (244,247,251)
                target = (246,172,55) if weight else ((238,104,97) if number >= 0 else (93,143,205))
                color = '#'+''.join(f'{round(a+(b-a)*intensity):02x}' for a,b in zip(neutral,target))
                data = escape(json.dumps({'1':3,'3':number}),quote=True)
                attrs = f' sdval="{number}" bgcolor="{color}" data-sheets-value="{data}"'
                if formula:
                    attrs += f' data-sheets-formula="{escape(formula,quote=True)}"'
                return f'<td align="center"{attrs}><font>{number:.3f}</font></td>'
            data = escape(json.dumps({'1':2,'2':str(value)}),quote=True)
            return f'<td data-sheets-value="{data}"><font>{escape(str(value))}</font></td>'
        rows=[]
        occupied_rows = {}
        for (row, column), value in self.cells.items():
            occupied_rows.setdefault(row, []).append((column, value))
        for row in range(1,self.row):
            entries = sorted(occupied_rows.get(row, []))
            rows.append('<tr>'+''.join(
                cell_html(*value).replace('<td', f'<td data-grid-column="{column}"', 1)
                for column, value in entries)+'</tr>')
        table='<table><colgroup span="'+str(self.width+3)+'" width="44"></colgroup>'+''.join(rows)+'</table>'
        embedding_rows=['<tr><td></td></tr>' for _ in range(5)]
        for token, values in zip(self.trace['vocabulary'],self.trace['global_weights']['token_embedding']):
            embedding_rows.append('<tr>'+cell_html(token_label(token))+''.join(cell_html(v,None,True) for v in values)+'</tr>')
        return '<html><head></head><body><A NAME="table0"><h1>Sheet 1: <em>Full Forward Pass</em></h1></A>'+table+'<A NAME="table1"><h1>Sheet 2: <em>Embeddings</em></h1></A><table>'+''.join(embedding_rows)+'</table></body></html>'


def export_trace_html(trace_path: Path, output: Path, show: str | None = None) -> Path:
    if show not in (None,"forward","training","kvcache"):
        raise ValueError("Unknown view: " + str(show))
    if show == "training" and not Path(trace_path).with_name("training-record.json").is_file():
        raise ValueError("Training-only export requires training-record.json; run python latte.py first")
    import re
    trace = json.loads(Path(trace_path).read_text(encoding='utf-8'))
    import tomllib
    config_path = Path(trace_path).resolve().parent / 'config.toml'
    if config_path.is_file():
        config = tomllib.loads(config_path.read_text(encoding='utf-8'))
        corpus_path = Path(config.get('data', {}).get('path', ''))
        if not corpus_path.is_absolute():
            corpus_path = Path(__file__).resolve().parent / corpus_path
        if corpus_path.is_file():
            trace['training_corpus'] = {'path': str(corpus_path), 'text': corpus_path.read_text(encoding='utf-8')}
    grid = TraceGrid(trace).build()
    if show in ('training','kvcache'):
        # Keep only the input/embedding scaffold required by shared controls.
        # Forward attention/FFN/logit matrices are not included in this document.
        boundary=grid.stages[1][0]
        controls={}
        for i,(name,address) in enumerate(getattr(grid,'generation_controls',{}).items()):
            match=re.fullmatch(r'R(\d+)C(\d+)',address)
            controls[name]=(boundary+2+i,2,grid.cells[int(match[1]),int(match[2])][0])
        selected=next(((r,c) for (r,c),(value,*_) in grid.cells.items() if value=='Selected vocabulary index · 1-based'),None)
        selected_value=grid.cells[selected[0]+1,selected[1]][0] if selected else 1
        grid.cells={key:value for key,value in grid.cells.items() if key[0]<boundary}
        grid.stages=grid.stages[:1]
        grid.put(boundary,2,'Selected vocabulary index · 1-based')
        grid.put(boundary+1,2,selected_value)
        for row,col,value in controls.values():grid.put(row,col,value)
        grid.generation_controls={name:f'R{row}C{col}' for name,(row,col,_) in controls.items()}
        grid.row=boundary+len(controls)+4
    html = build_workbook_html(grid.document(), 'One Small Latte')
    metadata = json.dumps(getattr(grid,'generation_controls',{})).replace('<','\\u003c')
    html = html.replace('</head>',f'<script id="generation-controls" type="application/json">{metadata}</script></head>',1)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    # The intermediate HTML stays in memory; preserve all existing exports.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temporary:
        source = Path(temporary)/'trace-base.html'
        source.write_text(html,encoding='utf-8')
        create_annotated_copy(source,output)
    # Keep a clean shell for rebuilding the same explorer as the sequence grows.
    runtime = Path(__file__).with_name('sequence_generation.js').read_text(encoding='utf-8')
    kv_runtime = Path(__file__).with_name('kv_cache.js').read_text(encoding='utf-8') if show in (None,'kvcache') else ''
    html = output.read_text(encoding='utf-8')
    shell = re.sub(r'<section class="worksheet active".*?</section>', '<section class="worksheet active" id="sheet-0" aria-label="Full Forward Pass" data-column-offset="0" data-row-offset="0">__FORWARD_TABLE__</section>', html, count=1, flags=re.S)
    shell = re.sub(r'<script id="generation-controls".*?</script>', '__GENERATION_CONTROLS__', shell, count=1, flags=re.S)
    # document.open preserves the Window's lexical bindings. Scope scripts so
    # successive generations can initialize the explorer without redeclarations.
    shell = re.sub(r'<script>(.*?)</script>', lambda m: '<script>(() => {\n' + m[1] + '\n})();</script>', shell, flags=re.S)
    shell = shell.replace('</body>', '__SEQUENCE_SCRIPT__</body>', 1)
    training_path = Path(trace_path).with_name('training-record.json')
    if training_path.is_file() and show in (None,'training'):
        training = with_corpus_usage(json.loads(training_path.read_text(encoding='utf-8')), training_path.parent)
        if training['architecture'] != trace['architecture'] or training['vocabulary'] != trace['vocabulary']:
            raise ValueError('Training recording architecture and vocabulary must match the forward trace')
        training_runtime = Path(__file__).with_name('training_replay.js').read_text(encoding='utf-8')
        training_metadata = json.dumps(training).replace('<', '\\u003c')
        html = html.replace('</body>', '<script id="training-record-data" type="application/json">' + training_metadata + '</script></body>', 1)
    else:
        training_runtime = ''
    metadata = json.dumps({'show': show, 'trace': trace, 'shell': shell, 'runtime': runtime, 'kvRuntime': kv_runtime, 'trainingRecord': training if training_runtime else None, 'trainingRuntime': training_runtime}).replace('<', '\\u003c')
    safe_runtime = runtime.replace('</script', '<\\/script')
    html = html.replace('</body>', f'<script id="sequence-generation-data" type="application/json">{metadata}</script>\n<script>{safe_runtime}</script>\n</body>', 1)
    if training_runtime:
        html = html.replace('</body>', '<script>' + training_runtime.replace('</script', '<\\/script') + '</script></body>', 1)
    html = html.replace('</body>', '<script>' + kv_runtime.replace('</script', '<\\/script') + '</script></body>', 1)
    output.write_text(html, encoding='utf-8')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--show',choices=('training',),default='training')
    args=parser.parse_args()
    output=args.output or args.trace.with_name(args.trace.name.removesuffix('.trace.json')+'-json.html')
    print(export_trace_html(args.trace,output,show=args.show))


if __name__=='__main__':
    main()
