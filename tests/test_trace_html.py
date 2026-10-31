import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from export_trace_html import EPSILON, TraceGrid, export_trace_html


def fixture():
    return {
        'architecture': {'model_dim':2,'qk_dim':1,'value_dim':1,'num_heads':1,
                         'num_layers':1,'ffn_dim':2,'context_length':4},
        'input': {'tokens':['a','b'],'token_ids':[0,1]}, 'vocabulary':['a','b'],
        'embeddings': {'selected_token_embeddings':[[1,2],[2,1]],
                       'selected_position_embeddings':[[0,0],[0,0]]},
        'global_weights': {'token_embedding':[[1,2],[2,1]],'position_embedding':[[0,0]]*4,
                           'final_norm':[1,1],'lm_head':[[1,0],[0,1]]},
        'layers':[{'layer':0,'weights':{
            'attention_norm':[1,1],'ffn_norm':[1,1],
            'wk_by_head':[[[0,0]]],'wq_by_head':[[[0,0]]],'wv_by_head':[[[0,0]]],
            'wo':[[0],[0]],'ffn_in':[[0,0],[0,0]],'ffn_out':[[0,0],[0,0]]}}],
    }


class TraceHTMLTests(unittest.TestCase):
    def test_sampling_vectors_share_rows_and_output_has_own_stage(self):
        trace = fixture()
        trace['generation'] = {'temperature':1,'top_k':2,'sampling_noise':[1,2]}
        grid = TraceGrid(trace).build()
        labels = {name:(row,column) for (row,column),(name,_,_) in grid.cells.items() if isinstance(name,str)}
        probability = labels['Top-k probabilities']
        noise = labels['Sampling noise · saved seeded draw']
        score = labels['Sampling score']
        edge = max(grid.right + grid.token_count - 1, 10)
        self.assertEqual(probability,(score[0],edge-4))
        self.assertEqual(noise,(score[0],edge-2))
        self.assertEqual(score[1],edge)
        self.assertEqual(labels['Temperature · Top-k · Seed'][1],edge)
        self.assertEqual(labels['Selected vocabulary index · 1-based'][1],edge)
        self.assertEqual(set(grid.generation_controls),{'temperature','top_k','seed'})
        self.assertEqual(grid.stages[-1][1],'GENERATION • Generated token')
        self.assertGreater(labels['Selected vocabulary index · 1-based'][0],score[0]+2)

    def test_aggregation_has_live_v_on_left_aligned_with_h(self):
        grid = TraceGrid(fixture()).build()
        stage_row = next(row for row,title in grid.stages if 'weighted value aggregation' in title)
        label_row = stage_row + 1
        left = grid.right - 1 - 2
        self.assertEqual(grid.cells[label_row,left][0], 'V · layer 0 · head 0')
        self.assertEqual(grid.cells[label_row,grid.right][0], 'H · layer 0 · head 0')
        self.assertTrue(grid.cells[label_row+1,left][1].startswith('=R'))
        self.assertIn(f'R{label_row+1}C{left}*',grid.cells[label_row+1,grid.right][1])

    def test_weights_are_right_aligned_with_one_column_gap(self):
        grid=TraceGrid(fixture()).build()
        for (_, column), (name, _, _) in grid.cells.items():
            if isinstance(name,str) and name.startswith(('WK ·','WQ ·','WV ·','W1 ·','W2 ·','Kᵀ ·')):
                width=1 if name.startswith('Kᵀ ·') else 2
                self.assertEqual(column+width,grid.right-1,name)

    def test_zero_projections_preserve_residual_and_causal_attention(self):
        grid=TraceGrid(fixture()).build()
        expected=np.array([[1,2],[2,1]]) / np.sqrt(2.5+EPSILON)
        np.testing.assert_allclose(grid.final_hidden,expected)
        np.testing.assert_allclose(grid.logits,expected)
        np.testing.assert_allclose(grid.probabilities.sum(axis=1),[1,1])
        lookup=[formula for _,formula,_ in grid.cells.values() if formula and 'EXACT' in formula]
        self.assertEqual(len(lookup),4)
        self.assertIn('EXACT(R5C,Embeddings!R6C1)',lookup[0])

    def test_single_view_exports_omit_unrelated_runtimes(self):
        import re
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'forward.trace.json'
            path.write_text(json.dumps(fixture()))
            for show in ('forward','kvcache'):
                html=export_trace_html(path,Path(temporary)/(show+'.html'),show=show).read_text()
                data=json.loads(re.search(r'<script id="sequence-generation-data" type="application/json">(.*?)</script>',html,re.S)[1])
                self.assertEqual(data['show'],show)
                self.assertEqual(data['trainingRuntime'],'')
                self.assertIsNone(data['trainingRecord'])
                self.assertEqual(bool(data['kvRuntime']),show=='kvcache')
                if show=='kvcache':
                    document=html.split('<script id="sequence-generation-data"')[0]
                    document=re.sub(r'<script\b[^>]*>.*?</script>', '', document, flags=re.S)
                    self.assertNotIn('FFN pre-activation',document)
                    self.assertNotIn('ATTENTION • Key projection',document)
            with self.assertRaisesRegex(ValueError,'training-record.json'):
                export_trace_html(path,Path(temporary)/'training.html',show='training')

    def test_export_without_excel_preserves_trace_and_existing_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            trace=directory/'forward.trace.json'
            content=json.dumps(fixture())
            trace.write_text(content)
            old=directory/'forward.html'
            old.write_text('preserve this version')
            output=export_trace_html(trace,directory/'forward-json.html')
            self.assertIn('renderWorkedCalculation',output.read_text())
            self.assertEqual(old.read_text(),'preserve this version')
            self.assertEqual(trace.read_text(),content)
            self.assertFalse(list(directory.glob('*.xlsx')))

    def test_visualization_controls_use_compact_screen_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace = directory / 'forward.trace.json'
            trace.write_text(json.dumps(fixture()))

            html = export_trace_html(trace, directory / 'forward-json.html').read_text()

            self.assertIn('className = "explorer-control-rail"', html)
            self.assertIn('controlRail.appendChild(sequenceEditor)', html)
            self.assertIn('controlRail.appendChild(toolbar)', html)
            self.assertIn('left: 18px', html)
            self.assertIn('right: 18px; bottom: 52px', html)
            self.assertIn('event.key === "ArrowLeft"', html)
