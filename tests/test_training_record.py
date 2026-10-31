import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import torch
import torch.nn.functional as F
from latte.config import load_config
from latte.model import TinyTransformer
from record_training import record
from tests.test_trace_html import fixture


class TrainingRecordTests(unittest.TestCase):
    def test_recorded_forward_gradients_and_adamw_update_are_real(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json'
            document=record(Path('configs/default.toml'),path,2)
            config=load_config('configs/default.toml')
            model=TinyTransformer(config.model,len(document['vocabulary']))
            step=document['steps'][0]
            model.load_state_dict({k:torch.tensor(v) for k,v in step['before'].items()})
            optimizer=torch.optim.AdamW(model.parameters(),lr=config.training.learning_rate,weight_decay=config.training.weight_decay)
            inputs=torch.tensor(step['inputs']);targets=torch.tensor(step['targets'])
            logits,_=model(inputs)
            torch.testing.assert_close(logits,torch.tensor(step['logits']))
            losses=F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').reshape_as(targets)
            torch.testing.assert_close(losses,torch.tensor(step['token_losses']))
            losses.mean().backward()
            for name,p in model.named_parameters():torch.testing.assert_close(p.grad,torch.tensor(step['gradients'][name]))
            optimizer.step()
            for name,p in model.named_parameters():torch.testing.assert_close(p,torch.tensor(step['after'][name]))
            self.assertEqual(document['steps'][1]['before'],step['after'])
            for start,ids in zip(step['starts'],step['inputs']):
                self.assertEqual(''.join(document['vocabulary'][i] for i in ids),document['corpus']['text'][start:start+len(ids)])
            self.assertEqual(json.loads(path.read_text()),document)

    def test_validation_loss_uses_held_out_text_and_final_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            document = record(Path('configs/default.toml'), path, 3)
            progress = json.loads((path.parent / 'training-progress.json').read_text())
            self.assertEqual([p['step'] for p in progress['validation_history']], [1, 2, 3])
            config = load_config('configs/default.toml')
            model = TinyTransformer(config.model, len(document['vocabulary']))
            model.load_state_dict({k: torch.tensor(v) for k, v in document['steps'][-1]['after'].items()})
            model.eval()
            lookup = {char: i for i, char in enumerate(document['vocabulary'])}
            held_out = torch.tensor([lookup[c] for c in document['corpus']['text'][document['corpus']['train_end']:]])
            length = config.model.context_length
            starts = torch.linspace(0, len(held_out)-length-1, min(16, len(held_out)-length)).long().unique()
            inputs = torch.stack([held_out[i:i+length] for i in starts])
            targets = torch.stack([held_out[i+1:i+length+1] for i in starts])
            with torch.no_grad():
                logits, _ = model(inputs)
                expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()
            self.assertAlmostEqual(progress['validation_loss'], expected, places=6)
            self.assertEqual(progress['validation_history'][-1]['loss'], progress['validation_loss'])

    def test_training_grid_has_targets_loss_and_no_sampling(self):
        trace=fixture();trace['training']={'targets':['b','a'],'losses':[.4,.6]}
        script="const {SequenceGrid}=require('./sequence_generation.js');const t=JSON.parse(require('fs').readFileSync(0,'utf8'));const g=new SequenceGrid(t,['a','b'],{}).build();console.log(JSON.stringify({stages:g.stages,table:g.table(),cells:[...g.cells.values()]}));"
        result=subprocess.run(['node','-e',script],input=json.dumps(trace),text=True,capture_output=True,check=True)
        grid=json.loads(result.stdout)
        self.assertIn('Correct next character',grid['table'])
        self.assertIn('Predicted next character',grid['table'])
        headers=[cell for cell in grid['cells'] if cell.get('tokenHeader')]
        self.assertEqual([cell['v'] for cell in headers],['a','b'])
        self.assertTrue(all(cell['r']==5 for cell in headers))
        names={cell['v']:cell for cell in grid['cells'] if isinstance(cell['v'],str)}
        self.assertEqual(names['Vocabulary weights']['r'],names['Logits']['r'])
        self.assertIn('data-training-character',grid['table'])
        self.assertIn('MATCH(MAX(',grid['table'])
        self.assertTrue(any(title=='OUTPUT • Vocabulary softmax' for _,title in grid['stages']))
        self.assertIn('Recorded loss before update',grid['table'])
        self.assertFalse(any('GENERATION' in title for _,title in grid['stages']))
        self.assertNotIn('Embeddings!',grid['table'])

    def test_step_update_retains_cells_and_only_reparses_changed_formulas(self):
        runtime=Path('html_runtime.js').read_text()
        method=runtime.split('    updateTrainingGrid(gridCells) {',1)[1].split('    commitSequenceInputs()',1)[0]
        script='''let parses=0,recalculations=0,writes=0,deferTrainingPaint=false;
const document={documentElement:{hasAttribute:()=>true}};
const key=(s,r,c)=>`${s}:${r}:${c}`;
class FormulaParser {constructor(f){this.f=f;}parse(){parses++;return this.f;}}
const writeCell=(c,v)=>{c.display=v;writes++;};const recalculate=()=>{recalculations++;};
const element={setAttribute(){}};
const weight={baseValue:1,formula:null,element};
const activation={baseValue:0,formula:'=R1C1*2',ast:'existing parsed formula',element};
const token={baseValue:0,formula:'=0.1',element};
const cells=new Map([[key('Full Forward Pass',1,1),weight],[key('Full Forward Pass',2,1),activation],[key('Full Forward Pass',3,1),token]]);
const api={updateTrainingGrid(gridCells){'''+method+'''};
api.updateTrainingGrid([{r:1,c:1,v:2,f:null},{r:2,c:1,v:0,f:'=R1C1*2'},{r:3,c:1,v:0,f:'=0.2'}]);
api.updateTrainingGrid([{r:1,c:1,v:3,f:null},{r:2,c:1,v:0,f:'=R1C1*2'},{r:3,c:1,v:0,f:'=0.2'}]);
console.log(JSON.stringify({parses,recalculations,weight:weight.baseValue,ast:activation.ast,retained:cells.get(key('Full Forward Pass',1,1))===weight}));'''
        result=subprocess.run(['node','-e',script],text=True,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout),dict(parses=1,recalculations=2,weight=3,ast='existing parsed formula',retained=True))

    def test_sparse_recording_matches_contiguous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            full=record(Path('configs/default.toml'),path/'full.json',6)
            sparse=record(Path('configs/default.toml'),path/'sparse.json',3,2,2)
            self.assertEqual([step['step'] for step in sparse['steps']],[2,4,6])
            self.assertEqual(sparse['steps'],[full['steps'][i] for i in [1,3,5]])
            for values in [(0,1,1),(1,0,1),(1,1,0)]:
                start,count,stride=values
                with self.assertRaises(ValueError):record(Path('configs/default.toml'),path/'bad.json',count,start,stride)
