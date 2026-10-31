from pathlib import Path
import tempfile
import time
import unittest
import shutil
import subprocess
import sys
from html.parser import HTMLParser

from fastapi.testclient import TestClient

from server import ROOT, create_app


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / 'Latte'
        shutil.copytree(ROOT, cls.root, ignore=shutil.ignore_patterns(
            '.git', '.venv', 'node_modules', '__pycache__', 'out', 'releases', '.artifact-work'))
        subprocess.run([sys.executable, str(cls.root / 'latte.py'), '--record', '1', '2', '1'],
                       cwd=cls.root, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.client = TestClient(create_app(self.root))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_old_truncated_loss_history_recovers_earlier_recorded_points(self):
        import json
        path = self.root / 'out' / 'legacy-history'
        path.mkdir(exist_ok=True)
        source = self.root / 'out' / 'default' / 'training-record.json'
        shutil.copyfile(source, path / 'training-record.json')
        history = [{'step': i, 'loss': 2.0 + (i % 2) * .4} for i in range(4018, 5018)]
        (path / 'training-progress.json').write_text(json.dumps({'step': 5017, 'history': history}))
        try:
            result = self.client.get('/api/runs/legacy-history/status').json()['progress']
            self.assertEqual(result['history'][0]['step'], 1)
            self.assertEqual(result['history'][-1]['step'], 5017)
            self.assertLessEqual(len(result['history']), 300)
            self.assertTrue(result['history_sparse'])
            self.assertTrue(result['history_averaged'])
            self.assertEqual(json.loads((path / 'training-progress.json').read_text())['history'], history)
        finally:
            shutil.rmtree(path)

    def test_index_preserves_shell_and_uses_react_without_embedded_recording(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="latte-root"', response.text)
        self.assertIn('data-latte-loading', response.text)
        self.assertIn('id="latte-loading"', response.text)
        self.assertLess(response.text.index('html[data-latte-loading]'), response.text.index('<script'))
        self.assertIn('type="module" src="/ui/', response.text)
        class ScriptTags(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                if tag == 'script':
                    self.ids.append(dict(attrs).get('id'))
        tags = ScriptTags()
        tags.feed(response.text)
        self.assertNotIn('training-record-data', tags.ids)
        self.assertNotIn('"trainingRecord":', response.text)
        self.assertNotIn('Replay recorded PyTorch training', response.text)
        original = (self.root / 'out/default/latte.html').read_text()
        self.assertLess(len(response.content), len(original.encode()))
        self.assertIn('annotation-overlay', response.text)

    def test_metadata_omits_tensors_and_steps_are_loaded_individually(self):
        metadata = self.client.get('/api/runs/default/metadata').json()
        self.assertTrue(metadata['steps'])
        self.assertEqual(set(metadata['steps'][0]), {'step', 'loss'})
        step = self.client.get('/api/runs/default/steps/0').json()
        self.assertEqual(step['step'], metadata['steps'][0]['step'])
        self.assertIn('gradients', step)
        self.assertIn('before', step)
        self.assertEqual(self.client.get('/api/runs/default/steps/-1').status_code, 404)
        self.assertEqual(self.client.get('/api/runs/default/steps/999999').status_code, 404)
        self.assertEqual(self.client.get('/api/runs/unknown/metadata').status_code, 404)

    def test_export_is_original_standalone_file(self):
        response = self.client.get('/api/runs/default/export')
        self.assertEqual(response.content, (self.root / 'out/default/latte.html').read_bytes())
        self.assertIn('attachment', response.headers['content-disposition'])

    def test_rejects_external_mutations_invalid_config_and_traversal(self):
        self.assertEqual(self.client.post('/api/runs', json={}, headers={'Origin': 'https://elsewhere.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/runs', json={'config': '../outside.toml'}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs', json={'count': 0}).status_code, 422)
        self.assertEqual(self.client.get('/api/runs/..invalid/metadata').status_code, 400)
        self.assertEqual(self.client.get('/api/runs', headers={'Host': 'elsewhere.example'}).status_code, 400)

    def test_delete_run_removes_listing_and_preserves_recoverable_files(self):
        path = self.root / 'out' / 'run-delete-test'
        path.mkdir(exist_ok=True)
        (path / 'sentinel.txt').write_text('keep this checkpoint')
        self.assertEqual(self.client.delete('/api/runs/default').status_code, 400)
        self.assertEqual(self.client.delete('/api/runs/run-delete-test', headers={'Origin':'https://elsewhere.example'}).status_code, 403)
        self.assertEqual(self.client.delete('/api/runs/run-delete-test').status_code, 200)
        self.assertFalse(path.exists())
        self.assertNotIn('run-delete-test', [item['id'] for item in self.client.get('/api/runs').json()])
        self.assertEqual(self.client.get('/api/runs/run-delete-test/status').status_code, 404)
        saved = list((self.root / 'out' / '.trash').glob('run-delete-test-*'))
        self.assertEqual(len(saved), 1)
        self.assertEqual((saved[0] / 'sentinel.txt').read_text(), 'keep this checkpoint')

    def test_browser_check_can_be_deleted(self):
        path = self.root / 'out' / 'browser-check'
        path.mkdir(exist_ok=True)
        (path / 'sentinel.txt').write_text('test fixture')
        self.assertEqual(self.client.delete('/api/runs/browser-check').status_code, 200)
        self.assertFalse(path.exists())
        self.assertNotIn('browser-check', [r['id'] for r in self.client.get('/api/runs').json()])
        self.assertTrue(list((self.root / 'out' / '.trash').glob('browser-check-*')))

    def test_run_corpus_uses_recorded_text_or_unfinished_run_copy(self):
        metadata = self.client.get('/api/runs/default/metadata').json()
        self.assertEqual(self.client.get('/api/runs/default/corpus').json()['text'], metadata['corpus']['text'])
        path = self.root / 'out' / 'run-corpus-test'
        path.mkdir(exist_ok=True)
        try:
            text = 'A stopped run still has its own corpus. 🦙'
            (path / 'corpus.txt').write_text(text, encoding='utf-8')
            self.assertEqual(self.client.get('/api/runs/run-corpus-test/corpus').json(), {'text': text})
            self.assertEqual(self.client.get('/api/runs/missing/corpus').status_code, 404)
        finally:
            shutil.rmtree(path)

    def test_old_run_settings_come_from_recording_and_saved_config(self):
        data = self.client.get('/api/runs/default/settings').json()
        metadata = self.client.get('/api/runs/default/metadata').json()
        self.assertEqual(data['model'], {key: metadata['architecture'][key] for key in data['model']})
        self.assertEqual(data['training']['steps'], metadata['steps'][-1]['step'])
        self.assertEqual(data['training']['record_every'], 1)
        self.assertEqual(data['training']['train_split'], .9)
        self.assertEqual(self.client.get('/api/runs/unknown/settings').status_code, 404)

    def test_can_stop_a_real_training_process(self):
        response = self.client.post('/api/runs', json={'count': 10000})
        self.assertEqual(response.status_code, 202)
        run_id = response.json()['id']
        self.assertEqual(self.client.post('/api/runs', json={'count': 2}).status_code, 409)
        self.assertEqual(self.client.delete(f'/api/runs/{run_id}').status_code, 409)
        stopped = self.client.post(f'/api/runs/{run_id}/stop')
        self.assertEqual(stopped.status_code, 200)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = self.client.get(f'/api/runs/{run_id}/status').json()['state']
            if state == 'stopped':
                break
            time.sleep(.05)
        self.assertEqual(state, 'stopped')

    def test_real_training_job_completes_without_blocking_api(self):
        response = self.client.post('/api/runs', json={'count': 2})
        self.assertEqual(response.status_code, 202)
        run_id = response.json()['id']
        output = self.root / 'out' / run_id
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = self.client.get(f'/api/runs/{run_id}/status').json()
                if status['state'] != 'running':
                    break
                time.sleep(.1)
            self.assertEqual(status['state'], 'complete', status)
            metadata = self.client.get(f'/api/runs/{run_id}/metadata').json()
            self.assertEqual(len(metadata['steps']), 2)
            self.assertEqual(self.client.get('/', params={'run': run_id}).status_code, 200)
        finally:
            import shutil
            shutil.rmtree(output)

    def test_custom_corpus_parameters_and_live_progress(self):
        import json
        text = 'A small llama learns from a little story. The sun is warm.\n' * 12
        response = self.client.post('/api/runs', json={
            'name': 'Custom corpus test', 'corpus': text,
            'model': {'model_dim': 3, 'qk_dim': 2, 'value_dim': 3, 'num_heads': 2,
                      'num_layers': 1, 'ffn_dim': 5, 'context_length': 4},
            'training': {'steps': 8, 'batch_size': 2, 'learning_rate': .005,
                         'record_every': 3, 'seed': 42, 'train_split': .8},
        })
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        selected = self.client.get(f'/api/runs/{run_id}/settings').json()
        self.assertEqual(selected['model'], {'model_dim':3,'qk_dim':2,'value_dim':3,'num_heads':2,'num_layers':1,'ffn_dim':5,'context_length':4})
        self.assertEqual(selected['training'], {'steps':8,'batch_size':2,'learning_rate':.005,'weight_decay':0,'seed':42,'train_split':.8,'record_every':3})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = self.client.get(f'/api/runs/{run_id}/status').json()
            if status['state'] != 'running':
                break
            time.sleep(.1)
        self.assertEqual(status['state'], 'complete', status)
        self.assertTrue(status['explorer_ready'])
        self.assertEqual(status['name'], 'Custom corpus test')
        self.assertEqual(status['progress']['phase'], 'complete')
        self.assertEqual(status['progress']['step'], 8)
        self.assertEqual(status['progress']['total_steps'], 8)
        self.assertEqual(status['progress']['recorded'], 4)
        self.assertEqual([point['step'] for point in status['progress']['history']], list(range(1, 9)))
        document = json.loads((self.root / 'out' / run_id / 'training-record.json').read_text())
        self.assertEqual(document['corpus']['text'], text)
        self.assertEqual(document['architecture']['qk_dim'], 2)
        self.assertEqual(document['optimizer']['learning_rate'], .005)
        self.assertEqual([step['step'] for step in document['steps']], [1, 4, 7, 8])
        self.assertTrue(all(len(step['inputs']) == 2 for step in document['steps']))
        self.assertEqual(self.client.get('/', params={'run': run_id}).status_code, 200)

    def test_custom_training_validation(self):
        self.assertEqual(self.client.post('/api/runs', json={'corpus': 'tiny'}).status_code, 400)
        self.assertEqual(self.client.post('/api/runs', json={'model': {'num_layers': 2}}).status_code, 400)
        for training in [{'steps': 0}, {'record_every': 0}, {'train_split': 1}, {'learning_rate': -1}]:
            self.assertEqual(self.client.post('/api/runs', json={
                'corpus': 'some training text ' * 40, 'training': training}).status_code, 422)

    def test_missing_build_and_recording_return_actionable_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(Path(directory))) as client:
                self.assertEqual(client.get('/').status_code, 503)
                self.assertEqual(client.get('/api/runs/missing/metadata').status_code, 404)
