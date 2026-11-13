"""Validate the published package without a server or PyTorch dependency."""
from html.parser import HTMLParser
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from scripts.build_pages import ROOT, build

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets=[]; self.ids=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name == 'id': self.ids.append(value)
            if name in ('href','src') and value:
                self.targets.append(value)

class PagesPackage(unittest.TestCase):
    def test_project_relative_pages_and_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            output=build(Path(folder)/'Latte')
            for name in ('index.html','training.html','run-locally.html'):
                parser=Links();html=(output/name).read_text();parser.feed(html)
                for target in parser.targets:
                    if target.startswith(('https:','http:','data:','#')):
                        continue
                    self.assertFalse(target.startswith('/'),target)
                    self.assertTrue((output/target.split('?')[0]).exists(),target)
            html=(output/'index.html').read_text()
            self.assertIn('data-latte-demo',html)
            self.assertIn('latte-root',html)
            parser=Links();parser.feed(html)
            self.assertNotIn('training-record-data',parser.ids)
            self.assertIn('demo.js',html)
            self.assertTrue((output/'.nojekyll').is_file())
            data=json.loads((output/'data/demo.json').read_text())
            self.assertEqual(data['settings']['training']['steps'],201)
            self.assertEqual(data['settings']['model']['num_layers'],2)
            self.assertEqual(len(data['steps']),6)
            self.assertTrue(data['run']['progress']['validation_history'])
            catalog = json.loads((output/'data/runs.json').read_text())
            self.assertEqual([entry['id'] for entry in catalog], ['demo', 'single-head', 'longer-context'])
            for entry in catalog:
                example = json.loads((output/'data'/entry['file']).read_text())
                self.assertEqual(example['run']['id'], entry['id'])
                self.assertEqual(example['run']['recorded_steps'], [step['step'] for step in example['steps']])
                self.assertEqual(example['settings']['model']['context_length'], len(example['steps'][0]['inputs'][0]))
            self.assertIn('?tab=training',(output/'training.html').read_text())
            self.assertFalse(list(output.rglob('*.pt')))
            self.assertFalse((output/'api').exists())

    def test_does_not_delete_unrelated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'keep').write_text('keep')
            with self.assertRaises(ValueError):build(path)
            self.assertEqual((path/'keep').read_text(),'keep')

    def test_builder_does_not_import_torch(self):
        command='import scripts.build_pages, sys; assert "torch" not in sys.modules'
        subprocess.run([sys.executable,'-c',command],cwd=ROOT,check=True)

if __name__=='__main__':unittest.main()
