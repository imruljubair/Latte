"""Validate the published package without a server or PyTorch dependency."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.build_pages import ROOT, build

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets=[]
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
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
                    self.assertTrue((output/target).exists(),target)
            html=(output/'index.html').read_text()
            self.assertIn('training-record-data',html)
            self.assertIn('demo.js',html)
            self.assertTrue((output/'.nojekyll').is_file())
            self.assertIn('stroke-dasharray="5 4"',(output/'training.html').read_text())
            self.assertIn('201 / 201',(output/'training.html').read_text())
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
