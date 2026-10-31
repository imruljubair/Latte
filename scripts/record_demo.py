"""Capture the public, deterministic training example used by GitHub Pages."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    output = ROOT / 'out/pages-demo'
    subprocess.run([sys.executable, str(ROOT/'latte.py'), '--config', str(ROOT/'demo/config.toml'),
                    '--record', '1', '6', '40', '--steps', '201', '--output-dir', str(output)], cwd=ROOT, check=True)
    destination = ROOT/'demo/data'
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ('training-record.json', 'training.trace.json', 'training-progress.json'):
        payload = (output/name).read_bytes()
        filename = name+'.gz'
        (destination/filename).write_bytes(gzip.compress(payload, mtime=0))
        files[filename] = hashlib.sha256(payload).hexdigest()
    manifest = {'schema':'latte-recorded-demo-v1', 'name':'One Small Latte · recorded training',
                'source':'Real CPU PyTorch training, AdamW, seed 1337', 'steps':201, 'recordedSteps':[1,41,81,121,161,201],
                'files':files}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved public demo recording in demo/data')

if __name__ == '__main__':
    main()
