"""Build the static Pages edition from a verified real training recording.

Only NumPy is required; publishing never trains or installs PyTorch.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
from html import escape
import json
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_trace_html import export_trace_html
from scripts.demo_presets import load_presets, preset_config, recording_dir


def navigation(active):
    return '<nav class="demo-nav" aria-label="Demo navigation">'+''.join(
        f'<a href="{url}"'+(' aria-current="page"' if label==active else '')+f'>{label}</a>'
        for label,url in [('Explorer','./'),('Training','training.html'),('Run locally','run-locally.html')])+'</nav>'


def document(title, body, active):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)} · One Small Latte</title><link rel="icon" href="logo.png"><link rel="stylesheet" href="demo.css"></head>'
            '<body class="demo-page"><header class="demo-header"><a class="demo-brand" href="./"><img src="logo.png" alt="">One Small Latte</a>'
            +navigation(active)+'</header><main>'+body+'</main></body></html>')


def build_recording(preset):
    directory = recording_dir(preset)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!='latte-recorded-demo-v1':
        raise ValueError('Invalid demo manifest')
    payloads={}
    for name in ('training-record.json','training.trace.json','training-progress.json'):
        raw=gzip.decompress((directory/(name+'.gz')).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=manifest['files'][name+'.gz']:
            raise ValueError('Demo checksum mismatch: '+name)
        payloads[name]=raw
    record=json.loads(payloads['training-record.json'])
    progress=json.loads(payloads['training-progress.json'])
    if [step['step'] for step in record['steps']]!=manifest['recordedSteps'] or progress['step']!=manifest['steps']:
        raise ValueError('Incomplete public training recording')
    if not progress.get('history') or not progress.get('validation_history'):
        raise ValueError('Missing loss curves')
    if not all('corpus_usage' in step for step in record['steps']):
        raise ValueError('Demo must include recorded corpus usage')
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        for name,payload in payloads.items():
            (folder/name).write_bytes(payload)
        html = export_trace_html(folder/'training.trace.json',folder/'index.html',show='training').read_text()
    match=re.search(r'<script id="sequence-generation-data" type="application/json">(.*?)</script>',html,re.S)
    bundle=json.loads(match[1])
    runtime=bundle['trainingRuntime'].replace('</script','<\\/script')
    html=html.replace('<script>'+runtime+'</script>','',1)
    html=re.sub(r'<script id="training-record-data".*?</script>','',html,count=1,flags=re.S)
    lean={'show':'training','trace':bundle['trace']}
    html=re.sub(r'<script id="sequence-generation-data".*?</script>',
                lambda _: '<script id="sequence-generation-data" type="application/json">'+json.dumps(lean).replace('<','\\u003c')+'</script>',html,count=1,flags=re.S)
    metadata={**{key:value for key,value in record.items() if key!='steps'},
              'corpus_usage':record['steps'][-1]['corpus_usage'],
              'steps':[{'step':step['step'],'loss':step['loss']} for step in record['steps']]}
    run = {
        'id': preset['id'], 'name': preset['name'], 'state': 'complete', 'explorer_ready': True,
        'log': f"{manifest['source']} · {progress['step']} optimizer steps · {len(record['steps'])} checkpoints.",
    }
    config = preset_config(preset)
    settings = {
        'model': {key: value for key, value in config['model'].items() if key != 'dropout'},
        'training': {key: config['training'][key] for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'seed')},
    }
    settings['training'].update(train_split=config['data']['train_split'], record_every=record['recording']['stride'])
    workspace = {
        'metadata': metadata, 'shell': bundle['shell'], 'steps': record['steps'],
        'run': {**run, 'progress': progress, 'recorded_steps': manifest['recordedSteps']}, 'settings': settings,
    }
    return html, workspace, run


def build(output):
    output=Path(output).resolve()
    if output==ROOT or output in ROOT.parents or (output.exists() and any(output.iterdir()) and not (output/'.latte-pages-build').is_file()):
        raise ValueError('Choose an empty output folder or a previous Latte Pages build')
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output/'.latte-pages-build').touch()
    (output/'data').mkdir()
    runs = []
    for preset in load_presets():
        bootstrap, workspace, run = build_recording(preset)
        filename = f"{preset['id']}.json"
        (output/'data'/filename).write_text(json.dumps(workspace, ensure_ascii=False))
        runs.append({**run, 'file': filename})
        if preset['id'] == 'demo':
            html = bootstrap
    (output/'data/runs.json').write_text(json.dumps(runs))

    dist=ROOT/'frontend/dist'
    entry=json.loads((dist/'.vite/manifest.json').read_text())['src/main.tsx']
    shutil.copytree(dist,output/'ui')
    # Local builds use /ui; published assets must resolve beneath /Latte/.
    for css in (output/'ui').rglob('*.css'):
        css.write_text(css.read_text().replace('url(/ui/assets/','url('))
    styles=''.join(f'<link rel="stylesheet" href="ui/{path}">' for path in entry.get('css',[]))
    html=re.sub(r'<html\b','<html data-latte-demo data-latte-loading',html,count=1)
    loading='<style>html[data-latte-loading] body>*{visibility:hidden!important}html[data-latte-loading] #latte-loading{visibility:visible!important;position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center;background:#f7f2eb;color:#826c5c;font:14px system-ui}</style>'
    html=re.sub(r'<head\b[^>]*>',lambda match:match[0]+loading,html,count=1)
    html=re.sub(r'<body\b[^>]*>',lambda match:match[0]+'<div id="latte-loading" role="status">Loading Latte…</div>',html,count=1)
    html=html.replace('</head>',styles+'<link rel="stylesheet" href="demo.css"></head>',1)
    html=html.replace('</body>','<div id="latte-root"></div>'+f'<script type="module" src="ui/{entry["file"]}"></script><script src="demo.js"></script></body>',1)
    (output/'index.html').write_text(html)
    (output/'training.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=./?tab=training"><title>Training · One Small Latte</title><a href="./?tab=training">Open Training</a>')
    for name in ('demo.css','demo.js'):
        shutil.copy2(ROOT/'demo'/name,output/name)
    shutil.copy2(ROOT/'assets/latte-logo.png',output/'logo.png')
    (output/'fonts').mkdir()
    for name in ('ArchitectsDaughter-Regular.ttf','Fraunces-Variable.ttf','OFL.txt','Fraunces-OFL.txt'):
        shutil.copy2(ROOT/'fonts'/name,output/'fonts'/name)
    guide=(ROOT/'demo/run-locally.html').read_text()
    (output/'run-locally.html').write_text(document('Run locally',guide,'Run locally'))
    (output/'.nojekyll').touch()
    print('Built static Latte demo:',output)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'_site')
    build(parser.parse_args().output)
