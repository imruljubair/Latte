"""Build the static Pages edition from a verified real training recording.

Only NumPy is required; publishing never trains or installs PyTorch.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
from html import escape
import json
import shutil
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_trace_html import export_trace_html


def navigation(active):
    return '<nav class="demo-nav" aria-label="Demo navigation">'+''.join(
        f'<a href="{url}"'+(' aria-current="page"' if label==active else '')+f'>{label}</a>'
        for label,url in [('Explorer','./'),('Training','training.html'),('Run locally','run-locally.html')])+'</nav>'


def document(title, body, active):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)} · One Small Latte</title><link rel="icon" href="logo.png"><link rel="stylesheet" href="demo.css"></head>'
            '<body class="demo-page"><header class="demo-header"><a class="demo-brand" href="./"><img src="logo.png" alt="">One Small Latte</a>'
            +navigation(active)+'</header><main>'+body+'</main></body></html>')


def chart(progress):
    series = [(progress['history'],'Training','#a56b35',''),(progress['validation_history'],'Validation','#538b9b',' stroke-dasharray="5 4"')]
    losses=[point['loss'] for points,*_ in series for point in points]
    low,high=min(losses),max(losses)
    span=max(.1,high-low);low=max(0,low-span*.08);high+=span*.08
    width,height=760,220
    x=lambda step:52+step/progress['total_steps']*680
    y=lambda loss:180-(loss-low)/(high-low)*160
    lines=['<svg class="demo-chart" viewBox="0 0 760 220" role="img" aria-label="Recorded training and validation loss versus optimizer step">']
    for ratio in (0,.5,1):
        loss=low+(high-low)*ratio
        lines.append(f'<path d="M52 {y(loss):.2f}H732" stroke="#e2d4c5" stroke-dasharray="4 5"/><text x="4" y="{y(loss)+4:.2f}">{loss:.2f}</text>')
    for points,label,color,dash in series:
        path=' '.join(f'{"M" if index==0 else "L"}{x(point["step"]):.2f} {y(point["loss"]):.2f}' for index,point in enumerate(points))
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.1" vector-effect="non-scaling-stroke"{dash}><title>{label}</title></path>')
    for step in (0,round(progress['total_steps']/2),progress['total_steps']):
        lines.append(f'<text x="{x(step):.2f}" y="208" text-anchor="middle">{step}</text>')
    lines.append('</svg>')
    return ''.join(lines)


def settings(title,values):
    return '<section class="demo-card"><h2>'+title+'</h2><dl>'+''.join(
        '<div><dt>'+escape(key.replace('_',' ').capitalize())+'</dt><dd>'+escape(str(value))+'</dd></div>'
        for key,value in values.items())+'</dl></section>'


def build(output):
    manifest=json.loads((ROOT/'demo/data/manifest.json').read_text())
    if manifest.get('schema')!='latte-recorded-demo-v1':
        raise ValueError('Invalid demo manifest')
    payloads={}
    for name in ('training-record.json','training.trace.json','training-progress.json'):
        raw=gzip.decompress((ROOT/'demo/data'/(name+'.gz')).read_bytes())
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
    output=Path(output).resolve()
    if output==ROOT or output in ROOT.parents or (output.exists() and any(output.iterdir()) and not (output/'.latte-pages-build').is_file()):
        raise ValueError('Choose an empty output folder or a previous Latte Pages build')
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output/'.latte-pages-build').touch()
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        for name,payload in payloads.items():
            (folder/name).write_bytes(payload)
        export_trace_html(folder/'training.trace.json',output/'index.html',show='training')
    html=(output/'index.html').read_text()
    html=html.replace('</head>','<link rel="stylesheet" href="demo.css"></head>',1)
    html=html.replace('</body>','<script src="demo.js"></script></body>',1)
    (output/'index.html').write_text(html)
    for name in ('demo.css','demo.js'):
        shutil.copy2(ROOT/'demo'/name,output/name)
    shutil.copy2(ROOT/'assets/latte-logo.png',output/'logo.png')
    (output/'fonts').mkdir()
    for name in ('ArchitectsDaughter-Regular.ttf','Fraunces-Variable.ttf','OFL.txt','Fraunces-OFL.txt'):
        shutil.copy2(ROOT/'fonts'/name,output/'fonts'/name)
    config=tomllib.loads((ROOT/'demo/config.toml').read_text())
    body='<p class="eyebrow">RECORDED TRAINING · COMPLETE</p><h1>A small model, learning.</h1><p>Explore a real 201-step training run. These are saved measurements; training new models runs on your computer.</p>'
    body+='<section class="demo-card"><h2>Training progress</h2><p>201 / 201 optimizer steps · 6 recorded checkpoints · batch of 4</p><div class="demo-legend"><span>━ Training</span><span>┄ Validation</span></div>'+chart(progress)
    body+=f'<p class="demo-metrics">Final training loss: {progress["loss"]:.4f} · Validation loss: {progress["validation_loss"]:.4f}</p><a href="./">Explore recorded matrices →</a></section>'
    body+='<div class="demo-columns">'+settings('Architecture',config['model'])+settings('Training settings',config['training'])+'</div>'
    body+='<section class="demo-card"><h2>Corpus</h2><p>The same fixed character corpus used by this recording; 90% training, 10% validation.</p><pre class="demo-corpus">'+escape(record['corpus']['text'])+'</pre></section>'
    body+='<section class="demo-card"><h2>Train with your own text</h2><p>Paste a corpus, choose the architecture, and watch live training and validation curves in the local Training tab.</p><a class="demo-button" href="run-locally.html">Run Latte locally →</a></section>'
    (output/'training.html').write_text(document('Recorded training',body,'Training'))
    guide=(ROOT/'demo/run-locally.html').read_text()
    (output/'run-locally.html').write_text(document('Run locally',guide,'Run locally'))
    (output/'.nojekyll').touch()
    print('Built static Latte demo:',output)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'_site')
    build(parser.parse_args().output)
