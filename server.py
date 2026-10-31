"""Local Latte application: existing visuals, React controls, on-demand recordings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import tomllib
from contextlib import asynccontextmanager
from functools import lru_cache
import json
import os
import signal
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from corpus_usage import with_corpus_usage
from training_history import loss_trend

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\Z')


class ModelSettings(BaseModel):
    model_dim: int = Field(default=4, ge=1, le=2048)
    qk_dim: int = Field(default=3, ge=1, le=2048)
    value_dim: int = Field(default=4, ge=1, le=2048)
    num_heads: int = Field(default=1, ge=1, le=64)
    num_layers: int = Field(default=1, ge=1, le=64)
    ffn_dim: int = Field(default=16, ge=1, le=8192)
    context_length: int = Field(default=8, ge=1, le=2048)


class TrainingSettings(BaseModel):
    steps: int = Field(default=100, ge=1, le=100000)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=.003, gt=0, le=1, allow_inf_nan=False)
    weight_decay: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    seed: int = Field(default=1337, ge=0, le=4294967295)
    train_split: float = Field(default=.9, gt=0, lt=1, allow_inf_nan=False)
    record_every: int = Field(default=10, ge=1, le=100000)


class RunRequest(BaseModel):
    config: str = 'configs/default.toml'
    name: str = Field(default='', max_length=80)
    corpus: str | None = Field(default=None, max_length=1000000)
    model: ModelSettings | None = None
    training: TrainingSettings | None = None
    start: int = Field(default=1, ge=1)
    count: int = Field(default=10, ge=1)
    stride: int = Field(default=1, ge=1)


@lru_cache(maxsize=4)
def read_record(path: str, modified: int) -> dict:
    return with_corpus_usage(json.loads(Path(path).read_text(encoding='utf-8')), Path(path).parent)


@lru_cache(maxsize=4)
def read_explorer(path: str, modified: int) -> tuple[str, dict]:
    html = Path(path).read_text(encoding='utf-8')
    # Live UI interactions should stay current even when a run embeds an older
    # presentation script. Standalone downloads retain their original file.
    presentation = (ROOT / 'presentation.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    html = re.sub(r'<script>(.*?)</script>',
                  lambda item: '<script>' + presentation + '</script>'
                  if 'function installCorpusTooltip(panel)' in item[1] else item[0], html, flags=re.S)
    match = re.search(r'<script id="sequence-generation-data" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise ValueError('Explorer is missing sequence metadata; regenerate it with latte.py')
    bundle = json.loads(match[1])
    # Update the live diagram's chooser even for previously exported run shells.
    current_runtime = (ROOT / 'html_runtime.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    bundle['shell'] = re.sub(r'<script>(.*?)</script>',
        lambda item: '<script>' + current_runtime + '</script>'
        if 'const cells = new Map();' in item[1] and 'const stageCards=' in item[1] else item[0],
        bundle['shell'], flags=re.S)
    current_annotations = (ROOT / 'annotation_overlay.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    bundle['shell'] = re.sub(r'<script>(.*?)</script>',
        lambda item: '<script>' + current_annotations + '</script>'
        if "Layer and operation hierarchy" in item[1] and 'const bands = [];' in item[1] else item[0],
        bundle['shell'], flags=re.S)
    current_zoom = (ROOT / 'annotated_zoom.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    bundle['shell'] = re.sub(r'<script>(.*?)</script>',
        lambda item: '<script>' + current_zoom + '</script>'
        if 'function onCanvas(event' in item[1] else item[0], bundle['shell'], flags=re.S)
    runtime = bundle.get('trainingRuntime', '').replace('</script', '<\\/script')
    if runtime:
        html = html.replace('<script>' + runtime + '</script>', '', 1)
    html = re.sub(r'<script id="training-record-data".*?</script>', '', html, count=1, flags=re.S)
    # The application fetches one recording step and the diagram shell separately.
    lean = {'show': 'training', 'trace': bundle['trace']}
    tag = '<script id="sequence-generation-data" type="application/json">'
    html = re.sub(r'<script id="sequence-generation-data".*?</script>',
                  lambda _: tag + json.dumps(lean).replace('<', '\\u003c') + '</script>',
                  html, count=1, flags=re.S)
    return html, {'shell': bundle['shell']}


def create_app(root: Path = ROOT) -> FastAPI:
    root = root.resolve()
    outputs = root / 'out'
    jobs: dict[str, tuple[subprocess.Popen, object]] = {}
    stopped: set[str] = set()
    job_lock = threading.Lock()

    def signal_job(process, force=False):
        try:
            if sys.platform != 'win32':
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif not force:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    @asynccontextmanager
    async def lifespan(app):
        yield
        for process, log in jobs.values():
            if process.poll() is None:
                signal_job(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    signal_job(process, force=True)
                    process.wait()
            log.close()

    app = FastAPI(title='Latte', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.middleware('http')
    async def local_requests(request: Request, call_next):
        # Local training mutations must come from this app, not another website.
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return PlainTextResponse('Cross-origin mutations are not allowed', status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-cache'
        return response

    def directory(run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(400, 'Invalid run ID')
        path = (outputs / run_id).resolve()
        if path.parent != outputs.resolve():
            raise HTTPException(400, 'Run must be inside the output directory')
        return path

    def recording(run_id: str) -> dict:
        path = directory(run_id) / 'training-record.json'
        if not path.is_file():
            raise HTTPException(404, 'Recording is not ready')
        return read_record(str(path), path.stat().st_mtime_ns)

    def explorer(run_id: str) -> tuple[str, dict]:
        path = directory(run_id) / 'latte.html'
        if not path.is_file():
            raise HTTPException(404, 'Explorer is not ready; run python latte.py first')
        return read_explorer(str(path), path.stat().st_mtime_ns)

    @app.get('/api/runs')
    def list_runs():
        ids = {p.name for p in outputs.iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)} if outputs.exists() else set()
        return [status(run_id) for run_id in sorted(ids | jobs.keys())]

    @app.post('/api/runs', status_code=202)
    def start_run(settings: RunRequest):
        with job_lock:
            return launch_run(settings)

    def launch_run(settings: RunRequest):
        config = (root / settings.config).resolve()
        if not config.is_relative_to(root) or not config.is_file() or config.suffix != '.toml':
            raise HTTPException(400, 'Config must be a TOML file inside Latte')
        if any(process.poll() is None for process, _ in jobs.values()):
            raise HTTPException(409, 'A training run is already active. Stop it or wait for completion.')
        custom = settings.corpus is not None
        if not custom and (settings.model is not None or settings.training is not None):
            raise HTTPException(400, 'Paste a corpus when providing model or training settings')
        model = settings.model or ModelSettings()
        training = settings.training or TrainingSettings()
        if custom:
            split = int(len(settings.corpus) * training.train_split)
            if min(split, len(settings.corpus) - split) < model.context_length + 2:
                raise HTTPException(400, 'Both training and validation text must have at least context length + 2 characters. Paste more text or reduce context length.')
        run_id = 'run-' + uuid.uuid4().hex[:12]
        output = directory(run_id)
        output.mkdir(parents=True)
        total_steps = None
        start, count, stride = settings.start, settings.count, settings.stride
        if custom:
            corpus_path = output / 'corpus.txt'
            corpus_path.write_text(settings.corpus, encoding='utf-8')
            raw = tomllib.loads(config.read_text(encoding='utf-8'))
            raw['model'] = {**model.model_dump(), 'dropout': 0.0}
            raw['data'] = {'path': str(corpus_path), 'tokenizer': 'char', 'train_split': training.train_split}
            raw['training'].update({key: value for key, value in training.model_dump().items()
                                    if key not in ('train_split', 'record_every')})
            raw['training'].update(device='cpu', eval_batches=5)
            raw['output']['directory'] = str(output)
            config = output / 'config.toml'
            config.write_text('\n\n'.join('[' + section + ']\n' + '\n'.join(
                key + ' = ' + json.dumps(value) for key, value in values.items())
                for section, values in raw.items()) + '\n', encoding='utf-8')
            total_steps = training.steps
            start, stride = 1, training.record_every
            count = 1 + (training.steps - 1) // stride
        (output / 'run-info.json').write_text(json.dumps({
            'name': settings.name.strip() or run_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'total_steps': total_steps,
            'record_every': stride,
            **({'model': model.model_dump(), 'training': training.model_dump()} if custom else {}),
        }), encoding='utf-8')
        command = [sys.executable, '-u', str(root / 'latte.py'), '--config', str(config),
                   '--output-dir', str(output), '--record', str(start), str(count), str(stride)]
        if total_steps is not None:
            command += ['--steps', str(total_steps)]
        log = (output / 'training.log').open('w', encoding='utf-8')
        try:
            process = subprocess.Popen(
                command,
                cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=sys.platform != 'win32',
            )
        except Exception:
            log.close()
            raise
        jobs[run_id] = process, log
        return status(run_id)

    @app.get('/api/runs/{run_id}/status')
    def status(run_id: str):
        path = directory(run_id)
        job = jobs.get(run_id)
        code = job[0].poll() if job else None
        if job:
            if run_id in stopped:
                state = 'stopping' if code is None else 'stopped'
            else:
                state = 'running' if code is None else 'complete' if code == 0 else 'failed'
            if code is not None:
                job[1].close()
        elif (path / 'latte.html').is_file():
            state = 'complete'
        elif not path.exists():
            raise HTTPException(404, 'Unknown run')
        else:
            state = 'incomplete'
        log = path / 'training.log'
        def saved_json(name):
            file = path / name
            try:
                return json.loads(file.read_text(encoding='utf-8')) if file.exists() else {}
            except (OSError, json.JSONDecodeError):
                return {}
        info = saved_json('run-info.json')
        progress = saved_json('training-progress.json')
        history = progress.get('history', [])
        # Older versions saved only the last 1,000 batches. Recover earlier
        # measured losses from saved records, without inventing missing batches.
        recorded = path / 'training-record.json'
        if history and history[0]['step'] > 1 and recorded.is_file():
            document = read_record(str(recorded), recorded.stat().st_mtime_ns)
            earlier = [{'step': point['step'], 'loss': point['loss']} for point in document['steps']
                       if point['step'] < history[0]['step']]
            if earlier:
                history = earlier + history
                progress['history_sparse'] = True
        if history:
            progress['history_averaged'] = progress.get('history_averaged', False) or len(history) > 300
            progress['history'] = loss_trend(history)
        return {'id': run_id, 'name': info.get('name', run_id),
                'created_at': info.get('created_at'), 'state': state, 'exit_code': code,
                'progress': progress, 'explorer_ready': (path / 'latte.html').is_file(),
                'log': log.read_text(encoding='utf-8')[-8000:] if log.exists() else ''}

    @app.post('/api/runs/{run_id}/stop')
    def stop_run(run_id: str):
        directory(run_id)
        if run_id not in jobs:
            raise HTTPException(404, 'No active training process for this run')
        process, _ = jobs[run_id]
        if process.poll() is None:
            stopped.add(run_id)
            signal_job(process)
        return status(run_id)

    @app.delete('/api/runs/{run_id}')
    def delete_run(run_id: str):
        with job_lock:
            path = directory(run_id)
            if run_id == 'default' or (outputs / run_id).is_symlink():
                raise HTTPException(400, 'The default example and linked directories cannot be deleted')
            job = jobs.get(run_id)
            if job and job[0].poll() is None:
                raise HTTPException(409, 'Stop training before deleting this run')
            if not path.is_dir():
                raise HTTPException(404, 'Unknown run')
            # Keep a recoverable copy outside the recent-run listing.
            trash = outputs / '.trash'
            trash.mkdir(exist_ok=True)
            if job:
                job[1].close()
            path.rename(trash / (run_id + '-' + uuid.uuid4().hex[:8]))
            jobs.pop(run_id, None)
            stopped.discard(run_id)
            read_record.cache_clear()
            read_explorer.cache_clear()
            return {'deleted': run_id}

    @app.get('/api/runs/{run_id}/corpus')
    def run_corpus(run_id: str):
        path = directory(run_id)
        # Completed recordings retain the exact text used for training.
        if (path / 'training-record.json').is_file():
            return {'text': recording(run_id)['corpus']['text']}
        corpus_file = path / 'corpus.txt'
        if corpus_file.is_file():
            return {'text': corpus_file.read_text(encoding='utf-8')}
        raise HTTPException(404, 'Corpus is not available for this run')

    @app.get('/api/runs/{run_id}/settings')
    def run_settings(run_id: str):
        path = directory(run_id)
        if not path.is_dir():
            raise HTTPException(404, 'Unknown run')
        def saved(name):
            file = path / name
            return json.loads(file.read_text(encoding='utf-8')) if file.is_file() else {}
        config = {}
        for name in ('config.toml', 'architecture.toml'):
            if (path / name).is_file():
                config = tomllib.loads((path / name).read_text(encoding='utf-8'))
                break
        if not config:
            # CLI runs may retain their project config only in the checkpoint.
            checkpoints = list(path.glob('*.pt'))
            checkpoint = path / 'model.pt'
            if not checkpoint.is_file() and len(checkpoints) == 1:
                checkpoint = checkpoints[0]
            if checkpoint.is_file():
                config = torch.load(checkpoint, map_location='cpu', weights_only=True).get('project_config', {})
        info, progress = saved('run-info.json'), saved('training-progress.json')
        document = recording(run_id) if (path / 'training-record.json').is_file() else {}
        model = {**config.get('model', {}), **info.get('model', {}), **document.get('architecture', {})}
        training = {**config.get('training', {}), **info.get('training', {})}
        training.update(document.get('optimizer', {}))
        if document.get('steps'):
            training['batch_size'] = len(document['steps'][0]['inputs'])
            training['steps'] = document['steps'][-1]['step']
        if 'seed' in document.get('sampling', {}):
            training['seed'] = document['sampling']['seed']
        training['steps'] = info.get('total_steps') or progress.get('total_steps') or training.get('steps')
        training['train_split'] = info.get('training', {}).get('train_split', config.get('data', {}).get('train_split'))
        training['record_every'] = info.get('record_every', document.get('recording', {}).get('stride'))
        return {'model': {key: model.get(key) for key in ModelSettings.model_fields},
                'training': {key: training.get(key) for key in TrainingSettings.model_fields}}

    @app.get('/api/runs/{run_id}/metadata')
    def metadata(run_id: str):
        data = recording(run_id)
        return {**{key: value for key, value in data.items() if key != 'steps'},
                'steps': [{'step': step['step'], 'loss': step['loss']} for step in data['steps']]}

    @app.get('/api/runs/{run_id}/steps/{index}')
    def get_step(run_id: str, index: int):
        steps = recording(run_id)['steps']
        if index < 0 or index >= len(steps):
            raise HTTPException(404, 'Recorded step index out of range')
        return steps[index]

    @app.get('/api/runs/{run_id}/shell', response_class=HTMLResponse)
    def get_shell(run_id: str):
        return explorer(run_id)[1]['shell']

    @app.get('/api/runs/{run_id}/export', response_class=HTMLResponse)
    def download_export(run_id: str):
        path = directory(run_id) / 'latte.html'
        if not path.is_file():
            raise HTTPException(404, 'Explorer is not ready')
        return HTMLResponse(path.read_text(encoding='utf-8'),
                            headers={'Content-Disposition': 'attachment; filename="latte.html"'})

    dist = root / 'frontend/dist'
    if dist.is_dir():
        app.mount('/ui', StaticFiles(directory=dist), name='ui')

    @app.get('/', response_class=HTMLResponse)
    def index(run: str = 'default'):
        manifest = dist / '.vite/manifest.json'
        if not manifest.is_file():
            raise HTTPException(503, 'Build the UI: cd frontend && npm ci && npm run build')
        html, _ = explorer(run)
        html = re.sub(r'<title>.*?</title>', '<title>One Small Latte</title>', html, count=1, flags=re.S)
        entry = json.loads(manifest.read_text())['src/main.tsx']
        styles = ''.join(f'<link rel="stylesheet" href="/ui/{path}">' for path in entry.get('css', []))
        html = re.sub(r'<html\b', '<html data-latte-loading', html, count=1)
        loading_style = '''<style>
html[data-latte-loading] body>*{visibility:hidden!important}
html[data-latte-loading] #latte-loading{visibility:visible!important;position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center;background:#f7f2eb;color:#826c5c;font:14px system-ui}
</style>'''
        # This gate precedes every export script and stylesheet: the browser
        # never paints the workbook used to bootstrap the live application.
        html = re.sub(r'<head\b[^>]*>', lambda match: match[0] + loading_style, html, count=1)
        html = re.sub(r'<body\b[^>]*>', lambda match: match[0] + '<div id="latte-loading" role="status">Loading Latte…</div>', html, count=1)
        html = html.replace('</head>', styles + '</head>', 1)
        html = html.replace('</body>', '<div id="latte-root"></div>'
                            f'<script type="module" src="/ui/{entry["file"]}"></script></body>', 1)
        return html

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    if args.open:
        threading.Timer(1, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    import uvicorn
    uvicorn.run(create_app(), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
