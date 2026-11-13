"""Capture real training runs for the public example configurations."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo_presets import ROOT, load_presets, preset_config, recording_dir


def record_preset(preset):
    config = preset_config(preset)
    total = config['training']['steps']
    stride = preset['stride']
    recorded_steps = list(range(1, total + 1, stride))
    if recorded_steps[-1] != total:
        recorded_steps.append(total)

    with TemporaryDirectory() as directory:
        output = Path(directory)
        config_file = output / 'config.toml'
        config_file.write_text('\n'.join(
            f'[{section}]\n' + '\n'.join(f'{key} = {json.dumps(value)}' for key, value in values.items()) + '\n'
            for section, values in config.items()
        ))
        subprocess.run([
            sys.executable, str(ROOT / 'latte.py'), '--config', str(config_file),
            '--record', '1', str((total - 1) // stride + 1), str(stride),
            '--steps', str(total), '--output-dir', str(output),
        ], cwd=ROOT, check=True)
        destination = recording_dir(preset)
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ('training-record.json', 'training.trace.json', 'training-progress.json'):
            payload = (output / name).read_bytes()
            if name == 'training.trace.json':
                trace = json.loads(payload)
                trace['checkpoint'] = config['output']['checkpoint_name']
                payload = json.dumps(trace, ensure_ascii=False).encode()
            filename = name + '.gz'
            (destination / filename).write_bytes(gzip.compress(payload, mtime=0))
            files[filename] = hashlib.sha256(payload).hexdigest()

    manifest = {
        'schema': 'latte-recorded-demo-v1',
        'name': preset['name'],
        'source': f"CPU PyTorch training, AdamW, seed {config['training']['seed']}",
        'steps': total,
        'recordedSteps': recorded_steps,
        'files': files,
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    presets = load_presets()
    parser = argparse.ArgumentParser(description=__doc__)
    available = [preset['id'] for preset in presets]
    parser.add_argument('presets', nargs='*', metavar='PRESET', help='Choose from: ' + ', '.join(available))
    args = parser.parse_args()
    unknown = set(args.presets) - set(available)
    if unknown:
        parser.error('Unknown preset: ' + ', '.join(sorted(unknown)))
    for preset in presets:
        if not args.presets or preset['id'] in args.presets:
            record_preset(preset)


if __name__ == '__main__':
    main()
