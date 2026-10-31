"""Set up and start a local Latte checkout, including prebuilt UI releases."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Latte requires Python 3.11 or newer')
    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        print('Creating Latte Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = ROOT / 'requirements.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / '.latte-requirements'
    if not stamp.exists() or stamp.read_text() != digest:
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        stamp.write_text(digest)
    if not (ROOT / 'frontend/dist/.vite/manifest.json').is_file():
        npm = shutil.which('npm')
        if not npm:
            parser.error('This source checkout needs Node.js to build the UI. Install Node.js or use a release with frontend/dist included.')
        subprocess.run([npm, 'ci'], cwd=ROOT / 'frontend', check=True)
        subprocess.run([npm, 'run', 'build'], cwd=ROOT / 'frontend', check=True)
    if not (ROOT / 'out/default/latte.html').is_file():
        subprocess.run([str(python), str(ROOT / 'latte.py')], cwd=ROOT, check=True)
    command = [str(python), str(ROOT / 'server.py'), '--port', str(args.port)]
    if not args.no_open:
        command.append('--open')
    print(f'Latte is starting at http://127.0.0.1:{args.port}', flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
