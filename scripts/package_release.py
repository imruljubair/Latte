"""Package Latte with its built UI; the recipient only needs Python."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not (ROOT / 'frontend/dist/.vite/manifest.json').is_file():
        raise SystemExit('Build the UI first: cd frontend && npm ci && npm run build')
    excluded = {'.git', '.venv', 'node_modules', '__pycache__', 'out', 'releases', 'backups',
                '.pytest_cache', '.artifact-work', 'test-results', 'playwright-report', '_site'}
    destination = ROOT / 'releases/Latte.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if path.is_file() and not excluded.intersection(relative.parts) and path.name != '.DS_Store':
                archive.write(path, Path('Latte') / relative)
    print(destination)


if __name__ == '__main__':
    main()
