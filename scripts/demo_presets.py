"""Configuration shared by recording and publishing the public examples."""
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_presets():
    return json.loads((ROOT / 'demo/presets.json').read_text())


def preset_config(preset):
    config = tomllib.loads((ROOT / 'demo/config.toml').read_text())
    for section, values in preset['overrides'].items():
        config[section].update(values)
    return config


def recording_dir(preset):
    directory = ROOT / 'demo/data'
    return directory if preset['id'] == 'demo' else directory / preset['id']
