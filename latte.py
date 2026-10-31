"""Latte — Language-model Architecture, Training & Tuning Explorer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import webbrowser

import torch

from latte.checkpoint import save_checkpoint
from latte.config import load_config, write_architecture_config
from latte.model import TinyTransformer
from latte.tokenizer import CharTokenizer
from record_training import record, update_progress
from export_trace_html import export_trace_html
from train import estimate_loss

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('architecture', nargs='?', help='Five digits: model dimension, head dimension (Q/K/V), heads, layers, FFN dimension; six-digit separate QK/V codes also supported')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.toml')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'out/default')
    parser.add_argument('--record', type=int, nargs=3, default=(1, 10, 1),
                        metavar=('START', 'COUNT', 'STRIDE'))
    parser.add_argument('--steps', type=int, help='Total optimizer steps, including the final recorded step')
    parser.add_argument('--open', action='store_true', help='Open the training explorer')
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    # Bundled config paths are relative to this independent project.
    os.chdir(ROOT)
    if args.architecture:
        generated = output / 'architecture.toml'
        try:
            write_architecture_config(config_path, generated, args.architecture)
        except ValueError as error:
            parser.error(str(error))
        config_path = generated
    config = load_config(config_path)
    start, count, stride = args.record
    document = record(config_path, output / 'training-record.json', count, start, stride, total_steps=args.steps)
    update_progress(output / 'training-record.json', phase='exporting')
    final = document['steps'][-1]
    tokenizer = CharTokenizer(document['vocabulary'])
    model = TinyTransformer(config.model, tokenizer.vocab_size)
    model.load_state_dict({name: torch.tensor(value) for name, value in final['after'].items()})
    encoded = torch.tensor(tokenizer.encode(document['corpus']['text']), dtype=torch.long)
    split = document['corpus']['train_end']
    losses = estimate_loss(model, encoded[:split], encoded[split:],
                           config.training.batch_size, config.model.context_length,
                           config.training.eval_batches, torch.device('cpu'))
    checkpoint = output / config.output.checkpoint_name
    save_checkpoint(checkpoint, model, tokenizer, config.to_dict(), final['step'], losses['validation'])
    trace = output / 'training.trace.json'
    prompt = ''.join(tokenizer.chars[index] for index in final['inputs'][0])
    subprocess.run([sys.executable, str(ROOT / 'export_trace.py'),
                    '--checkpoint', str(checkpoint), '--text', prompt,
                    '--output', str(trace)], check=True, cwd=ROOT)
    explorer = export_trace_html(trace, output / 'latte.html', show='training')
    update_progress(output / 'training-record.json', phase='complete')
    print(f'Latte · {len(document["steps"])} recorded steps · validation loss {losses["validation"]:.4f}')
    print(f'Explorer: {explorer}')
    if args.open:
        webbrowser.open(explorer.as_uri())


if __name__ == '__main__':
    main()
