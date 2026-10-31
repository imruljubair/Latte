"""Record a small real AdamW training run for the existing HTML explorer."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from latte.config import load_config
from latte.model import TinyTransformer
from latte.runtime import seed_everything
from latte.tokenizer import CharTokenizer
from corpus_usage import CorpusUsage
from training_history import loss_trend


def update_progress(output: Path, **values) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / 'training-progress.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def record(config_path: Path, output: Path, steps: int = 10, start: int = 1, stride: int = 1,
           total_steps: int | None = None) -> dict:
    config = load_config(config_path)
    if min(steps, start, stride) < 1:
        raise ValueError('Record start, count, and stride must be positive')
    selected = set(range(start, start + steps * stride, stride))
    last_step = start + (steps - 1) * stride
    if total_steps is not None:
        if total_steps < last_step:
            raise ValueError('Total training steps must include all requested recording steps')
        last_step = total_steps
        selected.add(last_step)
    update_progress(output, phase='initializing', step=0, total_steps=last_step, loss=None,
                    recorded=0, history=[], validation_loss=None, validation_history=[])
    if config.model.dropout:
        raise ValueError('Training replay currently requires dropout = 0 for the browser forward calculations')
    seed_everything(config.training.seed)
    text = Path(config.data.path).read_text(encoding='utf-8')
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(data) * config.data.train_split)
    validation = data[split:]
    data = data[:split]
    length = config.model.context_length
    if len(data) < length + 2:
        raise ValueError('Training corpus must exceed context length')
    model = TinyTransformer(config.model, tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    document = dict(format_version=1, architecture=model.architecture_dict(),
                    vocabulary=tokenizer.chars, corpus=dict(path=config.data.path, text=text, train_end=split),
                    optimizer=dict(name='AdamW', learning_rate=config.training.learning_rate,
                                   weight_decay=config.training.weight_decay), steps=[])
    document['recording'] = dict(start=start, count=len(selected), stride=stride)
    document['sampling'] = {'seed': config.training.seed}
    usage = CorpusUsage()
    history = []
    validation_history = []
    validation_loss = None
    evaluation_every = max(1, last_step // 100)
    # Fixed held-out windows, with no RNG calls or optimizer updates.
    validation_inputs = validation_targets = None
    if len(validation) > length:
        positions = torch.linspace(0, len(validation) - length - 1, min(16, len(validation) - length)).long().unique()
        validation_inputs = torch.stack([validation[i:i+length] for i in positions])
        validation_targets = torch.stack([validation[i+1:i+length+1] for i in positions])
    last_progress = 0.0
    for step in range(1, last_step + 1):
        capture = step in selected
        starts = torch.randint(len(data) - length - 1, (config.training.batch_size,))
        usage.add(starts.tolist(), length)
        inputs = torch.stack([data[i:i+length] for i in starts])
        targets = torch.stack([data[i+1:i+length+1] for i in starts])
        before = {name: p.detach().clone() for name, p in model.named_parameters()} if capture else None
        logits, _ = model(inputs)
        losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape_as(targets)
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: p.grad.detach().tolist() for name, p in model.named_parameters()} if capture else None
        optimizer.step()
        history.append({'step': step, 'loss': loss.item()})
        if validation_inputs is not None and (step == 1 or step == last_step or step % evaluation_every == 0):
            model.eval()
            with torch.no_grad():
                validation_logits, _ = model(validation_inputs)
                validation_loss = F.cross_entropy(validation_logits.flatten(0, 1), validation_targets.flatten()).item()
            model.train()
            validation_history.append({'step': step, 'loss': validation_loss})
        if step == 1 or step == last_step or time.monotonic() - last_progress >= .2:
            update_progress(output, phase='training', step=step, total_steps=last_step, loss=loss.item(),
                            recorded=len(document['steps']) + int(capture), history=loss_trend(history), history_averaged=len(history) > 300,
                            validation_loss=validation_loss, validation_history=validation_history[-1000:])
            last_progress = time.monotonic()
        if not capture:
            continue
        document['steps'].append(dict(step=step, starts=starts.tolist(), inputs=inputs.tolist(), corpus_usage=usage.snapshot(),
            targets=targets.tolist(), logits=logits.detach().tolist(), token_losses=losses.detach().tolist(),
            loss=loss.item(), before={name: p.tolist() for name, p in before.items()},
            gradients=gradients, after={name: p.detach().tolist() for name, p in model.named_parameters()}))
        print(f'Recorded step {step} ({len(document["steps"])}/{len(selected)}): loss {loss.item():.4f}')
    document['corpus_usage'] = usage.snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--record', type=int, nargs=3, metavar=('START', 'COUNT', 'STRIDE'))
    args = parser.parse_args()
    start, count, stride = args.record or (1, args.steps, 1)
    record(args.config, args.output, count, start, stride)
