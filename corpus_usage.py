"""Exact per-position input exposure counts, compacted as constant-count ranges."""
from __future__ import annotations

from pathlib import Path
import pickle
import threading
import tomllib


_REPLAY_LOCK = threading.Lock()


class CorpusUsage:
    def __init__(self):
        self.changes: dict[int, int] = {}
        self.total = 0

    def add(self, starts, length: int):
        for start in starts:
            start = int(start)
            for position, delta in ((start, 1), (start + length, -1)):
                value = self.changes.get(position, 0) + delta
                if value:
                    self.changes[position] = value
                else:
                    self.changes.pop(position, None)
            self.total += length

    def snapshot(self):
        count = previous = 0
        ranges = []
        for position, delta in sorted(self.changes.items()):
            if count and position > previous:
                ranges.append([previous, position, count])
            count += delta
            previous = position
        return {'ranges': ranges, 'max_count': max((item[2] for item in ranges), default=0),
                'used_positions': sum(end - start for start, end, _ in ranges),
                'total_uses': self.total}


def with_corpus_usage(document: dict, directory: Path) -> dict:
    """Recover legacy counts only when every saved batch matches seeded replay.

    Replay just model initialization and sampling, not training. RNG state and
    the original recording are left untouched. Unknown/mismatching seeds never
    produce an estimated heatmap.
    """
    if not document.get('steps'):
        return document
    if all('corpus_usage' in step for step in document['steps']):
        latest = max(document['steps'], key=lambda step: step['step'])
        return {**document, 'corpus_usage': latest['corpus_usage']}
    try:
        # Current recordings already contain counts; static exports need no PyTorch.
        import torch
        from latte.config import ModelConfig
        from latte.model import TinyTransformer
        seed = document.get('sampling', {}).get('seed')
        if seed is None and (directory / 'model.pt').is_file():
            checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
            seed = checkpoint['project_config']['training']['seed']
        if seed is None and (directory / 'config.toml').is_file():
            seed = tomllib.loads((directory / 'config.toml').read_text())['training']['seed']
        if seed is None:
            return document
        architecture = dict(document['architecture'])
        vocab_size = architecture.pop('vocab_size')
        if architecture.get('dropout', 0):
            return document
        steps = {step['step']: step for step in document['steps']}
        batch_size = len(document['steps'][0]['starts'])
        length = architecture['context_length']
        maximum = document['corpus']['train_end'] - length - 1
        usage = CorpusUsage()
        snapshots = {}
        with _REPLAY_LOCK, torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed))
            TinyTransformer(ModelConfig(**architecture), vocab_size)
            for index in range(1, max(steps) + 1):
                starts = torch.randint(maximum, (batch_size,)).tolist()
                if index in steps and starts != steps[index]['starts']:
                    return document
                usage.add(starts, length)
                if index in steps:
                    snapshots[index] = usage.snapshot()
        return {**document, 'corpus_usage': usage.snapshot(), 'sampling': {'seed': int(seed)},
                'steps': [{**step, 'corpus_usage': snapshots[step['step']]} for step in document['steps']]}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, pickle.UnpicklingError, EOFError):
        return document
