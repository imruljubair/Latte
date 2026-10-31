"""Bounded loss trends that retain the beginning and end of a training run."""
from math import fsum


def loss_trend(history: list[dict], limit: int = 300) -> list[dict]:
    if len(history) <= limit:
        return history
    first, last = history[0], history[-1]
    bins = {}
    for point in history[1:-1]:
        index = min(limit - 3, int((point['step'] - first['step']) * (limit - 2) / max(1, last['step'] - first['step'])))
        bins.setdefault(index, []).append(point)
    return [first, *[{'step': fsum(p['step'] for p in group) / len(group),
                      'loss': fsum(p['loss'] for p in group) / len(group)}
                     for group in bins.values()], last]
