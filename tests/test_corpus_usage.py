import copy
from pathlib import Path
import tempfile
import unittest

import torch

from corpus_usage import CorpusUsage, with_corpus_usage
from record_training import record


class CorpusUsageTests(unittest.TestCase):
    def test_overlaps_and_repeated_positions_are_counted_each_time(self):
        usage = CorpusUsage()
        usage.add([0, 2, 2], 3)
        self.assertEqual(usage.snapshot(), {
            'ranges': [[0, 2, 1], [2, 3, 3], [3, 5, 2]],
            'max_count': 3, 'used_positions': 5, 'total_uses': 9,
        })
        saved = usage.snapshot()
        usage.add([8], 2)
        self.assertEqual(saved['total_uses'], 9)
        self.assertEqual(usage.snapshot()['ranges'][-1], [8, 10, 1])

    def test_skipped_recording_steps_are_included_and_validation_is_unused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = record(Path('configs/default.toml'), root / 'dense/record.json', 5)
            sparse = record(Path('configs/default.toml'), root / 'sparse/record.json', 2, 2, 3)
            self.assertEqual(sparse['corpus_usage'], dense['corpus_usage'])
            self.assertEqual(sparse['corpus_usage']['total_uses'], 5 * 4 * 8)
            for step in sparse['steps']:
                expected = dense['steps'][step['step'] - 1]
                self.assertEqual(step['corpus_usage'], expected['corpus_usage'])
                self.assertEqual(step['after'], expected['after'])
                usage = step['corpus_usage']
                self.assertEqual(usage['total_uses'], step['step'] * 4 * 8)
                self.assertEqual(sum((b - a) * n for a, b, n in usage['ranges']), usage['total_uses'])
                self.assertTrue(all(0 <= a < b <= sparse['corpus']['train_end'] for a, b, _ in usage['ranges']))

    def test_verified_legacy_recovery_matches_actual_sampling_without_changing_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = record(Path('configs/default.toml'), root / 'record.json', 2, 2, 3)
            legacy = copy.deepcopy(document)
            del legacy['corpus_usage']
            for step in legacy['steps']:
                del step['corpus_usage']
            state = torch.random.get_rng_state().clone()
            recovered = with_corpus_usage(legacy, root)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
            self.assertEqual(recovered, document)
            self.assertTrue(all('corpus_usage' not in step for step in legacy['steps']))
            legacy['steps'][0]['starts'][0] = len(legacy['corpus']['text']) + 1
            rejected = with_corpus_usage(legacy, root)
            self.assertIs(rejected, legacy)
            self.assertTrue(all('corpus_usage' not in step for step in rejected['steps']))
            del legacy['sampling']
            self.assertIs(with_corpus_usage(legacy, root), legacy)
