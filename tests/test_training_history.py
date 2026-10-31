import unittest
from training_history import loss_trend


class TrainingHistoryTests(unittest.TestCase):
    def test_long_run_keeps_whole_range_and_averages_batch_noise(self):
        history = [{'step': step, 'loss': 2.0 + (0.5 if step % 2 else -0.5)} for step in range(1, 5018)]
        trend = loss_trend(history)
        self.assertLessEqual(len(trend), 300)
        self.assertEqual(trend[0], history[0])
        self.assertEqual(trend[-1], history[-1])
        self.assertTrue(all(a['step'] < b['step'] for a, b in zip(trend, trend[1:])))
        self.assertTrue(all(abs(point['loss'] - 2.0) < .04 for point in trend[1:-1]))
        self.assertLess(trend[1]['step'], 20)

    def test_short_and_sparse_histories_keep_real_measurement_positions(self):
        history = [{'step': n, 'loss': 3.0} for n in (1, 100, 5000)]
        self.assertEqual(loss_trend(history), history)
        self.assertEqual(loss_trend([]), [])
