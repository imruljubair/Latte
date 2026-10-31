import tempfile
import unittest
from pathlib import Path

from latte.config import load_config, write_architecture_config


class ArchitectureCodeTests(unittest.TestCase):
    def test_codes_preserve_training_and_set_dimensions(self):
        source = Path(__file__).resolve().parents[1] / 'configs/default.toml'
        baseline = load_config(source)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'architecture.toml'
            for code, expected in [('34324', (3, 4, 4, 3, 2, 4)), ('345324', (3, 4, 5, 3, 2, 4))]:
                write_architecture_config(source, destination, code)
                config = load_config(destination)
                self.assertEqual(tuple(getattr(config.model, key) for key in
                    ('model_dim', 'qk_dim', 'value_dim', 'num_heads', 'num_layers', 'ffn_dim')), expected)
                self.assertEqual(config.training, baseline.training)
                self.assertEqual(config.data, baseline.data)
            for code in ('34024', '1234', 'abcde'):
                with self.assertRaises(ValueError):
                    write_architecture_config(source, destination, code)
