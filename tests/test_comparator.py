"""Guard against a parity comparator that silently accepts incorrect peak values."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json

spec = importlib.util.spec_from_file_location(
    'parity', Path(__file__).resolve().parent / 'run_parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class ParityComparatorTests(unittest.TestCase):
    def test_changed_peak_values_fail(self):
        reference = {'peaks': [{'a': 1.0, 'b': 2.0, 'c': .9}]}
        for field in ('a', 'b', 'c'):
            with self.subTest(field=field):
                changed = {'peaks': [dict(reference['peaks'][0])]}
                changed['peaks'][0][field] += .001
                with self.assertRaisesRegex(AssertionError, field):
                    parity.compare(changed, reference)

    def test_missing_peak_and_nonfinite_values_fail(self):
        with self.assertRaises(AssertionError):
            parity.compare([], [{'a': 1.0}])
        for invalid in (float('nan'), float('inf'), None):
            with self.subTest(value=invalid), self.assertRaises(AssertionError):
                parity.compare(invalid, 1.0)

    def test_at_least_100_unique_reproducible_cases(self):
        suite = parity.load_cases(Path(__file__).parent / 'inputs')
        self.assertEqual(suite, parity.load_cases(Path(__file__).parent / 'inputs'))
        items = [item for group in suite for item in group['items']]
        self.assertGreaterEqual(len(items), 100)
        self.assertEqual(len(items), len({item['uid'] for item in items}))
        self.assertGreaterEqual(len({(tuple(item['x']), tuple(item['y'])) for item in items}), 100)

    def test_comparison_reads_edited_disk_input(self):
        # The runner must use user-visible files, never silently regenerate signals.
        with tempfile.TemporaryDirectory() as directory:
            for group in parity.load_cases(Path(__file__).parent / 'inputs'):
                for item in group['items']:
                    path = Path(directory) / (item['uid'] + '.json')
                    path.write_text(json.dumps({'config': group['config'], 'input': item}), encoding='utf-8')
            path = Path(directory) / 'test3_1_000.json'
            edited = json.loads(path.read_text(encoding='utf-8'))
            edited['input']['y'][0] = 12345.0
            path.write_text(json.dumps(edited), encoding='utf-8')
            loaded = parity.load_cases(directory)
            actual = next(item for group in loaded for item in group['items'] if item['uid'] == 'test3_1_000')
            self.assertEqual(actual['y'][0], 12345.0)


if __name__ == '__main__':
    unittest.main()
