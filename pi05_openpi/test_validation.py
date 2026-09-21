import json
from pathlib import Path
import tempfile
import unittest

from validation import episode_split, is_improvement


class ValidationTests(unittest.TestCase):
    def test_whole_episodes_deterministic_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / 'meta'
            meta.mkdir()
            rows = [{'episode_index': i, 'length': i + 2} for i in range(10)]
            (meta / 'episodes.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            manifest, indices = episode_split(directory, .1, 42)
            self.assertEqual((manifest, indices), episode_split(directory, .1, 42))
            self.assertEqual(len(manifest['val_episodes']), 1)
            self.assertFalse(set(indices['train']) & set(indices['val']))
            self.assertEqual(sorted(indices['train'] + indices['val']), list(range(65)))
            offset = 0
            for row in rows:
                part = 'val' if row['episode_index'] in manifest['val_episodes'] else 'train'
                self.assertTrue(set(range(offset, offset + row['length'])).issubset(indices[part]))
                offset += row['length']

    def test_only_finite_strict_improvements(self):
        best = float('inf')
        kept = []
        for step, loss in enumerate([3., 4., float('nan'), 2., 2., float('inf')], 1):
            if is_improvement(loss, best):
                kept.append(step)
                best = loss
        self.assertEqual(kept, [1, 4])

    def test_defaults(self):
        from run import parser
        args = parser().parse_args(['train'])
        self.assertEqual((args.horizon, args.steps, args.save_interval), (50, 10000, 2000))
        self.assertEqual(args.val_fraction, .1)


if __name__ == '__main__':
    unittest.main()
