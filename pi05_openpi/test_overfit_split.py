"""Selection must use original dataset indices and never masquerade as held-out evaluation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validation import episode_split, split_from_settings


class OverfitSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'meta').mkdir()
        self.rows = [{'episode_index': i, 'length': i + 2} for i in range(80)]
        self.write_rows()

    def write_rows(self):
        (self.root / 'meta/episodes.jsonl').write_text('\n'.join(map(json.dumps, self.rows)))

    def test_ten_seeded_whole_episodes_same_for_training_and_evaluation(self):
        split, indices = episode_split(self.root, .1, 42, 10)
        expected_ids = sorted(np.random.default_rng(42).permutation(80)[:10].tolist())
        self.assertEqual(split['train_episodes'], expected_ids)
        self.assertEqual(split['val_episodes'], expected_ids)
        self.assertEqual(split['evaluation_scope'], 'train_reconstruction')
        self.assertEqual(indices['train'], indices['val'])
        self.assertIsNot(indices['train'], indices['val'])
        expected_frames = []
        offset = 0
        for row in self.rows:
            if row['episode_index'] in expected_ids:
                expected_frames.extend(range(offset, offset + row['length']))
            offset += row['length']
        self.assertEqual(indices['train'], expected_frames)
        self.assertEqual(split['train_frames'], len(expected_frames))
        self.assertTrue(all(0 <= i < offset for i in indices['train']))
        self.assertEqual((split, indices), episode_split(self.root, .1, 42, 10))

    def test_normal_split_remains_disjoint_and_exhaustive(self):
        split, indices = episode_split(self.root, .1, 42)
        self.assertEqual(len(split['train_episodes']), 72)
        self.assertEqual(len(split['val_episodes']), 8)
        self.assertFalse(set(indices['train']) & set(indices['val']))
        self.assertEqual(sorted(indices['train'] + indices['val']),
                         list(range(sum(row['length'] for row in self.rows))))
        self.assertEqual((split, indices), episode_split(self.root, .1, 42, 0))

    def test_all_and_single_episode_boundaries(self):
        _, indices = episode_split(self.root, .1, 42, 80)
        self.assertEqual(indices['train'], list(range(sum(row['length'] for row in self.rows))))
        self.rows = [{'episode_index': 0, 'length': 3}]
        self.write_rows()
        split, indices = episode_split(self.root, .1, 42, 1)
        self.assertEqual(split['train_episodes'], [0])
        self.assertEqual(indices, {'train': [0, 1, 2], 'val': [0, 1, 2]})

    def test_invalid_counts(self):
        for count in (-1, 81, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                episode_split(self.root, .1, 42, count)

    def test_invalid_episode_metadata_rejected(self):
        self.rows[0]['episode_index'] = 99
        self.write_rows()
        with self.assertRaises(ValueError):
            episode_split(self.root, .1, 42, 10)

    def test_settings_route_overfit_and_legacy_modes(self):
        settings = {'dataset_root': self.root, 'val_fraction': .1, 'seed': 42}
        self.assertEqual(split_from_settings(settings), episode_split(self.root, .1, 42))
        settings['overfit_episodes'] = 10
        self.assertEqual(split_from_settings(settings), episode_split(self.root, .1, 42, 10))
        from run import parser
        self.assertEqual(parser().parse_args(['train']).overfit_episodes, 0)
        self.assertEqual(parser().parse_args(['train', '--overfit-episodes', '10']).overfit_episodes, 10)


if __name__ == '__main__':
    unittest.main()
