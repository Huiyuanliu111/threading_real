from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from chunk_selector.spatial_rule import SpatialRule, trajectory_progress
from chunk_selector.chunk_dataset import ChunkFeatureWriter, episode_split_indices
from chunk_selector.chunk_selector import ChunkSelector, ChunkSelectorConfig
from chunk_selector.mvt_features import selector_tokens, predict_with_selector


def line():
    return np.column_stack((np.linspace(0, 1, 101), np.zeros((101, 2))))


@pytest.mark.parametrize('task,split,fine_start', [('threading', .8, False), ('threading', .7, False), ('maze', .3, True)])
def test_fit_spatial_rule_and_soft_boundary(task, split, fine_start):
    rule = SpatialRule.fit([line(), line()], task=task, split_progress=split)
    probabilities, expected, _ = rule.label(line())
    assert expected[0] == (4 if fine_start else 10)
    assert expected[-1] == (10 if fine_start else 4)
    np.testing.assert_allclose(probabilities[round(split * 100)], [.5, .5], atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(1), 1)
    np.testing.assert_allclose(probabilities @ [4, 10], expected)
    np.testing.assert_allclose(rule.boundary_radius_m, split if fine_start else 1 - split)


def test_arc_length_node_ignores_stationary_dwell():
    xyz = line()
    paused = np.concatenate((xyz[:10], np.repeat(xyz[10:11], 200, axis=0), xyz[10:]))
    a = SpatialRule.fit([xyz], task='threading')
    b = SpatialRule.fit([paused], task='threading')
    assert a == b
    assert trajectory_progress(paused, 'frames')[10] != trajectory_progress(paused)[10]
    with pytest.raises(ValueError, match='stationary'):
        trajectory_progress(np.zeros((5, 3)))


def test_same_position_same_probability_after_reentry_and_all_integer_outputs():
    rule = SpatialRule.fit([line()], task='threading')
    probabilities, expected, _ = rule.label(np.array([[.9, 0, 0], [.1, 0, 0], [.9, 0, 0]]))
    np.testing.assert_array_equal(probabilities[0], probabilities[-1])
    radius = rule.boundary_radius_m
    distances = np.linspace(radius - .01, radius + .01, 701)
    xyz = np.column_stack((1 - distances, np.zeros((len(distances), 2))))
    expected = rule.label(xyz)[1]
    assert set(np.floor(expected + .5)) == set(range(4, 11))


def test_spatial_manifest_controls_training_split(tmp_path):
    path = tmp_path / 'features.h5'
    with ChunkFeatureWriter(path, feature_shape=(2, 4), candidate_chunks=(3, 10),
                            metadata={'label_source': 'spatial_rule', 'fit_episode_ids': ['b'],
                                      'validation_episode_ids': ['a']}) as writer:
        writer.append(np.zeros((3, 2, 4)), [0, 1, 0], episode_ids=['a', 'b', 'b'], decision_steps=[0, 0, 1])
    train, validation = episode_split_indices(path, val_ratio=.9, seed=999)
    np.testing.assert_array_equal(train, [1, 2])
    np.testing.assert_array_equal(validation, [0])


def test_pooling_preserves_view_order_and_ids():
    tokens = torch.cat((torch.ones(1, 16, 3), torch.full((1, 16, 3), 2)), dim=1)
    pooled, ids = selector_tokens(tokens, side=4, pool_grid=2)
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([1.] * 4 + [2.] * 4))
    assert ids['camera_ids'].tolist() == [0] * 4 + [1] * 4
    assert ids['spatial_ids'].tolist() == [0, 1, 2, 3] * 2


def test_label_real_format_preserves_video_rows_and_train_only_fit(tmp_path):
    from scripts.chunk_selector.label_spatial import create_labels
    import pyarrow.parquet as pq
    source = tmp_path / 'maze.h5'
    keys = [f'episode_{i:06d}' for i in range(5)]
    validation = np.random.default_rng(42).permutation(keys)[0]
    with h5py.File(source, 'w') as f:
        f.attrs['format'] = 'maze-mvt-pointcloud-v1'
        f.attrs['fixed_z_m'] = .1
        for key in keys:
            # A held-out episode far away must not move the fitted fine center.
            xy = line()[:, :2] + (10 if key == validation else 0)
            f.create_group(key).create_dataset('tcp_xy', data=xy)
    output = tmp_path / 'labels'
    summary = create_labels(source, output, task='maze')
    np.testing.assert_allclose(summary['rule']['center_xyz_m'], [0, 0, .1])
    table = pq.read_table(output / 'labels.parquet').to_pandas()
    assert len(table) == 505
    assert table[table.episode_key == validation].split.unique().tolist() == ['validation']
    assert summary['validation_episode_ids'] == [validation]
    assert validation not in summary['fit_episode_ids']


def test_real_mvt_selector_reuses_encoder(monkeypatch):
    from threading_task.mvt_arp_policy import ThreadingMVTARPPolicy
    policy = ThreadingMVTARPPolicy(horizon=10, n_action_steps=2, image_size=28,
                                   patch_size=14, hidden_dim=32, vit_mlp_dim=64,
                                   vit_depth=1, arp_depth=1, dropout=0, predict_gripper=False)
    control = torch.tensor([[[[.4, 0, 0], [.44, 0, 0], [.4, .04, 0]]]])
    obs = dict(points=control, colors=torch.ones_like(control) * .5,
               valid_points=torch.tensor([[3]]), agent_pos=torch.zeros(1, 1, 8),
               control_points=control)
    selector = ChunkSelector(ChunkSelectorConfig(input_dim=32, candidate_chunks=(3, 10),
        d_model=16, num_layers=1, n_heads=4, dim_feedforward=32, dropout=0,
        max_tokens=8, num_cameras=2, max_spatial_positions=4, selection_mode='expected'))
    for parameter in selector.head.parameters():
        parameter.data.zero_()
    calls = []
    original = policy._visual
    def visual(obs):
        calls.append(1)
        return original(obs)
    monkeypatch.setattr(policy, '_visual', visual)
    prediction, selection = predict_with_selector(policy, selector, obs)
    assert len(calls) == 1
    assert selection.chunk_sizes.item() == 7
    assert selection.continuous_chunk_sizes.item() == 6.5
    assert prediction['action_pred'].shape == (1, 10, 7)


def test_feature_cache_and_training_cli_use_spatial_probabilities(tmp_path, monkeypatch):
    from scripts.chunk_selector import extract_mvt_features as extraction
    from scripts.chunk_selector.label_spatial import create_labels
    from scripts.chunk_selector import train
    from chunk_selector.chunk_dataset import ChunkFeatureDataset

    source = tmp_path / 'maze.h5'
    with h5py.File(source, 'w') as f:
        f.attrs.update(format='maze-mvt-pointcloud-v1', fixed_z_m=.1)
        for index in range(2):
            group = f.create_group(f'episode_{index:06d}')
            xy = line()[::10, :2]
            group.create_dataset('tcp_xy', data=xy)
            group.create_dataset('points', data=np.ones((11, 3, 3), np.float32))
            group.create_dataset('colors', data=np.full((11, 3, 3), 128, np.uint8))
            group.create_dataset('valid_points', data=np.full(11, 3))
    labels_dir = tmp_path / 'labels'
    create_labels(source, labels_dir, task='maze')

    class FrozenPolicy:
        horizon, image_size, patch_size = 10, 28, 14
        def _visual(self, obs):
            assert not torch.is_grad_enabled()
            assert set(obs) == {'points', 'colors', 'valid_points'}
            return None, torch.ones(len(obs['points']), 8, 4), None

    monkeypatch.setattr(extraction, 'load_mvt_policy', lambda *args: FrozenPolicy())
    cache = tmp_path / 'features.h5'
    relocated_source = tmp_path / 'relocated_maze.h5'
    source.rename(relocated_source)
    assert extraction.extract(tmp_path / 'checkpoint', labels_dir, cache, batch_size=4,
                              dataset_path=relocated_source) == 22
    dataset = ChunkFeatureDataset(cache)
    assert dataset[0]['has_target_probabilities']
    assert dataset.feature_shape == (8, 4)
    assert dataset.metadata['label_source'] == 'spatial_rule'
    assert dataset.metadata['source_dataset'] == str(relocated_source)
    dataset.close()
    output = tmp_path / 'model'
    monkeypatch.setattr('sys.argv', ['train', str(cache), '--output-dir', str(output),
        '--device', 'cpu', '--epochs', '1', '--batch-size', '8', '--d-model', '16',
        '--dim-feedforward', '32', '--num-layers', '1'])
    assert train.main() == 0
    config = json.loads((output / 'chunk_selector_config.json').read_text())
    assert config['selection_mode'] == 'expected'
    assert config['metadata']['training_targets'] == 'explicit_probabilities'
    assert config['candidate_chunks'] == [4, 10]


def test_report_uses_recorded_camera_frame_indices(tmp_path):
    import cv2
    from scripts.chunk_selector.label_spatial import create_labels
    from scripts.chunk_selector.visualize_spatial import render_report
    source = tmp_path / 'maze.h5'
    raw = tmp_path / 'raw'
    raw.mkdir()
    with h5py.File(source, 'w') as f:
        f.attrs.update(format='maze-mvt-pointcloud-v1', fixed_z_m=.1, raw_root=str(raw),
                       bounds_m=[.15, -.4, -.15, .75, .3, .5])
        for index in range(2):
            key = f'episode_{index:06d}'
            trial = raw / key
            trial.mkdir()
            video = cv2.VideoWriter(str(trial / 'cam3.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
            assert video.isOpened()
            for frame in range(22):
                video.write(np.full((32, 32, 3), frame * 10, np.uint8))
            video.release()
            group = f.create_group(key)
            group.attrs['source_trial'] = key
            group.create_dataset('tcp_xy', data=line()[::10, :2])
            group.create_dataset('camera_frame_index', data=np.arange(11)[:, None] * 2)
            group.create_dataset('points', data=np.tile([[[.4, 0, .1]]], (11, 1, 1)))
            group.create_dataset('colors', data=np.full((11, 1, 3), 128, np.uint8))
            group.create_dataset('valid_points', data=np.ones(11, int))
    labels_dir = tmp_path / 'labels'
    create_labels(source, labels_dir, task='maze')
    output = tmp_path / 'report'
    report = render_report(labels_dir, output)
    assert report['missing_frames'] == 0
    for frame in report['frames']:
        assert frame['video_frame'] == frame['dataset_frame'] * 2
        assert (output / frame['image']).is_file()
        image = cv2.imread(str(output / frame['image']))
        assert abs(float(image.mean()) - frame['video_frame'] * 10) < 8
    assert (output / 'index.html').is_file()
    assert (output / 'distribution.png').is_file()

    from scripts.chunk_selector.export_spatial_frames import export_frames
    relocated = tmp_path / 'moved.h5'
    source.rename(relocated)
    for kind in ('video', 'pointcloud'):
        target = tmp_path / kind
        exported = export_frames(labels_dir, target, source_kind=kind, dataset=relocated)
        assert exported['missing_frames'] == 0
        assert exported['source_dataset'] == str(relocated)
        assert len(exported['overview_images']) == 2
        for frame in exported['frames']:
            path = target / frame['image']
            assert path.suffix == '.png'
            assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
            if kind == 'video':
                assert frame['video_frame'] == frame['dataset_frame'] * 2
                assert abs(cv2.imread(str(path)).mean() - frame['video_frame'] * 10) < 8
            else:
                assert frame['source_kind'] == 'pointcloud'
                assert cv2.imread(str(path)).shape == (468, 840, 3)
