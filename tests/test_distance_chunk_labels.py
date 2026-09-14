from __future__ import annotations

import json

import numpy as np
import pytest

from threading_task.tcp_chunk_labels import distance_chunk_targets


def targets(distances, candidates=(3, 10)):
    return distance_chunk_targets(
        np.column_stack((distances, np.zeros((len(distances), 2)))),
        endpoint_xyz_m=np.zeros(3), fine_radius_m=0.06,
        coarse_radius_m=0.12, candidate_chunks=candidates,
    )


def test_distance_boundaries_and_continuous_expectation():
    probabilities, expected, distances = targets([0, 0.06, 0.09, 0.12, 0.2])
    np.testing.assert_allclose(expected, [3, 3, 6.5, 10, 10])
    np.testing.assert_allclose(probabilities, [[1, 0], [1, 0], [.5, .5], [0, 1], [0, 1]])
    np.testing.assert_allclose(probabilities @ [3, 10], expected)
    np.testing.assert_allclose(distances, [0, .06, .09, .12, .2])


def test_nonuniform_candidates_and_history_independence():
    probabilities, expected, _ = targets([.09, .01, .09, .2], (1, 2, 4, 8, 20))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(probabilities @ [1, 2, 4, 8, 20], expected)
    np.testing.assert_allclose(expected, [10.5, 1, 10.5, 20])
    np.testing.assert_allclose(probabilities[0], targets([.09], (1, 2, 4, 8, 20))[0][0])


@pytest.mark.parametrize('override', [
    {'endpoint_xyz_m': [0, np.nan, 0]}, {'tcp_positions': [[0, 0]]},
    {'fine_radius_m': 0}, {'coarse_radius_m': .06}, {'coarse_radius_m': np.inf},
    {'candidate_chunks': (10, 3)}, {'candidate_chunks': (1, 2.5)},
])
def test_invalid_distance_configuration(override):
    args = dict(tcp_positions=[[0, 0, 0]], endpoint_xyz_m=[0, 0, 0],
                fine_radius_m=.06, coarse_radius_m=.12, candidate_chunks=(3, 10))
    args.update(override)
    with pytest.raises(ValueError):
        distance_chunk_targets(**args)


def test_distance_label_pipeline(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.chunk_selector import label_tcp_chunks as script

    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'data').mkdir()
    (dataset / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 10,
        'features': {'observation.state': {'shape': [8]}, 'action': {'shape': [7]}},
    }))
    # Interleaved episodes and unsorted frames exercise row alignment.
    positions = np.array([.09, .2, .06, .12])
    states = np.zeros((4, 8))
    states[:, 0] = positions
    pq.write_table(pa.table({
        'observation.state': states.tolist(), 'action': np.zeros((4, 7)).tolist(),
        'episode_index': [1, 0, 1, 0], 'frame_index': [1, 0, 0, 1],
        'index': [3, 0, 2, 1],
    }), dataset / 'data/part.parquet')
    class FakeFK:
        def __init__(self, path):
            pass

        def poses(self, joints):
            return joints[:, :3], None

    monkeypatch.setattr(script, 'UrdfForwardKinematics', FakeFK)
    schedule = tmp_path / 'schedule.yaml'
    schedule.write_text('frame: panda_link0\ntcp_frame: panda_hand_tcp\n'
                        'endpoint_xyz_m: [0, 0, 0]\nfine_radius_m: 0.06\n'
                        'fine_steps: 3\ncoarse_steps: 10\n')
    output = tmp_path / 'labels'
    summary = script.create_labels(
        dataset, output, urdf=tmp_path / 'fake.urdf', candidate_chunks=(3, 10),
        smoothing_window=1, label_smoothing_window=1, direction_speed_floor_mps=.005,
        label_method='distance', endpoint_schedule=schedule,
    )
    table = pq.read_table(output / 'labels.parquet')
    np.testing.assert_allclose(table['soft_chunk_size'].to_pylist(), [6.5, 10, 3, 10], atol=1e-6)
    np.testing.assert_allclose(table['endpoint_distance_m'].to_pylist(), positions)
    np.testing.assert_allclose(table['chunk_probability_3'].to_pylist(), [.5, 0, 1, 0], atol=1e-6)
    assert table['index'].to_pylist() == [3, 0, 2, 1]
    assert summary['coarse_radius_m'] == .12
    assert summary['label_rule_version'] == 'tcp_endpoint_distance_soft_v1'
    assert summary['history_latch'] is False
