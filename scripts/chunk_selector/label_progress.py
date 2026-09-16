#!/usr/bin/env python3
"""Label each complete episode by TCP arc length, and audit previous spatial labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from chunk_selector.mvt_data import read_trajectories
from chunk_selector.progress_rule import ProgressRule


def create_labels(dataset, output, *, task, h, H, split_progress=None,
                  transition_width=.1, urdf=None, previous_labels=None, seed=42, val_ratio=.2):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between zero and one')
    rule = ProgressRule(task, split_progress if split_progress is not None else
                        (.5 if task == 'threading' else .3), h, H, transition_width)
    trajectories = read_trajectories(dataset, task=task, urdf=urdf)
    keys = sorted(trajectories)
    if len(keys) < 2:
        raise ValueError('at least two episodes required')
    n_val = min(max(1, round(len(keys) * val_ratio)), len(keys) - 1)
    validation = set(np.random.default_rng(seed).permutation(keys)[:n_val])
    train = [k for k in keys if k not in validation]
    old = None
    if previous_labels is not None:
        previous_labels = Path(previous_labels).resolve()
        old_summary = json.loads((previous_labels / 'summary.json').read_text())
        if old_summary['rule']['task'] != task:
            raise ValueError('previous labels belong to another task')
        if set(old_summary['fit_episode_ids']) != set(train) or set(old_summary['validation_episode_ids']) != validation:
            raise ValueError('previous and new episode splits differ')
        old = pq.read_table(previous_labels / 'labels.parquet').to_pandas()
        if old.duplicated(['episode_key', 'frame_index']).any():
            raise ValueError('duplicate previous frame identity')
        old = old.set_index(['episode_key', 'frame_index'])
    tables, nodes = [], []
    offset = 0
    for episode_index, key in enumerate(keys):
        xyz = trajectories[key]
        probs, expected, progress = rule.label(xyz)
        fine = probs[:, 0]
        data = dict(index=np.arange(offset, offset + len(xyz)),
                    episode_index=np.full(len(xyz), episode_index), episode_key=[key]*len(xyz),
                    frame_index=np.arange(len(xyz)), split=['validation' if key in validation else 'train']*len(xyz),
                    trajectory_progress=progress, tcp_x_m=xyz[:, 0], tcp_y_m=xyz[:, 1], tcp_z_m=xyz[:, 2],
                    chunk_size=np.where(fine >= .5, h, H), soft_chunk_size=expected,
                    execution_steps=np.floor(expected+.5).astype(np.int64),
                    **{f'chunk_probability_{h}': fine, f'chunk_probability_{H}': probs[:, 1]})
        if old is not None:
            rows = old.loc[[(key, i) for i in range(len(xyz))]]
            np.testing.assert_allclose(rows[['tcp_x_m','tcp_y_m','tcp_z_m']], xyz, atol=1e-7)
            data['previous_p_fine'] = rows[f"chunk_probability_{old_summary['candidate_chunks'][0]}"].to_numpy()
        tables.append(pa.table(data))
        nodes.append(dict(episode_key=key, total_path_length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
                          frame_index=int(np.argmin(abs(progress-rule.split_progress))),
                          boundary_crossing_frames=(np.flatnonzero(np.diff(fine >= .5))+1).tolist(),
                          fine_frame_fraction=float((fine >= .5).mean())))
        offset += len(xyz)
    table = pa.concat_tables(tables)
    if old is not None and len(old) != offset:
        raise ValueError('previous labels have extra frames')
    summary = dict(label_source='progress_rule', rule=rule.to_dict(), source_dataset=str(dataset),
                   frame='panda_link0', tcp_frame='panda_hand_tcp',
                   urdf=str(Path(urdf).resolve()) if urdf else None, candidate_chunks=[h,H],
                   fit_episode_ids=train, validation_episode_ids=sorted(validation),
                   seed=seed, val_ratio=val_ratio, nodes=nodes, total_frames=offset,
                   previous_labels=str(previous_labels) if previous_labels else None,
                   online_progress_input=False)
    frame = table.to_pandas()
    if old is not None:
        summary['previous_hard_label_disagreement_fraction'] = float(
            ((frame.previous_p_fine >= .5) != (frame[f'chunk_probability_{h}'] >= .5)).mean())
    output.mkdir(parents=True)
    pq.write_table(table, output/'labels.parquet', compression='zstd')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_labels(frame, rule, output)
    return summary


def plot_labels(frame, rule, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4), constrained_layout=True)
    for _, rows in frame.groupby('episode_key'):
        axes[0].plot(rows.trajectory_progress, rows[f'chunk_probability_{rule.h}'], alpha=.15, color='tab:blue')
        axes[1].plot(rows.trajectory_progress, rows.soft_chunk_size, alpha=.15, color='tab:blue')
    if 'previous_p_fine' in frame:
        axes[0].scatter(frame.trajectory_progress, frame.previous_p_fine, s=2, alpha=.08,
                        color='tab:orange', label='Previous spatial labels')
        axes[0].plot([],[],color='tab:blue',label='New arc-length labels')
        axes[0].legend()
    for ax in axes:
        ax.set_xlabel('Normalized TCP arc length'); ax.set_xlim(0,1)
        ax.axvline(rule.split_progress, color='gray', linestyle='--'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Fine probability'); axes[1].set_ylabel('Expected chunk')
    fig.suptitle(f'{rule.task}: {len(frame.episode_key.unique())} episodes, split={rule.split_progress:.0%}')
    fig.savefig(output/'labels_by_arc_length.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=['threading','maze'],required=True)
    p.add_argument('--h',type=int,required=True); p.add_argument('--H',type=int,required=True)
    p.add_argument('--split-progress',type=float); p.add_argument('--transition-width',type=float,default=.1)
    p.add_argument('--previous-labels',type=Path)
    p.add_argument('--urdf',type=Path,default=ROOT.parent/'remote_controller/src/remote_controller/assets/panda/panda_arm.urdf')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--val-ratio',type=float,default=.2)
    s=create_labels(**vars(p.parse_args()))
    print(json.dumps({k:s[k] for k in ['rule','total_frames','previous_hard_label_disagreement_fraction'] if k in s},indent=2))
