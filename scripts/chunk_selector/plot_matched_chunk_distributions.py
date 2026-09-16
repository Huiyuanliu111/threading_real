#!/usr/bin/env python3
"""Render comparable Maze/Threading execution plots from their recorded traces."""
from collections import Counter
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASKS = [
    ('Maze', 4, 10, ROOT / 'data/analysis/selector_eval_20260914_142324/logs/maze10_selector_h4.jsonl', 'execution_steps'),
    ('Threading', 8, 20, ROOT / 'data/analysis/selector_eval_matched_20260916/logs/threading20_selector_h8_progress.jsonl', 'executed_steps'),
]


def plot(task, h, H, path, chunk_key):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    sequences = []
    for row in rows:
        if not sequences or row['cycle'] == 1 or row['episode'] != sequences[-1][-1]['episode']:
            sequences.append([])
        sequences[-1].append(row)
    if not rows:
        raise ValueError(f'Empty trace: {path}')
    width = max(row['cycle'] for row in rows)
    matrix = np.full((len(sequences), width), np.nan)
    for index, seq in enumerate(sequences):
        for row in seq:
            chunk = int(row[chunk_key])
            if not h <= chunk <= H or row['cycle'] < 1:
                raise ValueError('Invalid cycle or execution length')
            if np.isfinite(matrix[index, row['cycle'] - 1]):
                raise ValueError('Duplicate cycle within trace')
            matrix[index, row['cycle'] - 1] = chunk
    counts = Counter(int(row[chunk_key]) for row in rows)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10}):
        fig, (line, bar) = plt.subplots(1, 2, figsize=(16, 6.8),
                                       gridspec_kw={'width_ratios': [1.55, 1]})
        fig.subplots_adjust(left=.065, right=.98, bottom=.27, top=.80, wspace=.20)
        colors = plt.get_cmap('tab20')
        for i, seq in enumerate(sequences):
            line.plot([r['cycle'] for r in seq], [r[chunk_key] for r in seq],
                      'o-', color=colors(i % 20), markersize=3.5, linewidth=1.5,
                      alpha=.8, label=f'Trace {i+1}')
        line.set(title=f'Chunk per decision ({len(sequences)} traces)',
                 xlabel='Decision cycle', ylabel='Executed steps',
                 xlim=(.7, width+.3), ylim=(h-.6, H+.6))
        ticks = list(range(1,width+1)) if width <= 12 else [1]+list(range(5,width+1,5))
        line.set_xticks(ticks)
        line.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        line.grid(alpha=.2)
        line.legend(loc='upper center', bbox_to_anchor=(.5,-.18), ncol=5,
                    fontsize=8, frameon=False, columnspacing=1.4)
        xs = np.arange(h,H+1)
        values = [counts[int(x)] for x in xs]
        bars = bar.bar(xs, values, color='tab:blue', width=.78)
        for rect,x,value in zip(bars,xs,values):
            if value:
                bar.text(rect.get_x()+rect.get_width()/2, value+max(values)*.02,
                         f'{value}\n{value/len(rows):.1%}', ha='center', va='bottom', fontsize=9)
        bar.set(title=f'Frequency ({len(rows)} decisions)', xlabel='Executed chunk (steps)',
                ylabel='Decision count', ylim=(0,max(values)*1.22), xticks=xs)
        bar.yaxis.set_major_locator(MaxNLocator(integer=True))
        bar.grid(axis='y',alpha=.2);bar.set_axisbelow(True)
        fig.suptitle(f'{task} H{H} / selector h{h}: execution chunk distribution',y=.96,fontsize=16)
        fig.text(.5,.89,'Decision order only; not normalized TCP arc-length progress',ha='center',fontsize=11)
        output = path.with_name(path.stem+'_chunk_distribution.png')
        fig.savefig(output,dpi=180)
        plt.close(fig)
    print(json.dumps({'task':task,'traces':len(sequences),'decisions':len(rows),
                      'counts':dict(sorted(counts.items())),'output':str(output)}))


if __name__ == '__main__':
    for args in TASKS:
        plot(*args)
