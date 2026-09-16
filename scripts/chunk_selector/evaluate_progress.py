#!/usr/bin/env python3
"""Evaluate a selector on held-out episodes, plotted against measured TCP arc length."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from chunk_selector.chunk_dataset import ChunkFeatureDataset, episode_split_indices
from chunk_selector.chunk_selector import ChunkSelector
from scripts.chunk_selector.train import _selector_inputs


def evaluate(dataset, model_dir, labels_dir, *, device='cuda:0', batch_size=4):
    dataset, model_dir, labels_dir=map(Path,(dataset,model_dir,labels_dir))
    _, indices=episode_split_indices(dataset)
    model=ChunkSelector.from_pretrained(model_dir,device=device).eval()
    loader=DataLoader(ChunkFeatureDataset(dataset,indices),batch_size=batch_size,shuffle=False)
    output=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['features'].to(device),**_selector_inputs(batch,model.config,torch.device(device)))
            output.append(torch.softmax(logits,dim=-1).cpu().numpy())
    probabilities=np.concatenate(output)
    with h5py.File(dataset,'r') as f:
        episodes=f['episode_ids'].asstr()[:][indices]
        steps=f['decision_steps'][:][indices]
    labels=pq.read_table(labels_dir/'labels.parquet').to_pandas().set_index(['episode_key','frame_index'])
    rows=labels.loc[list(zip(episodes,steps))].reset_index()
    h,H=model.candidate_chunks
    rows['predicted_p_fine']=probabilities[:,0]
    rows['predicted_chunk']=probabilities@np.array([h,H])
    rows['predicted_execution_steps']=np.floor(rows.predicted_chunk+.5).astype(int)
    rule=json.loads((labels_dir/'summary.json').read_text())['rule']
    target=rows[f'chunk_probability_{h}'].to_numpy()
    metrics={'validation_frames':len(rows),'validation_episodes':len(set(episodes)),
             'probability_mae':float(np.abs(probabilities[:,0]-target).mean()),
             'chunk_mae':float(np.abs(rows.predicted_chunk-rows.soft_chunk_size).mean()),
             'hard_accuracy':float(((probabilities[:,0]>=.5)==(target>=.5)).mean())}
    for name,mask in [('pure_coarse',target==0),('pure_fine',target==1)]:
        metrics[name]={'frames':int(mask.sum()),'mean_predicted_p_fine':float(probabilities[mask,0].mean()) if mask.any() else None,
                       'misclassification_rate':float(((probabilities[mask,0]>=.5)!=(target[mask]>=.5)).mean()) if mask.any() else None}
    rows.to_csv(model_dir/'validation_predictions.csv',index=False)
    (model_dir/'validation_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for _,g in rows.groupby('episode_key'):
        ax.plot(g.trajectory_progress,g.predicted_chunk,alpha=.4,lw=1)
    ordered=rows.sort_values('trajectory_progress')
    ax.plot(ordered.trajectory_progress,ordered.soft_chunk_size,'k--',label='Target',lw=2)
    ax.set(xlabel='Normalized TCP arc length',ylabel='Expected chunk',title=f"{rule['task']}: held-out episode predictions",xlim=(0,1))
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(model_dir/'validation_chunks_by_arc_length.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path);p.add_argument('model_dir',type=Path);p.add_argument('labels_dir',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=4)
    evaluate(**vars(p.parse_args()))
