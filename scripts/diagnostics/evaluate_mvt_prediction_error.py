#!/usr/bin/env python3
"""Offline physical-unit prediction errors for MVT/PlanARP checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.arp.policy_loader import load_policy


def stats(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'median':float(np.median(x)),
            'p90':float(np.quantile(x,.9)),'max':float(x.max())}


def metrics(pred,target):
    error=np.linalg.norm(pred[...,:3]-target[...,:3],axis=-1)*1000
    return {'first_action_translation_error_mm':stats(error[:,0]),
            'per_action_translation_error_mm':stats(error),
            'chunk_translation_rms_mm':stats(np.sqrt(np.mean(error**2,axis=1))),
            'summed_translation_error_mm':stats(np.linalg.norm((pred[...,:3]-target[...,:3]).sum(axis=1),axis=-1)*1000),
            'per_axis_mae_mm':(np.abs(pred[...,:3]-target[...,:3]).mean(axis=(0,1))*1000).tolist(),
            'per_action_rotation_vector_error_deg':stats(np.linalg.norm(pred[...,3:6]-target[...,3:6],axis=-1)*180/np.pi),
            'per_future_step_translation_error_mm':[stats(error[:,i]) for i in range(error.shape[1])]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--all-windows',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=OmegaConf.create(OmegaConf.to_container(payload['cfg'].task.dataset,resolve=True))
    cfg.dataset_path=str(args.dataset.resolve())
    cfg.urdf_path=str(ROOT.parent/'remote_controller/src/remote_controller/assets/panda/panda_arm.urdf')
    del payload
    train=hydra.utils.instantiate(cfg);data=train.get_validation_dataset()
    indices=[]
    for ep in data.selected_episodes:
        candidates=[i for i,(e,_) in enumerate(data.sample_indices) if e==ep]
        if args.all_windows: indices.extend(candidates)
        else:
            for phase in (.1,.5,.85):
                frame=min(round(phase*(data.lengths[ep]-1)),data.lengths[ep]-data.horizon)
                indices.append(min(candidates,key=lambda i:abs(data.sample_indices[i][1]-frame)))
    indices=sorted(set(indices))
    policy=load_policy(args.checkpoint,device=args.device,weights='model')
    args.output.mkdir(parents=True)
    pred=[];target=[];origins=[];target_origins=[];pred_origins=[]
    started=time.monotonic()
    with torch.inference_mode():
        for batch in DataLoader(Subset(data,indices),batch_size=args.batch_size,shuffle=False,num_workers=0):
            obs={k:v.to(args.device) for k,v in batch['obs'].items()}
            result=policy.predict_action(obs,requested_steps=policy.horizon,prediction_mode='full_then_truncate',sample=False)
            a=result['action_pred'].cpu().numpy();y=batch['action'].numpy()
            assert a.shape==y.shape and np.isfinite(a).all()
            assert not batch['action_is_pad'].any()
            pred.append(a);target.append(y)
            origins.append(batch['obs']['control_points'][:,-1,0,:].numpy())
            target_origins.append(batch['target_control_points'][:,:,0,:].numpy())
            pred_origins.append(result['target_control_points'][:,:,0,:].cpu().numpy())
            print(f'{sum(len(v) for v in pred)}/{len(indices)} observations, {time.monotonic()-started:.1f}s',flush=True)
    pred=np.concatenate(pred).astype(np.float64);target=np.concatenate(target).astype(np.float64)
    origins=np.concatenate(origins);target_origins=np.concatenate(target_origins);pred_origins=np.concatenate(pred_origins)
    mean_action=np.concatenate([train.actions[e] for e in train.selected_episodes]).mean(axis=0)
    report={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'dataset':str(args.dataset.resolve()),'weights':'model','sampling':'deterministic sample=False',
            'observations':len(pred),'total_validation_windows':len(data),'episodes':data.selected_episodes,
            'horizon':policy.horizon,'all_windows':args.all_windows,'phases':None if args.all_windows else [.1,.5,.85],
            'elapsed_seconds':time.monotonic()-started,'prediction':metrics(pred,target),
            'zero_baseline':metrics(np.zeros_like(pred),target),
            'train_mean_baseline':metrics(np.broadcast_to(mean_action,pred.shape),target),
            'label_action_magnitude_mm':stats(np.linalg.norm(target[...,:3],axis=-1)*1000),
            'target_pose_vs_cumulative_action_consistency_mm':stats(np.linalg.norm(origins[:,None]+target[...,:3].cumsum(1)-target_origins,axis=-1)*1000),
            'predicted_absolute_target_error_mm':stats(np.linalg.norm(pred_origins-target_origins,axis=-1)*1000),
            'notes':['Offline only; no robot commands, augmentation or retraining.',
                     'Complete action chunks only; all selected episodes are held out by checkpoint split.',
                     'Frequency, point-cloud preprocessing, horizon and samples differ from OpenPI; not a controlled head-to-head comparison.']}
    with h5py.File(args.dataset,'r') as f: report['fps']=float(f.attrs['fps'])
    pairs=np.asarray([data.sample_indices[i] for i in indices])
    np.savez_compressed(args.output/'predictions.npz',prediction=pred,target=target,episodes=pairs[:,0],frames=pairs[:,1],origins=origins,target_origins=target_origins,predicted_origins=pred_origins)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
