"""Paired local-vs-SSH-server latency benchmark; recorded data, no robot/camera I/O."""
from pathlib import Path
import json,time,io,contextlib
from types import SimpleNamespace
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from deploy import LocalBackend,LocalPolicy
from deploy_remote import RemoteBackend
ROOT=Path(__file__).resolve().parent

def stats(values):
 v=np.asarray(values)*1000
 return dict(n=len(v),mean_ms=float(v.mean()),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),min_ms=float(v.min()),max_ms=float(v.max()))

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--remote-only',action='store_true');args=parser.parse_args()
 manifest=json.loads((ROOT.parent/'artifacts/openpi_mvt_matched_256/matched_observations.json').read_text())
 chosen=[manifest[i] for i in np.linspace(0,len(manifest)-1,100).round().astype(int)]
 cache={};samples=[]
 for r in chosen:
  ep=r['episode']
  if ep not in cache:
   cache[ep]=pq.read_table(ROOT/f'data/threading_tcp6_nosmooth_30hz/data/chunk-000/episode_{ep:06d}.parquet',columns=['observation.state','observation.images.exterior_image_1_left','observation.images.exterior_image_2_right'])
  row=cache[ep].slice(r['openpi_frame'],1).to_pylist()[0]
  obs={'agent_pos':torch.from_numpy(np.array(row['observation.state'],np.float32))[None,None]}
  for dst,key in [('frontview','observation.images.exterior_image_1_left'),('sideview','observation.images.exterior_image_2_right')]:
   a=np.array(Image.open(io.BytesIO(row[key]['bytes'])));obs[dst]=torch.from_numpy(a.transpose(2,0,1).copy())[None,None]
  samples.append(obs)
 checkpoint=ROOT/'checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000'
 started=time.perf_counter();local=None if args.remote_only else LocalBackend(checkpoint,ROOT/'deployment_config.json');load=time.perf_counter()-started
 remote=RemoteBackend(SimpleNamespace(policy_host='127.0.0.1',policy_port=18000,inference_timeout=5,checkpoint=Path('/home/huiyuan/threading_real/pi05_openpi/checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000')))
 # Wrap infer solely to capture server timing; both paths still pass through same action adapter.
 timings=[];original=remote.infer
 def capture(obs):
  result=original(obs);timings.append(float(result['server_timing']['infer_ms'])/1000);return result
 remote.infer=capture
 policies={'remote':LocalPolicy(remote)}
 if local is not None:policies={'local':LocalPolicy(local),**policies}
 warm={}
 for name,p in policies.items():
  warm[name]=[]
  for _ in range(5):
   s=time.perf_counter();a=p.predict_action(samples[0])['action'].numpy();warm[name].append(time.perf_counter()-s)
   assert a.shape==(1,50,7) and np.isfinite(a).all()
 print('WARMUP COMPLETE',warm,flush=True)
 timings.clear();records=[]
 try:
  for i,obs in enumerate(samples):
   row=dict(episode=chosen[i]['episode'],frame=chosen[i]['openpi_frame'])
   for name in ([k for k in ('local','remote') if k in policies] if i%2==0 else [k for k in ('remote','local') if k in policies]):
    with contextlib.redirect_stdout(io.StringIO()):
     s=time.perf_counter();a=policies[name].predict_action(obs)['action'].numpy();elapsed=time.perf_counter()-s
    assert a.shape==(1,50,7) and np.isfinite(a).all() and not a[:,:,6].any()
    row[name+'_seconds']=elapsed
   row['server_seconds']=timings[-1];records.append(row)
   if (i+1)%20==0:print(f'{i+1}/100 pairs complete',flush=True)
 finally:remote.close()
 report=dict(checkpoint=str(checkpoint),observations=100,horizon=50,denoising_steps=10,local_load_seconds=load,warmup_seconds=warm,
  local=None if args.remote_only else stats([r['local_seconds'] for r in records]),remote=stats([r['remote_seconds'] for r in records]),
  server_infer=stats([r['server_seconds'] for r in records]),
  remote_overhead=stats([r['remote_seconds']-r['server_seconds'] for r in records]),
  remote_faster_pairs=None if args.remote_only else sum(r['remote_seconds']<r['local_seconds'] for r in records),records=records,
  notes=[('100 remote-only observations across8 heldout episodes;5 warmup calls.' if args.remote_only else '100 exact paired recorded observations across8 heldout episodes; alternating local/remote order;5 warmup calls each.'),
         'End-to-end adapter wall time includes array conversion and output readiness. Remote includes SSH loopback forwarding, serialization, network and server inference.',
         'Image decode excluded equally; no camera capture or follower control. Ordinary desktop GPU workload not disabled.',
         'Same checkpoint, bf16 loading, full H50 and10 denoising steps; random flow draws not fixed, no output ensembling.'])
 (ROOT/('outputs/latency_remote_only.json' if args.remote_only else 'outputs/latency_comparison.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
