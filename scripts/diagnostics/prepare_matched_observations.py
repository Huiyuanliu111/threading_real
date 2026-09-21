"""Match held-out model observations by source trial and exact raw cam1 frame."""
from pathlib import Path
import json
import h5py
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/openpi_mvt_matched_256'
D=ROOT/'pi05_openpi/data/threading_tcp6_nosmooth_30hz'
arp=np.load(ROOT/'artifacts/mvt_cam1_epoch4_full_validation_error/predictions.npz')
raw=json.loads((D/'meta/raw_processing.json').read_text())
val=json.loads((ROOT/'pi05_openpi/outputs/server_training_config.json').read_text())['validation_split']['val_episodes']
rows=[]; ap=[]; at=[]; targets=[]; states=[]
with h5py.File(ROOT.parent/'data/datasets/threading_combined_80_mvt_cam1_7p5hz.h5') as f:
 for ep in sorted(set(val)&set(arp['episodes'].tolist())):
  g=f[f'episode_{ep:06d}']; trial=str(g.attrs['source_trial'])
  assert Path(raw['per_episode'][ep]['source']).as_posix().endswith('/'+trial)
  align=np.load(D/f'meta/alignment/episode_{ep:06d}.npz')
  lookup={int(v):i for i,v in enumerate(align['cam1_frame'])}
  tab=pq.read_table(D/f'data/chunk-000/episode_{ep:06d}.parquet',columns=['action','observation.state'])
  acts=np.array(tab['action'].to_pylist()); obs=np.array(tab['observation.state'].to_pylist())
  candidates=[]
  frames=g['camera_frame_index'][:,0]
  for idx in np.flatnonzero(arp['episodes']==ep):
   af=int(arp['frames'][idx]); camera=int(frames[af]); of=lookup.get(camera)
   if of is None or of+50>len(acts) or af+10>=len(frames): continue
   # Require all ten future interval endpoints to match exactly, no interpolation.
   if not np.array_equal(frames[af:af+11],align['cam1_frame'][of:of+41:4]): continue
   candidates.append((int(idx),af,of,camera))
  assert len(candidates)>=32
  for j in np.linspace(0,len(candidates)-1,32).round().astype(int):
   idx,af,of,camera=candidates[j]
   rows.append(dict(episode=int(ep),source_trial=trial,arp_frame=af,openpi_frame=of,cam1_frame=camera,
                    host_timestamp_ns=int(align['host_timestamp_ns'][of]),
                    duration_40_steps_seconds=float(align['action_dt_seconds'][of:of+40].sum()),arp_prediction_index=idx))
   ap.append(arp['prediction'][idx]);at.append(arp['target'][idx]);targets.append(acts[of:of+50]);states.append(obs[of])
  print(ep,len(candidates),'eligible; selected32')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'matched_observations.json').write_text(json.dumps(rows,indent=2)+'\n')
np.savez_compressed(OUT/'matched_reference.npz',arp_prediction=ap,arp_target=at,raw_target=targets,openpi_state=states)
print('selected',len(rows))
