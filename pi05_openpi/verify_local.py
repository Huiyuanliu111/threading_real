"""Recorded-frame local GPU smoke test; never connects to robot or cameras."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'threading_real/pi05_openpi'))
import numpy as np,torch,pyarrow.parquet as pq
from PIL import Image
import io
from deploy import LocalBackend,LocalPolicy
root=ROOT/'threading_real/pi05_openpi'
s=time.monotonic();backend=LocalBackend(root/'checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000',root/'deployment_config.json');policy=LocalPolicy(backend);load=time.monotonic()-s
row=pq.read_table(root/'data/threading_tcp6_nosmooth_30hz/data/chunk-000/episode_000000.parquet').slice(62,1).to_pylist()[0]
obs={'agent_pos':torch.from_numpy(np.array(row['observation.state'],dtype=np.float32))[None,None]}
for dest,key in [('frontview','observation.images.exterior_image_1_left'),('sideview','observation.images.exterior_image_2_right')]:
 im=np.array(Image.open(io.BytesIO(row[key]['bytes'])));obs[dest]=torch.from_numpy(im.transpose(2,0,1).copy())[None,None]
times=[]
for i in range(3):
 s=time.monotonic();a=policy.predict_action(obs)['action'].numpy();times.append(time.monotonic()-s)
 assert a.shape==(1,50,7) and np.isfinite(a).all() and not a[:,:,6].any()
 print('call',i,'seconds',times[-1],flush=True)
import jax,pinocchio,pyrealsense2
report=dict(checkpoint=backend.checkpoint,device=str(jax.devices()[0]),load_seconds=load,inference_seconds=times,shape=list(a.shape),finite=True,gripper_channel_zero=True,robot_connected=False)
(root/'outputs/local_deployment_smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
