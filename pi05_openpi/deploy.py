"""Local OpenPI no-selector deployment on the camera workstation."""
from pathlib import Path
import sys,time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))

def letterbox(image):
 h,w=image.shape[:2];scale=224/max(h,w);nh,nw=round(h*scale),round(w*scale)
 out=np.zeros((224,224,3),dtype=np.uint8)
 out[(224-nh)//2:(224-nh)//2+nh,(224-nw)//2:(224-nw)//2+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
 return out

class LocalPolicy(torch.nn.Module):
 uses_openpi=True
 action_mode='cartesian_delta';action_dim=7;horizon=50;n_obs_steps=1;n_action_steps=50
 rgb_keys=('sideview','frontview');image_shape=(3,224,224)
 def __init__(self,client):
  super().__init__();self.client=client
  m=client.get_server_metadata()
  if any(m.get(k)!=v for k,v in dict(kind='threading_openpi_tcp6',state_dim=9,action_dim=6,fps=30).items()):raise ValueError(f'Incompatible server metadata: {m}')
  if m.get('horizon') not in (10,50):raise ValueError('Unsupported model horizon')
  self.horizon=self.n_action_steps=m['horizon']
  self.metadata=m;self.task=m['task'];print('[openpi] model:',m,flush=True)
  self.image_profile=m.get('image_profile','224')
  if self.image_profile not in ('224','native640'):raise ValueError('Unsupported image profile')
  self.image_shape=(3,490,644) if self.image_profile=='native640' else (3,224,224)
  self.camera_views=m.get('camera_views','both')
  if self.camera_views not in ('both','cam1'):raise ValueError('Unsupported camera views')
  self.rgb_keys=('sideview',) if self.camera_views=='cam1' else ('sideview','frontview')
 def predict_action(self,obs):
  payload={'state':obs['agent_pos'][0,-1].cpu().numpy(),'prompt':self.task}
  cameras=[('side','sideview')] if self.camera_views=='cam1' else [('front','frontview'),('side','sideview')]
  for name,key in cameras:
   payload[name]=obs[key][0,-1].cpu().numpy().transpose(1,2,0)
  a=np.asarray(self.client.infer(payload)['actions'],dtype=np.float32)
  if a.shape!=(self.horizon,6) or not np.isfinite(a).all():raise ValueError('Invalid OpenPI actions')
  # Seventh runner channel is constant zero; it is not a model output.
  a=np.pad(a,((0,0),(0,1)))
  return {'action':torch.from_numpy(a[:self.n_action_steps]).unsqueeze(0)}

class LocalBackend:
 def __init__(self,checkpoint,run_config):
  import json
  from run import setup
  settings=json.loads(Path(run_config).read_text())
  if settings.get('physical_action_dim')!=6 or settings.get('physical_state_dim')!=9 or settings.get('horizon') not in (10,50):
   raise ValueError('Expected TCP6/state9 and horizon 10 or 50')
  config=setup(settings)
  import jax
  if jax.default_backend()!='gpu':raise RuntimeError('OpenPI requires the local CUDA backend')
  from openpi.policies.policy_config import create_trained_policy
  self.policy=create_trained_policy(config,Path(checkpoint),sample_kwargs={'num_steps':10})
  self.horizon=settings['horizon']
  self.checkpoint=str(Path(checkpoint).resolve())
  self.image_profile=settings.get('image_profile','224')
  self.camera_views=settings.get('camera_views','both')
 def get_server_metadata(self):
  return dict(kind='threading_openpi_tcp6',horizon=self.horizon,state_dim=9,action_dim=6,fps=30,checkpoint=self.checkpoint,task='insert the grasped block through the needle',image_profile=self.image_profile,camera_views=self.camera_views)
 def infer(self,obs):return self.policy.infer(obs)

class LockedGripper:
 def __init__(self,*args,**kwargs):pass
 def update(self,*args,**kwargs):pass

def main(backend_factory=None, configure_parser=None):
 from threading_real.scripts.deployment import cartesian
 from scripts.deployment.joint import ObservationFrame
 from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
 parser=cartesian.build_parser()
 parser.add_argument('--run-config',type=Path,default=Path(__file__).parent/'deployment_config.json')
 parser.set_defaults(gripper_force=70.0,prepare_initial_grasp=True,device='cpu',policy_hz=30,stream_hz=480,execute_steps=50,sync_require_target=False,image_size=224,synchronous=True,sync_timeout=5.0,weights='model')
 if configure_parser is not None:configure_parser(parser)
 args=parser.parse_args()
 if args.chunk_selector or args.aac or args.execution_schedule or args.prediction_mode!='full_then_truncate':raise ValueError('Only fixed chunks / full_then_truncate are supported')
 if args.policy_hz!=30 or args.image_size!=224 or args.pre_resize_image_size is not None:raise ValueError('Requires30Hz and full224 letterbox preprocessing')
 if args.move_to_training_start:raise ValueError('OpenPI deployment uses manual placement; automatic start/release is disabled')
 if not np.isclose(args.stream_hz/30,round(args.stream_hz/30)):raise ValueError('stream-hz must be a multiple of30 for exact action timing')
 fk=UrdfForwardKinematics(ROOT/'remote_controller/src/remote_controller/assets/panda/panda_arm.urdf')
 backend=LocalBackend(args.checkpoint,args.run_config) if backend_factory is None else backend_factory(args)
 policy=LocalPolicy(backend)
 from native_images import pad_native_rgb
 prepare_image=pad_native_rgb if policy.image_profile=='native640' else letterbox
 def observe(side,wrist,front,q,width,image_size,pre_resize_image_size=None,tcp_position=None):
  pos,rot=fk.poses(np.asarray(q,dtype=np.float64)[None])
  state=np.concatenate((pos[0],rot[0,:,:2].T.reshape(6))).astype(np.float32)
  side_image=torch.from_numpy(prepare_image(side).transpose(2,0,1).copy())
  front_image=None if policy.camera_views=='cam1' else torch.from_numpy(prepare_image(front).transpose(2,0,1).copy())
  return ObservationFrame(sideview=side_image,wrist=None,frontview=front_image,agent_pos=torch.from_numpy(state),timestamp=time.monotonic(),tcp_pos=torch.from_numpy(pos[0].astype(np.float32)))
 original=(cartesian.load_deployment_policy,cartesian.make_observation,cartesian.GripperController)
 cartesian.load_deployment_policy=lambda *a,**kw:policy
 cartesian.make_observation=observe;cartesian.GripperController=LockedGripper
 try:return cartesian.run(args)
 finally:
  cartesian.load_deployment_policy,cartesian.make_observation,cartesian.GripperController=original
  if hasattr(backend,"close"):backend.close()
if __name__=='__main__':raise SystemExit(main())
