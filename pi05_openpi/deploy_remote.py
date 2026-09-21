"""Local RGB/state capture and follower control, with remote OpenPI inference."""
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'vendor/openpi-client/src'))
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
import deploy

class RemoteBackend:
 def __init__(self,args):
  self.timeout=args.inference_timeout
  if self.timeout<=0:raise ValueError('inference-timeout must be positive')
  self.ws=connect(f'ws://{args.policy_host}:{args.policy_port}',compression=None,max_size=4*1024*1024,open_timeout=10,close_timeout=2)
  try:
   self.metadata=msgpack_numpy.unpackb(self.ws.recv(timeout=10))
   expected=dict(kind='threading_openpi_tcp6',horizon=50,state_dim=9,action_dim=6,fps=30)
   if any(self.metadata.get(k)!=v for k,v in expected.items()):raise ValueError(f'Incompatible remote model: {self.metadata}')
   if self.metadata['checkpoint']!=str(args.checkpoint):raise ValueError('Server checkpoint differs from requested path')
   self.packer=msgpack_numpy.Packer()
   # Compile before any robot or camera initialization; discard this synthetic prediction.
   shape=(490,644,3) if self.metadata.get('image_profile')=='native640' else (224,224,3)
   warmup={'front':np.zeros(shape,np.uint8),'side':np.zeros(shape,np.uint8),
           'state':np.array([0,0,0,1,0,0,0,1,0],np.float32),'prompt':self.metadata['task']}
   if self.metadata.get('camera_views')=='cam1':warmup.pop('front')
   started=time.monotonic();self._request(warmup,timeout=120)
   print(f'[remote] pre-robot warmup completed in {time.monotonic()-started:.3f}s',flush=True)
  except BaseException:
   self.ws.close();raise
 def get_server_metadata(self):return self.metadata
 def _request(self,obs,timeout):
  self.ws.send(self.packer.pack(obs))
  try:response=self.ws.recv(timeout=timeout)
  except TimeoutError:
   self.ws.close()
   raise RuntimeError(f'Remote inference exceeded {timeout}s; connection closed, no stale prediction accepted') from None
  if isinstance(response,str):raise RuntimeError(f'Remote policy error: {response}')
  result=msgpack_numpy.unpackb(response)
  a=np.asarray(result['actions'])
  if a.shape!=(50,6) or not np.isfinite(a).all():raise ValueError('Invalid remote action chunk')
  return result
 def infer(self,obs):
  started=time.monotonic();result=self._request(obs,self.timeout)
  print(f'[remote] roundtrip={time.monotonic()-started:.3f}s actions=50x6',flush=True)
  return result
 def close(self):self.ws.close()

def configure_parser(parser):
 parser.add_argument('--policy-host',default='127.0.0.1')
 parser.add_argument('--policy-port',type=int,default=18000)
 parser.add_argument('--inference-timeout',type=float,default=5.)
 parser.description=__doc__

if __name__=='__main__':raise SystemExit(deploy.main(backend_factory=RemoteBackend,configure_parser=configure_parser))
