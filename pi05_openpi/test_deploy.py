"""Robot-free contract tests for the OpenPI deployment adapter."""
import numpy as np
import torch
from threading_real.pi05_openpi.deploy import LocalPolicy,letterbox

class Client:
 def get_server_metadata(self):
  return dict(kind='threading_openpi_tcp6',horizon=50,state_dim=9,action_dim=6,fps=30,task='test')
 def infer(self,obs):
  assert obs['state'].shape==(9,)
  assert obs['front'].shape==(224,224,3) and obs['front'].dtype==np.uint8
  assert obs['side'][0,0,0]==3 and obs['front'][0,0,0]==7
  return {'actions':np.ones((50,6),dtype=np.float32)*.001}

def test_adapter():
 p=LocalPolicy(Client());p.n_action_steps=10
 obs={'agent_pos':torch.zeros(1,1,9),'frontview':torch.full((1,1,3,224,224),7,dtype=torch.uint8),'sideview':torch.full((1,1,3,224,224),3,dtype=torch.uint8)}
 result=p.predict_action(obs)['action'].numpy()
 assert result.shape==(1,10,7)
 np.testing.assert_allclose(result[...,:6],.001)
 assert not result[...,6].any()

def test_letterbox():
 result=letterbox(np.full((480,640,3),123,dtype=np.uint8))
 assert result.shape==(224,224,3)
 assert not result[:28].any() and not result[196:].any()
 assert np.all(result[28:196]==123)

def test_runner_preconnection_initialization(monkeypatch):
 """Exercise real runner setup; intercept the constructor before any robot I/O."""
 import importlib
 import pytest
 from threading_real.scripts.deployment import cartesian
 controller=importlib.import_module('remote_controller.RemoteControllerClient')
 policy=LocalPolicy(Client())
 policy.rgb_keys=("sideview",)
 monkeypatch.setattr(cartesian,'load_deployment_policy',lambda *a,**kw:policy)
 class ReachedConnectionBoundary(Exception):pass
 def stop_before_connect(*args,**kwargs):raise ReachedConnectionBoundary()
 monkeypatch.setattr(controller,'RemoteControllerClient',stop_before_connect)
 # Importing ARP here used to trigger unrelated h5py/zarr dependencies.
 import builtins
 original_import=builtins.__import__
 def checked_import(name,*args,**kwargs):
  if name=='threading_task.policy':raise AssertionError('OpenPI must skip ARP/GMM initialization')
  return original_import(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',checked_import)
 args=cartesian.build_parser().parse_args(['unused-local-checkpoint','--device','cpu','--policy-hz','30','--execute-steps','1'])
 with pytest.raises(ReachedConnectionBoundary):cartesian.run(args)

def test_staged_grasp_waits_for_placement(monkeypatch):
 from types import SimpleNamespace
 from threading_real.scripts.deployment import cartesian
 events=[]
 class Gripper:
  def gripper_release(self,speed,queue):events.append('open');return 0
  def wait_until_gripper_moving_finished(self,timeout):events.append('wait');return 'IDLE'
  def get_gripper_width(self):return .08
 def confirm(prompt,stop):
  assert events==['open','wait']
  assert 'Place the object' in prompt
  events.append('operator');return True
 monkeypatch.setattr(cartesian,'wait_for_episode_enter',confirm)
 assert cartesian.prepare_initial_grasp(Gripper(),SimpleNamespace(gripper_speed=.05,initial_grasp_width=.02),lambda:False)
 assert events==['open','wait','operator']

def test_staged_grasp_open_failure_aborts(monkeypatch):
 from types import SimpleNamespace
 import pytest
 from threading_real.scripts.deployment import cartesian
 class Gripper:
  def gripper_release(self,*args,**kwargs):return -4
  def decode_rpc_result(self,result):return 'EXECUTION_FAILED'
 monkeypatch.setattr(cartesian,'wait_for_episode_enter',lambda *args:pytest.fail('must not request placement after failed opening'))
 with pytest.raises(RuntimeError,match='opening failed'):
  cartesian.prepare_initial_grasp(Gripper(),SimpleNamespace(gripper_speed=.05),lambda:False)

def test_strict_tracking_does_not_confuse_sent_with_reached():
 import threading
 from types import SimpleNamespace
 import pytest
 from threading_real.scripts.deployment import cartesian
 streamer=SimpleNamespace(manager=SimpleNamespace(lock=threading.Lock(),completed=True))
 class Client:
  def get_latest_state(self,**kwargs):return {'q':np.zeros(7),'arm_state':'MOVING'},{}
  def get_tcp_pose_from_q(self,*args,**kwargs):return np.eye(4)
 target=np.eye(4);target[0,3]=.0005
 with pytest.raises(RuntimeError,match='translation_error=0.500 mm'):
  cartesian.wait_for_trackc_segment(streamer,Client(),None,target,position_tolerance=.0001,
   rotation_tolerance=.02,timeout=.005,settle_samples=1,poll_hz=1000,stop_requested=lambda:False,require_target=True)
 result=cartesian.wait_for_trackc_segment(streamer,Client(),None,np.eye(4),position_tolerance=.0001,
   rotation_tolerance=.02,timeout=.1,settle_samples=1,poll_hz=1000,stop_requested=lambda:False,require_target=True)
 assert result['within_tolerance']

def test_remote_timeout_discards_connection():
 import pytest
 import deploy_remote
 class Socket:
  closed=False
  def send(self,data):pass
  def recv(self,timeout):raise TimeoutError()
  def close(self):self.closed=True
 backend=object.__new__(deploy_remote.RemoteBackend)
 backend.ws=Socket();backend.packer=deploy_remote.msgpack_numpy.Packer()
 with pytest.raises(RuntimeError,match='no stale prediction accepted'):
  backend._request({'state':np.zeros(9)},timeout=.1)
 assert backend.ws.closed


def test_native_resolution_metadata_and_payload():
 class NativeClient:
  def get_server_metadata(self):
   return {**Client().get_server_metadata(), 'image_profile':'native640'}
  def infer(self,obs):
   assert obs['front'].shape==(490,644,3)
   assert obs['side'].shape==(490,644,3)
   assert obs['front'][5,2,0]==7
   assert obs['side'][5,2,0]==3
   return {'actions':np.zeros((50,6),np.float32)}
 p=LocalPolicy(NativeClient())
 assert p.image_shape==(3,490,644)
 obs={'agent_pos':torch.zeros(1,1,9),
      'frontview':torch.full((1,1,3,490,644),7,dtype=torch.uint8),
      'sideview':torch.full((1,1,3,490,644),3,dtype=torch.uint8)}
 assert p.predict_action(obs)['action'].shape==(1,50,7)


def test_cam1_only_deployment_does_not_read_or_send_front():
 class Cam1Client:
  def get_server_metadata(self):
   return {**Client().get_server_metadata(),'image_profile':'native640','camera_views':'cam1'}
  def infer(self,obs):
   assert set(obs)=={'side','state','prompt'}
   assert obs['side'].shape==(490,644,3)
   return {'actions':np.zeros((50,6),np.float32)}
 p=LocalPolicy(Cam1Client())
 assert p.rgb_keys==('sideview',)
 obs={'agent_pos':torch.zeros(1,1,9),'sideview':torch.zeros(1,1,3,490,644,dtype=torch.uint8)}
 assert p.predict_action(obs)['action'].shape==(1,50,7)


def test_h10_policy_preserves_all_ten_actions():
 class H10Client(Client):
  def get_server_metadata(self):
   return {**super().get_server_metadata(), 'horizon':10}
  def infer(self,obs):
   return {'actions':np.arange(60,dtype=np.float32).reshape(10,6)*.0001}
 p=LocalPolicy(H10Client())
 assert p.horizon==p.n_action_steps==10
 obs={'agent_pos':torch.zeros(1,1,9),'frontview':torch.zeros(1,1,3,224,224,dtype=torch.uint8),'sideview':torch.zeros(1,1,3,224,224,dtype=torch.uint8)}
 result=p.predict_action(obs)['action'].numpy()
 assert result.shape==(1,10,7)
 np.testing.assert_allclose(result[0,:,:6],np.arange(60,dtype=np.float32).reshape(10,6)*.0001)
 assert not result[0,:,6].any()


def test_h10_remote_rejects_wrong_chunk_length():
 import pytest
 from threading_real.pi05_openpi.deploy_remote import RemoteBackend
 from openpi_client import msgpack_numpy
 class Socket:
  def send(self,data):pass
  def recv(self,timeout):return msgpack_numpy.packb({'actions':np.zeros((50,6),np.float32)})
 p=RemoteBackend.__new__(RemoteBackend)
 p.ws=Socket();p.packer=msgpack_numpy.Packer();p.horizon=10
 with pytest.raises(ValueError,match='Invalid remote action chunk'):
  p._request({},1)
