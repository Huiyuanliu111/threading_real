"""Serve the trained six-action OpenPI policy over the official websocket protocol."""
import argparse,json
from pathlib import Path
from run import setup

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--run-config',required=True,type=Path);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
 a=p.parse_args();settings=json.loads(a.run_config.read_text())
 if settings.get('physical_action_dim')!=6 or settings.get('physical_state_dim')!=9 or settings.get('horizon') not in (10,50):raise ValueError('Expected TCP6/state9 and horizon 10 or 50')
 config=setup(settings)
 from openpi.policies.policy_config import create_trained_policy
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 policy=create_trained_policy(config,a.checkpoint,sample_kwargs={'num_steps':10})
 metadata=dict(kind='threading_openpi_tcp6',horizon=settings['horizon'],state_dim=9,action_dim=6,fps=30,checkpoint=str(a.checkpoint.resolve()),task='insert the grasped block through the needle',image_profile=settings.get('image_profile','224'),camera_views=settings.get('camera_views','both'))
 WebsocketPolicyServer(policy,host=a.host,port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
