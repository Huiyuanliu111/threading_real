"""Interactive gripper-only diagnostic: no policy load, cameras, arm init or motion."""
import argparse,json,socket,time,xmlrpc.client
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--server-url',default='http://10.157.175.22:8008/RPC2')
 p.add_argument('--width',type=float,default=.02)
 p.add_argument('--force',type=float,default=70.)
 p.add_argument('--speed',type=float,default=.02)
 p.add_argument('--epsilon',type=float,default=.01)
 p.add_argument('--execute',action='store_true',help='explicitly enable interactive opening and grasping')
 p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/grasp_diagnostics.jsonl')
 a=p.parse_args()
 if not (0<a.width<=.08 and 0<a.force<=70 and 0<a.speed<=.05 and 0<=a.epsilon<a.width):
  p.error('Require0<width<=.08,0<force<=70,0<speed<=.05,0<=epsilon<width (exclude empty closure).')
 socket.setdefaulttimeout(30)
 rpc=xmlrpc.client.ServerProxy(a.server_url)
 def record(event,**extra):
  r=dict(time=time.time(),event=event,gripper_state=int(rpc.getGripperState()),cached_width_m=float(rpc.getGripperWidth()),arm_state=int(rpc.getArmState()),**extra)
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r,indent=2),flush=True)
  return r
 start=record('status',target_width_m=a.width,force_n=a.force,speed_m_s=a.speed,epsilon_m=a.epsilon)
 if not a.execute:return 0
 if start['arm_state']!=0:raise RuntimeError('Arm must be IDLE; no commands sent.')
 input('Clear the gripper travel area. Enter to OPEN ONLY; Ctrl+C cancels: ')
 result=rpc.gripperRelease(a.speed)
 r=record('open_result',rpc_result=result)
 if result!=0 or r['gripper_state']!=0:raise RuntimeError('Opening failed; no grasp attempted.')
 input('Place object between open fingers, clear your hands, then Enter to GRASP; Ctrl+C cancels: ')
 result=rpc.graspO(a.width,a.speed,a.force,a.epsilon,a.epsilon)
 r=record('grasp_result',rpc_result=result)
 if result!=0 or r['gripper_state']!=4:
  print('FAILED: no retries or arm actions. Check physical contact and controller [graspO] logs.')
  return 1
 print('HOLDING confirmed. Gripper remains closed; no robot policy will start.')
 return 0
if __name__=='__main__':raise SystemExit(main())
