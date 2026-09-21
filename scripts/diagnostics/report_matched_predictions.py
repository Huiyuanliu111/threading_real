"""Report native and common-label equal-duration errors on matched observations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_openpi_mvt_10step import metrics,stats
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/openpi_mvt_matched_256'

def main():
 rows=json.loads((OUT/'matched_observations.json').read_text())
 d=np.load(OUT/'matched_reference.npz');o=np.load(OUT/'openpi_predictions.npz')
 assert o['prediction'].shape==(256,2,50,6)
 np.testing.assert_allclose(o['state'],d['openpi_state'],rtol=0,atol=1e-7)
 pred=o['prediction'].astype(np.float64);raw=d['raw_target'].astype(np.float64)
 p=pred.reshape(-1,50,6);t=np.repeat(raw,2,axis=0)
 grouped=pred[:,:,:40,:3].reshape(256,2,10,4,3).sum(3)
 common=raw[:,:40,:3].reshape(256,10,4,3).sum(2)
 pi_common=metrics(grouped.reshape(-1,10,3),np.repeat(common,2,axis=0))
 arp_common=metrics(d['arp_prediction'],common)
 nativepi=metrics(p[:,:10],t[:,:10]);nativearp=metrics(d['arp_prediction'],d['arp_target'])
 full=np.load(ROOT/'artifacts/mvt_cam1_epoch4_full_validation_error/predictions.npz')
 origins=full['origins'][[r['arp_prediction_index'] for r in rows]]
 eps=np.array([r['episode'] for r in rows]);byepisode=[]
 for ep in sorted(set(eps)):
  m=eps==ep
  byepisode.append(dict(episode=int(ep),openpi=metrics(grouped[m].reshape(-1,10,3),np.repeat(common[m],2,axis=0)),arp=metrics(d['arp_prediction'][m],common[m])))
 # Pair errors per observation, averaging seed errors (never averaging actions).
 oe=np.linalg.norm(grouped-common[:,None],axis=-1).mean((1,2))
 ae=np.linalg.norm(d['arp_prediction'][...,:3]-common,axis=-1).mean(1)
 result=dict(observations=256,episodes=sorted(set(eps.tolist())),openpi_inferences=512,arp_inferences_reused=256,
   matching='Exact source_trial and cam1 raw frame, including all10 future interval boundaries. Uniform32 per heldout episode; no error-based selection.',
   actual_duration_seconds=stats([r['duration_40_steps_seconds'] for r in rows]),
   state_origin_difference_mm=stats(np.linalg.norm(origins-d['openpi_state'][:,:3],axis=-1)*1000),
   native_label_vs_common_raw_label_mm=stats(np.linalg.norm(d['arp_target'][...,:3]-common,axis=-1)*1000),
   openpi_native10=nativepi,arp_native10=nativearp,
   openpi_common_duration=pi_common,arp_common_duration=arp_common,
   openpi_by_seed=[metrics(grouped[:,k],common) for k in range(2)],by_episode=byepisode,
   paired_observations_openpi_lower_step_distance=int((oe<ae).sum()),
   notes=['OpenPI step2000, two original RGB views, 30Hz. ARP epoch4, one physical RGB-D view, 7.5Hz.',
          'No new smoothing, ROI, retraining, augmentation or interpolation. Existing model-specific input preprocessing retained.',
          'OpenPI first40 predicted and raw-label deltas are summed in nonoverlapping groups of4. Both models scored against identical raw label at10 matching intervals.',
          'Native10 covers0.333s versus1.333s. Common-duration comparison covers40 OpenPI steps versus10 ARP steps.',
          'ARP historical robot state/labels originate from time-grid joint interpolation; OpenPI uses nearest measured robot row. Camera frames match exactly; robot state difference reported.',
          'Translation direction angle is not end-effector orientation error. Default excludes only vectors<=1e-9m; secondary threshold requires both>0.1mm.',
          'OpenPI seed outputs are scored separately; no action ensembling. Reused ARP predictions from deterministic full validation evaluation.',
          'Overlapping windows and shared episodes are correlated; no significance or real-robot success claim.'])
 (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 匹配观测比较：OpenPI 与单视角点云 ARP','',
 '8 条共同验证轨迹，每条均匀选取 32 个观测，共 256 对；OpenPI 每个观测运行 2 个噪声种子，共 512 次。ARP 复用已完成的确定性推理结果。',
 '按 source_trial 和原始 cam1 帧编号精确配对；10 个未来时间段的相机帧端点也逐一匹配。','',
 '## 同一时间跨度、同一原始标签（主要比较）','',
 'OpenPI 前 40 步每 4 步累加一次，对应 ARP 10 步，共约 1.333 秒。两者均与未平滑原始数据的相同位移标签比较。','',
 '|指标（均值）|OpenPI|单视角点云 ARP|','|---|---:|---:|']
 for key,label in [('per_step_distance_mm','每个 0.133 秒段距离误差 mm'),('per_step_magnitude_error_mm','每段移动长度误差 mm'),('per_step_direction_deg','每段方向夹角 °'),('per_step_direction_over_0p1mm_deg','两向量均 >0.1 mm 时方向夹角 °'),('endpoint_distance_mm','累计终点距离误差 mm'),('endpoint_direction_deg','累计位移方向夹角 °')]:
  lines.append(f"|{label}|{pi_common[key]['mean']:.3f}|{arp_common[key]['mean']:.3f}|")
 lines+=['','## 原生前 10 步（时间跨度不同）','','|指标（均值）|OpenPI 0.333 秒|ARP 1.333 秒|','|---|---:|---:|']
 for key,label in [('per_step_distance_mm','每步距离误差 mm'),('per_step_direction_deg','每步方向夹角 °'),('endpoint_distance_mm','累计终点距离误差 mm'),('endpoint_direction_deg','累计位移方向夹角 °')]:
  lines.append(f"|{label}|{nativepi[key]['mean']:.3f}|{nativearp[key]['mean']:.3f}|")
 lines+=['','## 同时间跨度逐段结果','','|段|OpenPI 距离 mm|ARP 距离 mm|OpenPI 方向 °|ARP 方向 °|','|---|---:|---:|---:|---:|']
 for a,b in zip(pi_common['by_step'],arp_common['by_step']):
  lines.append(f"|{a['step']}|{a['distance_mm']['mean']:.3f}|{b['distance_mm']['mean']:.3f}|{a['direction_deg']['mean']:.2f}|{b['direction_deg']['mean']:.2f}|")
 lines+=['','## 方法和限制','',
 '没有新增平滑、ROI、插值或数据增强；保留模型原有预处理（包括 ARP 固定空间范围和体素化）。OpenPI 两个种子分别评分，不对动作取平均。',
 'ARP 历史标签来自固定时间网格上的关节插值，OpenPI 使用最近实测机器人状态。因此相机帧完全一致，机器人状态仍有轻微差异。',
 f"对应 TCP 起点差异均值 {result['state_origin_difference_mm']['mean']:.4f} mm；ARP 原标签与本次公共原始标签的每段差异均值 {result['native_label_vs_common_raw_label_mm']['mean']:.4f} mm。",
 '方向是平移向量夹角，不是姿态角；0° 同向、90° 垂直、180° 反向。方向阈值不影响距离统计。完整有效样本数、P90、分 episode 和分种子结果见 report.json。',
 '观测匹配不等于输入模态完全相同：OpenPI 用两路 RGB，ARP 用一路 RGB-D；本结果比较现有完整系统，不隔离架构因素。窗口可能重叠，不能视为全部独立实验。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
 fig,ax=plt.subplots(1,3,figsize=(15,4))
 for report,label in [(pi_common,'OpenPI: 4 x 30Hz steps'),(arp_common,'ARP: 1 x 7.5Hz step')]:
  x=np.arange(1,11)
  ax[0].plot(x,[s['distance_mm']['mean'] for s in report['by_step']],'-o',label=label)
  ax[1].plot(x,[s['direction_deg']['mean'] for s in report['by_step']],'-o',label=label)
  ax[2].plot(x,[s['cumulative_distance_mm']['mean'] for s in report['by_step']],'-o',label=label)
 for a,title,y in zip(ax,['Translation error','Translation direction error','Cumulative position error'],['mm','degrees','mm']):
  a.set(title=title,xlabel='Matched interval (0.133s)',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
 fig.suptitle('256 matched observations / same raw labels / same time span');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
 print(json.dumps({k:result[k] for k in ['state_origin_difference_mm','native_label_vs_common_raw_label_mm','paired_observations_openpi_lower_step_distance']},indent=2))
 print('\n'.join(lines[:30]))
if __name__=='__main__':main()
