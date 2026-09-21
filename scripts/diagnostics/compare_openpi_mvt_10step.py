"""Compare saved physical translation predictions; does not run inference or resample data."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/openpi_vs_mvt_10step'

def stats(a):
    a = np.asarray(a).ravel()
    a = a[np.isfinite(a)]
    return dict(count=int(a.size), mean=float(a.mean()) if a.size else None,
                median=float(np.median(a)) if a.size else None,
                p90=float(np.percentile(a, 90)) if a.size else None)

def angle(p, t, threshold=1e-9):
    pn, tn = np.linalg.norm(p, axis=-1), np.linalg.norm(t, axis=-1)
    mask = (pn > threshold) & (tn > threshold)
    result = np.full(pn.shape, np.nan)
    result[mask] = np.degrees(np.arccos(np.clip(
        np.sum(p*t, axis=-1)[mask] / (pn[mask]*tn[mask]), -1, 1)))
    return result

def metrics(p, t):
    p, t = p[..., :3].astype('float64'), t[..., :3].astype('float64')
    assert p.shape == t.shape and np.isfinite(p).all() and np.isfinite(t).all()
    err = np.linalg.norm(p-t, axis=-1)*1000
    angles = angle(p, t)
    cp, ct = np.cumsum(p, axis=1), np.cumsum(t, axis=1)
    return dict(windows=len(p), steps=p.shape[1],
        per_step_distance_mm=stats(err),
        per_step_magnitude_error_mm=stats(abs(np.linalg.norm(p,axis=-1)-np.linalg.norm(t,axis=-1))*1000),
        per_step_direction_deg=stats(angles),
        per_step_direction_over_0p1mm_deg=stats(angle(p,t,0.0001)),
        label_step_magnitude_mm=stats(np.linalg.norm(t,axis=-1)*1000),
        chunk_rms_mm=stats(np.sqrt(np.mean(err**2,axis=1))),
        endpoint_distance_mm=stats(np.linalg.norm(cp[:,-1]-ct[:,-1],axis=-1)*1000),
        endpoint_direction_deg=stats(angle(cp[:,-1],ct[:,-1])),
        endpoint_magnitude_error_mm=stats(abs(np.linalg.norm(cp[:,-1],axis=-1)-np.linalg.norm(ct[:,-1],axis=-1))*1000),
        zero_baseline_step_distance_mm=stats(np.linalg.norm(t,axis=-1)*1000),
        by_step=[dict(step=i+1,distance_mm=stats(err[:,i]),direction_deg=stats(angles[:,i]),
                      cumulative_distance_mm=stats(np.linalg.norm(cp[:,i]-ct[:,i],axis=-1)*1000)) for i in range(p.shape[1])])

def main():
    source = ROOT/'pi05_openpi/outputs/visual_sensitivity_2000_tcp6'
    saved = np.load(source/'actions.npz')
    records = [json.loads(line) for line in (source/'records.jsonl').read_text().splitlines()]
    records = [r for r in records if r['condition']=='baseline']
    targets, predictions, cache = [], [], {}
    for r in records:
        ep = r['episode']
        if ep not in cache:
            path = ROOT/f'pi05_openpi/data/threading_tcp6_nosmooth_30hz/data/chunk-000/episode_{ep:06d}.parquet'
            cache[ep] = np.asarray(pq.read_table(path,columns=['action'])['action'].to_pylist())
        predictions.append(saved[r['actions_key']])
        targets.append(cache[ep][r['frame']:r['frame']+50])
    p,t = np.array(predictions),np.array(targets)
    # Cross-check recovered labels against the previously reported full-chunk error.
    reference = json.loads((source/'label_error.json').read_text())
    assert np.isclose(metrics(p,t)['chunk_rms_mm']['mean'],reference['chunk_translation_rms_mean_mm'],atol=1e-8)
    arp = np.load(ROOT/'artifacts/mvt_cam1_epoch4_full_validation_error/predictions.npz')
    result = dict(notes=[
        'OpenPI step2000: 30Hz, 24 states x 2 noise seeds, first10 of original50 predictions, 0.333s.',
        'Single physical camera MVT ARP epoch4: 7.5Hz, all2251 heldout complete windows, 10steps, 1.333s.',
        'Native10-step comparison is descriptive, not controlled: different durations, observations, validation samples and preprocessing.',
        'Direction means translation-vector angle, not tool orientation. Angles omit only either-vector norm <=1e-9m; valid counts reported.',
        'Additional direction metric requires both norms >0.1mm to show near-zero sensitivity; distance metrics retain all samples.',
        'No smoothing, ROI, resampling, new inference, or noise averaging. OpenPI two seeds are separately scored.',
        'Optional same-duration summary sums each consecutive4 OpenPI translation deltas over first40 steps; native predictions and labels unchanged. Samples are still not paired.'
    ], openpi_10=metrics(p[:,:10],t[:,:10]), arp_10=metrics(arp['prediction'],arp['target']),
       openpi_40_grouped_by_4=metrics(p[:,:40,:3].reshape(-1,10,4,3).sum(axis=2),t[:,:40,:3].reshape(-1,10,4,3).sum(axis=2)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# OpenPI 与单视角点云 ARP：前 10 步平移误差','',
           'OpenPI：2000 步权重，24 个验证状态 × 2 个噪声种子；ARP：epoch4 权重，2251 个验证窗口。',
           'OpenPI 10 步覆盖 0.333 秒；ARP 10 步覆盖 1.333 秒。样本和时间尺度不同，不能据此直接排名。','',
           '|步|OpenPI 距离误差 mm|ARP 距离误差 mm|OpenPI 方向夹角 °|ARP 方向夹角 °|',
           '|---|---:|---:|---:|---:|']
    for a,b in zip(result['openpi_10']['by_step'],result['arp_10']['by_step']):
        lines.append(f"|{a['step']}|{a['distance_mm']['mean']:.3f}|{b['distance_mm']['mean']:.3f}|{a['direction_deg']['mean']:.2f}|{b['direction_deg']['mean']:.2f}|")
    lines+=['','距离误差 = ‖预测平移向量 − 标签平移向量‖；方向夹角 = 两向量夹角（0°同向，90°垂直，180°反向），不是末端姿态角误差。',
            '累计终点误差先分别累加整个 chunk 的平移向量再比较。完整均值、中位数、P90 和有效样本数见 report.json。',
            '方向统计仅排除任一向量长度 ≤ 1e-9 m 的未定义方向；另存双向量均 > 0.1 mm 的敏感性统计。距离保留全部样本。',
            '没有平滑、ROI、重采样，也没有对两个噪声种子的预测取平均。额外提供 OpenPI 前 40 步每 4 步相加的 1.333 秒统计；仅作时间尺度参考，未对齐观测。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    for name in ('openpi_10','arp_10','openpi_40_grouped_by_4'):
        print(name,json.dumps({k:v for k,v in result[name].items() if k!='by_step'}))

if __name__ == '__main__':
    main()
