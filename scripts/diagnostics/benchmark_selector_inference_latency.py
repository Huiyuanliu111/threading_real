#!/usr/bin/env python3
"""Time deployed full-horizon selector inference and nested ARP decoding offline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark_main_arp_latency import ROOT, CHECKPOINTS, sha256, load_threading, load_maze, MazeMVTDataset
from chunk_selector import mvt_features
from chunk_selector.mvt_features import predict_with_selector

SELECTORS = {
    'threading': 'training_runs/threading_selector_H20_h8_progress_20260916_134300/model',
    'maze': 'data/analysis/selector_eval_20260914_142324/models/maze_h4',
}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=CHECKPOINTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--calls', type=int, default=10)
    args = parser.parse_args()
    if min(args.batches, args.calls) < 1 or args.warmup < 0:
        parser.error('invalid repeat counts')
    torch.manual_seed(42)
    np.random.seed(42)
    checkpoint = ROOT / CHECKPOINTS[args.task]
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if args.task == 'threading':
        from chunk_selector.chunk_selector import ChunkSelector
        cfg = OmegaConf.to_container(payload['cfg'].task.dataset, resolve=True)
        dataset = hydra.utils.instantiate(cfg)
        policy = load_threading(checkpoint, device='cuda:0', weights='model')
    else:
        from maze_selector import ChunkSelector
        launch = json.loads((checkpoint.parent.parent/'launch.json').read_text())
        cfg = dict(dataset_path=str(ROOT/'data/datasets/maze_train49_mvt_7p5hz.h5'),
                   horizon=payload['policy_config']['horizon'],
                   val_ratio=launch['arguments']['val_ratio'], seed=launch['arguments']['seed'])
        dataset = MazeMVTDataset(**cfg)
        policy = load_maze(checkpoint, device='cuda:0', weights='model')
    del payload
    selector_path = ROOT / SELECTORS[args.task]
    selector = ChunkSelector.from_pretrained(selector_path, device='cuda:0').eval()
    obs = {key: value.unsqueeze(0).cuda() for key, value in dataset[0]['obs'].items()}
    for _ in range(args.warmup):
        predict_with_selector(policy, selector, obs)
    torch.cuda.synchronize()
    events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(2)]
              for key in ('total', 'decode', 'visual', 'selector')}
    original_generate, original_visual = policy.policy.generate, policy._visual
    original_select = mvt_features.select_from_visual
    def generate(*pos, **kw):
        events['decode'][0].record()
        result = original_generate(*pos, **kw)
        events['decode'][1].record()
        return result
    def visual(*pos, **kw):
        events['visual'][0].record()
        result = original_visual(*pos, **kw)
        events['visual'][1].record()
        return result
    def select(*pos, **kw):
        events['selector'][0].record()
        result = original_select(*pos, **kw)
        events['selector'][1].record()
        return result
    mvt_features.select_from_visual = select
    policy.policy.generate, policy._visual = generate, visual
    rows = []
    try:
        for batch in range(args.batches):
            for call in range(args.calls):
                torch.cuda.synchronize()
                wall = time.perf_counter()
                events['total'][0].record()
                prediction, selection = predict_with_selector(policy, selector, obs)
                events['total'][1].record()
                events['total'][1].synchronize()
                wall_ms = (time.perf_counter() - wall)*1000
                timings = {f'{key}_ms': start.elapsed_time(end) for key,(start,end) in events.items()}
                assert prediction['action_pred'].shape[1] == policy.horizon
                rows.append(dict(batch=batch, call=call, wall_ms=wall_ms,
                                 selected_steps=int(selection.chunk_sizes[0]), **timings))
                del prediction, selection
            print(json.dumps({'task': args.task, 'batch': batch,
                              **{key: float(np.mean([r[key] for r in rows[-args.calls:]]))
                                 for key in ('total_ms','decode_ms','visual_ms','selector_ms','wall_ms')}}), flush=True)
    finally:
        policy.policy.generate, policy._visual = original_generate, original_visual
        mvt_features.select_from_visual = original_select
    summary = {}
    for key in ('total_ms','decode_ms','visual_ms','selector_ms','wall_ms'):
        batches = [float(np.mean([r[key] for r in rows if r['batch']==b])) for b in range(args.batches)]
        summary[key] = dict(mean=float(np.mean(batches)), batch_sd=float(np.std(batches,ddof=1)))
    summary['decode_fraction_of_total'] = summary['decode_ms']['mean']/summary['total_ms']['mean']
    summary['selector_fraction_of_total'] = summary['selector_ms']['mean']/summary['total_ms']['mean']
    ep,frame = dataset.sample_indices[0]
    report = dict(task=args.task, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  selector=str(selector_path),
                  selector_files_sha256={str(p.relative_to(selector_path)):sha256(p)
                                         for p in selector_path.rglob('*') if p.is_file()},
                  script_sha256=sha256(__file__), dataset_config=cfg, split='train',
                  dataset_index=0, episode=dataset.episode_keys[ep], frame=frame,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  dtype=str(next(policy.parameters()).dtype), prediction_mode='full_then_truncate',
                  horizon=policy.horizon, batch_size=1, warmup=args.warmup,
                  batches=args.batches, calls_per_batch=args.calls,
                  scope='GPU-resident observations -> rendering/vision -> selector -> full-horizon ARP -> action postprocessing. Excludes data loading, camera, transfer, guards, robot execution. Nested CUDA events in same call, outer synchronization only. Selector boundary is select_from_visual: visual-token preparation/pooling plus selector.select; excludes shared visual encoder and subsequent chunk_sizes.max().item().',
                  measurements=rows, summary=summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
