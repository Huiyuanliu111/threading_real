#!/usr/bin/env python3
"""Backfill logged metrics only; no model/data artifact upload or training restart."""
import argparse
import ast
import json
import math
from pathlib import Path
import re


def read_metrics(log, validation):
    merged = {}
    for match in re.finditer(r'Step (\d+): (\{[^\n]+\})', log.read_text()):
        step = int(match.group(1))
        metrics = ast.literal_eval(match.group(2))
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in metrics.values()):
            raise ValueError(f'Nonfinite training metrics at step {step}')
        merged.setdefault(step, {}).update(metrics)
    validations = [json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
    for row in validations:
        if not math.isfinite(row['val_loss']):
            raise ValueError(f'Nonfinite validation loss: {row}')
        merged.setdefault(row['step'], {}).update(val_loss=row['val_loss'], val_improved=int(row['improved']))
    if not merged:
        raise ValueError('No metrics found')
    return merged, validations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-config', required=True, type=Path)
    p.add_argument('--log', required=True, type=Path)
    p.add_argument('--validation', required=True, type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--project', default='threading_pi05_openpi')
    args = p.parse_args()
    settings = json.loads(args.run_config.read_text())
    merged, validations = read_metrics(args.log, args.validation)
    import wandb
    previous = json.loads(args.receipt.read_text()) if args.receipt.exists() else {}
    if previous.get('status') == 'completed':
        print(json.dumps(previous, indent=2))
        return
    run_id = previous.get('run_id', wandb.util.generate_id())
    receipt = {'status': 'uploading', 'run_id': run_id, 'project': args.project,
               'source_log': str(args.log.resolve()), 'experiment': settings['exp_name']}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    keys = ('mode','horizon','steps','batch_size','fsdp_devices','learning_rate','warmup_steps',
            'seed','fps','physical_action_dim','physical_state_dim','gripper_removed','save_interval',
            'repo_id','base_checkpoint','validation_split','training_image_augmentation')
    run = wandb.init(project=args.project, id=run_id, resume='allow', mode='online',
                     name=settings['exp_name'], tags=['historical-backfill','tcp6','lora'],
                     notes='Backfilled after user stopped training. Original step numbers retained; timestamps are upload time. Only logged steps are available (step 1 and every 20 steps). No interpolated values. No checkpoint or dataset uploaded.',
                     config={**{k: settings[k] for k in keys if k in settings},
                             'historical_backfill': True, 'training_status': 'stopped_by_user',
                             'original_wandb_enabled': settings.get('wandb',False)})
    receipt['url'] = run.url
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    run.define_metric('train_step')
    for key in ('loss','grad_norm','param_norm','val_loss','val_improved'):
        run.define_metric(key, step_metric='train_step')
    start = run.step
    for step, metrics in sorted(merged.items()):
        if step >= start:
            run.log({'train_step': step, **metrics}, step=step)
    best = min(validations, key=lambda r: r['val_loss']) if validations else None
    summary = {'last_logged_step':max(merged), 'train_loss_points':sum('loss' in m for m in merged.values()),
               'validation_points':len(validations), 'training_status':'stopped_by_user',
               'best_checkpoint_step':best['step'] if best else None,
               'best_val_loss':best['val_loss'] if best else None}
    run.summary.update(summary)
    url=run.url
    run.finish()
    receipt.update(status='completed',url=url,**summary)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
