#!/usr/bin/env python3
"""Run the existing trainer with bounded local checkpoint storage."""
import logging
import os
from pathlib import Path
import runpy
import shutil


def prune_checkpoints(current: Path, keep: int) -> list[Path]:
    """Remove older numbered saves only within the current run, after completion."""
    if keep < 1:
        raise ValueError('keep must be positive')
    current = Path(current).resolve()
    if not current.name.isdigit() or current.parent.name != 'checkpoints':
        raise ValueError(f'Unexpected checkpoint location: {current}')
    completed = sorted(
        (p for p in current.parent.iterdir()
         if p.name.isdigit() and p.is_dir() and not p.is_symlink()
         and int(p.name) <= int(current.name)),
        key=lambda p: int(p.name), reverse=True,
    )
    removed = []
    for old in completed[keep:]:
        shutil.rmtree(old)
        removed.append(old)
    return removed


def main() -> None:
    from lerobot.common import train_utils
    from lerobot.common.wandb_utils import WandBLogger

    keep = int(os.environ.get('PI05_KEEP_CHECKPOINTS', '2'))
    if keep < 1:
        raise ValueError('PI05_KEEP_CHECKPOINTS must be positive')
    original_update = train_utils.update_last_checkpoint

    def update_and_prune(checkpoint_dir):
        original_update(checkpoint_dir)
        for old in prune_checkpoints(Path(checkpoint_dir), keep):
            logging.info('Removed superseded checkpoint %s', old)

    train_utils.update_last_checkpoint = update_and_prune
    # Keep W&B scalar metrics; local saves are authoritative. Uploading every
    # model also retains large artifact-cache copies and races with retention.
    WandBLogger.log_policy = lambda self, checkpoint_dir: None
    print(f'Checkpoint retention: latest {keep}; W&B model uploads disabled, metrics enabled', flush=True)
    runpy.run_path(str(Path(__file__).with_name('train_with_state_dropout.py')), run_name='__main__')


if __name__ == '__main__':
    main()
