"""Opt-in host-clock diagnostics; no robot commands or control changes."""
import json
from pathlib import Path
import time

import numpy as np


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class SyncDiagnostics:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.session = str(time.time_ns())
        self.episode = self.cycle = 0
        self.last_wait_sample = float('-inf')
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.emit('session', clock='workstation monotonic seconds',
                      scope='host observations and UDP sends, NOT controller receive acknowledgements',
                      camera_time='read start/end only; not sensor exposure timestamps')

    def emit(self, event, *, state=None, info=None, T=None, streamer=None, **fields):
        if self.path is None:
            return
        row = dict(event=event, monotonic_s=time.monotonic(), wall_time_s=time.time(),
                   session=self.session, episode=self.episode, cycle=self.cycle, **fields)
        if state is not None:
            row['state'] = {k: state[k] for k in ('q', 'dq', 'K_F_ext_hat', 'arm_state') if k in state}
        if info is not None:
            row['state_info'] = info
        if T is not None:
            row['fk_tcp_T'] = T
        if streamer is not None and hasattr(streamer, 'get_diagnostics'):
            row['sender'] = streamer.get_diagnostics()
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, default=_json) + '\n')

    def wait_sample(self, **fields):
        now = time.monotonic()
        if now - self.last_wait_sample >= .05:
            self.emit('wait_sample', **fields)
            self.last_wait_sample = now
