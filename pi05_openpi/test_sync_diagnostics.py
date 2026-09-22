import json
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from threading_real.scripts.deployment.sync_diagnostics import SyncDiagnostics
from threading_real.scripts.deployment.cartesian import wait_for_trackc_segment
from threading_real.pi05_openpi.analyze_sync import summarize


def test_timeout_preserves_samples_and_event(tmp_path):
    path=tmp_path/'sync.jsonl';diag=SyncDiagnostics(path);diag.episode=1;diag.cycle=1
    streamer=SimpleNamespace(manager=SimpleNamespace(lock=threading.Lock(),completed=True))
    class Client:
        def get_latest_state(self,**kwargs):return {'q':np.zeros(7),'dq':np.ones(7)*.02,'arm_state':'MOVING'}, {'age':.001}
        def get_tcp_pose_from_q(self,*args,**kwargs):return np.eye(4)
    T=np.eye(4);T[0,3]=.01
    with pytest.raises(RuntimeError,match='timed out'):
        wait_for_trackc_segment(streamer,Client(),None,T,position_tolerance=.001,
                                rotation_tolerance=.02,timeout=.005,settle_samples=1,poll_hz=1000,
                                stop_requested=lambda:False,require_target=True,diagnostics=diag)
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert rows[-1]['event']=='wait_timeout'
    report=summarize(rows)[0]
    assert report['timeout']
    assert report['last_wait']['error_mm']==pytest.approx(10)
    assert report['last_wait']['max_abs_dq_rad_s']==pytest.approx(.02)


def test_disabled_diagnostics_do_not_touch_sender():
    class Sender:
        def get_diagnostics(self):raise AssertionError('disabled')
    SyncDiagnostics(None).emit('test',streamer=Sender())
