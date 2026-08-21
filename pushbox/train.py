"""Train ARP on PushBox teleop demos.

Usage:
  cd pushbox/pushbox
  python train.py --config-name=arp
  python train.py --config-name=arp training.debug=true training.device=cpu
"""
import atexit
import os
import pathlib
import signal
import sys
import traceback
from datetime import datetime

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

_CRASH_LOG = None


def _crash_handler(exc_type, exc_value, exc_tb):
    """Log unhandled Python exceptions to a crash file."""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    msg = f"[{datetime.now().isoformat()}] UNHANDLED EXCEPTION\n{tb_str}\n"
    sys.stderr.write(msg)
    if _CRASH_LOG:
        try:
            with open(_CRASH_LOG, "a") as f:
                f.write(msg)
        except Exception:
            pass


def _signal_handler(signum, frame):
    """Log signals (SIGKILL/OOM can't be caught, but SIGTERM/SIGSEGV can)."""
    sig_name = signal.Signals(signum).name
    msg = f"[{datetime.now().isoformat()}] RECEIVED SIGNAL: {sig_name} ({signum})\n{traceback.format_stack(frame)}"
    sys.stderr.write(msg)
    if _CRASH_LOG:
        try:
            with open(_CRASH_LOG, "a") as f:
                f.write(msg)
        except Exception:
            pass
    sys.exit(128 + signum)


# Install handlers early
sys.excepthook = _crash_handler
for sig in (signal.SIGTERM, signal.SIGSEGV, signal.SIGABRT, signal.SIGINT):
    try:
        signal.signal(sig, _signal_handler)
    except (ValueError, OSError):
        pass  # not available in some envs (e.g. sub-interpreters)

import hydra
from omegaconf import OmegaConf

PUSHBOX_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PUSHBOX_ROOT))

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(PUSHBOX_ROOT / "pushbox" / "configs"))
def main(cfg: OmegaConf):
    global _CRASH_LOG

    # Set crash log path inside Hydra's output dir
    output_dir = os.getcwd()
    _CRASH_LOG = os.path.join(output_dir, "crash.log")

    workspace_target = str(
        cfg.get("_target_", "pushbox.workspace.PushBoxARPWorkspace")
    )
    workspace_cls = hydra.utils.get_class(workspace_target)
    print(f"[train] workspace: {workspace_target}")
    workspace = workspace_cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
