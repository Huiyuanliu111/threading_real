"""PushBox task: simulation environment + ARP imitation learning."""


def _get_pushbox_env():
    from envs.pushbox_env import PushBoxEnv  # noqa: PLC0415
    return PushBoxEnv


# Expose PushBoxEnv at package level for convenience, but avoid eager import to keep
# training / non-sim modules from pulling in robosuite/MuJoCo at startup.
def __getattr__(name: str):
    if name == "PushBoxEnv":
        return _get_pushbox_env()
    raise AttributeError(f"module 'pushbox' has no attribute '{name}'")


__all__ = ["PushBoxEnv"]
