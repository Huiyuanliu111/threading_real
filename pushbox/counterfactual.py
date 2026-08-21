"""Task-agnostic primitives for state-level counterfactual chunk labels."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

import numpy as np
import torch


SnapshotT = TypeVar("SnapshotT")


@dataclass
class RNGSnapshot:
    """Python, NumPy, and Torch RNG state used to make branches comparable."""

    python_state: object
    numpy_state: tuple
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor] | None

    @classmethod
    def capture(cls) -> "RNGSnapshot":
        return cls(
            python_state=random.getstate(),
            numpy_state=np.random.get_state(),
            torch_state=torch.random.get_rng_state().clone(),
            cuda_states=(
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        )

    def restore(self) -> None:
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.random.set_rng_state(self.torch_state)
        if self.cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_states)


@dataclass
class BranchOutcome:
    """Minimal task-independent result returned by one candidate branch."""

    success: bool
    progress: float = 0.0
    collision: bool = False
    dropped: bool = False
    timeout: bool = False
    environment_steps: int = 0
    policy_calls: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UtilityWeights:
    """Default priorities put success and safety well ahead of speed."""

    success: float = 100.0
    progress: float = 1.0
    collision: float = 30.0
    dropped: float = 100.0
    timeout: float = 50.0
    environment_step: float = 0.01
    policy_call: float = 0.1

    def score(self, outcome: BranchOutcome) -> float:
        return float(
            self.success * float(outcome.success)
            + self.progress * float(outcome.progress)
            - self.collision * float(outcome.collision)
            - self.dropped * float(outcome.dropped)
            - self.timeout * float(outcome.timeout)
            - self.environment_step * int(outcome.environment_steps)
            - self.policy_call * int(outcome.policy_calls)
        )


@dataclass
class CounterfactualResult:
    candidate_chunks: tuple[int, ...]
    outcomes: tuple[BranchOutcome, ...]
    utilities: np.ndarray
    best_class: int
    best_chunk: int


class CounterfactualChunkSampler(Generic[SnapshotT]):
    """Run every candidate from one restored simulator/RNG root.

    `run_branch(prefix_actions, chunk)` is deliberately task-specific.  It should
    execute the prefix, continue the same frozen policy for the configured
    lookahead, and return task metrics as a BranchOutcome.
    """

    def __init__(
        self,
        *,
        candidate_chunks: Sequence[int],
        snapshot_fn: Callable[[], SnapshotT],
        restore_fn: Callable[[SnapshotT], None],
        run_branch: Callable[[Any, int], BranchOutcome],
        utility: Callable[[BranchOutcome], float] | None = None,
        tie_break: Literal["smaller", "larger"] = "smaller",
        tie_tolerance: float = 1.0e-8,
    ):
        candidates = tuple(int(value) for value in candidate_chunks)
        if not candidates or tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_chunks must be unique and strictly increasing")
        if tie_break not in ("smaller", "larger"):
            raise ValueError("tie_break must be 'smaller' or 'larger'")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance cannot be negative")
        self.candidate_chunks = candidates
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.run_branch = run_branch
        self.utility = utility or UtilityWeights().score
        self.tie_break = tie_break
        self.tie_tolerance = float(tie_tolerance)

    def _prefix(self, actions: Any, chunk: int) -> Any:
        prefix = actions[:chunk]
        if len(prefix) != chunk:
            raise ValueError(
                f"Action prediction contains {len(actions)} steps, cannot evaluate chunk={chunk}"
            )
        return prefix

    def sample(self, max_actions: Any) -> CounterfactualResult:
        if len(max_actions) < max(self.candidate_chunks):
            raise ValueError(
                f"Need at least {max(self.candidate_chunks)} predicted actions, "
                f"got {len(max_actions)}"
            )
        root_snapshot = copy.deepcopy(self.snapshot_fn())
        root_rng = RNGSnapshot.capture()
        outcomes: list[BranchOutcome] = []
        utilities: list[float] = []
        try:
            for chunk in self.candidate_chunks:
                self.restore_fn(copy.deepcopy(root_snapshot))
                root_rng.restore()
                outcome = self.run_branch(self._prefix(max_actions, chunk), chunk)
                if not isinstance(outcome, BranchOutcome):
                    raise TypeError("run_branch must return BranchOutcome")
                score = float(self.utility(outcome))
                if not np.isfinite(score):
                    raise ValueError(f"Utility for chunk={chunk} is not finite: {score}")
                outcomes.append(outcome)
                utilities.append(score)
        finally:
            self.restore_fn(copy.deepcopy(root_snapshot))
            root_rng.restore()

        utility_array = np.asarray(utilities, dtype=np.float32)
        maximum = float(utility_array.max())
        tied = np.flatnonzero(
            np.isclose(
                utility_array,
                maximum,
                rtol=0.0,
                atol=self.tie_tolerance,
            )
        )
        best_class = int(tied[0] if self.tie_break == "smaller" else tied[-1])
        return CounterfactualResult(
            candidate_chunks=self.candidate_chunks,
            outcomes=tuple(outcomes),
            utilities=utility_array,
            best_class=best_class,
            best_chunk=self.candidate_chunks[best_class],
        )


def robosuite_flat_state(env: Any) -> np.ndarray:
    """Capture the MuJoCo flat state used by robosuite/MimicGen reset_to."""
    raw_env = getattr(env, "env", env)
    if not hasattr(raw_env, "sim"):
        raise AttributeError("Environment does not expose a robosuite sim")
    return np.asarray(raw_env.sim.get_state().flatten()).copy()


def restore_robosuite_flat_state(env: Any, state: np.ndarray) -> Any:
    """Restore through the wrapper so observations and controllers are refreshed."""
    if not hasattr(env, "reset_to"):
        raise AttributeError("Environment wrapper does not expose reset_to")
    return env.reset_to({"states": np.asarray(state).copy()})
