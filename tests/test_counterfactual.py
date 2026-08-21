from __future__ import annotations

import numpy as np

from pushbox.counterfactual import (
    BranchOutcome,
    CounterfactualChunkSampler,
)


def test_counterfactual_branches_share_root_state_and_rng():
    environment = {"position": 0.0}
    branch_starts = []
    first_random_values = []

    def snapshot():
        return environment["position"]

    def restore(position):
        environment["position"] = position

    def run_branch(actions, _chunk):
        branch_starts.append(environment["position"])
        first_random_values.append(np.random.random())
        environment["position"] += float(np.asarray(actions).sum())
        return BranchOutcome(
            success=environment["position"] == 2.0,
            progress=-abs(environment["position"] - 2.0),
            environment_steps=len(actions),
        )

    np.random.seed(123)
    sampler = CounterfactualChunkSampler(
        candidate_chunks=(1, 2, 3),
        snapshot_fn=snapshot,
        restore_fn=restore,
        run_branch=run_branch,
    )
    result = sampler.sample(np.ones(3, dtype=np.float32))

    assert branch_starts == [0.0, 0.0, 0.0]
    assert first_random_values[0] == first_random_values[1] == first_random_values[2]
    assert result.best_chunk == 2
    assert environment["position"] == 0.0
    assert np.random.random() == np.random.RandomState(123).random()


def test_counterfactual_tie_break_can_prefer_larger_chunk():
    sampler = CounterfactualChunkSampler(
        candidate_chunks=(1, 2, 4),
        snapshot_fn=lambda: 0,
        restore_fn=lambda _snapshot: None,
        run_branch=lambda actions, chunk: BranchOutcome(success=True),
        tie_break="larger",
    )
    result = sampler.sample(np.zeros(4))
    assert result.best_class == 2
    assert result.best_chunk == 4
