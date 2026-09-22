#!/usr/bin/env python3
"""Weight measured ARP latencies by the recorded main-experiment selector traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data/analysis/arp_decode_latency_20260922"
TASKS = {
    "threading": ("data/analysis/selector_eval_matched_20260916/logs/threading20_selector_h8_progress.jsonl", "executed_steps", 20),
    "maze": ("data/analysis/selector_eval_20260914_142324/logs/maze10_selector_h4.jsonl", "execution_steps", 10),
}


def main():
    results = {}
    for task, (relative, key, horizon) in TASKS.items():
        path = ROOT / relative
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows and all(row["executed"] for row in rows)
        traces = []
        for row in rows:
            if not traces or row["cycle"] == 1 or row["episode"] != traces[-1][-1]["episode"]:
                traces.append([])
            traces[-1].append(row)
        counts = Counter(int(row[key]) for row in rows)
        assert all(1 <= k <= horizon for k in counts)
        reports = [json.loads((OUTPUT / f"{task}{suffix}.json").read_text())
                   for suffix in ("", "_intermediate")]
        for field in ("checkpoint_sha256", "dataset_config", "episode", "frame", "weights",
                      "gpu", "torch", "cuda", "dtype", "warmup_per_length", "batches", "calls_per_batch"):
            assert reports[0][field] == reports[1][field], (task, field)
        measurements = {}
        for report in reports:
            for k, value in report["summary"].items():
                assert int(k) not in measurements
                measurements[int(k)] = value
        assert set(counts) <= set(measurements)
        full_ms = measurements[horizon]["mean_ms"]
        total_full_ms = len(rows) * full_ms
        total_short_ms = sum(n * measurements[k]["mean_ms"] for k, n in counts.items())
        saved_ms = total_full_ms - total_short_ms
        per_trace = []
        for i, trace in enumerate(traces):
            full = len(trace) * full_ms
            short = sum(measurements[int(r[key])]["mean_ms"] for r in trace)
            per_trace.append({"trace": i + 1, "episode_label": trace[0]["episode"],
                              "calls": len(trace), "full_ms": full,
                              "required_ms": short, "saved_ms": full - short})
        results[task] = {
            "source": relative, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "last_timestamp": max(row["timestamp"] for row in rows),
            "horizon": horizon, "calls": len(rows), "traces": len(traces),
            "lengths": [{"k": k, "count": n, "probability": n / len(rows),
                         **measurements[k]} for k, n in sorted(counts.items())],
            "full_ms_per_call": full_ms, "weighted_required_ms_per_call": total_short_ms / len(rows),
            "saved_ms_per_call": saved_ms / len(rows), "saved_fraction": saved_ms / total_full_ms,
            "full_total_s": total_full_ms / 1000, "required_total_s": total_short_ms / 1000,
            "saved_total_s": saved_ms / 1000,
            "full_s_per_trace": total_full_ms / len(traces) / 1000,
            "required_s_per_trace": total_short_ms / len(traces) / 1000,
            "saved_s_per_trace": saved_ms / len(traces) / 1000,
            "per_trace": per_trace,
            "assumption": "Counterfactual: retain the recorded selector lengths and call counts; substitute measured short-sequence decoding latency. Existing deployment generates the full horizon and realizes no such saving.",
        }
    (OUTPUT / "selector_weighted.json").write_text(json.dumps(results, indent=2) + "\n")
    for task, result in results.items():
        print(task, json.dumps({k: v for k, v in result.items() if k not in ("lengths", "per_trace")}))


if __name__ == "__main__":
    main()
