#!/usr/bin/env python3
"""Write spatial label distributions and exact source-video node frames to HTML/PNG."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from chunk_selector.mvt_data import video_provenance


def render_report(labels_dir: Path, output: Path, *, raw_root: Path | None = None,
                  episodes: list[str] | None = None, frame_offset: int = 3):
    if frame_offset < 1:
        raise ValueError("frame_offset must be positive")
    labels_dir, output = Path(labels_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = json.loads((labels_dir / "summary.json").read_text())
    table = pq.read_table(labels_dir / "labels.parquet").to_pandas()
    rule = summary["rule"]
    h, H = summary["candidate_chunks"]
    keys = sorted(table.episode_key.unique()) if episodes is None else episodes
    if not keys or set(keys) - set(table.episode_key):
        raise ValueError("requested episodes are absent from labels")
    output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(table.soft_chunk_size, bins=np.linspace(h, H, 29), color="#2486aa")
    axes[0].set(xlabel="Continuous expected chunk", ylabel="Frames", title="Soft label distribution")
    counts = table.execution_steps.value_counts().reindex(range(h, H + 1), fill_value=0)
    axes[1].bar(counts.index, counts.values, color="#2486aa")
    axes[1].set(xlabel="Rounded execution steps", ylabel="Frames", xticks=range(h, H + 1))
    fig.savefig(output / "distribution.png", dpi=150)
    plt.close(fig)
    entries, html = [], []
    nodes = {node["episode_key"]: node for node in summary["nodes"]}
    with h5py.File(summary["source_dataset"], "r") as source:
        for key in keys:
            rows = table[table.episode_key == key].sort_values("frame_index")
            node = nodes[key]
            frame = node["frame_index"]
            xyz = rows[["tcp_x_m", "tcp_y_m", "tcp_z_m"]].to_numpy()
            chunks = rows.soft_chunk_size.to_numpy()
            fig = plt.figure(figsize=(12, 4), layout="constrained")
            ax = fig.add_subplot(121, projection="3d")
            points = ax.scatter(*xyz.T, c=chunks, cmap="viridis", vmin=h, vmax=H, s=10)
            ax.scatter(*xyz[frame], marker="*", color="red", s=120, label="Progress node")
            center = np.asarray(rule["center_xyz_m"])
            ax.scatter(*center, marker="x", color="black", s=50, label="Fine center")
            ax.set(xlabel="TCP X (m)", ylabel="TCP Y (m)", zlabel="TCP Z (m)")
            ax.legend(fontsize=8)
            fig.colorbar(points, ax=ax, shrink=.65, label="Expected chunk")
            ax = fig.add_subplot(122)
            ax.plot(rows.trajectory_progress, chunks, label="Soft expectation")
            ax.step(rows.trajectory_progress, rows.execution_steps, alpha=.4, label="Rounded steps")
            ax.axvline(node["progress"], color="red", ls="--", label="Progress node")
            for index in node["boundary_crossing_frames"]:
                ax.axvline(rows.trajectory_progress.iloc[index], color="grey", lw=.6, alpha=.6)
            ax.set(xlabel=f"Trajectory progress ({rule['progress_mode']})", ylabel="Chunk",
                   ylim=(h - .3, H + .3), title=f"{key} / {rows.split.iloc[0]}")
            ax.legend(fontsize=8)
            fig.savefig(output / f"{key}.png", dpi=130)
            plt.close(fig)
            html.append(f'<section id="{escape(key)}"><h2>{escape(key)}</h2>'
                        f'<p>Progress node: row {frame}; progress {node["progress"]:.3f}; '
                        f'fine frames {node["fine_frame_fraction"]:.1%}; '
                        f'spatial crossings: {escape(str(node["boundary_crossing_frames"]))}</p>'
                        f'<img class="plot" src="{key}.png"><div class="frames">')
            try:
                paths, camera_indices = video_provenance(source, key, raw_root=raw_root)
                chosen = sorted(set([max(0, frame - frame_offset), frame, min(len(rows) - 1, frame + frame_offset)]
                                    + node["boundary_crossing_frames"]))
                for camera, path in enumerate(paths):
                    capture = cv2.VideoCapture(str(path)) if path.is_file() else None
                    try:
                        for row in chosen:
                            video_frame = int(camera_indices[row, camera])
                            entry = dict(episode_key=key, dataset_frame=row, video=str(path),
                                         video_frame=video_frame, progress_node=row == frame,
                                         spatial_crossing=row in node["boundary_crossing_frames"])
                            caption = f"{path.name}: row {row} / video frame {video_frame}"
                            if capture is not None and capture.isOpened():
                                capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                                ok, image = capture.read()
                            else:
                                ok = False
                            if ok:
                                name = f"{key}_camera{camera}_row{row}.png"
                                if not cv2.imwrite(str(output / name), image):
                                    raise OSError(f"could not write {name}")
                                entry.update(status="ok", image=name)
                                html.append(f'<figure><img src="{name}"><figcaption>{escape(caption)}</figcaption></figure>')
                            else:
                                entry["status"] = "missing_or_unreadable_video"
                                html.append(f'<p>{escape(caption)}: video unavailable ({escape(str(path))})</p>')
                            entries.append(entry)
                    finally:
                        if capture is not None:
                            capture.release()
            except (ValueError, KeyError) as error:
                entries.append(dict(episode_key=key, status="missing_provenance", error=str(error)))
                html.append(f'<p>{escape(str(error))}</p>')
            html.append('</div></section>')
    report = dict(rule=rule, labels_dir=str(labels_dir), frames=entries,
                  available_frames=sum(item["status"] == "ok" for item in entries),
                  missing_frames=sum(item["status"] != "ok" for item in entries))
    (output / "video_frames.json").write_text(json.dumps(report, indent=2) + "\n")
    nav = ' '.join(f'<a href="#{escape(key)}">{escape(key)}</a>' for key in keys)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Spatial rule labels</title>'
        '<style>body{font:16px sans-serif;margin:24px;max-width:1250px} '
        '.plot{width:100%}.frames{display:flex;flex-wrap:wrap}figure{margin:8px} '
        'figure img{width:360px}nav a{display:inline-block;margin:4px}section{border-top:1px solid #ccc}</style>'
        f'<h1>{escape(rule["task"])} spatial rule</h1>'
        f'<p>h={h}, H={H}; progress split={rule["split_progress"]}; '
        f'boundary radius={rule["boundary_radius_m"]:.4f} m; '
        f'transition width={rule["transition_width_m"]:.4f} m.</p>'
        '<p>Red star/dashed line: requested progress node. Grey lines: actual spatial crossings. '
        'Images use stored camera_frame_index; offsets are dataset rows.</p>'
        f'<p>Available video frames: {report["available_frames"]}; missing: {report["missing_frames"]}.</p>'
        f'<img class="plot" src="distribution.png"><nav>{nav}</nav>' + ''.join(html))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument("--frame-offset", type=int, default=3)
    args = parser.parse_args()
    report = render_report(**vars(args))
    print(json.dumps({k: report[k] for k in ("available_frames", "missing_frames")}))


if __name__ == "__main__":
    main()
