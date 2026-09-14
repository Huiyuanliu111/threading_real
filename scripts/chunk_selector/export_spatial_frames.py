#!/usr/bin/env python3
"""Export standalone spatial-node PNGs and episode contact sheets, without HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import h5py
import numpy as np

from chunk_selector.mvt_data import video_provenance


def _caption(image, text):
    panel = np.full((image.shape[0] + 48, image.shape[1], 3), 245, np.uint8)
    panel[48:] = image
    for line, value in enumerate(text):
        cv2.putText(panel, value, (8, 18 + 20 * line), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (25, 25, 25), 1, cv2.LINE_AA)
    return panel


def _write(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def export_frames(labels_dir, output, *, source_kind="video", raw_root=None,
                  episodes=None, frame_offset=3, device="cpu", dataset=None):
    if source_kind not in {"video", "pointcloud"}:
        raise ValueError("source_kind must be video or pointcloud")
    if frame_offset < 1:
        raise ValueError("frame_offset must be positive")
    labels_dir, output = Path(labels_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = json.loads((labels_dir / 'summary.json').read_text())
    nodes = {node['episode_key']: node for node in summary['nodes']}
    keys = sorted(nodes) if episodes is None else episodes
    if not keys or set(keys) - set(nodes):
        raise ValueError("requested episodes are absent from labels")
    output.mkdir(parents=True)
    entries, sheets = [], []
    dataset = Path(dataset or summary['source_dataset']).expanduser().resolve()
    with h5py.File(dataset, 'r') as source:
        if source_kind == "pointcloud":
            import torch
            from threading_task.mvt_renderer import OrthographicMVTRenderer
            renderer = OrthographicMVTRenderer(scene_bounds=source.attrs['bounds_m']).to(device)
        for key in keys:
            group, node = source[key], nodes[key]
            frame = node['frame_index']
            length = len(group['camera_frame_index'])
            selected = sorted(set([max(0, frame-frame_offset), frame, min(length-1, frame+frame_offset)]
                                  + node['boundary_crossing_frames']))
            panels = []
            if source_kind == "video":
                paths, indices = video_provenance(source, key, raw_root=raw_root)
                streams = list(enumerate(paths))
            else:
                streams = [(0, None)]
            for camera, path in streams:
                capture = cv2.VideoCapture(str(path)) if path is not None and path.is_file() else None
                try:
                    for row in selected:
                        role = '/'.join(name for name, match in (
                            ('progress_node', row == frame),
                            ('spatial_crossing', row in node['boundary_crossing_frames'])) if match) or 'neighbor'
                        entry = dict(episode_key=key, dataset_frame=row, role=role, source_kind=source_kind)
                        if source_kind == 'video':
                            video_frame = int(indices[row, camera])
                            entry.update(video=str(path), video_frame=video_frame)
                            ok = False
                            if capture is not None and capture.isOpened():
                                capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
                                ok, image = capture.read()
                            if not ok:
                                entries.append({**entry, 'status': 'missing_or_unreadable_video'})
                                continue
                            name = f'{key}_camera{camera}_row{row:04d}_video{video_frame:05d}.png'
                            caption = f'VIDEO {path.name} frame {video_frame}'
                        else:
                            valid = int(group['valid_points'][row])
                            points = torch.as_tensor(group['points'][row, :valid].astype(np.float32), device=device)[None]
                            colors = torch.as_tensor(group['colors'][row, :valid].astype(np.float32), device=device)[None] / 255
                            with torch.inference_mode():
                                rendered = renderer(points, colors)[0, :, 3:6]
                            rgb = ((rendered.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                            image = cv2.cvtColor(np.concatenate(list(rgb), axis=1), cv2.COLOR_RGB2BGR)
                            name = f'{key}_row{row:04d}_pointcloud_top_left.png'
                            caption = 'POINT CLOUD (not video): top | left'
                            # Keep the provenance visible even outside the contact sheet.
                            image = _caption(image, [caption, f'{key} / row {row} / {role}'])
                            entry['camera_frame_indices'] = group['camera_frame_index'][row].tolist()
                        _write(output / name, image)
                        entries.append({**entry, 'status': 'ok', 'image': name})
                        width = 420
                        thumb = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
                        panels.append(_caption(thumb, [caption, f'row {row} / {role}']))
                finally:
                    if capture is not None:
                        capture.release()
            if panels:
                height = max(panel.shape[0] for panel in panels)
                # Wrap at four panels so multi-crossing episodes remain readable.
                columns = min(4, len(panels))
                sheet = np.full((((len(panels) + columns - 1) // columns) * height, columns * 420, 3), 245, np.uint8)
                for i, panel in enumerate(panels):
                    y, x = (i // columns) * height, (i % columns) * 420
                    sheet[y:y+len(panel), x:x+420] = panel
                name = f'{key}_overview.png'
                _write(output / name, sheet)
                sheets.append(name)
    manifest = dict(source_dataset=str(dataset), labels_dir=str(labels_dir),
                    source_kind=source_kind, frames=entries, overview_images=sheets,
                    available_frames=sum(e['status'] == 'ok' for e in entries),
                    missing_frames=sum(e['status'] != 'ok' for e in entries))
    (output / 'frames.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-kind', choices=('video', 'pointcloud'), default='video')
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--dataset', type=Path, help='override source HDF5 after moving the dataset')
    parser.add_argument('--episodes', nargs='+')
    parser.add_argument('--frame-offset', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    report = export_frames(**vars(parser.parse_args()))
    print(json.dumps({k: report[k] for k in ('available_frames', 'missing_frames')}))


if __name__ == '__main__':
    main()
