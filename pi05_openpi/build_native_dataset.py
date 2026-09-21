"""Restore original RGB pixels using verified frame alignment; preserve numeric labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from transforms import FRONT, SIDE


def build_episode(job):
    source, output, report = job
    source, output = Path(source), Path(output)
    ep = report['episode_index']
    rel = f'data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet'
    table = pq.read_table(source / rel)
    alignment = np.load(source / f'meta/alignment/episode_{ep:06d}.npz')
    stats = {}
    for cam, key in [('cam1', SIDE), ('cam3', FRONT)]:
        indices = alignment[f'{cam}_frame']
        assert len(indices) == len(table) and np.all(np.diff(indices) >= 0)
        video = Path(report['source']) / f'{cam}.mp4'
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(video)
        rows, next_frame, bgr = [], 0, None
        sums, squares = np.zeros(3), np.zeros(3)
        low, high = np.full(3, 255), np.zeros(3)
        try:
            for index in indices:
                while next_frame <= index:
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError(f'Could not decode {video} frame {index}')
                    next_frame += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape != (480, 640, 3):
                    raise ValueError(f'Unexpected source size: {video}: {rgb.shape}')
                ok, png = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                assert ok
                rows.append({'bytes': png.tobytes(), 'path': None})
                sums += rgb.sum(axis=(0, 1), dtype=np.float64)
                squares += np.square(rgb, dtype=np.float64).sum(axis=(0, 1))
                low = np.minimum(low, rgb.min(axis=(0, 1)))
                high = np.maximum(high, rgb.max(axis=(0, 1)))
        finally:
            capture.release()
        count = len(table) * 480 * 640
        mean = sums / count
        stats[key] = {k: np.asarray(v)[:, None, None].tolist() for k, v in {
            'min': low / 255, 'max': high / 255, 'mean': mean / 255,
            'std': np.sqrt(np.maximum(squares / count - mean**2, 0)) / 255,
        }.items()}
        stats[key]['count'] = [len(table)]
        i = table.schema.get_field_index(key)
        table = table.set_column(i, table.schema.field(i), pa.array(rows, type=table.schema.field(i).type))
    destination = output / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    # Numeric columns and row order must remain byte-for-byte equivalent in Arrow.
    original = pq.read_table(source / rel)
    for key in original.column_names:
        if key not in (FRONT, SIDE):
            assert original[key].equals(table[key]), (ep, key)
    return ep, stats


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=here/'data/threading_tcp6_nosmooth_30hz')
    p.add_argument('--output', type=Path, default=here/'data/threading_tcp6_native640_30hz')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    processing = json.loads((source/'meta/raw_processing.json').read_text())
    output.mkdir(parents=True)
    shutil.copytree(source/'meta', output/'meta')
    # Do not expose a seemingly complete dataset until all episodes are written.
    (output/'meta/info.json').unlink()
    (output/'meta/verification.json').unlink(missing_ok=True)
    jobs = [(str(source), str(output), report) for report in processing['per_episode']]
    stats = {r['episode_index']: r for r in map(json.loads, (source/'meta/episodes_stats.jsonl').read_text().splitlines())}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for completed, (ep, image_stats) in enumerate(pool.map(build_episode, jobs), 1):
            stats[ep]['stats'].update(image_stats)
            print(f'[{completed}/{len(jobs)}] episode {ep}: restored 640x480 RGB; numeric columns unchanged', flush=True)
    (output/'meta/episodes_stats.jsonl').write_text(''.join(json.dumps(stats[k])+'\n' for k in sorted(stats)))
    processing.update(repo_id='threading_real/threading_tcp6_native640_30hz',
                      resize='none; original RGB 640x480 stored, model pads to 644x490',
                      source_numeric_dataset=str(source), image_encoding='PNG lossless, original decoded RGB pixels')
    (output/'meta/raw_processing.json').write_text(json.dumps(processing, indent=2)+'\n')
    provenance = output/'meta/gripper_removal.json'
    if provenance.exists():
        record = json.loads(provenance.read_text())
        record['all_other_columns_including_image_bytes_exact'] = False
        record['image_note'] = 'Images replaced by original 640x480 decoded frames; numeric columns unchanged from source_numeric_dataset'
        provenance.write_text(json.dumps(record, indent=2)+'\n')
    info = json.loads((source/'meta/info.json').read_text())
    for key in (FRONT, SIDE):
        info['features'][key]['shape'] = [480, 640, 3]
    (output/'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
