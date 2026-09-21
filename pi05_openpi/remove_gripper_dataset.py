#!/usr/bin/env python3
"""Remove only gripper columns from the verified raw-derived dataset; preserve every frame and RGB byte."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transforms import ACTION_NAMES, STATE_NAMES, validate_info

HERE = Path(__file__).resolve().parent


def convert(source, output, repo_id):
    info = json.loads((source / 'meta/info.json').read_text())
    processing = json.loads((source / 'meta/raw_processing.json').read_text())
    assert info['fps'] == 30 and info['codebase_version'] == 'v2.1'
    assert processing['smoothing'] is None and processing['roi'] is None
    assert info['features']['action']['names'] == ACTION_NAMES + ['dgripper']
    assert info['features']['observation.state']['names'] == STATE_NAMES + ['gripper_width']
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(source / 'meta', output / 'meta', ignore=shutil.ignore_patterns(
        'info.json', 'verification.json', 'build_progress.json', 'replaced_15hz.json'))
    frames = 0
    for path in sorted((source / 'data').rglob('*.parquet')):
        old = pq.read_table(path)
        new = old
        metadata = dict(old.schema.metadata)
        hf = json.loads(metadata[b'huggingface'])
        for key, width in [('observation.state', 9), ('action', 6)]:
            values = np.asarray(old[key].to_pylist(), dtype=np.float32)
            assert values.shape[1] == width + 1
            array = pa.FixedSizeListArray.from_arrays(pa.array(values[:, :width].ravel()), width)
            new = new.set_column(new.schema.get_field_index(key), key, array)
            hf['info']['features'][key]['length'] = width
        metadata[b'huggingface'] = json.dumps(hf).encode()
        new = new.replace_schema_metadata(metadata)
        dest = output / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new, dest)
        saved = pq.read_table(dest)
        for key in old.column_names:
            if key in ('observation.state', 'action'):
                np.testing.assert_array_equal(np.array(saved[key].to_pylist()), np.array(old[key].to_pylist())[:, :-1])
            else:
                assert saved[key].equals(old[key]), (path, key)
        frames += len(saved)
    assert frames == info['total_frames']
    stats_path = output / 'meta/episodes_stats.jsonl'
    rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
    for row in rows:
        for key in ('observation.state', 'action'):
            for name in ('min', 'max', 'mean', 'std'):
                row['stats'][key][name] = row['stats'][key][name][:-1]
    stats_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for key, names in [('observation.state', STATE_NAMES), ('action', ACTION_NAMES)]:
        info['features'][key]['shape'] = [len(names)]
        info['features'][key]['names'] = names
    processing.update(repo_id=repo_id, gripper_removed=True, physical_action_dim=6, physical_state_dim=9,
        state='measured q -> Panda FK -> xyz, rotation columns 0 and 1; no gripper',
        action='next selected measured TCP pose minus current; base-frame rotation log; no gripper')
    (output / 'meta/raw_processing.json').write_text(json.dumps(processing, indent=2)+'\n')
    report = {'source': str(source.resolve()), 'frames': frames, 'episodes': info['total_episodes'],
              'removed_columns': ['observation.state[9]: gripper_width', 'action[6]: dgripper'],
              'remaining_numeric_values_exact': True, 'all_other_columns_including_image_bytes_exact': True,
              'smoothing': None, 'roi': None, 'frame_filter': None, 'fps': 30}
    (output / 'meta/gripper_removal.json').write_text(json.dumps(report, indent=2)+'\n')
    validate_info(info)
    (output / 'meta/info.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE / 'data/threading_full_nosmooth_30hz')
    p.add_argument('--output', type=Path, default=HERE / 'data/threading_tcp6_nosmooth_30hz')
    p.add_argument('--repo-id', default='threading_real/threading_tcp6_nosmooth_30hz')
    args = p.parse_args()
    convert(args.source, args.output, args.repo_id)
