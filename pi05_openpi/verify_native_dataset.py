"""Verify restored native RGB against raw videos and numeric labels against v4."""
import io
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from transforms import FRONT, SIDE


def verify(root):
    root = Path(root)
    info = json.loads((root/'meta/info.json').read_text())
    proc = json.loads((root/'meta/raw_processing.json').read_text())
    source = Path(proc['source_numeric_dataset'])
    total, checks = 0, 0
    for report in proc['per_episode']:
        ep = report['episode_index']
        rel = f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
        columns = ['observation.state','action','timestamp','frame_index','episode_index','index','task_index']
        before, after = (pq.read_table(p/rel,columns=columns) for p in (source,root))
        assert before.equals(after), ep
        total += len(after)
        if ep not in (0, info['total_episodes']//2, info['total_episodes']-1):
            continue
        table = pq.read_table(root/rel,columns=[FRONT,SIDE])
        alignment = np.load(root/f'meta/alignment/episode_{ep:06d}.npz')
        for cam,key in [('cam1',SIDE),('cam3',FRONT)]:
            cap=cv2.VideoCapture(str(Path(report['source'])/(cam+'.mp4')))
            try:
                for frame in [0,len(table)//2,len(table)-1]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(alignment[cam+'_frame'][frame]))
                    ok,bgr=cap.read();assert ok
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                    saved=np.asarray(Image.open(io.BytesIO(table[key][frame].as_py()['bytes'])))
                    assert saved.shape==(480,640,3)
                    np.testing.assert_array_equal(saved,rgb)
                    checks+=1
            finally:cap.release()
    assert total==info['total_frames']
    record=dict(episodes=info['total_episodes'],frames=total,
                numeric_columns_exact=True,raw_video_pixel_checks=checks,
                stored_image_hw=[480,640],model_image_hw=[490,644],resize=False)
    (root/'meta/native_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path,nargs='?',default=Path(__file__).parent/'data/threading_tcp6_native640_30hz')
    verify(p.parse_args().dataset)
