import json
import numpy as np
import pytest
from chunk_selector.progress_rule import ProgressRule
from chunk_selector.chunk_dataset import ChunkFeatureWriter, episode_split_indices


@pytest.mark.parametrize('task,split,h,H,start,end', [
    ('threading',.5,8,20,20,8), ('maze',.3,20,50,20,50)])
def test_direction_and_transition(task,split,h,H,start,end):
    xyz=np.column_stack((np.linspace(0,1,101), np.zeros((101,2))))
    p,c,s=ProgressRule(task,split,h,H).label(xyz)
    assert c[0]==start and c[-1]==end
    np.testing.assert_allclose(p[round(split*100)], [.5,.5], atol=1e-6)
    np.testing.assert_allclose(p.sum(1),1)
    assert np.all(np.diff(c) <= 1e-6) if task=='threading' else np.all(np.diff(c) >= -1e-6)
    assert set(np.unique(c[s < split-.05])) == {start}
    assert set(np.unique(c[s > split+.05])) == {end}


def test_pause_and_return_use_path_not_position_or_frames():
    xyz=np.array([[0,0,0],[1,0,0],[1,0,0],[0,0,0]], dtype=float)
    p,c,s=ProgressRule('threading',.5,8,20).label(xyz)
    np.testing.assert_allclose(s,[0,.5,.5,1])
    assert p[0,0]==0 and p[-1,0]==1
    assert c[1]==c[2]
    with pytest.raises(ValueError,match='stationary'):
        ProgressRule('maze',.3,20,50).label(np.zeros((5,3)))


@pytest.mark.parametrize('split,width', [(0,.1),(.5,0),(.1,.3),(.5,float('nan'))])
def test_invalid_transition(split,width):
    with pytest.raises(ValueError):
        ProgressRule('threading',split,8,20,width)


def test_progress_manifest_preserves_episode_split(tmp_path):
    path=tmp_path/'cache.h5'
    with ChunkFeatureWriter(path, feature_shape=(2,4),candidate_chunks=(8,20),
        metadata={'label_source':'progress_rule','fit_episode_ids':['b'],'validation_episode_ids':['a']}) as w:
        w.append(np.zeros((3,2,4)),[0,1,0],episode_ids=['a','b','b'],decision_steps=[0,0,1])
    train,val=episode_split_indices(path,seed=999,val_ratio=.9)
    np.testing.assert_array_equal(train,[1,2]);np.testing.assert_array_equal(val,[0])


def test_label_pipeline_maze(tmp_path):
    import h5py
    import pyarrow.parquet as pq
    from scripts.chunk_selector.label_progress import create_labels
    source=tmp_path/'source.h5'
    with h5py.File(source,'w') as f:
        f.attrs['format']='maze-mvt-pointcloud-v1'; f.attrs['fixed_z_m']=.1
        for i in range(5):
            f.create_group(f'episode_{i:06d}').create_dataset('tcp_xy',data=np.column_stack((np.linspace(0,1,101),np.zeros(101))))
    output=tmp_path/'labels'
    summary=create_labels(source,output,task='maze',h=20,H=50)
    assert summary['total_frames']==505 and len(summary['validation_episode_ids'])==1
    rows=pq.read_table(output/'labels.parquet').to_pandas()
    for _,group in rows.groupby('episode_key'):
        assert group.execution_steps.iloc[0]==20 and group.execution_steps.iloc[-1]==50
        assert np.all(np.diff(group.soft_chunk_size)>=-1e-6)
    with pytest.raises(FileExistsError):
        create_labels(source,output,task='maze',h=20,H=50)
