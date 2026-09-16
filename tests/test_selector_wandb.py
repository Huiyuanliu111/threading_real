import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from chunk_selector.chunk_dataset import ChunkFeatureWriter


@pytest.mark.parametrize('fail', [False, True])
def test_online_run_records_metrics_and_failure(tmp_path, monkeypatch, fail):
    from scripts.chunk_selector import train
    source=tmp_path/'features.h5'
    with ChunkFeatureWriter(source,feature_shape=(2,4),candidate_chunks=(8,20),
        metadata={'label_source':'progress_rule','fit_episode_ids':['train'],'validation_episode_ids':['val']}) as writer:
        writer.append(np.ones((4,2,4),dtype=np.float32),[0,1,0,1],
                      episode_ids=['train','train','val','val'],decision_steps=[0,1,0,1],
                      target_probabilities=[[1,0],[0,1],[1,0],[0,1]])
    run=SimpleNamespace(id='test',url='https://wandb.ai/test/run',entity='test',project='test',
                        settings=SimpleNamespace(mode='online'),summary={},define_metric=Mock(),log=Mock(),finish=Mock())
    init=Mock(return_value=run)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    output=tmp_path/'model'
    monkeypatch.setattr(sys,'argv',['train',str(source),'--output-dir',str(output),
        '--device','cpu','--epochs','1','--batch-size','2','--d-model','8',
        '--num-layers','1','--dim-feedforward','16','--wandb-mode','online'])
    if fail:
        monkeypatch.setattr(train,'_run_epoch',Mock(side_effect=RuntimeError('test failure')))
        with pytest.raises(RuntimeError,match='test failure'):
            train.main()
        run.finish.assert_called_once_with(exit_code=1)
    else:
        assert train.main()==0
        run.finish.assert_called_once_with()
        metrics=run.log.call_args.args[0]
        assert metrics['epoch']==1 and 'validation/chunk_mae' in metrics
        assert run.summary['best_epoch']==1
        assert (output/'chunk_selector.safetensors').exists()
    assert init.call_args.kwargs['mode']=='online'
    assert init.call_args.kwargs['config']['selector_config']['candidate_chunks']==(8,20)
    assert json.loads((output/'wandb_run.json').read_text())['mode']=='online'
