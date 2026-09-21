"""Current-camera vs training embeddings from deployed OpenPI vision encoder."""
from pathlib import Path
import json,io,time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw
from deploy import LocalBackend,letterbox
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/live_visual_similarity'

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),min=float(x.min()),median=float(np.median(x)),max=float(x.max()),p05=float(np.quantile(x,.05)))

def main():
 cfg=json.loads((ROOT/'deployment_config.json').read_text());split=cfg['validation_split']
 refs=[];images={'front':[],'side':[]}
 for ep,n in enumerate(split['episode_lengths']):
  frames=np.linspace(0,n-1,5).round().astype(int)
  table=pq.read_table(ROOT/f'data/threading_tcp6_nosmooth_30hz/data/chunk-000/episode_{ep:06d}.parquet',columns=['observation.images.exterior_image_1_left','observation.images.exterior_image_2_right'])
  for f in frames:
   r=table.slice(int(f),1).to_pylist()[0]
   refs.append(dict(episode=ep,frame=int(f),split='train' if ep in split['train_episodes'] else 'val'))
   for cam,key in [('front','observation.images.exterior_image_1_left'),('side','observation.images.exterior_image_2_right')]:images[cam].append(np.array(Image.open(io.BytesIO(r[key]['bytes']))))
 for i in range(10):
  refs.append(dict(live=i,split='live'))
  for cam in images:images[cam].append(letterbox(np.array(Image.open(OUT/f'live_{i:02d}_{cam}.png'))))
 backend=LocalBackend(ROOT/'checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000',ROOT/'deployment_config.json')
 from openpi.shared.nnx_utils import module_jit
 import jax.numpy as jnp
 encode=module_jit(backend.policy._model.PaliGemma.img.__call__,static_argnames=('train',))
 features={};start=time.monotonic()
 for cam,ims in images.items():
  pooled=[];spatial=[]
  for i in range(0,len(ims),2):
   batch=np.stack(ims[i:i+2]).astype(np.float32)/255*2-1
   tokens,_=encode(jnp.asarray(batch),train=False)
   tokens=np.asarray(tokens,dtype=np.float32)
   assert tokens.shape[1]==256 and np.isfinite(tokens).all()
   pooled.extend(tokens.mean(1))
   grid=tokens.reshape(len(batch),4,4,4,4,-1).mean((2,4))
   spatial.extend(grid.reshape(len(batch),-1))
   if i%100==0:print(cam,i,'/',len(ims),round(time.monotonic()-start,1),flush=True)
  features[cam]={'global':norm(np.array(pooled)),'spatial4x4':norm(np.array(spatial))}
 train=np.array([i for i,r in enumerate(refs) if r['split']=='train']);val=np.array([i for i,r in enumerate(refs) if r['split']=='val']);live=np.arange(400,410)
 report=dict(checkpoint=backend.checkpoint,train_images_per_camera=len(train),val_images_per_camera=len(val),live_images_per_camera=10,
  method='Actual deployed SigLIP image tokens before language fusion; cosine of mean token embedding and flattened4x4 spatial mean-pooled tokens. Full images, no ROI, augmentation or smoothing.',cameras={})
 for cam in images:
  result={}
  for kind,feat in features[cam].items():
   scores=feat[live]@feat[train].T;vs=feat[val]@feat[train].T;ts=feat[train]@feat[train].T
   eps=np.array([refs[i]['episode'] for i in train]);ts[eps[:,None]==eps[None,:]]=-np.inf
   best=scores.max(1);baseline=vs.max(1)
   result[kind]=dict(live_nearest_train=stats(best),val_nearest_train=stats(baseline),train_nearest_other_episode=stats(ts.max(1)),
    live_below_val_p05=int((best<np.quantile(baseline,.05)).sum()),live_percentiles_in_val=[float((baseline<=v).mean()*100) for v in best],
    top5=[[dict(**refs[train[j]],cosine=float(scores[k,j])) for j in np.argsort(scores[k])[::-1][:5]] for k in range(10)])
  report['cameras'][cam]=result
  scores=features[cam]['spatial4x4'][live]@features[cam]['spatial4x4'][train].T
  best=np.argsort(scores[0])[::-1][:5]
  panel=Image.new('RGB',(224*6,254),'white');draw=ImageDraw.Draw(panel)
  for col,(idx,label) in enumerate([(400,'LIVE')]+[(int(train[j]),f"ep{refs[train[j]]['episode']} f{refs[train[j]]['frame']} cos={scores[0,j]:.3f}") for j in best]):
   panel.paste(Image.fromarray(images[cam][idx]),(col*224,30));draw.text((col*224+4,8),label,fill='black')
  panel.save(OUT/f'nearest_{cam}.png')
 # Compare whether physical camera labels agree, using the same learned encoder.
 for cam,other in [('front','side'),('side','front')]:
  wrong=features[cam]['spatial4x4'][live]@features[other]['spatial4x4'][train].T
  report['cameras'][cam]['opposite_camera_nearest_train']=stats(wrong.max(1))
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 np.savez_compressed(OUT/'features.npz',**{cam+'_'+kind:f for cam,d in features.items() for kind,f in d.items()})
 (OUT/'samples.json').write_text(json.dumps(refs,indent=2)+'\n')
 for cam,d in report['cameras'].items():
  print(cam,json.dumps({k:{x:v for x,v in r.items() if x!='top5'} if k!='opposite_camera_nearest_train' else r for k,r in d.items()}),flush=True)
if __name__=='__main__':main()
