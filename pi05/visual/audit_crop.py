#!/usr/bin/env python3
"""Render real raw-video frames through the checkpoint's exact crop transform."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from threading_real.pi05.visual.crop import load, transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    paths=[root/'outputs'/name/'checkpoints/005000/pretrained_model/visual_preprocessing.json'
           for name in ['threading_crop_v1','threading_crop_chunk20_v1']]
    config=load(paths[0]);assert config==load(paths[1])
    episodes=[Path('/home/huiyuan/threading_new/threading_new_1/episode_001'),
              Path('/home/huiyuan/threading_new/threading_new_1/episode_025'),
              Path('/home/huiyuan/threading_new/threading_new_2/episode_015')]
    records=[]
    for e,episode in enumerate(episodes):
        for phase in [0,.5,.95]:
            fig,axes=plt.subplots(2,3,figsize=(13,8),gridspec_kw={'width_ratios':[1.7,1.1,1]})
            for row,camera in enumerate(['cam1','cam3']):
                cap=cv2.VideoCapture(str(episode/f'{camera}.mp4'))
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                index=round((n-1)*phase);cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,bgr=cap.read();cap.release()
                if not ok:raise RuntimeError(f'{episode}/{camera} frame {index}')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                x0,y0,x1,y1=config['cameras'][camera]['roi']
                roi=rgb[y0:y1,x0:x1];model=transform(rgb,camera,config)
                stem=f'e{e}_{int(phase*100):02d}_{camera}'
                cv2.imwrite(str(a.output/f'{stem}_raw.png'),bgr)
                cv2.imwrite(str(a.output/f'{stem}_input224.png'),cv2.cvtColor(model,cv2.COLOR_RGB2BGR))
                for ax,img in zip(axes[row],[rgb,roi,model]):
                    ax.imshow(img,interpolation='nearest');ax.set_axis_off()
                axes[row,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#00e54c',linewidth=2))
                axes[row,0].set_title(f'{camera} raw 640x480; frame {index}, {index/fps:.2f}s')
                axes[row,1].set_title(f'ROI [{x0},{y0},{x1},{y1}]\n{x1-x0}x{y1-y0} native pixels')
                axes[row,2].set_title('Model input: 224x224\nblack padding retained')
                records.append(dict(episode=str(episode),phase=phase,camera=camera,frame=index,seconds=index/fps,
                    roi=[x0,y0,x1,y1],raw=f'{stem}_raw.png',input=f'{stem}_input224.png'))
            fig.suptitle(f'{episode.parent.name}/{episode.name} | {phase:.0%} of video duration (not path progress)',fontsize=13)
            fig.tight_layout();fig.savefig(a.output/f'episode{e}_{int(phase*100):02d}.png',dpi=140);plt.close(fig)
    (a.output/'manifest.json').write_text(json.dumps(dict(config=config,records=records),indent=2)+'\n')
    html='''<!doctype html><meta charset="utf-8"><title>pi05 裁剪审计</title>
<style>body{font-family:sans-serif;background:#182029;color:#eee;margin:24px}select{padding:8px} .row{display:flex;gap:24px;margin:20px 0;align-items:start} .raw{position:relative;width:640px;height:480px}.raw img{width:640px;height:480px}.roi{position:absolute;border:2px solid #00e54c;box-sizing:border-box} .input img{width:224px;height:224px;image-rendering:pixelated} .zoom img{width:448px;height:448px;image-rendering:pixelated} small{color:#bac5cf}</style>
<h1>pi05 实际裁剪：chunk 10 / 20 配置相同</h1>
<p>左：原始 640×480 图像，绿框是固定 ROI。中：模型输入，原生 224×224。右：仅为查看细节的 2× 放大，不增加模型分辨率。</p>
<label>视频轨迹 <select id="ep"></select></label> <label>视频时间位置 <select id="phase"><option value="0">开始 0%</option><option value="0.5" selected>中段 50%</option><option value="0.95">末段 95%</option></select></label>
<p>时间百分比不是 TCP 路程进度；相机各按视频长度选帧，此页用于裁剪检查，不做跨相机时间同步评估。</p>
<div id="rows"></div><p>cam1：260×300 → 194×224，左右各补 15 px 黑边。cam3：530×265 → 224×112，上下各补 56 px 黑边。</p>
<script>const data=DATA;const episodes=[...new Set(data.records.map(r=>r.episode))];
episodes.forEach((e,i)=>ep.add(new Option(e.split('/').slice(-2).join('/'),i)));
function render(){rows.innerHTML='';for(const r of data.records.filter(r=>r.episode===episodes[+ep.value]&&r.phase===+phase.value)){
let [x0,y0,x1,y1]=r.roi;rows.innerHTML+=`<h2>${r.camera} — 帧 ${r.frame}，${r.seconds.toFixed(2)} 秒</h2><div class="row"><div class="raw"><img src="${r.raw}"><div class="roi" style="left:${x0}px;top:${y0}px;width:${x1-x0}px;height:${y1-y0}px"></div></div><div class="input"><img src="${r.input}"><p>实际 224×224 输入</p></div><div class="zoom"><img src="${r.input}"><p>2× 查看（非模型分辨率）</p></div></div>`;}}
ep.onchange=phase.onchange=render;render();</script>'''
    html=html.replace('DATA',json.dumps(dict(records=records)))
    (a.output/'index.html').write_text(html)
    print(a.output.resolve())

if __name__=='__main__':main()
