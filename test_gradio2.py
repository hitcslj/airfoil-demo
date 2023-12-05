import torchkeras 
from torchkeras.data import download_baidu_pictures 
# download_baidu_pictures('猫咪表情包',100)

import gradio as gr
from PIL import Image
import time,os
from pathlib import Path 
root_path = 'data/airfoil/picked_uiuc_img'
file_paths = []
## 使用os.walk()函数遍历文件夹
for root, dirs, files in os.walk(root_path):
  for file in files:
      file_path = os.path.join(root, file)
      # Do something with the file_path
      file_paths.append(file_path)

def show_img(path):
    return Image.open(path)

def fn_before(idx,path):
    idx = (idx - 1)%len(file_paths)
    path = file_paths[int(idx)]
    img = show_img(path)
    return idx,img,path

def fn_next(idx,path): 
    idx = (idx+1)%len(file_paths)
    path = file_paths[int(idx)]
    img = show_img(path)
    return idx,img,path


def get_default_msg():
    return "欢迎使用机翼编辑软件！"

nameDict = [
    'leading edge radius',
    'pressure crest location x-coordinate',
    'pressure crest location y-coordinate',
    'curvatures at the pressuresurface crest ',
    'angle of the pressure surface at the trailing edge',
    'suction crest location x-coordinate',
    'suction crest location y-coordinate',
    'curvatures at suction surface crest locations',
    'angle of the suction surface at the trailing edge',
    'diff between gt and parsec airfoil'
]

with gr.Blocks() as demo:
    msg = gr.TextArea(value=get_default_msg(), lines=3, max_lines=5)
    with gr.Row():
        with gr.Column():
          slider1 = gr.Slider(0, 10, step=0.1,label=nameDict[0],value=1)
          slider2 = gr.Slider(0, 10, step=0.1,label=nameDict[1],value=1)
          slider3 = gr.Slider(0, 10, step=0.1,label=nameDict[2],value=1)
          slider4 = gr.Slider(0, 10, step=0.1,label=nameDict[3],value=1)
          slider5 = gr.Slider(0, 10, step=0.1,label=nameDict[4],value=1)
          slider6 = gr.Slider(0, 10, step=0.1,label=nameDict[5],value=1)
          slider7 = gr.Slider(0, 10, step=0.1,label=nameDict[6],value=1)
          slider8 = gr.Slider(0, 10, step=0.1,label=nameDict[7],value=1)
          slider9 = gr.Slider(0, 10, step=0.1,label=nameDict[8],value=1)
          slider10 = gr.Slider(0, 10, step=0.1,label=nameDict[9],value=1)
        img = gr.Image(value=show_img(file_paths[0]), type='pil')
    gr.Audio(sources=["microphone","upload"],format="wav")
    with gr.Row():
        with gr.Row():
            bn_before = gr.Button("上一张")
            bn_next = gr.Button("下一张")
        with gr.Row():
            idx = gr.Number(value = 1,label='当前索引')
    # 在线更换路径
    
    path = gr.Text(file_paths[int(idx.value)], lines=1, label='当前图片路径') 
    bn_before.click(fn_before,
                    inputs=[idx,path],
                    outputs=[idx,img,path])
    bn_next.click(fn_next,
                  inputs=[idx],
                  outputs=[idx,img,path])

demo.launch()
