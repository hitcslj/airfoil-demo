import random
import gradio as gr
from PIL import Image
import os
from scipy.io.wavfile import write
from audio_api import audio2parsec


root_path = 'data/airfoil/picked_uiuc_keypoint'
file_paths = []
## 使用os.walk()函数遍历文件夹
for root, dirs, files in os.walk(root_path):
  for file in files:
      file_path = os.path.join(root, file)
      # Do something with the file_path
      file_paths.append(file_path)

def show_img(path):
    return Image.open(path)

def fn_before(idx):
    idx = (idx - 1)%len(file_paths)
    path = file_paths[int(idx)]
    img = show_img(path)
    return idx,img

def fn_sample(idx):
    idx = random.randint(0,len(file_paths)-1)
    path = file_paths[int(idx)]
    img = show_img(path)
    return idx,img


def fn_next(idx): 
    idx = (idx+1)%len(file_paths)
    path = file_paths[int(idx)]
    img = show_img(path)
    return idx,img

def process_audio(input_audio):
    rate, y = input_audio
    write("output2.wav", rate, y)
    prompt = audio2parsec("output2.wav")
    return prompt

def infer(input_img):
    return input_img




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

title = "# 机翼编辑软件"
with gr.Blocks() as demo:
    gr.Markdown(title)
    with gr.Column():
        with gr.Accordion("Physical parameters", open=False):
            slider0 = gr.Slider(0, 10, step=0.1,label=nameDict[0],value=1)
            slider1 = gr.Slider(0, 10, step=0.1,label=nameDict[1],value=1)
            slider2 = gr.Slider(0, 10, step=0.1,label=nameDict[2],value=1)
            slider3 = gr.Slider(0, 10, step=0.1,label=nameDict[3],value=1)
            slider4 = gr.Slider(0, 10, step=0.1,label=nameDict[4],value=1)
            slider5 = gr.Slider(0, 10, step=0.1,label=nameDict[5],value=1)
            slider6 = gr.Slider(0, 10, step=0.1,label=nameDict[6],value=1)
            slider7 = gr.Slider(0, 10, step=0.1,label=nameDict[7],value=1)
            slider8 = gr.Slider(0, 10, step=0.1,label=nameDict[8],value=1)
            slider9 = gr.Slider(0, 10, step=0.1,label=nameDict[9],value=1)
        with gr.Accordion("Audio edit parameters", open=False):
            input_audio = gr.Audio(sources=["microphone","upload"],format="wav")
            with gr.Row():
              bn_param = gr.Button("start edit")
              param = gr.Textbox()
            bn_param.click(process_audio,
                            inputs=[input_audio],
                            outputs=[param])
    with gr.Row():
      img = gr.Image(height=600,width=600,value=show_img(file_paths[0]), type='pil')
      img_out = gr.Image(height=600,width=600,value=show_img(file_paths[0]), type='pil')
    
    with gr.Row():
        with gr.Row():
            bn_before = gr.Button("pre")
            bn_samlpe = gr.Button("sample")
            bn_next = gr.Button("next")
        with gr.Row():
            idx = gr.Number(value = 1,label='cur idx')

    bn_before.click(fn_before,
                    inputs=[idx],
                    outputs=[idx,img])
    bn_samlpe.click(fn_sample,
                inputs=[idx],
                outputs=[idx,img])
    bn_next.click(fn_next,
                  inputs=[idx],
                  outputs=[idx,img])

    ## 新建一个button, 执行input_image，得到output_image
    submit_button = gr.Button("infer")
    submit_button.click(infer,
                        inputs=[img],
                        outputs=[img_out])

demo.launch()
