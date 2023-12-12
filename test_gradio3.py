import os
import torch
import random
import gradio as gr
from PIL import Image
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from audio_api import audio2parsec
from models import AE_AB,AE_A_variable,AE_B_Attention
from utils import get_name,get_params,get_path,get_point,point2img

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

param2idx = {name:i for i,name in enumerate(nameDict)}


modelA = AE_A_variable() # 编辑模型，input : source_keypoint,source_params,target_keypoint     output: target_params
modelB = AE_B_Attention() # 重建模型，input : target_keypoint,target_params   output: target_full
model = AE_AB(modelA,modelB) # input : source_keypoint,source_params,target_keypoint     output: target_full
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('weights/logs_edit_AB/cond_ckpt_epoch_100.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
model.to(device)
model.eval()

img_paths = get_path('data/airfoil/picked_uiuc_keypoint')
point_paths = get_path('data/airfoil/picked_uiuc')
name2params = get_params('data/airfoil/parsec_params.txt')


def show_img(path):
    return Image.open(path)

def fn_before(idx):
    idx = (idx - 1)%len(img_paths)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img

def fn_sample(idx):
    idx = random.randint(0,len(img_paths)-1)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img


def fn_next(idx): 
    idx = (idx+1)%len(img_paths)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img

# def process_audio(inputs):
#     input_audio,params_slider = inputs[0],inputs[1:]
#     rate, y = input_audio
#     write("output2.wav", rate, y)
#     prompt = audio2parsec("output2.wav")
#     for name,strength in prompt.items():
#         idx = param2idx[name]
#         params_slider[idx] = strength
    
#     ops = [prompt]+params_slider
#     return ops

def process_audio(input_audio):
    rate, y = input_audio
    write("output2.wav", rate, y)
    prompt = audio2parsec("output2.wav")
    return prompt

@torch.no_grad()
def infer(input_image,idx,slider0,slider1,slider2,slider3,slider4,slider5,slider6,slider7,slider8,slider9):
    path = point_paths[int(idx)]
    params_slider = [slider0,slider1,slider2,slider3,slider4,slider5,slider6,slider7,slider8,slider9]
    ## ----- 首先，实现编辑物理量的逻辑 ----- ##
    data = get_point(path)
    source_params = torch.FloatTensor(name2params[get_name(path)])
    source_keypoint = data['keypoint'] # [20,2]                
    source_params = source_params.unsqueeze(-1) #[10,1]
    source_keypoint = source_keypoint.unsqueeze(0) # [1,20,2]
    source_params = source_params.unsqueeze(0) # [1,10,1]
    source_keypoint = source_keypoint.to(device) 
    source_params = source_params.to(device)
    target_params_pred,target_point_pred = model.editing_param(source_keypoint, source_params,params_slider)  
    output_img = point2img(target_point_pred[0].cpu().numpy())
    # print(output_img.shape)
    plt.imsave('output.png', output_img, cmap='gray')
    output_img = Image.open('output.png')
    return output_img






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
              paramTex = gr.Textbox()
            bn_param.click(process_audio,
                            inputs=[input_audio] ,
                            outputs=[paramTex])
    with gr.Row():
      img = gr.Image(height=600,width=600,value=show_img(img_paths[0]), type='pil')
      img_out = gr.Image(height=600,width=600,value=show_img(img_paths[0]), type='pil')
    
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
    ips = [img,idx,slider0,slider1,slider2,slider3,slider4,slider5,slider6,slider7,slider8,slider9]
    submit_button.click(infer,
                        inputs=ips,
                        outputs=[img_out])

demo.launch()
