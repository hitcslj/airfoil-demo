import os
import torch
import random
import gradio as gr
from PIL import Image,ImageOps
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from audio_api import audio2parsec
from models import AE_AB,AE_A_variable,AE_B_Attention
from utils import get_name,get_params,get_path,get_point,point2img,point2img_new
# from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
#                           get_latest_points_pair, get_valid_mask,
#                           on_change_single_global_state)
from PIL import ImageDraw

nameDict = [
    'leading edge radius(前缘半径)',
    'pressure crest location x-coordinate(下表面峰值x)',
    'pressure crest location y-coordinate(下表面峰值y)',
    'curvatures at the pressuresurface crest(下表面曲率)',
    'angle of the pressure surface at the trailing edge(下表面后缘角)',
    'suction crest location x-coordinate(上表面峰值x)',
    'suction crest location y-coordinate(上表面峰值y)',
    'curvatures at suction surface crest locations(上表面曲率)',
    'angle of the suction surface at the trailing edge(上表面后缘角)',
    'diff between gt and parsec airfoil(parsec机翼与真实机翼差值)'
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

img_paths = get_path('data/airfoil/picked_uiuc_img')
point_paths = get_path('data/airfoil/picked_uiuc')
name2params = get_params('data/airfoil/parsec_params.txt')
global_points = []
target_size = (600, 600)

def clear():
    global global_points
    global_points = []
    return None, None


def show_img(path):
    img = Image.open(path)
    # padded_img = ImageOps.pad(img, target_size, method=Image.BOX, color=0)
    return  img

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

def process_audio(input_audio):
    rate, y = input_audio
    ## 保存音频
    print('save audio')
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
    point2img_new(target_point_pred[0].cpu().numpy())
    output_img = Image.open('output.png')


    # # ## ----- 其次，实现编辑控制点的逻辑 ----- ##
    # if len(global_points)==2:
    #     point1 = global_points[0]
    #     point2 = global_points[1]
    #     point1 = torch.FloatTensor(point1).unsqueeze(0).unsqueeze(0).to(device)
    #     point2 = torch.FloatTensor(point2).unsqueeze(0).unsqueeze(0).to(device)
    #     target_point_pred = model.editing_point(source_keypoint, source_params,point1,point2)
    #     point2img_new(target_point_pred[0].cpu().numpy())
    #     output_img = Image.open('output.png')


    return output_img

def get_points_with_draw(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0)
    global global_points
    print((x, y))
    global_points.append([x, y])    
    # print(type(image)) #转成 PIL.Image
    image = Image.fromarray(image)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)

    # 得到两个点，需要画线 TODO

    return image

def mirror(x):
    return x

quick_start_cn = """
        ## 快速开始
        1. 选择下方的airfoil example。
        2. 单击infer按钮， 对齐分辨率。
        3. 调整上方栏的Physical parameters，对机翼进行编辑。
        """
advanced_usage_cn = """
        ## 高级用法
        1. 使用大语言模型，语言转为文字，然后单击 `start audio to param` 。
        2. 单击 `Add Points` 添加关键点。
        3. 单击 `编辑灵活区域` 创建mask并约束未mask区域保持不变。
        """


title = "# 机翼编辑软件"
with gr.Blocks(theme=gr.themes.Soft()) as demo:
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
              bn_param = gr.Button("start audio to param")
              paramTex = gr.Textbox()
            bn_param.click(process_audio,
                            inputs=[input_audio] ,
                            outputs=[paramTex])
    with gr.Row():
      # img = ImageMask()  # NOTE: hard image size code here.
      # img_out = gr.ImageEditor(label="Output Image")
      img_in = gr.Image(label="Input Image")
      img_out = gr.Image(label="Output Image")
    with gr.Row():
        with gr.Row():
            bn_before = gr.Button("pre")
            bn_samlpe = gr.Button("sample")
            bn_next = gr.Button("next")
        with gr.Row():
            idx = gr.Number(value = 1,label='cur idx')
    # 新建一个button, 执行input_image，得到output_image
    img_in.select(get_points_with_draw, [img_in], img_in)
    with gr.Row():
        with gr.Column(scale=1, min_width=10):
            enable_add_points = gr.Button('Add Points')
        with gr.Column(scale=1, min_width=10):
            clear_points = gr.Button('clear')
        with gr.Column(scale=1, min_width=10):
            submit_button = gr.Button("infer")
    
    ips = [img_in,idx,slider0,slider1,slider2,slider3,slider4,slider5,slider6,slider7,slider8,slider9]
    bn_before.click(fn_before,
                    inputs=[idx],
                    outputs=[idx,img_in])
    bn_samlpe.click(fn_sample,
                inputs=[idx],
                outputs=[idx,img_in])
    bn_next.click(fn_next,
                  inputs=[idx],
                  outputs=[idx,img_in])
    clear_points.click(clear, outputs=[img_in,img_out])
    submit_button.click(infer,
                    inputs=ips,
                    outputs=[img_out])
    gr.Markdown("## Airfoil Examples")
    gr.Examples(
        examples=['data/airfoil/picked_uiuc_img/2032c.png','data/airfoil/picked_uiuc_img/a18.png'],
        inputs=[img_in],
        fn=mirror,
        outputs=[img_out],
        cache_examples=True
     )
    # Instruction
    with gr.Row():
        with gr.Column():
            quick_start_markdown = gr.Markdown(quick_start_cn)
        with gr.Column():
            advanced_usage_markdown = gr.Markdown(advanced_usage_cn)
if __name__=="__main__":
  demo.queue().launch(share=True)
