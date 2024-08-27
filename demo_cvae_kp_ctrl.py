import os
import torch
import random
import gradio as gr
from PIL import Image,ImageOps
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from audio_api import audio2parsec
from models import AE_A_Parsec,AE_A_Keypoint,CVAE,AE_AB_Parsec,AE_AB_Keypoint
from utils import get_name,get_params,get_path,get_point_cvae,point2img, generate_3D_from_dat,interpolate, bezier_curve, point2img, pixel_to_coordinate
import numpy as np
from PIL import ImageDraw
from utils import Fit_airfoil
import time
from scipy.signal import savgol_filter
from scipy.interpolate import make_lsq_spline, splev


nameDict =[
    '前缘半径' ,
    '上表面峰值' ,
    '下表面峰值' ,
    '后缘角']


param2idx = {
    '前缘半径':0,
    '上表面峰值':2,
    '下表面峰值':5,
    '后缘角':9}

p2idx = [0,2,5,9]

# 设置随机数种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device used:", device)

# 生成模型，input : target_params,target_keypoint   output: target_full
cvae = CVAE().to(device) 

# 编辑物理量模型，input : source_params,target_params,source_keypoint     output: target_keypoint
modelA_p = AE_A_Parsec().to(device) 

# 编辑物理量+生成， input : source_params,target_params,source_keypoint     output: target_full
model_p = AE_AB_Parsec(modelA_p,cvae).to(device) 
checkpoint = torch.load('weights/logs_edit_parsec_AB/ckpt_epoch_1000.pth', map_location='cpu')
model_p.load_state_dict(checkpoint['model'], strict=True)
model_p.eval()

# 编辑控制点模型，input : source_keypoint,target_keypoint,source_params     output: target_params
modelA_k = AE_A_Keypoint().to(device) 

# 编辑控制点+生成，input : source_keypoint,source_params,target_keypoint     output: target_full
model_k = AE_AB_Keypoint(modelA_k,cvae).to(device) 
checkpoint = torch.load('weights/logs_edit_keypoint_AB/cond_ckpt_epoch_1000.pth', map_location='cpu')
model_k.load_state_dict(checkpoint['model'], strict=True)
model_k.eval()

#生成翼型的dat文件路径
tmp_dat_path = 'generate_result/airplane_tmp111.dat'

img_paths = get_path('data/airfoil/supercritical_airfoil_img')
point_paths = get_path('data/airfoil/supercritical_airfoil')
name2params = get_params('data/airfoil/parsec_params_direct.txt')
global_points = []
target_size = (600, 600)

def clear():
    global global_points
    global_points = []
    return None, None


def show_img(path):
    img = Image.open(path)
    padded_img = ImageOps.pad(img, target_size, method=Image.BOX, color=(0, 0, 0)) # Padding 黑色区域
    return  padded_img

def fn_before(idx):
    idx = (idx - 1)%len(img_paths)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img

def fn_sample(idx):
    clear()
    idx = random.randint(0,len(img_paths)-1)
    path = point_paths[int(idx)]
    data = get_point_cvae(path)
    img = point2img(data["full"])
    return idx,img


def fn_next(idx): 
    idx = (idx+1)%len(img_paths)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img

def process_audio(input_audio,slider0,slider1,slider2,slider3):
    print('process audio')
    #print("input_audio",input_audio)
    rate, y = input_audio
    # 保存音频
    print('save audio')
    write("generate_result/output.wav", rate, y)
    name,strength = audio2parsec("generate_result/output.wav")
    if strength == -1:
        return name,slider0,slider1,slider2,slider3
    if name=='前缘半径':
        slider0 = strength
    elif name=='上表面峰值':
        slider1 = strength
    elif name=='下表面峰值':
        slider2 = strength
    elif name=='后缘角':
        slider3 = strength
    return name+'增加'+str(strength)+'倍',slider0,slider1,slider2,slider3

def prepare2airfoil(pred):
    # 上下表面
    upper = pred[:128][::4]
    upper[0][0],upper[0][1]=1,0
    mid = pred[128:129]
    mid[0][0],mid[0][1]=0,0
    low = pred[129:][::4]
    low[-1][0],low[-1][1]=1,0
    low[:,0] = upper[:,0][::-1]
   
    # 适配3D keypoint
    # 将上下表面和中点都concat在一起
    keypoint_3d = np.concatenate((upper,mid,low),axis=0)
    # print(keypoint_3d.shape)
    return keypoint_3d

def get_indices(center_index, left_n, right_n, total_points):
    indices = []
    for i in range(center_index - left_n, center_index + right_n + 1):
        # 处理索引超出范围的情况
        index = i % total_points
        indices.append(index)
    return indices

@torch.no_grad()
def infer(input_image,idx,slider0,slider1,slider2,slider3, slider_l, slider_r, slider_f):
    path = point_paths[int(idx)]
    params_slider = [slider0,slider1,slider2,slider3]
    ## ----- 首先，实现编辑物理量的逻辑 ----- ##
    data = get_point_cvae(path)
    print(f"Path: {path}")
    source_params = torch.FloatTensor(name2params[get_name(path)])
    source_keypoint = data['keypoint'] # [26,2]                
    source_params = source_params.unsqueeze(-1) #[11,1]
    source_keypoint = source_keypoint.unsqueeze(0) # [1,26,2]
    source_params = source_params.unsqueeze(0) # [1,11,1]
    source_keypoint = source_keypoint.to(device) 
    source_params = source_params.to(device)
    target_params = source_params.clone()
    for i,strength in enumerate(params_slider):
        target_params[:,p2idx[i]] *= strength
    target_point_pred = model_p.editing_params(source_params, target_params, source_keypoint)  

    #output_img = point2img(target_point_pred[0].cpu().numpy())
    # output_img = show_img('generate_result/output.png')

    # ## ----- 其次，实现编辑控制点的逻辑 ----- ##
    if len(global_points)==2:
        point1 = global_points[0]
        point2 = global_points[1]
        print("slider_f")
        print(slider_f)
        point1 = pixel_to_coordinate(point1[0], point1[1], data['full'], slider_f).to(device) 
        point2 = pixel_to_coordinate(point2[0], point2[1], data['full'], slider_f).to(device) 
        # 从两个点坐标获得对某个keypoint的修改
        target_keypoint = source_keypoint.clone() # [1,26,2]
        distances = torch.norm(target_keypoint[0] - point1, dim=1)
        nearest_index = torch.argmin(distances)
        print(f"最近的索引: {nearest_index}")
        print(f"Start: {point1} End: {point2}")
        print(f"Before kp: {target_keypoint[:,nearest_index]}")
        target_keypoint[:,nearest_index, 1] = point2[0, 1]
        print(f"After kp: {target_keypoint[:,nearest_index]}")
        #target_point_pred = model_k.editing_point(source_keypoint, target_keypoint, source_params)
        
        #指定区域的控制点优化
        target_point_pred_new = model_k.editing_point(source_keypoint, target_keypoint, source_params)
        mask = torch.zeros_like(target_point_pred_new[0])
        center_index = nearest_index * 10
        total_points = mask.shape[0]
        indices  = get_indices(center_index, slider_l, slider_r, total_points)
        mask[indices,:] = 1
        # print(mask.shape)
        # print(indices)
        target_point_pred[0] = target_point_pred_new[0] * mask + target_point_pred[0] * (1-mask)

        #output_img = point2img(target_point_pred[0].cpu().numpy())
        # output_img = show_img('generate_result/output.png')

    # 3D model
    #由于推理生成的点云连续性太差，使用bezier曲线拟合,进行平滑
    target_datt = target_point_pred[0].cpu().numpy().astype("float32")
    t_curve = np.linspace(0, 1, len(target_datt))
    target_datt = bezier_curve(target_datt, t_curve)

    output_img = point2img(target_datt)

    np.savetxt(tmp_dat_path, target_datt)

    #使用滤波后的预测点矩阵生成3D飞机模型
    # generate_3D_from_dat(target_datt)

    model_airplane = gr.Model3D(value='generate_result/airplane_tmp.stl',
                                label='Output Airplane',
                                clear_color = [0,0,0,1],
                                camera_position=(-103,83,1200))

    ## 这里可以计算物理量的误差
    target_parsec_features =  Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features # 11个物理量
    error = ''
    # import pdb; pdb.set_trace()
    for i,strength in enumerate(params_slider):
        idx = p2idx[i]
        name = nameDict[i]
        s_e = source_params[0][idx].cpu().numpy().item()
        t_e = target_parsec_features[idx].item()
        ratio = abs(t_e/(s_e+1e-9))
        error += f'{name}: source_param {s_e:.4f}, target_param {t_e:.4f}, ratio {ratio:.2f}' + '\n'
        

    return output_img,error,model_airplane

def draw_arrow(draw, start, end, arrow_size=10, color=(255, 0, 0), arrow_fill=True):
    # 绘制箭头的主体线段
    draw.line(start + end, fill=color, width=2)
    
    # 绘制箭头的头部
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length > 0:
        ux = dx / length
        uy = dy / length
        # 箭头头部的三个顶点
        points = [
            (end[0] - ux * arrow_size - uy * arrow_size, end[1] - uy * arrow_size + ux * arrow_size),
            (end[0], end[1]),
            (end[0] - ux * arrow_size + uy * arrow_size, end[1] - uy * arrow_size - ux * arrow_size)
        ]
        draw.polygon(points, fill=color if arrow_fill else None)

def get_points_with_draw(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (255, 255, 0)
    global global_points
    print((x, y))
    global_points.append([x, y])    
    # print(type(image)) #转成 PIL.Image
    # image = Image.fromarray(image)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)

    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)

    if len(global_points)==2:
        draw_arrow(draw, global_points[0], global_points[1])
    # 得到两个点，需要画线 TODO （source -> target）

    return image


def reset(slider0,slider1,slider2,slider3):
    slider0  = 1
    slider1  = 1
    slider2  = 1
    slider3  = 1
    return slider0,slider1,slider2,slider3
   

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
# gr.themes.builder()

title = "# 机翼编辑软件"
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(title)
    with gr.Column():
        with gr.Accordion("物理参数", open=False):
            with gr.Row():
              with gr.Column(min_width=200):
                img0 = gr.Image(value='assets/example_parsec_0.png',show_label=False,show_download_button=False)
                slider0 = gr.Slider(0, 10, step=0.1,label=nameDict[0],value=1)
              with gr.Column(min_width=200):
                img1 = gr.Image(value='assets/example_parsec_1.png',show_label=False,show_download_button=False)
                slider1 = gr.Slider(0, 10, step=0.1,label=nameDict[1],value=1)
              with gr.Column(min_width=200):
                img2 = gr.Image(value='assets/example_parsec_2.png',show_label=False,show_download_button=False)
                slider2 = gr.Slider(0, 10, step=0.1,label=nameDict[2],value=1)
              with gr.Column(min_width=200):
                img3 = gr.Image(value='assets/example_parsec_3.png',show_label=False,show_download_button=False)
                slider3 = gr.Slider(0, 10, step=0.1,label=nameDict[3],value=1)
            reset_param = gr.Button("reset")
            reset_param.click(reset,
                            inputs=[slider0,slider1,slider2,slider3],
                            outputs=[slider0,slider1,slider2,slider3])
        with gr.Accordion("语音控制", open=False):
            input_audio = gr.Audio(sources=["microphone","upload"],format="wav")
            with gr.Row():
              bn_param = gr.Button("start audio to param")
              paramTex = gr.Textbox()
            bn_param.click(process_audio,
                            inputs=[input_audio,slider0,slider1,slider2,slider3] ,
                            outputs=[paramTex,slider0,slider1,slider2,slider3])
        with gr.Accordion("控制点修改范围", open=False):
            with gr.Row():
                slider_l = gr.Slider(0, 100, step=1, label="右侧控制点修改范围", value=20)
                slider_r = gr.Slider(0, 100, step=1, label="左侧侧控制点修改范围", value=20)
                slider_f = gr.Slider(0, 5, step=0.1, label="拖拽力度", value=2)
    with gr.Row():
      img_in = gr.Image(label="Input Airfoil",width=600,type='pil')
      img_out = gr.Image(label="Output Airfoil",width=600,type='pil')
      ## 3D model show
      model_airplane = gr.Model3D(label='Output Airplane',camera_position=(-103,83,7000)) # 调位姿
    with gr.Row():
        with gr.Row():

            bn_before = gr.Button("前一个")
            bn_samlpe = gr.Button("随机")
            bn_next = gr.Button("后一个")
        with gr.Row():
            error = gr.Textbox(label='物理误差')
            idx = gr.Number(value = 1,label='cur idx',visible=False)
    # 新建一个button, 执行input_image，得到output_image
    img_in.select(get_points_with_draw, [img_in], img_in)
    with gr.Row():
        with gr.Column(scale=1, min_width=10):
            enable_add_points = gr.Button('加点')
        with gr.Column(scale=1, min_width=10):
            clear_points = gr.Button('清空')
        with gr.Column(scale=1, min_width=10):
            submit_button = gr.Button("生成")

    ## 编辑后物理参数的显示
    ips = [img_in,idx,slider0,slider1,slider2,slider3,slider_l,slider_r,slider_f]
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
                    outputs=[img_out,error,model_airplane])
    gr.Markdown("## Airfoil Examples")
    gr.Examples(
        examples=['data/airfoil/demo/supercritical_airfoil_img/air05_000001.png',
        'data/airfoil/demo/supercritical_airfoil_img/air05_000002.png'],
        inputs=[img_in]
     )
    # Instruction
    with gr.Row():
        with gr.Column():
            quick_start_markdown = gr.Markdown(quick_start_cn)
        with gr.Column():
            advanced_usage_markdown = gr.Markdown(advanced_usage_cn)
if __name__=="__main__":
  demo.queue().launch(share=True)
  print('http://localhost:7860?__theme=dark')
  