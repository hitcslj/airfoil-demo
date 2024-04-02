import os
import torch
import random
import gradio as gr
from PIL import Image,ImageOps
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from audio_api import audio2parsec
from models import AE_A_Parsec,AE_A_Keypoint,Diff_AB_Keypoint,Diff_AB_Parsec,VAE
from models import get_diffusion
from utils import get_name,get_params,get_path,get_point_diffusion,point2img
from cquav.wing.airfoil import load_airfoils_collection
from cquav.wing.airfoil import Airfoil
from cquav.wing.profile import AirfoilSection
from cquav.wing.rect_console import RectangularWingConsole
import cadquery as cq
import numpy as np
from PIL import ImageDraw
import trimesh
from utils import Fit_airfoil

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


airfoils_collection = load_airfoils_collection()
airfoil_data = airfoils_collection["NACA 6 series airfoils"]["NACA 64(3)-218 (naca643218-il)"]

airfoil = Airfoil(airfoil_data) # 将这个airfoil对象的profile属性

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE().to(device) # latent_space(b,128)->(b,257*2)上进行diffusion
checkpoint = torch.load('weights/vae/ckpt_epoch_30000.pth', map_location='cpu')
vae.load_state_dict(checkpoint['model'])
vae.eval()


diffusion = get_diffusion().to(device) # 生成模型，condition : target_params(b,11),target_keypoint(b,26*2)   output: target_full_latent(b,128)
diffusion.load_state_dict(torch.load('weights/dit/model-airfoil-300000.pth'))
diffusion.eval()


modelA_p = AE_A_Parsec().to(device) # 编辑物理量模型，input : source_params(b,11,1),target_params(b,11,1),source_keypoint(b,26,2)     output: target_keypoint(b,26,2)
modelA_p_checkpoint = torch.load('weights/logs_edit_parsec/cond_ckpt_epoch_10000.pth', map_location='cpu')
modelA_p.load_state_dict(modelA_p_checkpoint['model'],strict=True)
modelA_p.eval()


model_p = Diff_AB_Parsec(modelA_p, vae, diffusion).to(device) # input : source_params(b,11,1),target_params(b,11,1),source_keypoint(b,26,2)      output: target_full(b,257,2)
model_p.eval()


modelA_k = AE_A_Keypoint().to(device) # 编辑控制点模型，input : source_keypoint(b,26,2),target_keypoint(b,26,2),source_params(b,11,2)     output: target_params(b,11,1)
modelA_k_checkpoint = torch.load('weights/logs_edit_keypoint/cond_ckpt_epoch_10000.pth', map_location='cpu')
modelA_k.load_state_dict(modelA_k_checkpoint['model'],strict=True)
modelA_k.eval()

model_k = Diff_AB_Keypoint(modelA_k,vae, diffusion).to(device) # input : source_keypoint(b,26,2),target_keypoint(b,26,2),source_params(b,11,2)     output: target_full(b,257,2)
model_k.eval()


img_paths = get_path('data/airfoil/demo/supercritical_airfoil_img')
point_paths = get_path('data/airfoil/demo/supercritical_airfoil')
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
    idx = random.randint(0,len(img_paths)-1)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img


def fn_next(idx): 
    idx = (idx+1)%len(img_paths)
    path = img_paths[int(idx)]
    img = show_img(path)
    return idx,img

def process_audio(input_audio,slider0,slider1,slider2,slider3):
    print('process audio')
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


@torch.no_grad()
def infer(input_image,idx,slider0,slider1,slider2,slider3):
    path = point_paths[int(idx)]
    params_slider = [slider0,slider1,slider2,slider3]
    ## ----- 首先，实现编辑物理量的逻辑 ----- ##
    data = get_point_diffusion(path)
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

    point2img(target_point_pred[0].cpu().numpy())
    output_img = show_img('generate_result/output.png')

    # ## ----- 其次，实现编辑控制点的逻辑 ----- ##
    if len(global_points)==2:
        point1 = global_points[0]
        point2 = global_points[1]
        point1 = torch.FloatTensor(point1).unsqueeze(0).unsqueeze(0).to(device)
        point2 = torch.FloatTensor(point2).unsqueeze(0).unsqueeze(0).to(device)
        # 从两个点坐标获得对某个keypoint的修改
        target_keypoint = source_keypoint.clone() # [1,26,2]

        target_keypoint[:,6,1]*=5
        target_point_pred = model_k.editing_point(source_keypoint, target_keypoint, source_params)
        point2img(target_point_pred[0].cpu().numpy())
        output_img = show_img('generate_result/output.png')

    # 3D model
    # 处理得到 (51,)
    keypoint_3d = prepare2airfoil(target_point_pred[0].cpu().numpy())
    # print(keypoint_3d.shape)
    # print(keypoint_3d)
    airfoil_x = airfoil.profile['A'].copy()
    airfoil_y = airfoil.profile['B'].copy()
    try:
      airfoil.profile['A'] =keypoint_3d[:,0]  # (51,)
      airfoil.profile['B'] =keypoint_3d[:,1]  # (51,)
      airfoil_section = AirfoilSection(airfoil, chord=200)
      wing_console = RectangularWingConsole(airfoil_section, length=800)
      assy = cq.Assembly()
      assy.add(wing_console.foam, name="foam", color=cq.Color("lightgray"))
      assy.add(wing_console.front_box, name="left_box", color=cq.Color("yellow"))
      assy.add(wing_console.central_box, name="central_box", color=cq.Color("yellow"))
      assy.add(wing_console.rear_box, name="right_box", color=cq.Color("yellow"))
      assy.add(wing_console.shell, name="shell", color=cq.Color("lightskyblue2"))
      # show(assy, angular_tolerance=0.1)
      assy.save(path='generate_result/output3d.stl',exportType='STL')
    except:
      airfoil.profile['A'] = airfoil_x  # (51,)
      airfoil.profile['B'] = airfoil_y  # (51,)
      airfoil_section = AirfoilSection(airfoil, chord=200)
      wing_console = RectangularWingConsole(airfoil_section, length=800)
      assy = cq.Assembly()
      assy.add(wing_console.foam, name="foam", color=cq.Color("lightgray"))
      assy.add(wing_console.front_box, name="left_box", color=cq.Color("yellow"))
      assy.add(wing_console.central_box, name="central_box", color=cq.Color("yellow"))
      assy.add(wing_console.rear_box, name="right_box", color=cq.Color("yellow"))
      assy.add(wing_console.shell, name="shell", color=cq.Color("lightskyblue2"))
      # show(assy, angular_tolerance=0.1)
      assy.save(path='generate_result/output3d.stl',exportType='STL')
    model3d =  gr.Model3D(value='generate_result/output3d.stl',label='Output 3D Airfoil',camera_position=(-90,2,800))
    ## 这里可以计算物理量的误差
    target_parsec_features =  Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features # 11个物理量
    error = ''
    for i,strength in enumerate(params_slider):
        idx = p2idx[i]
        name = nameDict[i]
        s_e = source_params[0][idx].cpu().numpy().item()
        t_e = target_parsec_features[idx].item()
        ratio = abs(t_e/(s_e+1e-9))
        error += f'{name}: source_param {s_e:.4f}, target_param {t_e:.4f}, ratio {ratio:.2f}' + '\n'
        

    return output_img,model3d,error

def get_points_with_draw(image, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (255, 255, 0)
    global global_points
    print((x, y))
    global_points.append([x, y])    
    # print(type(image)) #转成 PIL.Image
    image = Image.fromarray(image)
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)

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
    with gr.Row():
      # img = ImageMask()  # NOTE: hard image size code here.
      # img_out = gr.ImageEditor(label="Output Image")
      img_in = gr.Image(label="Input Airfoil",width=600,height=600)
      img_out = gr.Image(label="Output Airfoil",width=600,height=600)
      ## 3D model show
      # model3d = gr.Model3D(label='Output 3D Airfoil',value='assets/airfoil.stl',camera_position=(270,0,None)) # 调位姿
      model3d = gr.Model3D(label='Output 3D Airfoil',camera_position=(-90,2,800)) # 调位姿
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
    ips = [img_in,idx,slider0,slider1,slider2,slider3]
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
                    outputs=[img_out,model3d,error])
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