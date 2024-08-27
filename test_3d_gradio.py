import gradio as gr

import aerosandbox as asb
import aerosandbox.numpy as np
from shutil import which
from utils import generate_3D_from_dat,get_point_cvae,get_path
import random


path = "generate_result/airplane_tmp2.stl"
point_paths = get_path('data/airfoil/supercritical_airfoil')
wing_dats = [get_point_cvae(point_paths[i])['full'] for i in range(len(point_paths))]

def fn_sample(idx):
  idx = random.randint(0,len(wing_dats)-1)
  wing_dat = wing_dats[idx]
  generate_3D_from_dat(wing_dat, stl_path = path)
  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-103,83,1200)) 
  return idx,model3d

# def infer(slider_bar_x,slider_bar_y,slider_bar_z):
#   model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(slider_bar_x,slider_bar_y,slider_bar_z)) # 调位姿
#   return model3d

def infer(slider_bar):
  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-103,83,1200)) # 调位姿
  return model3d


with gr.Blocks() as demo:

  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-103,83,1200)) # 调位姿
  slider_bar = gr.Slider(1, len(wing_dats), step=1,label='slider_bar_x',value=1)
  # slider_bar_x = gr.Slider(-180, 180, step=1,label='slider_bar_x',value=-103)
  # slider_bar_y = gr.Slider(-180, 180, step=1,label='slider_bar_y',value=83)
  # slider_bar_z = gr.Slider(1, 10000, step=1,label='slider_bar_z',value=7000)
  submit_button = gr.Button()
  bn_samlpe = gr.Button("随机")
  idx = gr.Number(value = 1,label='cur idx',visible=False)
  print('fuck')
  bn_samlpe.click(fn_sample,
                inputs=[idx],
                outputs=[idx,model3d])
  submit_button.click(infer,
                inputs=[slider_bar],
                outputs=[model3d])

if __name__ == '__main__':
  demo.launch(share=False)
  print('http://127.0.0.1:7860/?__theme=dark')
  








  def get_indices(center_index, n, total_points):
    indices = []
    for i in range(center_index - n, center_index + n + 1):
        # 处理索引超出范围的情况
        index = i % total_points
        indices.append(index)
    return indices
  
  get_indices(10, 20, 258)