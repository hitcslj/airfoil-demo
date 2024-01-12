import gradio as gr


def infer(slider_bar):
  path = 'output3d.stl'
  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-90,2,slider_bar)) # 调位姿
  return model3d

with gr.Blocks() as demo:

  model3d = gr.Model3D(value='airfoil_ag36.stl',label='Output 3D Airfoil',camera_position=(-90,2,500)) # 调位姿
  slider_bar = gr.Slider(500, 1000, step=10,label='slider_bar',value=500)
  submit_button = gr.Button()
  submit_button.click(infer,
                inputs=[slider_bar],
                outputs=[model3d])

if __name__ == '__main__':
  demo.launch(share=False)
  print('http://127.0.0.1:7861/?__theme=dark')
  
