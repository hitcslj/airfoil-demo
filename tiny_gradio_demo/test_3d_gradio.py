import gradio as gr

path = 'generate_result/airplane.stl'
def infer(slider_bar):
  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-103,83,slider_bar)) # 调位姿
  return model3d

with gr.Blocks() as demo:

  model3d = gr.Model3D(value=path,label='Output 3D Airfoil',camera_position=(-103,83,7000)) # 调位姿
  slider_bar = gr.Slider(1, 10000, step=1,label='slider_bar',value=8000)
  submit_button = gr.Button()
  submit_button.click(infer,
                inputs=[slider_bar],
                outputs=[model3d])

if __name__ == '__main__':
  demo.launch(share=False)
  print('http://127.0.0.1:7861/?__theme=dark')
  
