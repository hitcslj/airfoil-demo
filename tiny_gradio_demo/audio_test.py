import gradio as gr
from scipy.io.wavfile import write
from audio_api import audio2parsec

nameDict = [
    '前缘半径',
    '上表面峰值',
    '下表面峰值',
    '后缘角'
]


def process_audio(input_audio,slider0,slider1,slider2,slider3):
    print('process audio')
    rate, y = input_audio
    # 保存音频
    print('save audio')
    write("output.wav", rate, y)
    name,strength = audio2parsec("output.wav")
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



with gr.Blocks(theme=gr.themes.Soft()) as demo:
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


    with gr.Accordion("Audio edit parameters", open=False):
        input_audio = gr.Audio(sources=["microphone","upload"],format="wav")
        with gr.Row():
          bn_param = gr.Button("start audio to param")
          paramTex = gr.Textbox()
        bn_param.click(process_audio,
                        inputs=[input_audio] ,
                        outputs=[paramTex,slider0,slider1,slider2,slider3])

if __name__ == "__main__":
    demo.launch(share=True)