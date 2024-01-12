import gradio as gr
from scipy.io.wavfile import write
from audio_api import audio2parsec

def process_audio(input_audio):
    print('process audio')
    rate, y = input_audio
    # 保存音频
    write("output2.wav", rate, y)
    prompt = audio2parsec("output2.wav")
    return prompt

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Accordion("Audio edit parameters", open=False):
        input_audio = gr.Audio(sources=["microphone","upload"],format="wav")
        with gr.Row():
          bn_param = gr.Button("start audio to param")
          paramTex = gr.Textbox()
        bn_param.click(process_audio,
                        inputs=[input_audio] ,
                        outputs=[paramTex])

if __name__ == "__main__":
    demo.launch(share=False)