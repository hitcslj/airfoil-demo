from math import log2, pow
import os

import numpy as np
from scipy.fftpack import fft

import gradio as gr

from scipy.io.wavfile import write
from audio_api import audio2parsec 




def main_note(audio,input_image):
    rate, y = audio
    write("output.wav", rate, y)
    prompt = audio2parsec("output.wav")
    return input_image,prompt


airfoil_point_input = []
airfoil_point_output = []


demo = gr.Interface(
    main_note,
    inputs=[gr.Audio(sources=["microphone","upload"],format="wav"),gr.Image(height=200,width=200)],
    outputs=[gr.Image(height=200,width=200),gr.Textbox()],
    # examples=[
    #     ["output.wav","data/airfoil/picked_uiuc_img/2032c.png"],
    #     ["data/airfoil/picked_uiuc_img/2032c.png","2"],
    # ],
)

# def fn(input_image):
#     return input_image

# demo = gr.Interface(
#     fn,
#     inputs=[gr.Image()],
#     outputs=[gr.Image()],
# )


if __name__ == "__main__":
    demo.launch()

