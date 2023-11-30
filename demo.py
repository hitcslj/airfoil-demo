from math import log2, pow
import os

import numpy as np
from scipy.fftpack import fft

import gradio as gr

from scipy.io.wavfile import write
from audio_api import audio2parsec 


A4 = 440
C0 = A4 * pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_pitch(freq):
    h = round(12 * log2(freq / C0))
    n = h % 12
    return name[n]


def main_note(audio):
    rate, y = audio
    write("output.wav", rate, y)
    prompt = audio2parsec("output.wav")


    if len(y.shape) == 2:
        y = y.T[0]
    N = len(y)
    T = 1.0 / rate
    yf = fft(y)
    yf2 = 2.0 / N * np.abs(yf[0 : N // 2])
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    volume_per_pitch = {}
    total_volume = np.sum(yf2)
    for freq, volume in zip(xf, yf2):
        if freq == 0:
            continue
        pitch = get_pitch(freq)
        if pitch not in volume_per_pitch:
            volume_per_pitch[pitch] = 0
        volume_per_pitch[pitch] += 1.0 * volume / total_volume
    volume_per_pitch = {k: float(v) for k, v in volume_per_pitch.items()}
    return volume_per_pitch,prompt


demo = gr.Interface(
    main_note,
    inputs=[gr.Audio(sources=["microphone"],format="wav")],
    outputs=[gr.Label(num_top_classes=4),gr.Textbox()]
    # examples=[
    #     [os.path.join(os.path.dirname(__file__),"audio/recording1.wav")],
    #     [os.path.join(os.path.dirname(__file__),"audio/cantina.wav")],
    # ],
)

if __name__ == "__main__":
    demo.launch()

