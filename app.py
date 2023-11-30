import gradio as gr
from XEdu.hub import Workflow as wf

model = wf(task='MMEdu',checkpoint='model/Oracle.onnx')
classes = ['人','大']

def instruct_and_predict(instruction, input_img=None, input_audio=None):
    if input_img is None:
        return "", "请按照上方的指示开始绘图。"
    else:
        result = model.inference(data=input_img) 
        result = model.format_output(lang="zh")
        score = result['置信度']
        result_text = classes[result['标签']]
        feedback = ""
        if result['置信度'] < 0.99:
            feedback = f"不太对哦，我都判断不准确了！"
        else:    
            if instruction == result_text:
                feedback = "恭喜！你画得很好！"
            else:
                feedback = f"不太对哦，我期望的是：{instruction}，但你画的像：{result_text}。"

        return result_text,score, feedback

label_input = gr.Textbox(label="我希望你画的是")
Image_input = gr.Image(shape=(128, 128), source="canvas", label="画板")
label_output1 = gr.Textbox(label="你画的甲骨文是")
label_output2 = gr.Textbox(label="识别置信度是")
Image_output = gr.Textbox(label="反馈")
audio_input = gr.Audio(sources=["microphone"])

demo = gr.Interface(fn=instruct_and_predict, 
    inputs=[label_input,Image_input,audio_input],
    outputs=[label_output1,label_output2,Image_output],
    live=False,
    title="甲骨文学习小游戏",
    description="请在下方文本框输入你希望绘制的甲骨文（“人”或“大”），然后在画板上进行绘制，查看结果。",
    theme="default"
)

demo.launch(share=True)