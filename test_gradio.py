import gradio as gr 
import pandas as pd 
from skimage import data
from PIL import Image
from models import AE_AB,AE_A_variable,AE_B_Attention
import torch

 
modelA = AE_A_variable()
modelB = AE_B_Attention()
model = AE_AB(modelA,modelB)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checkpoint = torch.load('weights/logs_edit_AB/cond_ckpt_epoch_100.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()



def predict(img):
    result = model.predict(source=img)
    df = pd.Series(result[0].names).to_frame()
    df.columns = ['names']
    df['probs'] = result[0].probs
    df = df.sort_values('probs',ascending=False)
    res = dict(zip(df['names'],df['probs']))
    return res
gr.close_all() 
demo = gr.Interface(fn = predict,inputs =[],outputs=[], 
                     )
demo.launch()