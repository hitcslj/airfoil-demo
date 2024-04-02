import torch
from torch import nn
from utils import Fit_airfoil
import numpy as np

def de_norm(data):
    mean = torch.tensor([0.50194553,0.01158151]).to(data.device)
    std = torch.tensor([0.35423523,0.03827245]).to(data.device)
    return (data+1)/2 *std + mean

def norm(data):
    mean = torch.tensor([0.50194553,0.01158151]).to(data.device)
    std = torch.tensor([0.35423523,0.03827245]).to(data.device)
    return (data-mean)/std * 2 -1
    

class Diff_AB_Parsec(nn.Module):
    def __init__(self,modelA,vae,diffusion) -> None:
        super().__init__()
        self.modelA = modelA
        self.vae = vae
        self.diffusion = diffusion


    def forward(self,source_param,target_param,source_keypoint): 
        # import pdb;pdb.set_trace()
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint) # source_params(b,11,1),target_params(b,11,1),source_keypoint(b,26,2)
        target_keypoint_pred = norm(target_keypoint_pred)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # target_params(b,11),target_keypoint(b,26*2) target_full_latent(b,128)
        samples = self.diffusion.sample_ddim(batch_size=1, device=device, y=target_param.squeeze(-1), y2=target_keypoint_pred.reshape(-1,26*2)).to(device)
        # samples = torch.randn(1,128).to(device)
        with torch.no_grad():
            samples = self.vae.decode(samples.squeeze(-1))  # latent_space(b,128)->(b,257*2)上进行diffusion
        target_point_pred = de_norm(samples.reshape(-1,257,2))
        return target_point_pred
    

    def editing_params(self,source_param,target_param,source_keypoint): 
        # import pdb;pdb.set_trace()
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint) # source_params(b,11,1),target_params(b,11,1),source_keypoint(b,26,2)
        target_keypoint_pred = norm(target_keypoint_pred)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # target_params(b,11),target_keypoint(b,26*2) target_full_latent(b,128)
        samples = self.diffusion.sample_ddim(batch_size=1, device=device, y=target_param.squeeze(-1), y2=target_keypoint_pred.reshape(-1,26*2)).to(device)
        # samples = torch.randn(1,128).to(device)
        with torch.no_grad():
            samples = self.vae.decode(samples.squeeze(-1))  # latent_space(b,128)->(b,257*2)上进行diffusion
        target_point_pred = de_norm(samples.reshape(-1,257,2))
        return target_point_pred
 
    def refine_forward(self,source_param,target_param,source_keypoint,refine_iteration=1):  
        ans = []
        # 编辑控制点
        for _ in range(refine_iteration):
          target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
          target_param_ = target_param.expand(-1,-1,2) # (b,11,2)
          condition = torch.cat((target_param_,target_keypoint_pred),dim=1)
          target_point_pred = self.modelB.forward3(condition) 
          source_keypoint = target_point_pred[:,::10]
          f_pred = Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features
          source_param = torch.from_numpy(f_pred).type(source_param.dtype).unsqueeze(0).unsqueeze(-1).cuda()
          ans.append(target_point_pred)
        # print('-------------------')
        return ans



class Diff_AB_Keypoint(nn.Module):
    def __init__(self,modelA,vae,diffusion) -> None:
        super().__init__()
        self.modelA = modelA
        self.vae = vae
        self.diffusion = diffusion

    def forward(self,source_keypoint,target_keypoint,source_param): # y就是物理参数
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
        target_param_pred = target_param_pred.expand(-1,-1,2)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.forward3(condition) 
        return target_param_pred,target_point_pred

    def editing_point(self,source_keypoint,target_keypoint,source_param): # y就是物理参数
        # import pdb; pdb.set_trace()
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param.expand(-1,-1,2))
        # target_keypoint = norm(target_keypoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = self.diffusion.sample_ddim(batch_size=1, device=device, y=target_param_pred.squeeze(-1), y2=target_keypoint.reshape(-1,26*2)).to(device)
        with torch.no_grad():
            samples = self.vae.decode(samples.squeeze(-1))
        target_point_pred = de_norm(samples.reshape(-1,257,2))
        return target_point_pred
    

    def refine_forward(self,source_keypoint,target_keypoint,source_param,refine_iteration=1): #  
        ans = []
        # 编辑控制点
        for _ in range(refine_iteration):
          target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
          target_param_pred = target_param_pred.expand(-1,-1,2)
          condition = torch.cat((target_param_pred,target_keypoint),dim=1)
          target_point_pred = self.modelB.forward3(condition) 
          source_keypoint = target_point_pred[:,::10]
          f_pred = Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features
          source_param = torch.from_numpy(f_pred).type(source_param.dtype).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).cuda()
          ans.append(target_point_pred)
        # print('-------------------')
        return ans