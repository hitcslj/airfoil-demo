import torch
from torch import nn
from utils import Fit_airfoil

class AE_AB_Parsec(nn.Module):
    '''Editing parsec'''
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def editing_params(self,source_param,target_param,source_keypoint): 
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
        target_param = target_param.expand(-1,-1,2) # (b,11,2)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_point_pred

    def forward(self,source_param,target_param,source_keypoint): 
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
        target_param = target_param.expand(-1,-1,2) # (b,11,2)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_keypoint_pred,target_point_pred


    def refine_forward(self,source_param,target_param,source_keypoint,refine_iteration=1): #  
        ans = []
        for _ in range(refine_iteration):
          target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
          target_param_ = target_param.expand(-1,-1,2) # (b,11,2)
          condition = torch.cat((target_param_,target_keypoint_pred),dim=1)
          target_point_pred = self.modelB.sample(condition) 
          source_keypoint = target_point_pred[:,::10]
          f_pred = Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features
          source_param = torch.from_numpy(f_pred).type(source_param.dtype).unsqueeze(0).unsqueeze(-1).cuda()
          ans.append(target_point_pred)
        return ans


class AE_AB_Keypoint(nn.Module):
    '''editing keypoint'''
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
    


    def editing_point(self,source_keypoint,target_keypoint,source_param): # y就是物理参数
        source_param = source_param.expand(-1,-1,2)
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
        target_param_pred = target_param_pred.expand(-1,-1,2)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_point_pred

    def forward(self,source_keypoint,target_keypoint,source_param):  
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
        target_param_pred = target_param_pred.expand(-1,-1,2)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.sample(condition) 
        return target_param_pred,target_point_pred



    def refine_forward(self,source_keypoint,target_keypoint,source_param,refine_iteration=1): #  
        ans = []
        for _ in range(refine_iteration):
          target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
          target_param_pred = target_param_pred.expand(-1,-1,2)
          condition = torch.cat((target_param_pred,target_keypoint),dim=1)
          target_point_pred = self.modelB.sample(condition) 
          source_keypoint = target_point_pred[:,::10]
          f_pred = Fit_airfoil(target_point_pred[0].cpu().numpy()).parsec_features
          source_param = torch.from_numpy(f_pred).type(source_param.dtype).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2).cuda()
          ans.append(target_point_pred)
        return ans
