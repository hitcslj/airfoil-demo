import torch
from torch import nn
 
class AE_AB(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def forward(self,source_keypoint,target_keypoint,source_params,perturb_target_idx=None,strength=0.1): # y就是物理参数
        target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
        if perturb_target_idx is not None:
            target_params_pred[:,perturb_target_idx] *= (1+strength)
        target_params_pred_ = target_params_pred.expand(-1,-1,2)

        target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
        return target_params_pred,target_point_pred


    def editing_param(self,source_keypoint,source_params,parmas=None): # y就是物理参数
        for i,strength in enumerate(parmas):
           source_params[:,i] *= (1+strength)
        source_params_ = source_params.expand(-1,-1,2)
        target_point_pred = self.modelB(source_keypoint,source_params_) 
        return source_params,target_point_pred
    
    def editing_point(self,source_keypoint,source_params,point1=None,point2=None): # y就是物理参数
        target_keypoint = source_keypoint.clone()
        # target_keypoint.shape  (1,20,2)

        # target_keypoint[:,point1[0],0] += point1[1]

        target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
        target_params_pred_ = target_params_pred.expand(-1,-1,2)

        target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
        return target_params_pred,target_point_pred

