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
