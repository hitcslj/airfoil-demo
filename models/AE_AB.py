import torch
from torch import nn
from utils import Fit_airfoil


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

    def refine_forward(self,source_keypoint,target_keypoint,source_params,refine_iteration=1): #  
        target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
        target_params_pred_ = target_params_pred.expand(-1,-1,2)
        ans = []
        for _ in range(refine_iteration):
          target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
          target_keypoint = target_point_pred[:,::10]
          ans.append(target_point_pred)

        return target_params_pred,ans

    def refine_forward2(self,source_keypoint,target_keypoint,source_params,refine_iteration=1): #  
        ans = []
        for _ in range(refine_iteration):
          target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
          print(target_params_pred)
          target_params_pred_ = target_params_pred.expand(-1,-1,2)
          target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
          target_keypoint = target_point_pred[:,::10]
          ans.append(target_point_pred)
        print('-------------------')
        return target_params_pred,ans
    
    def refine_forward3(self,source_keypoint,target_keypoint,source_params,target_params,refine_iteration=1): #  
        ans = []
        for _ in range(refine_iteration):
          # target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
          # print(target_params_pred)
          target_params_pred_ = target_params.expand(-1,-1,2)
          target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
          target_keypoint = target_point_pred[:,::10]
          ans.append(target_point_pred)
        # print('-------------------')
        return target_params_pred_,ans
    
    def refine_forward4(self,source_keypoint,target_keypoint,source_params,target_params,refine_iteration=1): #  
        ans = []
        # 编辑控制点
        for _ in range(refine_iteration):
          target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
          target_params_pred_ = target_params_pred.expand(-1,-1,2)
          target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
          source_keypoint = target_point_pred[:,::10]
          f_pred = Fit_airfoil(target_point_pred[0].cpu().numpy())
          source_params = torch.from_numpy(f_pred.parsec_features[:10]).type(source_params.dtype).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 1).cuda()
          ans.append(target_point_pred)
        # print('-------------------')
        return target_params_pred_,ans

    def editing_param(self,source_keypoint,source_params,parmas=None): # y就是物理参数
        for i,strength in enumerate(parmas):
           source_params[:,i] *= strength
        source_params_ = source_params.expand(-1,-1,2)
        target_point_pred = self.modelB(source_keypoint,source_params_) 
        return source_params,target_point_pred
    
    def editing_point(self,source_keypoint,source_params,point1=None,point2=None): # y就是物理参数
        target_keypoint = source_keypoint.clone()
        point1 /= 600
        point2 /= 600
        # target_keypoint.shape  (1,20,2)

        # target_keypoint[:,point1[0],0] += point1[1]

        target_params_pred = self.modelA(source_keypoint,target_keypoint,source_params)
        target_params_pred_ = target_params_pred.expand(-1,-1,2)

        target_point_pred = self.modelB(target_keypoint,target_params_pred_) 
        return target_params_pred,target_point_pred


class AE_AB_parsec(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def forward(self,source_param,target_param,source_keypoint): 
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
        target_param = target_param.expand(-1,-1,2) # (b,11,2)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.forward3(condition) 
        return target_keypoint_pred,target_point_pred
    

    def editing_params(self,source_param,target_param,source_keypoint): 
        # import pdb;pdb.set_trace()
        target_keypoint_pred = self.modelA(source_param,target_param,source_keypoint)
        target_param = target_param.expand(-1,-1,2) # (b,11,2)
        condition = torch.cat((target_param,target_keypoint_pred),dim=1)
        target_point_pred = self.modelB.forward3(condition) 
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



class AE_AB_keypoint(nn.Module):
    def __init__(self,modelA,modelB) -> None:
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB


    def forward(self,source_keypoint,target_keypoint,source_param): # y就是物理参数
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
        target_param_pred = target_param_pred.expand(-1,-1,2)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.forward3(condition) 
        return target_param_pred,target_point_pred

    def editing_point(self,source_keypoint,target_keypoint,source_param): # y就是物理参数
        # import pdb; pdb.set_trace()
        source_param = source_param.expand(-1,-1,2)
        target_param_pred = self.modelA(source_keypoint,target_keypoint,source_param)
        target_param_pred = target_param_pred.expand(-1,-1,2)
        condition = torch.cat((target_param_pred,target_keypoint),dim=1)
        target_point_pred = self.modelB.forward3(condition) 
        return target_param_pred,target_point_pred
    

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