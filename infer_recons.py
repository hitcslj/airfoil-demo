import argparse
import os
import torch
import torch.nn as nn

from tqdm import tqdm 
from torch.utils.data import DataLoader

import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from models import AE_AB,AE_A_variable,AE_B_Attention
from dataload import AirFoilDataset2, EditingAirFoilDataset
import math 
from utils import Fit_airfoil
from collections import Counter


nameDict = [
    'leading edge radius',
    'pressure crest location x-coordinate',
    'pressure crest location y-coordinate',
    'curvatures at the pressuresurface crest ',
    'angle of the pressure surface at the trailing edge',
    'suction crest location x-coordinate',
    'suction crest location y-coordinate',
    'curvatures at suction surface crest locations',
    'angle of the suction surface at the trailing edge',
    'diff between gt and parsec airfoil'
]


parsec_features = ['rle','xup','yup','yxxup','xlo','ylo','yxxlo','yteup','ytelo','alphate','betate']



def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default=['airfoil'],# ['cocodatasets'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='airfoil')# 'cocodatasets')
    parser.add_argument('--data_root', default='./data/airfoil/picked_uiuc/')
    parser.add_argument('--num_workers',type=int,default=4)

    # io
    parser.add_argument('--checkpoint_path', default='weights/logs_edit_AB/cond_ckpt_epoch_100.pth',help='Model checkpoint path')

    parser.add_argument('--log_dir', default='test_result/refine_editing2',
                        help='Dump dir to save visual result')

    parser.add_argument('--eval', default=False, action='store_true')

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()


    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()

    return epoch 

    
class Tester:
    @staticmethod
    def get_model(args):
        modelA = AE_A_variable()
        modelB = AE_B_Attention()
        return modelA, modelB
    
    @torch.no_grad()
    def evaluate_one_epoch(self, model,criterion, dataloader,device, epoch, args,
                           perturb_target_idx,strength):
        """验证一个epoch"""
        model.eval()
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = total_loss1 = total_loss2 = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            
            source_keypoint = data['origin_input'] # [b,20,2]
            target_keypoint = data['editing_input'] # [b,20,2]
            source_params = data['params'] # [b,10]
            target_params = data['gt'] # [b,10]
            target_point = data['origin_trans'] # [b,20,2]
            
            source_params = source_params.unsqueeze(-1) #[b,10,1]
            target_params = target_params.unsqueeze(-1) #[b,10,1]

            source_keypoint = source_keypoint.to(device) 
            target_keypoint = target_keypoint.to(device) 
            source_params = source_params.to(device)
            target_params = target_params.to(device)
            target_point = target_point.to(device)

            # # AE
            target_params_pred,target_point_pred = model(source_keypoint, target_keypoint, source_params,perturb_target_idx=perturb_target_idx,strength=
            strength) 

            loss1 = criterion(target_params_pred,target_params)
            loss2 = criterion(target_point_pred,target_point)
            # loss,_ = criterion(data,output)
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss += loss1.item()+loss2.item()
            total_pred += source_keypoint.shape[0]
            # 判断样本是否预测正确
            distances = torch.norm(target_point - target_point_pred,dim=2) #(B,200)
            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 200个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*200).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count
            
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        avg_loss1 = total_loss1 / total_pred
        avg_loss2 = total_loss2 / total_pred
        
        s = f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss} ,avg_loss1: {avg_loss1},avg_loss2: {avg_loss2}, perturb_target_idx: {perturb_target_idx}, strength: {strength}"
        
        # print(s)
        with open(os.path.join(args.log_dir,'eval_result.txt'),'a') as f:
            f.write(s+'\n')


    def test(self,args):
        loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        import matplotlib.pyplot as plt
        """Run main training/testing pipeline."""
        test_dataset = AirFoilDataset2(split='test')
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        n_data_test = len(test_loader.dataset)
        test_loader = tqdm(test_loader)
        print(f"length of validating dataset: {n_data_test}")
        modelA, model_B = self.get_model(args)
        model = AE_AB(modelA,model_B)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        epoch = load_checkpoint(args, model)
        model.eval()
        os.makedirs(args.log_dir, exist_ok=True)
        perturb_target_idx,strength  = 1,5
        refine_iteration = 3
        recons_parsec_features_error = [Counter() for _ in range(refine_iteration)]
        num = 0
        recons_error = [0]*refine_iteration


        for step, data in enumerate(test_loader):
            fig, axes = plt.subplots(1, 2+refine_iteration, figsize=(20, 5))
            ax1, ax2, *axi = axes
            with torch.no_grad():
                source_keypoint = data['keypoint'] # [b,20,2]
                target_keypoint = data['keypoint'] # [b,20,2]
                source_params = data['params'] # [b,10]
                target_params = data['params'] # [b,10]
                target_point = data['gt'] # [b,200,2]
                source_point = data['gt'] # [b,200,2]
                
                source_params = source_params.unsqueeze(-1) #[b,10,1]
                target_params = target_params.unsqueeze(-1) #[b,10,1]

                source_keypoint = source_keypoint.to(device) 
                target_keypoint = target_keypoint.to(device) 
                source_params = source_params.to(device)
                target_params = target_params.to(device)
                target_point = target_point.to(device)
                source_point = source_point.to(device)

                # target_params_pred,ans = model.refine_forward2(source_keypoint, target_keypoint, source_params,refine_iteration=refine_iteration)  
                target_params_pred,ans = model.refine_forward3(source_keypoint, target_keypoint, source_params,target_params,refine_iteration=refine_iteration)  

                origin_x = source_point[0,:,0].cpu().numpy()
                origin_y = source_point[0,:,1].cpu().numpy()
                ax1.scatter(origin_x, origin_y, color='red', marker='o')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_title('Source airfoil')
                
                ax2.set_title('Target airfoil')
                trans_x = target_point[0,:,0].cpu().numpy()
                trans_y = target_point[0,:,1].cpu().numpy()
                ax2.scatter(trans_x, trans_y, color='red', marker='o')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                
                
                for i, ax in enumerate(axi):
                    outputs2_x = ans[i][0,:,0].cpu().numpy()
                    outputs2_y = ans[i][0,:,1].cpu().numpy()
                    l = loss(ans[i],target_point)*10**5 
                    ## 计算ans[i], target_point 物理量的差异
                    f_pred = Fit_airfoil(ans[i][0].cpu().numpy())
                    f_gt = Fit_airfoil(target_point[0].cpu().numpy())
                   
                    recons_error[i] += l
                    num += 1
                    for j in range(11):
                      recons_parsec_features_error[i][parsec_features[j]] += abs(f_pred.parsec_features[j] - f_gt.parsec_features[j])
                    # print(abs(f_pred.parsec_features - f_gt.parsec_features)/(f_gt.parsec_features+1e-9))
                    
                    # Loss： l也打印出来
                    ax.scatter(outputs2_x, outputs2_y, color='red', marker='o')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title('Target pred airfoil refine iter:{}\nLoss: {:.4f}'.format(i, l))
                ## 给图片设置总的Title
                fig.suptitle(f'{args.log_dir}/{step}_editing_params_{perturb_target_idx}_strength_{strength}', fontsize=16)

                fig.tight_layout()

                plt.savefig(f'{args.log_dir}/{step}_editing_params_{perturb_target_idx}_strength_{strength}.png', format='png')
                plt.close()
            # break
        # 将 recons_parsec_features_error 以及 recons_error写入文件
                
        with open(os.path.join(args.log_dir,'recons_parsec_features_error.txt'),'w') as f:
            for i in range(refine_iteration):
              f.write(f'iter{i}\n')
              for k,v in recons_parsec_features_error[i].items():
                f.write(f'{k}:{v/(num*refine_iteration)}\n')
        with open(os.path.join(args.log_dir,'recons_error.txt'),'w') as f:
            for i in range(refine_iteration):
              f.write(f'iter{i}\n')
              f.write(f'{recons_error[i]/(num*10**5*refine_iteration)}\n')

        # self.evaluate_one_epoch(model,nn.MSELoss(),test_loader,device,epoch,args,perturb_target_idx,strength)
if __name__ == '__main__':
    opt = parse_option()
    # cudnn
    # 启用cudnn
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    tester = Tester()
    tester.test(opt)