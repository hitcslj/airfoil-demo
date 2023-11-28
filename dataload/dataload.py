import torch
from torch.utils.data import Dataset,DataLoader
import os
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

class AirFoilDataset(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))
        
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        # input = data[::10]
        # params = torch.FloatTensor(params)
        # return {'input':input,'output':data,'params':params}
        return data
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m


class AirFoilDataset2(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        # params = []
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))
                # params.append(list(map(float,name_params[1:])))
        # ## 求params每个物理量的范围
        # params = np.array(params)
        # self.params_min = np.min(params,axis=0)
        # self.params_max = np.max(params,axis=0)
        # print('params_min:',self.params_min)
        # print('params_max:',self.params_max)
    
    def calMidLine(self,data):
        n = data.shape[0]//2 # data是torch.tensor,shape为[200,2]
        up = data[1:n, 1]
        low =  data[-n+1:, 1].flip(0)
        return torch.stack([data[1:n, 0], (up+low) / 2], dim=1)

    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        input = data[::10] # 20个点
        mid_data = self.calMidLine(data) # [99,2]
        mid_input = self.calMidLine(input) # [9,2]
        # params = params[0:9] # 9个参数
        params = torch.FloatTensor(params)
        return {'input':input,'output':data,'params':params,'mid_input':mid_input,'mid_output':mid_data}
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m

class AirFoilDatasetEdit(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        txt_mpt = {}
        with open('data/airfoil/%s.txt' % split) as f:
              for line in f.readlines():
                name = line.rstrip().strip('\n')
                txt_mpt[name] = os.path.join(datapath,name + '.dat') 
        self.txt_mpt = txt_mpt
        self.params = {}
        self.pairs = []
        # params = []
        with open('data/airfoil/predict_parsec.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                source_name = name_params[0].split('/')[-1].split('.')[0]
                target_name = name_params[1].split('/')[-1].split('.')[0]
                self.params[(source_name,target_name)] = list(map(float,name_params[2:]))
                self.pairs.append((source_name,target_name))
    
    def calMidLine(self,data):
        n = data.shape[0]//2 # data是torch.tensor,shape为[200,2]
        up = data[1:n, 1]
        low =  data[-n+1:, 1].flip(0)
        return torch.stack([data[1:n, 0], (up+low) / 2], dim=1)

    def __getitem__(self, index):
        """Get current batch for input index"""
        source_name,target_name = self.pairs[index]
        txt_path = self.txt_mpt[target_name]
        params = self.params[(source_name,target_name)]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        input = data[::10] # 20个点
        mid_data = self.calMidLine(data) # [99,2]
        mid_input = self.calMidLine(input) # [9,2]
        # params = params[0:9] # 9个参数
        params = torch.FloatTensor(params)
        return {'input':input,'output':data,'params':params,'mid_input':mid_input,'mid_output':mid_data}
    
    def __len__(self):
        return len(self.pairs) # 主要编辑的pair有多少


class AirFoilDatasetRandom(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = './data/airfoil/picked_uiuc',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        # params = []
        with open('data/airfoil/parsec_params.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))
                # params.append(list(map(float,name_params[1:])))
        # ## 求params每个物理量的范围
        # params = np.array(params)
        # self.params_min = np.min(params,axis=0)
        # self.params_max = np.max(params,axis=0)
        # print('params_min:',self.params_min)
        # print('params_max:',self.params_max)
    
    # def calMidLine(self,data):
    #     n = data.shape[0]//2 # data是torch.tensor,shape为[200,2]
    #     up = data[1:n, 1]
    #     low =  data[-n+1:, 1].flip(0)
    #     return torch.stack([data[1:n, 0], (up+low) / 2], dim=1)

    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = []
        with open(txt_path) as file:
            # 逐行读取文件内容
            for line in file:
                # 移除行尾的换行符，并将每行拆分为两个数值
                values = line.strip().split()
                # 将数值转换为浮点数，并添加到数据列表中
                data.append([float(values[0]), float(values[1])])
        if len(data) == 201:
            data = data[:200]
        elif len(data) > 201:
            assert len(data) < 201, f'data is not 200 ! {txt_path}'
        # data = self.pc_norm(data)
        data = torch.FloatTensor(data)
        # input = data[::10] # 20个点
        # 先位置随机，后控制点的数量随机
        # 从data中随机选取20个点
        randomIdx = random.sample(range(0,200),20)
        input = data[randomIdx]

        # mid_data = self.calMidLine(data) # [99,2]
        # mid_input = self.calMidLine(input) # [9,2]
        # params = params[0:9] # 9个参数
        params = torch.FloatTensor(params)
        return {'input':input,'output':data,'params':params}
    
    def __len__(self):
        return len(self.txt_list)
    
    def pc_norm(self,pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / m


if __name__=='__main__':
    dataset = AirFoilDatasetEdit(split='test')
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4)
    for step,data in enumerate(dataloader):
        # print(data)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        with torch.no_grad():
            x_sample = data['input'] # [b,20,2]
            x_physics = data['params'] # [b,10]
            x_mid = data['mid_input'] # [b,9,2]
            x_physics = x_physics.unsqueeze(-1) #[b,10,1]
            x_physics = x_physics.expand(-1,-1,2) #[b,10,2]
            x_gt = data['output'] # [b,200,2]
            mid_gt = data['mid_output'] # [b,99,2]
            data = torch.cat((x_gt,mid_gt),dim=1) # [b,299,2]
            origin_x = data[0,:,0].cpu().numpy()
            origin_y = data[0,:,1].cpu().numpy()
            ax1.scatter(origin_x, origin_y, color='red', marker='o')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            # ax1.set_aspect('equal')
            # ax1.axis('off')
            ax1.set_title('Original Data')

            fig.tight_layout()


            plt.savefig(f'data.png', format='png')
            plt.close()
        break