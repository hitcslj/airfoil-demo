import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import torch
from scipy.interpolate import splev,splprep
from scipy import optimize
import matplotlib.pyplot as plt


def norm(data):
    mean = np.array([0.50194553,0.01158151])
    std = np.array([0.35423523,0.03827245])
    return (data-mean)/std * 2 -1 

def denorm(data):
    mean = np.array([0.50194553,0.01158151])
    std = np.array([0.35423523,0.03827245])
    return (data+1)/2 * std + mean

def get_path(root_path):
  file_paths = []
  ## 使用os.walk()函数遍历文件夹
  for root, dirs, files in os.walk(root_path):
    for file in files:
        file_path = os.path.join(root, file)
        # Do something with the file_path
        file_paths.append(file_path)
  return file_paths

def get_name(path):
    return path.split('/')[-1].split('.')[0]



def get_point_diffusion(path):
    data = []
    with open(path) as file:
        # 逐行读取文件内容
        for line in file:
            # 移除行尾的换行符，并将每行拆分为两个数值
            values = line.strip().split()
            # 将数值转换为浮点数，并添加到数据列表中
            data.append([float(values[0]), float(values[1])])
    
    data = np.array(data)
    upper = data[:128][::4]
    mid = data[128:129]
    low = data[129:][::4]
    low[-1][0]=1
    low[-1][1]=0
    keypoint_3d = np.concatenate((upper,mid,low),axis=0)
    data = norm(data) # 必须得norm才能送给diffusion
    data = torch.FloatTensor(data)
    keypoint = data[::10] # 20个点

    return {'keypoint':keypoint,'full':data, 'keypoint_3d':keypoint_3d}

def get_point_cvae(path):
    data = []
    with open(path) as file:
        # 逐行读取文件内容
        for line in file:
            # 移除行尾的换行符，并将每行拆分为两个数值
            values = line.strip().split()
            # 将数值转换为浮点数，并添加到数据列表中
            data.append([float(values[0]), float(values[1])])
    
    data = np.array(data)
    upper = data[:128][::4]
    mid = data[128:129]
    low = data[129:][::4]
    low[-1][0]=1
    low[-1][1]=0
    keypoint_3d = np.concatenate((upper,mid,low),axis=0)
    data = torch.FloatTensor(data)
    keypoint = data[::10] # 20个点    

    return {'keypoint':keypoint,'full':data, 'keypoint_3d':keypoint_3d}


def get_params(txt_path):
    params = {}
    with open(txt_path) as f:
      for line in f.readlines():
          name_params = line.rstrip().strip('\n').split(',')
          # 取出路径的最后一个文件名作为key
          name = get_name(name_params[0])
          params[name] = list(map(float,name_params[1:]))
    return params


def point2img(data):
    ## 将data可视化，要求x,y尺度一致
    plt.plot(data[:,0], data[:,1])
    plt.gca().set_aspect('equal', adjustable='box')

    # 将x尺度保持不变，y尺度显示的时候resize成原来的1/5

    plt.gcf().set_facecolor('black')  # 设置背景色为黑色
    # 去除坐标轴
    plt.xticks([])
    plt.yticks([])

    # remove box, and save img
    plt.box(False)
     
    file_path = 'generate_result/output.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()


class Fit_airfoil():
    '''
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    '''
    def __init__(self,airfoil,iLE=128):
        self.iLE = iLE
        self.tck, self.u  = splprep(airfoil.T,s=0)

        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0,1]
        ytelo = airfoil[-1,1]
        alphate, betate = self.get_te_angle()

        self.parsec_features = np.array([rle,xup,yup,yxxup,xlo,ylo,yxxlo,
                                         yteup,ytelo,alphate,betate]) 
        
        # 超临界翼型的特征
        xaft, yaft, yxxaft = self.get_aftload()
        # print(xaft, yaft, yxxaft)

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu,yu = splev(uLE, self.tck,der=1) # dx/du
        xuu,yuu = splev(uLE, self.tck,der=2) # ddx/du^2
        K = abs(xu*yuu-xuu*yu)/(xu**2+yu**2)**1.5 # curvature
        return 1/K
    
    def get_up(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0,self.u[self.iLE]),method='bounded')
        uup = res.x
        xup ,yup = splev(uup, self.tck)

        xu,yu = splev(uup, self.tck, der=1) # dx/du
        xuu,yuu = splev(uup, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(self.u[self.iLE],1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

    def get_te_angle(self):
        xu,yu = splev(0, self.tck, der=1)
        yx = yu/xu
        alphate = np.arctan(yx)

        xu,yu = splev(1, self.tck, der=1)
        yx = yu/xu
        betate = np.arctan(yx)

        return alphate, betate
    
    # 后加载位置
    def get_aftload(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0.75,1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

# if __name__ == '__main__':
#   ## 遍历data/airfoil/picked_uiuc/下的所有文件.dat文件

#   root_path = 'data/airfoil/picked_uiuc'
#   file_paths = []
#   ## 使用os.walk()函数遍历文件夹
#   for root, dirs, files in os.walk(root_path):
#       for file in files:
#           file_path = os.path.join(root, file)
#           # Do something with the file_path
#           file_paths.append(file_path)

#   ## 并行处理allData中的文件
#   with Pool(processes=8) as pool:
#       pool.map(point2img, file_paths)