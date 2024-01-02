import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import torch


def get_name(path):
    return path.split('/')[-1].split('.')[0]

def get_point(path):
    data = []
    with open(path) as file:
        
        i = 0
        # 逐行读取文件内容
        for line in file:
            if i==0:
                i+=1
                continue
            # 移除行尾的换行符，并将每行拆分为两个数值
            values = line.strip().split()
            # 将数值转换为浮点数，并添加到数据列表中
            data.append([float(values[0]), float(values[-1])]) # 超临界数据集里
    data = torch.FloatTensor(data)
    keypoint = data[::10] # 20个点
    return {'keypoint':keypoint,'full':data}

def airfoil2imgScale(txt_path):
    data = get_point(txt_path)['full'] # (256,2)
    ## 将data可视化，要求x,y尺度一致
    plt.plot(data[:,0], data[:,1])
    plt.gca().set_aspect('equal', adjustable='box')
    # 去除坐标轴
    plt.xticks([])
    plt.yticks([])

    # remove box, and save img
    plt.box(False)
    name = get_name(txt_path)
    file_path = 'data/airfoil/supercritical_airfoil_img/' + name + '.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()
    # plt.show()
 


if __name__ == '__main__':
  ## 遍历data/airfoil/picked_uiuc/下的所有文件.dat文件

  root_path = 'data/airfoil/supercritical_airfoil'
  file_paths = []
  ## 使用os.walk()函数遍历文件夹
  for root, dirs, files in os.walk(root_path):
      for file in files:
          file_path = os.path.join(root, file)
          # Do something with the file_path
          file_paths.append(file_path)

  ## 并行处理allData中的文件
  with Pool(processes=8) as pool:
      pool.map(airfoil2imgScale, file_paths)
