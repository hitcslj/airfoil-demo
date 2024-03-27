import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

source_path = 'data/airfoil/supercritical_airfoil'
target_path = 'data/airfoil/supercritical_airfoil_img'
 

def read_file(file_path):
    data = []
    with open(file_path) as file:
        for i,line in enumerate(file):
            if i==0:continue
            values = line.strip().split()
            data.append([float(values[0]), float(values[1])])
    return np.array(data)

def point2img_new(data,file_path):
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
     
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()

for root, dirs, files in os.walk(source_path):
  for file in files:
      file_path = os.path.join(root, file)
      data = read_file(file_path)
      img_path = os.path.join(target_path, file.replace('.dat','.png'))
      point2img_new(data,img_path)

      