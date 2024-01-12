import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import torch

 
# def airfoil2img(txt_path):
#     data = []
#     with open(txt_path) as f:
#         for line in f:
#             values = line.strip().split()
#             data.append([float(values[0]), float(values[1])])
#     data = np.array(data)
#     img = point2img(data)
#     # save the binary image
#     name = txt_path.split('/')[-1].split('.')[0]
#     file_path = 'data/airfoil/picked_uiuc/' + name + '.png'
#     plt.imsave(file_path, img, cmap='gray')
#     return img

# def point2img(data):
#     # Normalize the coordinates
#     data[:, 0] = (data[:, 0] - min(data[:, 0])) / (max(data[:, 0]) - min(data[:, 0]))
#     data[:, 1] = (data[:, 1] - min(data[:, 1])) / (max(data[:, 1]) - min(data[:, 1]))

#     data[:,1] /= 5
#     # Create a 200x200 binary image
#     img = np.zeros((40, 200))

#     # Map the normalized coordinates to the image grid
#     x_indices = (data[:, 0] * 199).astype(int)
#     y_indices = (data[:, 1] * 199).astype(int)

#     # Set the corresponding pixels to 1
#     img[y_indices, x_indices] = 255
#     # img[y_indices[::10],x_indices[::10]] = 20

#     # 将图像外围padding
#     img = np.pad(img, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
#     return img
 
def airfoil2imgScale(txt_path):
    data = get_point(txt_path)['full'] # (200,2)
    ## 将data可视化，要求x,y尺度一致
    plt.plot(data[:,0], data[:,1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gcf().set_facecolor('black')  # 设置背景色为黑色
    # 去除坐标轴
    plt.xticks([])
    plt.yticks([])

    # remove box, and save img
    plt.box(False)
    name = txt_path.split('/')[-1].split('.')[0]
    file_path = 'data/airfoil/picked_uiuc_img3/' + name + '.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()
    # plt.show()

def point2img_new(data):
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
     
    file_path = 'output.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()




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



def get_point(path):
    data = []
    with open(path) as file:
        # 逐行读取文件内容
        for line in file:
            # 移除行尾的换行符，并将每行拆分为两个数值
            values = line.strip().split()
            # 将数值转换为浮点数，并添加到数据列表中
            data.append([float(values[0]), float(values[1])])
    
    data = np.array(data)
    data = data[:200]
    upper = data[:100][::4]
    mid = data[100:101]
    low = data[101:][::4]
    low[-1][0]=1
    low[-1][1]=0
    keypoint_3d = np.concatenate((upper,mid,low),axis=0)
    # data = self.pc_norm(data)
    data = torch.FloatTensor(data)
    keypoint = data[::10] # 20个点
    # 适配3D keypoint
    # 将上下表面和中点都concat在一起
    

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
#       pool.map(airfoil2imgScale, file_paths)


if __name__ == '__main__':
    path = 'data/airfoil/picked_uiuc/2032c.dat'
    data = get_point(path)
    print(data['keypoint_3d'])