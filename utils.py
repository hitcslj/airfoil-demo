import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool


def airfoil2img(txt_path):
    data = []
    with open(txt_path) as f:
        for line in f:
            values = line.strip().split()
            data.append([float(values[0]), float(values[1])])
    data = np.array(data)

    # Normalize the coordinates
    data[:, 0] = (data[:, 0] - min(data[:, 0])) / (max(data[:, 0]) - min(data[:, 0]))
    data[:, 1] = (data[:, 1] - min(data[:, 1])) / (max(data[:, 1]) - min(data[:, 1]))

    # Create a 200x200 binary image
    img = np.zeros((200, 200))

    # Map the normalized coordinates to the image grid
    x_indices = (data[:, 0] * 199).astype(int)
    y_indices = (data[:, 1] * 199).astype(int)

    # Set the corresponding pixels to 1
    img[y_indices, x_indices] = 1

    # save the binary image
    name = txt_path.split('/')[-1].split('.')[0]
    file_path = 'data/airfoil/picked_uiuc_img/' + name + '.png'
    plt.imsave(file_path, img, cmap='gray')
 
if __name__ == '__main__':
  ## 遍历data/airfoil/picked_uiuc/下的所有文件.dat文件

  root_path = 'data/airfoil/picked_uiuc'
  file_paths = []
  ## 使用os.walk()函数遍历文件夹
  for root, dirs, files in os.walk(root_path):
      for file in files:
          file_path = os.path.join(root, file)
          # Do something with the file_path
          file_paths.append(file_path)

  ## 并行处理allData中的文件
  with Pool(processes=8) as pool:
      pool.map(airfoil2img, file_paths)
