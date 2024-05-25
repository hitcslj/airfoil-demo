import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import torch
from scipy.interpolate import splev,splprep
from scipy import optimize
import matplotlib.pyplot as plt
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy.interpolate import splev, splrep,splprep
from PIL import Image, ImageDraw


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
        # file_path = root + '/' + file
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

# def point2img(data):
#     ## 将data可视化，要求x,y尺度一致
#     plt.plot(data[:,0], data[:,1])
#     plt.gca().set_aspect('equal', adjustable='box')

#     # 将x尺度保持不变，y尺度显示的时候resize成原来的1/5

#     plt.gcf().set_facecolor('black')  # 设置背景色为黑色
#     # 去除坐标轴
#     plt.xticks([])
#     plt.yticks([])

#     # remove box, and save img
#     plt.box(False)
     
#     file_path = 'generate_result/output.png'
#     plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
#     # Clear the plot cache
#     plt.clf()


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


def interpolote_up(data, x_coords):
    x = data[:,0][::-1]
    y = data[:,1][::-1]
    spl = splrep(x, y, s = 0)
    y_interp = splev(x_coords, spl)
    x_coords = x_coords[::-1]
    y_interp = y_interp[::-1]
    return np.array([x_coords, y_interp]).T

def interpolote_down(data, x_coords):
    x = data[:,0]
    y = data[:,1]
    spl = splrep(x, y, s = 0)
    y_interp = splev(x_coords, spl)
    return np.array([x_coords, y_interp]).T


def interpolate(data,s_x = 128, t_x = 129):
    # bspline 插值
    up = data[:s_x]  # 上表面原来为100个点, 插值为128个点
    mid = data[s_x:s_x+1]
    down = data[s_x+1:]  # 下表面原来为100个点，插值为128个点

    theta = np.linspace(np.pi, 2*np.pi, t_x)
    x_coords = (np.cos(theta) + 1.0) / 2
    x_coords = x_coords[1:]  # 从小到大
    
    up_interp = interpolote_up(up, x_coords)

    down_interp = interpolote_down(down, x_coords)

    # 组合上下表面
    interpolated_data = np.concatenate((up_interp, mid, down_interp))
    
    # x,y = data[:,0],data[:,1]
    # x2,y2 = interpolated_data[:,0],interpolated_data[:,1]

    # # 可视化验证
    # plt.plot(x, y, 'o', x2, y2)
    # plt.show()
    # plt.savefig('interpolated.png')
    return interpolated_data

def bernstein_poly(i, n, t):
    return np.math.comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    curve = np.zeros((len(t), control_points.shape[1]))  # 初始化曲线数组
    for i in range(len(t)):
        for j in range(n + 1):
            curve[i] += bernstein_poly(j, n, t[i]) * control_points[j]
    return curve


def point2img(data):
    # 图像尺寸
    width, height = 600, 600
    background_color = (0, 0, 0)  # 黑色背景
    point_color = (255, 255, 255)  # 白色点
    line_color = (255, 255, 255)
    # 创建一个新的图像
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # 找到数据的范围
    min_x, max_x = min(data[:, 0]), max(data[:, 0])
    min_y, max_y = min(data[:, 1]), max(data[:, 1])

    # 定义缩放比例
    scale_x = width / (max_x - min_x)
    scale_y = height / (max_y - min_y) / 5  # y方向缩小为1/5
    y_offset = (height - (max_y - min_y) * scale_y) / 2
    # 绘制散点
    # for x, y in data:
    #     # 将数据点转换为图像坐标
    #     px = int((x - min_x) * scale_x)
    #     py = int((y - min_y) * scale_y + y_offset)
    #     draw.point((px, height - py), fill=point_color)
    points = [(int((x - min_x) * scale_x), int((y - min_y) * scale_y + y_offset)) for x, y in data]
    
    # 绘制线段
    for i in range(len(points) - 1):
        draw.line((points[i][0], height - points[i][1], points[i+1][0], height - points[i+1][1]), fill=line_color, width=3)
    # 保存图像
    file_path = 'generate_result/output.png'
    # img.save(file_path)
    return img# , scale_x, scale_y, y_offset


def pixel_to_coordinate(pixel_x, pixel_y ,data):
    # 找到数据的范围
    min_x, max_x = min(data[:, 0]), max(data[:, 0])
    min_y, max_y = min(data[:, 1]), max(data[:, 1])
    # 图像尺寸
    width, height = 600, 600
    # 定义缩放比例
    scale_x = width / (max_x - min_x)
    scale_y = height / (max_y - min_y) / 5  # y方向缩小为1/5
    # y方向的偏移量，使点在图像中间
    y_offset = (height - (max_y - min_y) * scale_y) / 2
    # 将像素坐标转换为数据坐标
    data_x = pixel_x / scale_x + min_x
    data_y = (height - (pixel_y - y_offset)) / scale_y + min_y
    return torch.FloatTensor([[data_x, data_y]])


def generate_3D_from_dat(wing_dat, stl_path = "generate_result/airplane_tmp.stl"):
    """
    wing_dat: 定义机翼的 [x, y] 坐标的 Nx2 数组, 点应始于后缘的上表面，继续向前越过上表面，环绕前缘，继续向后越过下表面，然后在下表面的后缘处结束。
    stl_path: 暂存stl文件路径, 用于gradio展示
    """

    #wing_airfoil = asb.Airfoil("sd7037")
    wing_airfoil = asb.Airfoil(coordinates=wing_dat)
    tail_airfoil = asb.Airfoil("naca0010")

    ### Define the 3D geometry you want to analyze/optimize.
    # Here, all distances are in meters and all angles are in degrees.
    airplane = asb.Airplane(
        name="Peter's Glider",
        xyz_ref=[0, 0, 0],  # CG location
        wings=[
            asb.Wing(
                name="Main Wing",
                xyz_le=[0, 0, 0],  # Coordinates of the wing's leading edge
                symmetric=True,  # Should this wing be mirrored across the XZ plane?
                xsecs=[  # The wing's cross ("X") sections
                    asb.WingXSec(  # Root
                        xyz_le=[0, 0, 0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        chord=0.18,
                        twist=2,  # degrees
                        airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                        control_surface_is_symmetric=True,
                        # Flap (ctrl. surfs. applied between this XSec and the next one.)
                        control_surface_deflection=0,  # degrees
                    ),
                    asb.WingXSec(  # Mid
                        xyz_le=[0.01, 0.6, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=wing_airfoil,
                        control_surface_is_symmetric=False,  # Aileron
                        control_surface_deflection=0,
                    ),
                    # asb.WingXSec(  # Tip
                    #     xyz_le=[0.08, 1, 0.1],
                    #     chord=0.08,
                    #     twist=-2,
                    #     airfoil=wing_airfoil,
                    # ),
                ]
            ),
            asb.Wing(
                name="Horizontal Stabilizer",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=tail_airfoil,
                        control_surface_is_symmetric=True,  # Elevator
                        control_surface_deflection=0,
                    ),
                    asb.WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=tail_airfoil
                    )
                ]
            ).translate([0.6, 0, 0.06]),
            asb.Wing(
                name="Vertical Stabilizer",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=tail_airfoil,
                        control_surface_is_symmetric=True,  # Rudder
                        control_surface_deflection=0,
                    ),
                    asb.WingXSec(
                        xyz_le=[0.04, 0, 0.15],
                        chord=0.06,
                        twist=0,
                        airfoil=tail_airfoil
                    )
                ]
            ).translate([0.6, 0, 0.07])
        ],
        fuselages=[
            asb.Fuselage(
                name="Fuselage",
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                        radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
                    )
                    for xi in np.cosspace(0, 1, 30)
                ]
            )
        ]
    )
    
    airplane.export_cadquery_geometry(filename=stl_path)

    return airplane


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