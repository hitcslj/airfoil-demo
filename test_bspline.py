import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev, splprep
from scipy.interpolate import UnivariateSpline

def get_indices(center_index, left_n, right_n, total_points):
    indices = []
    for i in range(center_index - left_n, center_index + right_n + 1):
        # 处理索引超出范围的情况
        index = i % total_points
        indices.append(index)
    return indices

# 使用numpy读取二维数组文件
control_points = np.loadtxt('/home/bingxing2/ailab/wangshuai/airfoil-demo_ws/data/airfoil/picked_uiuc/a18.dat')

# 确保控制点是闭合的
control_points = np.vstack([control_points, control_points[0]])

# 改变其中20个控制点
center_index = 50
slider_l, slider_r = 10,10
total_points = len(control_points)
change_indices = get_indices(center_index, slider_l, slider_r, total_points)
buffer_indices = get_indices(center_index, slider_l+20, slider_r+20, total_points)

for idx in change_indices:
    control_points[idx,1] += 0.02

# 使用B样条曲线进行局部拟合
def fit_bspline(points, degree=3, smoothness=0.01):
    tck, u = splprep(points.T, s=smoothness, k=degree)
    return tck

# 局部拟合改变的20个控制点
local_points = control_points[change_indices]
local_points = control_points[buffer_indices]
tck = fit_bspline(local_points)

# 生成B样条曲线的点
bspline_points = np.array(splev(np.linspace(0, 1, len(buffer_indices)), tck)).T

# 拼接拟合后的曲线和未改变的部分
new_control_points = control_points.copy()
#new_control_points[change_indices] = bspline_points
new_control_points[buffer_indices] = bspline_points

# 绘制完整的翼型
plt.figure()
#plt.plot(control_points[:, 0], control_points[:, 1], 'ro', label='Original Control Points')
plt.plot(new_control_points[:, 0], new_control_points[:, 1], 'b-', label='Modified Control Points')
#plt.plot(bspline_points[:, 0], bspline_points[:, 1], 'g-', label='B-spline Fit')
plt.legend()
plt.title('Modified Airfoil with B-spline Fit')
plt.axis('equal')
plt.savefig('airfoil_modified.png',dpi=300)
plt.show()

