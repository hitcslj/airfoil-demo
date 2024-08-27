import numpy as np
import matplotlib.pyplot as plt

f0 = "/root/airfoil-demo/airfoil-demo_local/data/airfoil/supercritical_airfoil/air05_000001.dat"
# 定义机翼的二维坐标控制点
control_points = np.loadtxt(f0)

# 绘制控制点和连续曲线
plt.style.use('dark_background')
plt.figure(figsize=(10, 2))
plt.plot(control_points[:, 0], control_points[:, 1], 'w-')
#plt.title('Airfoil Profile')
# plt.ylim(-0.1, 0.1)  # 手动设置 y 轴范围
plt.grid(False)
#plt.axis('equal')
plt.axis('off')  # 关闭坐标轴显示
plt.savefig('./generate_result/test.png',dpi=300, bbox_inches='tight')