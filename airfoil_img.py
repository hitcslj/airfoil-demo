import matplotlib.pyplot as plt
import numpy as np

from utils import get_point

path = 'data/airfoil/picked_uiuc/2032c.dat'

data = get_point(path)['full'] # (200,2)

## 将data可视化，要求x,y尺度一致

plt.plot(data[:,0], data[:,1])
plt.gca().set_aspect('equal', adjustable='box')
# 去除坐标轴
plt.xticks([])
plt.yticks([])

# remove box, and save img
plt.box(False)
plt.savefig('out.png', dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.show()