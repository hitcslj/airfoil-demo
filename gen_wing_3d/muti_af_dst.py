import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

# 定义翼型
airfoil = asb.Airfoil("naca0012")

# 定义翼型分布函数
def linear_chord_distribution(y, b, c0, c_tip):
    return c0 - (c0 - c_tip) * (y / b)

def polynomial_chord_distribution(y, b, c0, a1, a2):
    return c0 * (1 + a1 * (y / b) + a2 * (y / b)**2)

def elliptic_chord_distribution(y, b, c0):
    return c0 * np.sqrt(1 - (y / b)**2)

def exponential_chord_distribution(y, b, c0, k):
    return c0 * np.exp(-k * (y / b))

# 参数设置
b = 5  # 翼展长度
c0 = 2  # 翼根翼弦长度
c_tip = 0.5  # 翼尖翼弦长度
a1 = 0.1  # 多项式系数1
a2 = -0.05  # 多项式系数2
k = 0.5  # 指数衰减系数

# 生成展向位置
y_positions = np.linspace(0, b, 100)

# 计算不同分布方式的翼弦长度
linear_chord = linear_chord_distribution(y_positions, b, c0, c_tip)
polynomial_chord = polynomial_chord_distribution(y_positions, b, c0, a1, a2)
elliptic_chord = elliptic_chord_distribution(y_positions, b, c0)
exponential_chord = exponential_chord_distribution(y_positions, b, c0, k)

# 绘制翼型分布
plt.figure()
plt.plot(y_positions, linear_chord, label='Linear Chord')
plt.plot(y_positions, polynomial_chord, label='Polynomial Chord')
plt.plot(y_positions, elliptic_chord, label='Elliptic Chord')
plt.plot(y_positions, exponential_chord, label='Exponential Chord')
plt.xlabel('Spanwise Position (y)')
plt.ylabel('Chord Length (c)')
plt.title('Airfoil Distribution Along Span')
plt.legend()
plt.grid(True)
plt.show()

# 定义三维翼型
wings = {
    "Linear Chord": asb.Wing(
        name="Linear Chord Wing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, y, 0],  # 前缘位置
                chord=chord,  # 翼弦长度
                airfoil=airfoil  # 翼型
            )
            for y, chord in zip(y_positions, linear_chord)
        ]
    ),
    "Polynomial Chord": asb.Wing(
        name="Polynomial Chord Wing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, y, 0],  # 前缘位置
                chord=chord,  # 翼弦长度
                airfoil=airfoil  # 翼型
            )
            for y, chord in zip(y_positions, polynomial_chord)
        ]
    ),
    "Elliptic Chord": asb.Wing(
        name="Elliptic Chord Wing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, y, 0],  # 前缘位置
                chord=chord,  # 翼弦长度
                airfoil=airfoil  # 翼型
            )
            for y, chord in zip(y_positions, elliptic_chord)
        ]
    ),
    "Exponential Chord": asb.Wing(
        name="Exponential Chord Wing",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0, y, 0],  # 前缘位置
                chord=chord,  # 翼弦长度
                airfoil=airfoil  # 翼型
            )
            for y, chord in zip(y_positions, exponential_chord)
        ]
    )
}

# 绘制三维翼型
for name, wing in wings.items():
    wing.draw()
