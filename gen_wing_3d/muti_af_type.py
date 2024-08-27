import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

# 定义不同翼型
airfoil_root = asb.Airfoil("naca4412")  # 翼根翼型
airfoil_mid = asb.Airfoil("naca0012")  # 中间翼型
airfoil_tip = asb.Airfoil("naca2412")  # 翼尖翼型

# 获取翼型坐标
airfoil_root_coords = airfoil_root.coordinates
airfoil_mid_coords = airfoil_mid.coordinates
airfoil_tip_coords = airfoil_tip.coordinates

# 分离上下表面坐标
def separate_coordinates(coords):
    upper = coords[coords[:, 1] >= 0]
    lower = coords[coords[:, 1] < 0]
    return upper, lower

airfoil_root_upper, airfoil_root_lower = separate_coordinates(airfoil_root_coords)
airfoil_mid_upper, airfoil_mid_lower = separate_coordinates(airfoil_mid_coords)
airfoil_tip_upper, airfoil_tip_lower = separate_coordinates(airfoil_tip_coords)

# 定义翼型分布函数
def blended_airfoil_distribution(y, b, c0, c_tip, airfoil_root_upper, airfoil_root_lower, airfoil_mid_upper, airfoil_mid_lower, airfoil_tip_upper, airfoil_tip_lower):
    # 定义翼型过渡位置
    y_transition1 = b * 0.3
    y_transition2 = b * 0.7
    
    # 定义翼弦长度分布
    chord = np.where(
        y < y_transition1,
        c0 - (c0 - c_tip) * (y / b),
        np.where(
            y < y_transition2,
            c_tip + (c0 - c_tip) * ((y - y_transition1) / (b - y_transition1)),
            c_tip
        )
    )
    
    # 定义翼型分布
    airfoil = []
    for y_pos in y:
        if y_pos < y_transition1:
            airfoil.append(asb.Airfoil(coordinates=airfoil_root_coords))
        elif y_pos < y_transition2:
            # 使用插值方法生成中间翼型
            t = (y_pos - y_transition1) / (y_transition2 - y_transition1)
            upper = (1 - t) * airfoil_root_upper + t * airfoil_mid_upper
            lower = (1 - t) * airfoil_root_lower + t * airfoil_mid_lower
            airfoil.append(asb.Airfoil(coordinates=np.vstack([upper, lower])))
        else:
            airfoil.append(asb.Airfoil(coordinates=airfoil_tip_coords))
    
    return chord, airfoil

# 参数设置
b = 10  # 翼展长度
c0 = 1  # 翼根翼弦长度
c_tip = 0.5  # 翼尖翼弦长度

# 生成展向位置
y_positions = np.linspace(0, b, 100)

# 计算翼弦长度和翼型分布
chord_lengths, airfoils = blended_airfoil_distribution(y_positions, b, c0, c_tip, airfoil_root_upper, airfoil_root_lower, airfoil_mid_upper, airfoil_mid_lower, airfoil_tip_upper, airfoil_tip_lower)

# 绘制翼型分布
plt.figure()
plt.plot(y_positions, chord_lengths)
plt.xlabel('Spanwise Position (y)')
plt.ylabel('Chord Length (c)')
plt.title('Blended Airfoil Distribution')
plt.grid(True)
plt.show()

# 定义三维翼型
wing = asb.Wing(
    name="Blended Airfoil Wing",
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, y, 0],  # 前缘位置
            chord=chord,  # 翼弦长度
            airfoil=airfoil  # 翼型
        )
        for y, chord, airfoil in zip(y_positions, chord_lengths, airfoils)
    ]
)

# 绘制三维翼型
wing.draw()