import aerosandbox as asb
import aerosandbox.numpy as np
from shutil import which

avl_is_present = which('avl') is not None

sd7037 = asb.Airfoil("sd7037")

airplane = asb.Airplane(
    name="Vanilla",
    wings=[
        asb.Wing(
            name="Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[1.5, 0, 0],     # 前缘位置
                    chord=1,                # 弦长
                    twist=2,                # 弯度
                    airfoil=sd7037,         # 翼型，根据文档中的内容，目前支持UIUC和NACA四位数翼型，以及自定义的dat文件
                ),                          # 自定义的话，要遵循国际的格式标准，从翼型上翼面前缘开始往后取坐标，到达后缘再沿下翼面往前取坐标
                asb.WingXSec(
                    xyz_le=[0.5, 5, 1],
                    chord=0.6,
                    twist=2,
                    airfoil=sd7037,
                )
            ]
        ),
        asb.Wing(
            name="H-stab",
            symmetric=True,
            xyz_le=[4, 0, 0],
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 1.25, 0],
                    chord=0.42
                ),
            ]
        ),
        asb.Wing(
            name="V-stab",
            xyz_le=[4, 0, 0],
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=0.7,
                ),
                asb.WingXSec(
                    xyz_le=[0.14, 0, 1],
                    chord=0.42
                )
            ]
        )
    ],
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[xi * 5 - 0.5, 0, 0],
                    radius=asb.Airfoil("naca0024").local_thickness(x_over_c=xi)
                )
                for xi in np.cosspace(0, 1, 50)
            ]
        )
    ]
)

airplane.export_cadquery_geometry(filename='airplane.stl')

# vlm = asb.VortexLatticeMethod(
#     airplane=airplane, # 上面设置的模型
#     op_point=asb.OperatingPoint(
#         velocity=-25,  # 速度，m/s
#         alpha=5,  # 迎角，角度制
#     )
# )


# aero = vlm.run()  # Returns a dictionary
# for k, v in aero.items():
#     print(f"{k.rjust(4)} : {v}")

# vlm.draw(show_kwargs=dict(jupyter_backend="static"))
