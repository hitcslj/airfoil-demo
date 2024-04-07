import aerosandbox as asb
import aerosandbox.numpy as np
from shutil import which

xfoil_is_present = which('xfoil') is not None
print(xfoil_is_present)
airfoil = asb.Airfoil("dae51") # 已经下载到本地的UIUC

from aerosandbox.tools.pretty_plots import plt, show_plot, set_ticks # 这个语法和matplotlib差不多，少量地方改了。

fig, ax = plt.subplots()
airfoil.draw(show=False)
set_ticks(0.1, 0.05, 0.1, 0.05)
show_plot()

if xfoil_is_present:  # 只有正常加载XFoil时才能用

    analysis = asb.XFoil(
        airfoil=airfoil,
        Re=3e5,
        xfoil_command="xfoil",
        # XFoil必须在PATH或本目录下
    )

    point_analysis = analysis.alpha(
        alpha=3
    )

    from pprint import pprint

    print("\nPoint analysis:")
    pprint(point_analysis)

    sweep_analysis = analysis.alpha(
        alpha=np.linspace(0, 15, 6)
    )
    print("\nSweep analysis:")
    pprint(sweep_analysis)

    cl_analysis = analysis.cl(
        cl=1.2
    )
    print("\nFixed-CL analysis:")
    pprint(cl_analysis)