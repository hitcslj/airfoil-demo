from cquav.wing.airfoil import load_airfoils_collection
from cquav.wing.airfoil import Airfoil
from cquav.wing.profile import AirfoilSection
from cquav.wing.rect_console import RectangularWingConsole
import cadquery as cq
# from jupyter_cadquery import show
from utils import get_point




airfoils_collection = load_airfoils_collection()
airfoil_data = airfoils_collection["NACA 6 series airfoils"]["NACA 64(3)-218 (naca643218-il)"]



path = 'data/airfoil/picked_uiuc/ag38.dat'
data = get_point(path)
keypoint_3d = data['keypoint_3d']

airfoil = Airfoil(airfoil_data) # 将这个airfoil对象的profile属性，传入二维机翼的点集
# print(airfoil.profile['A']  - keypoint_3d[:,0])
# print(airfoil.profile['B'] - keypoint_3d[:,1] )
# 传入二维机翼的点集
airfoil.profile['A'] =keypoint_3d[:,0]  # (51,)
airfoil.profile['B'] =keypoint_3d[:,1]  # (51,)


airfoil_section = AirfoilSection(airfoil, chord=200)
wing_console = RectangularWingConsole(airfoil_section, length=800)
assy = cq.Assembly()
assy.add(wing_console.foam, name="foam", color=cq.Color("lightgray"))
assy.add(wing_console.front_box, name="left_box", color=cq.Color("yellow"))
assy.add(wing_console.central_box, name="central_box", color=cq.Color("yellow"))
assy.add(wing_console.rear_box, name="right_box", color=cq.Color("yellow"))
assy.add(wing_console.shell, name="shell", color=cq.Color("lightskyblue2"))
# show(assy, angular_tolerance=0.1)
assy.save(path='airfoil_ag38.stl',exportType='STL')