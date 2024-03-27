import os
import shutil

source_path = 'data/airfoil/supercritical_airfoil'
target_path = 'data/airfoil/super_critical_airfoil2'
os.makedirs(target_path, exist_ok=True)

for root, dirs, files in os.walk(source_path):
  # 将source_path路径下的所有文件copy到target_path路径下
  for file in files:
    if 'air05' in file:
      file_path = os.path.join(root, file)
      target_file_path = file_path.replace(source_path, target_path)
      shutil.copyfile(file_path, target_file_path)
      


