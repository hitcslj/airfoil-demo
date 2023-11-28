# airfoil-demo
Airfoil demo. Editing keypoint &amp; parameters

An airfoil parameter visualization panel built upon [visTorch](https://github.com/koulanurag/visTorch).

- `src_A` 文件夹里面是受控情况下机翼变化的demo(半个机翼变化)
- `src_B` 文件夹里面是受控情况下机翼变化的demo(整个机翼变化)

## Installation

```bash
conda create --name airfoil python=3.8
conda activate airfoil
pip install -r requirements.txt
cd src_A
python setup.py install
cd ..
```

## Dataset

请将数据集存放在 `dataset` 文件夹下, 默认数据集为 `dataset/picked_uiuc/*.dat`

## Usage

在项目的根文件夹下:

```bash
# Start Controlled Scene
python ./src_A/app.py
# Start Not Controlled Scene
python .\src_B\app.py
# More Help
python .\src_A\app.py -h
```

使用浏览器访问终端里面生成的网页即可.

## Debug

- 提示找不到数据集
  - 因为数据集存放在代码目录的父目录,请在**项目根目录**里面使用`src_A(B)/app.py`路径启动程序