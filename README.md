# HandGestureRecognition

基于cv与机器学习的实时手势识别系统。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.11-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10.21-blue" />
  <img src="https://img.shields.io/badge/NumPy-1.26-orange?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2.2.3-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-开发中-yellow" />
</p>

## 项目简介

本项目是一个基于 Python 的实时手势识别系统，结合计算机视觉与机器学习技术，通过摄像头采集手部图像，利用 MediaPipe 提取手部关键点，并使用RandomForest模型对不同手势进行分类识别。

项目旨在探索计算机视觉与人机交互技术的结合，实现自然、直观的手势交互方式。

## 核心功能

- **手部检测：** 使用 MediaPipe 实时检测手部位置及关键点。
- **数据采集：** 提取手部关键点特征，构建手势数据集。
- **模型训练：** 基于采集的数据训练手势分类模型。
- **实时识别：** 通过摄像头实时识别不同手势。
- **多手势支持：** 支持多种常见手势的识别。

## 支持的手势

| 手势 | 标签 |
|---|---|
| 握拳 | `fist` |
| OK 手势 | `ok` |
| 张开手掌 | `palm` |
| 剪刀手 | `peace` |
| 点赞 | `thumbs_up` |

## 技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目主要开发语言 |
| OpenCV | 摄像头调用及图像处理 |
| MediaPipe | 手部关键点检测 |
| NumPy | 数值计算及特征处理 |
| Pandas | 数据集管理 |
| Scikit-learn | 机器学习模型训练与评估 |

## 模型测试结果

目前模型在测试集上的表现：

| 指标 | 结果 |
|---|---:|
| 测试样本数量 | 200 |
| 测试集准确率 | 99.50% |
| 握拳识别召回率 | 100% |
| OK 手势识别召回率 | 100% |
| 张开手掌识别召回率 | 97% |
| 剪刀手识别召回率 | 100% |
| 点赞识别召回率 | 100% |

注：以上为当前测试集结果，实际识别效果可能受到光照、拍摄角度及手部位置等因素影响。

## 项目结构

```text
HandGestureRecognition/
│
├── data/
│   └── hand_gestures.csv    # 手势数据集
│
├── .gitignore
├── collect_data.py          # 手势数据采集
├── main.py                 # 实时手势识别
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

## 环境要求

- Python 3.12
- camera
- Windows / Linux

## 安装依赖

克隆项目：

```bash
git clone https://github.com/Nayo0o0o/HandGestureRecognition.git
```

进入项目目录：

```bash
cd HandGestureRecognition
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

## 项目运行

运行数据（左右手）采集程序：

```bash
python collect_data.py
python collect_left_data.py
```

运行实时手势识别程序：

```bash
python main.py
```

注：模型训练脚本及模型文件需要根据实际项目文件配置。


## 项目地址

GitHub：[HandGestureRecognition](https://github.com/Nayo0o0o/HandGestureRecognition)

---

**贡献者：** Nayo

