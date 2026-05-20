# BLIP-2 论文复现任务清单

> **论文:** BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
> **任务:** 图像描述（Image Captioning）—— 输入图片，输出英文描述

---

## 一、环境与数据准备

| 编号 | 任务 | 详细说明 | 关键配置/工具 |
|------|------|----------|--------------|
| 1 | 环境搭建 | 配置 Python 环境，安装 PyTorch、Transformers、Pillow、NumPy 等依赖 | `pip install torch transformers pillow numpy` |
| 2 | 下载数据集 | 从 Kaggle 下载 Flickr8k 数据集，放入 `data/` 目录 | [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)，只需前 200 张图片 |

---

## 二、代码实现（核心任务）

| 编号 | 模块 | 需要做什么 | 技术细节 |
|------|------|----------|----------|
| 3 | 数据加载 | 读取 Flickr8k 前 200 张图片及对应的 5 条英文描述（caption） | 图片预处理（resize/normalize），文本 tokenization，构建 Dataset/DataLoader |
| 4 | Mini Q-Former | 实现简化版 Q-Former（Querying Transformer） | BLIP-2 的核心创新模块：用可学习的 query tokens 通过交叉注意力从冻结的图像特征中提取与文本相关的视觉信息 |
| 5 | 投影层 | 实现线性投影层，将 Q-Former 输出映射到语言模型 embedding 空间 | 维度对齐：CLIP ViT-B/32 输出 512 维 → OPT-125M embedding 维度（768） |
| 6 | 完整模型组装 | 串联三个模块 | `CLIP ViT-B/32（冻结）→ Q-Former（可训练）→ 投影层（可训练）→ OPT-125M（冻结）` |
| 7 | 训练循环 | 实现完整训练流程 | Loss: Cross-Entropy Loss；只训练 Q-Former + 投影层，其余冻结 |
| 8 | 推理/生成脚本 | 给定测试图片，生成英文描述 | 测试 3-5 张图片，对比 ground-truth 和模型生成的 caption |

---

## 三、模型架构

```
图片 (Image)
  ↓
冻结的 Vision Encoder (CLIP ViT-B/32, HuggingFace: openai/clip-vit-base-patch32)
  ↓
可训练的 Mini Q-Former（学生实现的核心模块）
  ↓
可训练的 Projection Layer（维度对齐）
  ↓
冻结的 Language Decoder (OPT-125M, HuggingFace: facebook/opt-125m)
  ↓
英文描述 (Caption)
```

---

## 四、输出交付物

| 编号 | 交付物 | 具体要求 |
|------|--------|----------|
| 9 | 实验报告 | 按 `report/report_template.md` 填写：实验背景、方法、结果分析 |
| 10 | 训练日志/截图 | 记录训练 loss 曲线或 loss 数值变化 |
| 11 | 测试样例展示 | 3-5 张测试图片的 ground-truth vs 生成结果对比 |
| 12 | Git 提交记录 | 模块化提交，约 10 次左右小粒度 commit |
| 13 | AI 对话记录 | 使用 entir.io 记录所有开发过程 |

---

## 五、关键约束

| 约束项 | 说明 |
|--------|------|
| 训练参数量 | 只训练 Q-Former + 投影层，Vision Encoder 和 Language Decoder 全部冻结 |
| 数据量 | 仅用 Flickr8k 前 200 张图片，不要求达到原论文性能 |
| 最低完成标准 | ① 能读取数据 ② 能构建模型 ③ 能跑通训练循环 |
| 模型来源 | CLIP: `openai/clip-vit-base-patch32`，OPT: `facebook/opt-125m`，均从 HuggingFace 加载 |

---

## 六、环境配置（版本清单）

### 核心依赖

| 包名 | 推荐版本 | 用途 | 备注 |
|------|----------|------|------|
| **Python** | **3.10** | 运行环境 | Windows 上最稳定，避免 3.13 兼容性问题 |
| **PyTorch** | **2.7.0** | 深度学习框架 | 踩坑提示：不要用 2.8.0，OPT-125M 输出有 bug |
| **torchvision** | **0.22.0** | 图像预处理/transforms | 与 PyTorch 2.7.0 配套 |
| **transformers** | **4.52.2** | 加载 CLIP 和 OPT 预训练模型 | HuggingFace 库 |
| **Pillow** | **>=10.0** | 图片读取 | 处理 Flickr8k 图像 |
| **numpy** | **>=1.24** | 数值计算 | transformers 依赖 |
| **tqdm** | **>=4.65** | 训练进度条 | 非必须，但强烈推荐 |
| **matplotlib** | **>=3.7** | 画 loss 曲线 | 实验报告用 |

### 安装命令

#### GPU 版（有 NVIDIA 显卡）

```bash
# 1. 安装 PyTorch (CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 2. 安装其他依赖
pip install transformers==4.52.2 pillow numpy tqdm matplotlib
```

#### CPU 版（无 NVIDIA 显卡 / 临时调试）

```bash
pip install torch==2.7.0 torchvision==0.22.0
pip install transformers==4.52.2 pillow numpy tqdm matplotlib
```

#### 一键安装（推荐创建 `requirements.txt`）

```txt
torch==2.7.0
torchvision==0.22.0
transformers==4.52.2
pillow>=10.0
numpy>=1.24
tqdm>=4.65
matplotlib>=3.7
```

```bash
pip install -r requirements.txt
```

### GPU 驱动要求（GPU 版）

| 项目 | 要求 |
|------|------|
| NVIDIA 驱动 | >= 572.xx（支持 CUDA 12.8） |
| CUDA Toolkit | 不需要单独安装，PyTorch 自带 CUDA runtime |

### 验证安装

```python
import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")

# 测试加载模型（首次运行会自动下载）
from transformers import CLIPVisionModel, OPTForCausalLM
clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
opt = OPTForCausalLM.from_pretrained("facebook/opt-125m")
print("模型加载成功！")
```

### HuggingFace 模型（训练时自动下载）

| 模型 | HuggingFace ID | 大小 | 作用 |
|------|---------------|------|------|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | ~340 MB | 冻结的视觉编码器 |
| OPT-125M | `facebook/opt-125m` | ~500 MB | 冻结的语言解码器 |

> 如果下载慢，可设置 HuggingFace 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 踩坑记录

| 问题 | 解决方案 |
|------|----------|
| PyTorch 2.8.0 + OPT-125M 输出错误 | 降级到 PyTorch 2.7.0 |
| CLIPVisionModel 加载时报 key 不匹配警告 | 不影响使用，可以忽略；或改用 `CLIPModel.from_pretrained()` 然后取 `.vision_model` |
| Windows 中文路径写入失败 | 用英文路径或 Git Bash 操作 |
