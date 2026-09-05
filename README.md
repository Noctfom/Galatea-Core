<div align="center">

<img src="docs/图片/logo.png" alt="Galatea Logo" width="50%">

# 🌟 Galatea-Core

**基于 Transformer + PPO 的游戏王通用 AI 训练框架**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Release: v3.6.2](https://img.shields.io/badge/Release-v3.6.2-brightgreen.svg)](docs/changelog.md)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](README_EN.md) | 简体中文

</div>

---

## ✨ 特性亮点

- 🧠 **通用 AI 模型** - 不依赖特定卡组，自动解析 Lua 脚本学习卡片效果
- 🎮 **完整 WebUI** - 一站式训练、测试、管理控制台
- 📦 **一键包支持** - 内置 Python 环境，双击即可启动
- 🔥 **高效训练** - 固定中央批量推理 + CPU/CUDA 双模式 + 联盟训练机制
- 🔐 **模型身份管理** - 动态文件前缀、自动 UUID、内置轮次与完整 ONNX 制品组
- 🧩 **动作协议 V2** - 顺序多选、宏动作组合、响应语义与约束特征统一进入策略网络
- 👁️ **决策可视化** - 全息回放系统，深入理解 AI 思考过程

---

## 🖥️ 功能展示

<div align="center">

| | |
|:---:|:---:|
| ![卡组生态大盘](docs/图片/卡组生态大盘.png) | ![TensorBoard](docs/图片/TensorBoard.png) |
| **📈 卡组生态大盘** | **📉 训练流形图** |
| ![启动与监控中枢](docs/图片/启动与监控中枢.png) | ![全息回放](docs/图片/全息回放.png) |
| **⚔️ 启动与监控中枢** | **👁️ 全息读心回放** |

</div>

---

## 🚀 快速开始

### Windows 用户

#### 一键包（推荐，无需配置 Python 环境）

1. 下载整合包并解压
2. 双击 `一键包启动Webui.bat`
3. 浏览器自动打开 WebUI 界面

当前 Windows 一键包内置 PyTorch 2.9.1 + CUDA 13.0，面向 RTX 20/30/40/50
系列及 GTX 16 系列。没有兼容 NVIDIA 显卡时，`auto` 模式会回落 CPU；GTX 10
系列及更早显卡如需 GPU 训练，应自行更换兼容的旧版 PyTorch/CUDA 环境。

#### 构建发布用一键包

维护者双击 `构建一键包.bat` 即可执行依赖、运行资源、内核和 CUDA 探针，成功后
在项目根目录生成 `Galatea_Core_Vx.x.x.zip`。压缩包包含便携 Python、卡片数据库、
Lua 脚本和卡组，不包含本机模型、日志、回放、缓存及 Git 开发数据。

#### 手动安装（开发者）

```bash
# 克隆仓库
git clone https://github.com/Noctfom/Galatea-Core.git
cd Galatea-Core

# 安装 PyTorch（以下与当前 Windows 一键包一致；旧显卡请选择兼容版本）
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
# 纯 CPU 环境可改用：https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 准备资源文件
python main.py update --data

# 启动 WebUI
streamlit run app.py
```

### Linux 用户

```bash
# 克隆仓库
git clone https://github.com/Noctfom/Galatea-Core.git
cd Galatea-Core

# 一键环境安装 + 启动（自动检测 GPU/CUDA、创建虚拟环境）
chmod +x setup.sh
./setup.sh               # 安装依赖 & 启动 WebUI
./setup.sh --train       # 安装依赖 & 启动 CLI 训练
./setup.sh --duel        # 安装依赖 & 启动竞技场

# 或手动安装
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py update --data
streamlit run app.py
```

📖 **详细教程**: [快速上手指南](docs/quickstart.md)

---

## 🖥️ WebUI 功能

| 模块 | 功能 |
|------|------|
| 📈 **卡组生态大盘** | 胜率统计、克制矩阵 |
| 📉 **训练流形图** | 内嵌 TensorBoard |
| ⚔️ **启动与监控中枢** | 一键训练/竞技场 |
| 🗃️ **资产与卡组管理** | 卡组上传、泛用卡池、权重调度、在线抓取 |
| 🔄 **资源同步中枢** | 自动更新卡库 |
| 🧠 **语义知识库引擎** | Lua 脚本解析 |
| 📁 **存储与日志仓库** | 处理项目各类文件 |
| 👁️ **全息读心回放** | AI 决策可视化 |
| 📦 **模型部署与打包** | 导入导出模型包 |

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| [🚀 快速上手](docs/quickstart.md) | 5 分钟完成安装和首次训练 |
| [📚 功能详解](docs/features.md) | WebUI 和命令行完整指南 |
| [🔧 架构设计](docs/architecture.md) | 技术原理和核心算法 |
| [🧬 特殊处理逻辑](docs/special_handling.md) | 框架独特特性的实现细节 |
| [🗺️ 后续路线](docs/roadmap.md) | 主框架认知与组卡模块 TODO |
| [📝 更新日志](docs/changelog.md) | 版本更新历史 |

---

## 🛠️ 命令行

```bash
# 训练
python main.py train --dir ./models --additional-iterations 1000 --model-prefix galatea --device auto --no_compile

# 竞技场
python main.py duel --p0 ./models/galatea_iter_100.pth --num 100

# 更新资源
python main.py update --data

# 语义解析
python main.py parse --script_dir ./script
```

---

## 📋 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| Python | 3.9+ | 3.11 |
| GPU | 可选（支持纯 CPU 训练） | RTX 3060 12GB+ |
| RAM | 16GB | 32GB+ |
| 硬盘 | 10GB | SSD |
| Linux | GLIBC ≥ 2.35（Ubuntu 22.04+） | — |

---

## 🤝 社区

- **QQ 群**: 492420925
- **GitHub Issues**: [提交问题](https://github.com/Noctfom/Galatea-Core/issues)
- 📧 联系作者：noctfom114514@outlook.com

---

## 📄 许可证

本项目采用 [GNU General Public License v3.0](LICENSE)。

---

## 🙏 致谢

- [YGOProCore](https://github.com/Fluorohydride/ygopro-core) - YGOPRO 核心引擎，万物之源
- [MDPro3](https://code.moenext.com/sherry_chaos/MDPro3) - MDPro3，目前优先适配的端
- [YGOPro 官方脚本库](https://github.com/Fluorohydride/ygopro-scripts) - 官方 Lua 脚本仓库，卡片效果解析的基础
- [萌卡 MyCard](https://github.com/mycard/ygopro-database) - cards.cdb 卡片数据库来源
- [百鸽 YGOCDB](https://ygocdb.com/) - 卡片图片渲染 API 与数据查询
- [YGOProDeck](https://ygoprodeck.com/) - 在线卡组数据爬取来源
- [YugiohAi](https://github.com/crispy-chiken/YugiohAi) - 致敬同路线开发者，在正式版后的迭代优化中参考了该项目
- [ygo-agent](https://github.com/sbl1996/ygo-agent) - 致敬同路线开发者，在正式版后的迭代优化中参考了该项目

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

</div>
