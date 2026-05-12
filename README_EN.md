<div align="center">

![Galatea Logo](docs/图片/logo.png)

# 🌟 Galatea-Core

**Yu-Gi-Oh! Universal AI Training Framework based on Transformer + PPO**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

English | [简体中文](README.md)

</div>

---

## ✨ Key Features

- 🧠 **Universal AI Model** - Deck-agnostic, automatically parses Lua scripts to learn card effects
- 🎮 **Complete WebUI** - All-in-one training, testing, and management console
- 📦 **One-Click Package** - Built-in Python environment, just double-click to start
- 🔥 **Efficient Training** - Async inference + Mixed precision + League training mechanism
- 👁️ **Decision Visualization** - Holographic replay system to understand AI thinking process

---

## 🖥️ Showcase

<div align="center">

| | |
|:---:|:---:|
| ![Deck Ecosystem Dashboard](docs/图片/卡组生态大盘.png) | ![TensorBoard](docs/图片/TensorBoard.png) |
| **📈 Meta Dashboard** | **📉 Training Manifold** |
| ![Launch & Monitor Hub](docs/图片/启动与监控中枢.png) | ![Holographic Replay](docs/图片/全息回放.png) |
| **⚔️ Launch & Monitor Hub** | **👁️ Holographic Replay** |

</div>

---

## 🚀 Quick Start

### One-Click Package Users (Recommended)

1. Download and extract the integrated package
2. Double-click `一键包启动Webui.bat`
3. Browser automatically opens the WebUI

### Developers

```bash
# Clone repository
git clone https://github.com/Noctfom/Galatea-Core.git
cd Galatea-Core

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit tensorboard numpy pandas psutil rich

# Prepare resource files
python main.py update --data

# Start WebUI
streamlit run app.py
```

📖 **Detailed Tutorial**: [Quick Start Guide](docs/quickstart.md) (Chinese)

---

## 🖥️ WebUI Features

| Module | Function |
|--------|----------|
| 📈 **Meta Dashboard** | Win rate stats, counter matrix |
| 📉 **Training Manifold** | Embedded TensorBoard |
| ⚔️ **Launch & Monitor Hub** | One-click training/arena |
| 🗃️ **Assets & Deck Management** | Deck upload, staple pool, weight scheduling, online fetching |
| 🔄 **Resource Sync Hub** | Auto-update card database |
| 🧠 **Semantic Knowledge Engine** | Lua script parsing |
| 👁️ **Holographic Replay** | AI decision visualization |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [🚀 Quick Start](docs/quickstart.md) | Installation and first training in 5 minutes (Chinese) |
| [📚 Feature Guide](docs/features.md) | Complete WebUI and CLI guide (Chinese) |
| [🔧 Architecture](docs/architecture.md) | Technical principles and core algorithms (Chinese) |
| [🧬 Special Handling](docs/special_handling.md) | Implementation details of unique features (Chinese) |
| [📝 Changelog](docs/changelog.md) | Version history (Chinese) |

> 💡 Documentation is currently available in Chinese only. English translations are in progress. Contributions are welcome!

---

## 🛠️ CLI Commands

```bash
# Training
python main.py train --dir ./models --steps 1000 --async_infer --no_compile

# Arena
python main.py duel --p0 ./models/galatea_iter_100.pth --num 100

# Update resources
python main.py update --data

# Semantic parsing
python main.py parse --script_dir ./script
```

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| GPU | GTX 1060 6GB | RTX 3060 12GB+ |
| RAM | 16GB | 32GB+ |
| Disk | 10GB | SSD |

---

## 🤝 Community

- **QQ Group**: 492420925
- **GitHub Issues**: [Submit an Issue](https://github.com/Noctfom/Galatea-Core/issues)
- 📧 Contact Author: noctfom114514@outlook.com

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

<div align="center">

**If this project helps you, please give it a ⭐ Star!**

</div>
