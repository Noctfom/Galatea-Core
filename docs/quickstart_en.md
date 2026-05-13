# 🚀 Quick Start Guide

> Zero to first AI training in **5-10 minutes**.

---

## 📋 Table of Contents

- [One-Click Launch](#one-click-launch)
- [Sync Resources & Card DB](#sync-resources--card-db)
- [Build Semantic Knowledge Base](#build-semantic-knowledge-base)
- [Prepare Decks](#prepare-decks)
- [Adjust Deck Weights (Optional)](#adjust-deck-weights-optional)
- [Check Special Announce Pool (Optional)](#check-special-announce-pool-optional)
- [RuleBot Stress Test (Optional)](#rulebot-stress-test-optional)
- [Start Training](#start-training)
- [View Training Results](#view-training-results)
- [Arena Model Testing](#arena-model-testing)
- [[Alternate] Using Packaged Models](#alternate-using-packaged-models)

---

## One-Click Launch

### Windows Users

Double-click `一键包启动Webui.bat`. Browser opens automatically at `http://localhost:8501`.

![One-Click Launch](图片/一键包启动窗口.png)

> 💡 If browser doesn't open, manually visit `http://localhost:8501`

### Linux Users

Use the bundled Python environment to launch WebUI:

```bash
# Enter project directory
cd Galatea-Core

# Use built-in python_env to launch Streamlit
./python_env/bin/python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
```

> ⚠️ Linux `ocgcore.so` is not fully tested. Report engine issues on GitHub Issues.

---

## Sync Resources & Card DB

Go to **🔄 Resource Sync Hub**:

1. Check **🃏 Update CDB Database & Official Lua Scripts**
2. Click **🚀 Start Sync**

![Resource Sync](图片/资源同步.png)

Wait a few minutes. This pulls the latest `cards.cdb` and `script/` from official repos.

---

## Build Semantic Knowledge Base

Go to **🧠 Semantic Knowledge Engine**:

1. ⚠️ **Important**: Check **🌐 Sync Base KB from Github** (must for first time, greatly speeds up parsing)
2. Click **🧠 Start Extracting Card Semantics**

![Semantic KB](图片/语义知识库.png)

> 📖 See [Special Handling - Semantic KB](special_handling.md#语义化模块semantic-kb) for Hash clustering details.

Wait for parsing to complete (first time may take several minutes). Results saved to `knowledge_base.json`.

---

## Prepare Decks

Multiple ways:

### Method 1: Use bundled test decks

The one-click package includes several `.ydk` test decks ready to use.

### Method 2: Manual upload

Go to **🗃️ Assets & Deck Management → 📂 Deck & Pool Manager**:
- Drag your `.ydk` files into the upload area
- Create subfolders for categorization (e.g. `tier1_meta`, `fun_decks`)

### Method 3: Online fetch (Recommended)

Go to **🗃️ Assets & Deck Management → 🌐 Online Fetcher**:
- Select target label (e.g. `🏆 Tournament TCG`, `⚔️ Meta Decks`)
- Set fetch quantity
- Click fetch to batch download decks automatically

![Online Fetch](图片/在线爬取卡组.png)

---

## Adjust Deck Weights (Optional)

Go to **🗃️ Assets & Deck Management → ⚖️ Dynamic Pool Weights**:

- Set weight for each pool (0.0 ~ 10.0)
- Higher weight = AI trains more in that environment
- Adjust anytime, takes effect next game

![Weight Adjustment](图片/权重调整界面.png)

> 📖 See [Special Handling - Global Weights](special_handling.md#卡组权重调整global-weights)

---

## Check Special Announce Pool (Optional)

Go to **🗃️ Assets & Deck Management → 🃏 Meta Staples (142 Cache)**:

- View current staple cards (default includes common handtraps: Ash Blossom, Maxx "C", etc.)
- Add/remove cards as needed

![Staple Configuration](图片/泛用卡组配置.png)

> 📖 See [Special Handling - 142 Announce Pool](special_handling.md#142-宣言池包装逻辑)

---

## RuleBot Stress Test (Optional)

Go to **⚔️ Launch & Monitor Hub → 🛠️ Rules Self-Check**:

- Set game count (recommend 50-100)
- Start to verify engine & script stability
- Run once before first training to ensure environment is healthy

---

## Start Training

Go to **⚔️ Launch & Monitor Hub → 🔥 Start Training (Train)**.

### Parameter Classification

Galatea-Core's training parameters are divided into three tiers:

| Tier | Parameters | Characteristics |
|------|-----------|-----------------|
| 🧠 **Brain Structure** | `d_model`, `n_heads`, `n_layers` | Like AI's brain capacity — fixed after definition (model architecture is locked) |
| ⚙️ **Training Config** | `batch_size`, `mini_batch`, `workers`, `timeout`, `async_infer`, `no_compile`, plus RL hyperparameters below | Can be freely adjusted when resuming training |
| 🗂️ **Environment Config** | Deck weights, virtual pools, staple pool | Fully decoupled from training module, adjustable in real-time during training |

> ℹ️ When resuming, brain structure params are auto-locked from checkpoint. Training and environment configs can be freely modified.

### Parameter Quick Reference

| Parameter | Description | Beginner | Memory/VRAM Impact |
|-----------|-------------|----------|---------------------|
| **Checkpoint Loader** | "None" = train from scratch | None | - |
| **d_model** | 🧠 Feature dimension (bigger = smarter) | 256 | ⬆️ VRAM |
| **n_heads** | 🧠 Attention heads (more = broader reference) | 4 | ⬆️ VRAM |
| **n_layers** | 🧠 Transformer layers (more = deeper thinking) | 2 | ⬆️ VRAM |
| **Target Iterations** | Total training iterations | 1000+ | - |
| **Batch Size** | Steps per collection | 4096 | ⬆️ RAM (primary) |
| **Mini Batch** | GPU training batch | 256-512 | ⬆️ VRAM (primary) |
| **Workers** | Parallel worker processes | 4 (by CPU cores) | ⬆️ RAM (primary) |
| **Timeout** | Worker single-collection timeout | 300s | - |
| **Async Inference** | ✅ Enable (saves VRAM) | On | ⬇️ VRAM |
| **Disable Compile** | ✅ Enable (Windows required) | On | - |

> 💡 **RAM/VRAM Tuning Rule**: `batch_size` and `workers` consume RAM, `mini_batch` consumes VRAM. If RAM blows up, reduce workers first. If VRAM blows up, reduce mini_batch or enable async inference. batch_size relates to learning smoothness (total training data volume) — don't set it too low. workers and mini_batch are more about speed optimization. Worker count must not exceed physical CPU cores! Can reduce appropriately when async inference is enabled.

### Recommended Configurations

#### 🧪 Beginner Test (Quick Validation)

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| n_layers | 2 |
| Target Iterations | 100 |
| Batch Size | 2048 |
| Mini Batch | 128 |
| Workers | 2 |
| Timeout | 300 |

#### 🏆 Full Training (Competitive)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| n_layers | 6 |
| Target Iterations | 5000+ |
| Batch Size | 16384 |
| Mini Batch | 256 |
| Workers | 6-8 |
| Timeout | 600 |

Click **🔥 Start Training Process**.

![Launch & Monitor Hub](图片/启动与监控中枢.png)

> 📖 Full CLI parameter reference: [Feature Guide - CLI Mode](features.md#命令行模式)

---

## View Training Results

### Live Logs

After training starts, view real-time logs at the bottom of **⚔️ Launch & Monitor Hub**.

### TensorBoard Curves

Go to **📉 Training Manifold**, click **🚀 Start TensorBoard** to view:

| Key Metric | Ideal Trend |
|------------|-------------|
| `Train/Total_Loss` | Decreasing |
| `Train/Entropy` | Slowly decreasing |
| `Rollout/Average_Reward` | Increasing |
| `League_Overall/WinRate_Total` | Increasing |

### Meta Dashboard

Go to **📈 Meta Dashboard** to view win rate stats and counter matrix.

---

## Arena Model Testing

After training, go to **⚔️ Launch & Monitor Hub → 🏟️ Start Arena (Duel)**:

1. **P0 Model**: Select your trained `.pth` model
2. **P1 Model**: Select "None" to fight RuleBot
3. Set game count and thought log frequency
4. Click **⚔️ Start Arena Process**

Results viewable in **📈 Meta Dashboard**. AI decision process replayable in **👁️ Holographic Replay**.

![Holographic Replay](图片/全息回放.png)

---

## [Alternate] Using Packaged Models

If you received a `.gkg` packaged model:

1. Go to **📦 Model Deployment → 📤 Unpack & Import**
2. Place `.gkg` in local packages folder
3. Select to unpack, check desired model/knowledge files
4. Click import

Imported models appear in `./models/`. Jump to [Arena Testing](#arena-model-testing) or [Holographic Replay](#view-training-results).

---

## Next Steps

- 📚 Read [Feature Guide](features.md) for detailed module usage
- 🔧 Read [Architecture](architecture.md) for framework internals
- 🧬 Read [Special Handling](special_handling.md) for unique features

Issues? [GitHub Issues](https://github.com/Noctfom/Galatea-Core/issues) or QQ Group **492420925**

📧 Contact: noctfom114514@outlook.com
