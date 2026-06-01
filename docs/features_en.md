# 📚 Feature Guide

> Complete guide to all Galatea-Core modules, including WebUI and CLI tools.

---

## 📋 Table of Contents

- [WebUI Module Details](#webui-module-details)
  - [Meta Dashboard](#meta-dashboard)
  - [Training Manifold](#training-manifold)
  - [Launch & Monitor Hub](#launch--monitor-hub)
  - [Assets & Deck Management](#assets--deck-management)
  - [Resource Sync Hub](#resource-sync-hub)
  - [Semantic Knowledge Engine](#semantic-knowledge-engine)
  - [Storage & Logs](#storage--logs)
  - [Holographic Replay](#holographic-replay)
  - [Model Deployment & Packaging](#model-deployment--packaging)
- [CLI Mode](#cli-mode)
- [Model Deployment Tool](#model-deployment-tool)
- [Parameter Reference](#parameter-reference)
- [FAQ](#faq)

---

## WebUI Module Details

### Meta Dashboard

**Overview**: Real-time win rate statistics for each deck during training.

#### Main Features

1. **Overall Dashboard**
   - Overall Win Rate: all decks ranked by total win rate
   - Going First Win Rate: each deck's win rate when going first
   - Going Second Win Rate: each deck's breakthrough rate when going second
   - Counter Matrix: mutual counter relationships between decks

2. **Dual Deck Comparison**
   - Select two decks for detailed comparison
   - View direct head-to-head records
   - Analyze first/second advantage

#### Usage

1. Select **📈 Meta Dashboard** in sidebar
2. Use slider to select iteration range for analysis
3. Switch between environment pools (if multiple deck folders exist)
4. View statistics

![Meta Dashboard](图片/卡组生态大盘.png)

---

### Training Manifold

**Overview**: Embedded TensorBoard for real-time training metric monitoring.

#### Key Metrics

| Metric | Meaning | Ideal Trend |
|--------|---------|-------------|
| `Train/Total_Loss` | Total loss | Decreasing |
| `Train/Policy_Loss` | Policy loss | Decreasing |
| `Train/Value_Loss` | Value loss | Decreasing |
| `Train/Entropy` | Exploration entropy | Slowly decreasing |
| `Rollout/Average_Reward` | Average reward | Increasing |
| `League_Overall/WinRate_Total` | Total win rate | Increasing |

#### Usage

1. Select **📉 Training Manifold** in sidebar
2. Click **🚀 Start TensorBoard**
3. Wait for TensorBoard service to start
4. View curves in embedded window, or open link in new tab

![TensorBoard](图片/TensorBoard.png)

---

### Launch & Monitor Hub

**Overview**: Core control panel for training and arena.

#### Three Function Tabs

##### 🔥 Start Training (Train)

Configure and launch AI training tasks.

**Checkpoint Loader**:
- Select existing `.pth` model to resume training
- Select "None" to train from scratch

**Model Architecture Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| d_model | Feature dimension | 256 |
| n_heads | Attention heads | 4 |
| n_layers | Transformer layers | 2 |

> ⚠️ When resuming, architecture params are auto-locked from checkpoint.

**Training Environment Config**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| Target Iterations | Total training iterations | 5000 |
| Batch Size | Steps per update | 4096 |
| Mini Batch | GPU training batch | 512 |
| Workers | Parallel processes | 4 |
| Worker Device | Device for worker inference | cpu |

**Advanced Toggles**:
- **Async Inference**: Central GPU server for inference, drastically saves VRAM
- **Disable Compile**: Must enable on Windows

##### 🏟️ Start Arena (Duel)

Test trained models in battle.

**Config**:
- P0 Model: Attacking AI model
- P1 Model: Opponent (select "None" for RuleBot)
- Game Count: Total test games
- Thought Frequency: Save AI decision record every N games

##### 🛠️ Rules Self-Check (Stress Test)

High-speed pure RuleBot battles to verify engine and script stability.

#### Live Terminal Mapping

Bottom of page shows real-time training/arena logs:
- **Full Terminal Logs**: Complete output
- **Alert Extraction**: Auto-extract error messages

![Launch & Monitor Hub](图片/启动与监控中枢.png)

---

### Assets & Deck Management

**Overview**: Manage deck files, staple pools, environment weights, and virtual mix pools.

#### 🃏 Meta Staples (142 Cache)

When AI encounters effects requiring card announcement, it prioritizes this pool.

**Features**:
- Add card: enter 8-digit card code
- Remove card: select and delete unwanted cards
- Preview list: view all cards in current pool

> 📖 See [Special Handling - 142 Announce Pool](special_handling.md#142-宣言池包装逻辑)

#### 📂 Deck & Pool Management

Manage `.ydk` deck files in `decks/` directory.

**Features**:
- Upload decks: supports `.ydk` format
- Create subfolders: as different "environment pools"
- Holographic deck viewer: visualize deck contents
- Batch operations: move, delete deck files

#### 🌐 Online Dynamic Environment Builder

Auto-fetch decks from online libraries like YGOProDeck with scheduled auto-updates.

![Online Fetch](图片/在线爬取卡组.png)

**Features**:
- Select target labels (Tournament TCG/OCG, Meta Decks, Casual, etc.)
- Set fetch quantity and depth mode (Latest/Historical Random)
- One-click fetch auto-creates environment pool
- **Background Daemon**: Set global cycle interval to auto-pull new decks from API periodically, ensuring AI always faces the latest meta

**Available Labels**:
| Label | Source |
|-------|--------|
| 🏆 Tournament TCG/OCG | Event top decks |
| ⚔️ Meta Decks | Current meta staples |
| 🎉 Non-Meta | Rogue brews |
| 📺 Anime Decks | Character-themed decks |
| 🕰️ Special Formats | Edison/Goat/Speed Duel |

#### ⚖️ Dynamic Environment Weight Scheduler

Control sampling probability for each environment pool during training.

**Features**:
- Set weight per pool (0.0 ~ 10.0)
- Categorized by source (Online/Virtual/Local pools)
- Bulk set applies to all pools in category at once
- Takes effect immediately, Workers read on next game

> 📖 See [Special Handling - Global Weights](special_handling.md#卡组权重调整global-weights)

#### 🧠 Virtual Mix Pool Builder

Create cross-pool mixing recipes, allowing decks from different physical pools to battle each other.

**Features**:
- Create virtual mix pool recipes
- Mix multiple physical pools with per-pool weights
- Automatically appears in global weights panel after creation

> 📖 See [Special Handling - Virtual Mix Pools](special_handling.md#虚拟伪装池模块virtual-mix-pools)

---

### Resource Sync Hub

**Overview**: Sync latest card data and scripts from official repositories.

#### Sync Targets

- **Update Core Code**: Pull latest framework code from GitHub
- **Update CDB & Scripts**: Sync `cards.cdb` and `script/` from MyCard/official repos

#### Advanced Options

- **Script Repo Source**: Specify custom script repository URL
- **Force Overwrite**: Override local modifications

![Resource Sync](图片/资源同步.png)

---

### Semantic Knowledge Engine

**Overview**: Parse Lua scripts to extract semantic features for AI learning.

#### ⚙️ Execution Hub

Scan all Lua scripts in `script/` directory to extract semantic information.

**Options**:
- **Physical Clear**: Delete local old data, re-parse everything
- **Github Sync**: Pull base KB from remote repository

#### 🧬 Custom Hash Explorer

View special effects compressed through Hash clustering.

**Features**:
- View underlying Lua logic for each hash tag
- View all cards sharing the same logic

#### 🔍 Card Semantic Viewer

Enter card code to view AI-perspective semantic features.

![Semantic KB](图片/语义知识库.png)

---

### Storage & Logs

**Overview**: Manage all files generated during system operation.

#### Managed File Types

| Tab | Directory | File Type |
|-----|-----------|-----------|
| System Logs | `./system_logs/` | `.log` |
| Thought Records | `./ai_thoughts/` | `.json` |
| Model Storage | `./models/` | `.pth` |
| Match Data | `./web_data/` | `.csv` |
| TensorBoard | `./runs/` | Binary |

#### Common Functions

- View/preview file contents
- Export (download) files
- Batch delete
- One-click purge (requires confirmation)

![Storage & Logs](图片/存储与日志仓库.png)

---

### Holographic Replay

**Overview**: Visualize AI decision process to deeply understand AI "thinking".

#### Interface Components

1. **Board View**
   - True coordinate card layout
   - Dynamic display of hand, field, graveyard zones
   - Highlight AI-selected cards and targets

2. **Decision List**
   - Show all available actions
   - Show confidence (probability) for each action
   - Highlight AI's final chosen action

3. **Playback Controls**
   - Previous/Next step
   - Auto-play (adjustable speed)
   - Timeline slider

#### Vision Toggles

- 👁️ P1 Hand: Show/hide opponent hand
- 👁️ P0 Hand: Show/hide own hand
- 👁️ P1 Set: Show/hide opponent face-downs
- 👁️ P0 Set: Show/hide own face-downs
- 🔄 P1 Flip: Rotate opponent perspective 180°

![Holographic Replay](图片/全息回放.png)

---

### 📦 Model Deployment & Packaging

**Overview**: Package trained models and knowledge bases into `.gkg` deployment packages for easy sharing and import.

#### 📥 Export Package

Package models as .gkg format:

- Select `.pth` models to package (multi-select supported)
- Optionally include knowledge base (`knowledge_base.json`) and staple pool (`meta_staples.json`)
- Auto-generate manifest file (`manifest.json`)
- Custom package name support

#### 📤 Unpack & Selective Import

Import external .gkg packages into current system:

- Support direct unpack from local filesystem (no upload needed)
- Fine-grained selection of models and config files to import
- Staging area management (preview before import decision)

---

## CLI Mode

Galatea-Core also supports full command-line operation.

### Training Command

```bash
python main.py train [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--dir` | Model save directory | `./models` |
| `--steps` | Total training iterations | 1000 |
| `--deck_dir` | Deck directory | `./decks` |
| `--d_model` | Feature dimension | 256 |
| `--n_heads` | Attention heads | 4 |
| `--n_layers` | Transformer layers | 2 |
| `--resume` | Checkpoint to resume from | - |
| `--batch_size` | Steps per collection | 4096 |
| `--mini_batch` | GPU training batch | 512 |
| `--workers` | Parallel processes | 4 |
| `--worker_device` | Worker device | cpu |
| `--async_infer` | Enable async inference | - |
| `--use_onnx` | Enable ONNX inference acceleration | - |
| `--no_compile` | Disable compilation | - |
| `--standard_core` | Disable ghost byte parsing (for custom cores) | - |
**Examples**:

```bash
# Basic training
python main.py train --dir ./models --steps 1000 --no_compile

# Advanced config
python main.py train \
  --dir ./models \
  --batch_size 16384 \
  --mini_batch 512 \
  --workers 6 \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --async_infer \
  --no_compile

# Resume training
python main.py train \
  --resume ./models/galatea_iter_100.pth \
  --steps 5000 \
  --async_infer \
  --no_compile
```

### Arena Command

```bash
python main.py duel [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--p0` | P0 model path | - |
| `--p1` | P1 model path (None = RuleBot) | - |
| `-n, --num` | Game count | 100 |
| `--device` | Inference device | cpu |
| `--deck_dir` | Deck directory | `./decks` |
| `--thought_freq` | Thought log save frequency | 0 |
| `--standard_core` | Disable ghost byte parsing (for custom cores) | - |
**Examples**:

```bash
# AI vs RuleBot
python main.py duel --p0 ./models/galatea_iter_100.pth --num 100

# AI vs AI
python main.py duel --p0 ./models/model_a.pth --p1 ./models/model_b.pth --num 100

# Save thought records
python main.py duel --p0 ./models/galatea_iter_100.pth --thought_freq 5 --num 100
```

### Self-Check Command

```bash
python main.py play [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --num` | Game count | 10 |
| `--deck_dir` | Deck directory | `./decks` |
| `--standard_core` | Disable ghost byte parsing (for custom cores) | - |

### Update Command

```bash
python main.py update [options]
```

| Option | Description |
|--------|-------------|
| `--core` | Update core code |
| `--data` | Update card DB and scripts |
| `--repo` | Specify script repo source |
| `--force` | Force overwrite local changes |

### Semantic Parse Command

```bash
python main.py parse [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--script_dir` | Lua script directory | `./script` |
| `--output` | Output file path | `knowledge_base.json` |
| `--clear` | Clear local KB | - |
| `--sync` | Pull base library from remote | - |

---

## Model Deployment Tool

Galatea-Core provides standalone model packaging and deployment tools.

### Launch Deployment Tool

```bash
python deploy_tool.py
```

### Features

#### 1. Package New Model (.gkg)

Package trained models into `.gkg` deployment packages containing:
- Model files (`.pth`)
- Knowledge base (`knowledge_base.json`)
- Staple pool (`meta_staples.json`)
- Manifest file (`manifest.json`)

#### 2. Unpack & Import

Extract `.gkg` packages and import into current system:
- Model files extracted to `./models/`
- Knowledge base files overwritten to root directory

---

## Parameter Reference

### Model Architecture Parameters

| Parameter | Description | Tuning Advice |
|-----------|-------------|---------------|
| `d_model` | Feature dimension | Higher = smarter, but computation increases exponentially. Must be divisible by `n_heads` |
| `n_heads` | Attention heads | Typically 4 or 8. More heads = stronger complex relationship handling |
| `n_layers` | Transformer layers | 2 for quick experiments, 4-6 for competitive decks |
| `vocab_size` | Card vocabulary size | Just needs to exceed total card ID count |

### Training Parameters

| Parameter | Description | Resource Impact |
|-----------|-------------|-----------------|
| `batch_size` | Steps per collection | ⬆️ **RAM** (primary) — larger = more stable but needs more memory |
| `mini_batch` | GPU training batch | ⬆️ **VRAM** (primary) — larger = faster updates |
| `workers` | Parallel processes | ⬆️ **RAM** (primary) — adjust by CPU cores, typically 4-12 |
| `timeout` | Worker single-collection timeout | Prevents zombie processes, default 300s, large decks can use 600s |
| `async_infer` | Async inference | ⬇️ **VRAM** (significant savings) — GPU centralized inference, workers don't load model |
| `use_onnx` | ONNX inference acceleration | ⬆️ **Collection Speed** (30%+ faster) — Workers use ONNX Runtime for high-speed inference |
| `no_compile` | Disable compilation | Recommended for Windows or legacy environments |

#### RL Soul Hyperparameters (Deep Tuning)

These parameters can be adjusted in WebUI or CLI for fine-grained PPO algorithm control:

| Parameter | CLI Flag | Description | Default | Tuning Advice |
|-----------|----------|-------------|---------|---------------|
| Discount Factor | `--gamma` | Future reward weighting | 0.998 | Higher = more emphasis on long-term, good for long games |
| Learning Rate | `--lr` | Neural plasticity speed | 1e-4 | Too high = unstable, too low = slow convergence |
| Exploration Coef | `--entropy` | Curiosity/exploration strength | 0.03 | Encourages trying new actions, auto-decays with training |
| GAE Lambda | `--gae_lambda` | Generalized Advantage Estimation λ | 0.95 | Balances bias-variance tradeoff, generally don't change |
| PPO Clip Epsilon | `--clip_eps` | Policy update clipping threshold | 0.2 | Limits single update magnitude, prevents overshooting |

---

## FAQ

### Q1: CUDA Out of Memory (VRAM insufficient)

**Solutions**:

```bash
# Option 1: Enable async inference (recommended)
python main.py train --async_infer --worker_device cpu

# Option 2: Reduce mini_batch
python main.py train --mini_batch 256

# Option 3: Reduce workers
python main.py train --workers 2
```

### Q2: Out of Memory (RAM insufficient)

**Solutions**:

```bash
# Option 1: Reduce workers
python main.py train --workers 2

# Option 2: Reduce Batch Size
python main.py train --batch_size 4096
```

### Q3: Windows torch.compile error

**Solution**: Add `--no_compile` flag, or check "Disable Model Compilation" in WebUI.

### Q4: Training speed is slow

**Possible causes and solutions**:

1. **No GPU**: Ensure PyTorch correctly detects CUDA
2. **Too many workers**: Reduce `--workers`
3. **Async inference not enabled**: Add `--async_infer`

### Q5: Model not converging

**Possible causes and solutions**:

1. **Learning rate too high**: Adjust `LR` parameter in `trainer.py`
2. **Batch too small**: Increase `--batch_size`
3. **Too few decks**: Add more decks to `decks/` directory

### Q6: Missing cards.cdb or script directory

**Solution**: Use built-in update tool:

```bash
python main.py update --data
```

Or click sync in WebUI's **🔄 Resource Sync Hub**.

---

## Next Steps

- 🔧 Read [Architecture](architecture.md) for framework internals
- 📝 View [Changelog](changelog.md) for version history
- 🚀 Return to [Quick Start](quickstart.md) to begin training
