# 📚 Feature Guide

> Complete guide to all Galatea-Core modules, including WebUI and CLI tools.

> This document applies to **Galatea-Core v3.6.3**.

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
| `Train/Total_Loss` | Combined PPO policy, value, and entropy objective | Need not decrease monotonically; watch for non-finite values and persistent discontinuities |
| `Train/Policy_Loss` | Clipped policy objective | May oscillate around zero and is not a standalone playing-strength metric |
| `Train/Value_Loss` | Error between value estimates and returns | Narrowing over the long term, with substantial short-term noise allowed |
| `Train/Entropy` | Policy exploration uncertainty | May decrease slowly, but should not collapse toward zero too early |
| `Train/Approx_KL` | Approximate old/new policy divergence | Small and stable; avoid persistent spikes |
| `Train/Clip_Fraction` | Fraction of samples clipped by PPO | Moderate; avoid staying near 0 or 1 |
| `Train/Explained_Variance` | How much return variance the value network explains | Rise from 0 toward 1; persistent negatives need investigation |
| `Train/Gradient_Norm` | Total gradient norm before clipping | Finite and without sustained abnormal spikes |
| `Rollout/Average_Reward` | Mean reward of sampled games | Prefer smoothed trends within the same opponent category |
| `League_Overall/WinRate_Total` | Win rate over the mixed league pool | Read alongside Rule/Self/Hist splits to avoid opponent-mixture bias |
| `Performance/Rollout_Steps_Per_Second` | Valid rollout samples collected per second | Stable or increasing |
| `Performance/Collection_Seconds` | Collection duration for the iteration | Stable at comparable batch and opponent composition |
| `Performance/PPO_Update_Seconds` | PPO update duration for the iteration | Stable at comparable sample counts |

#### Usage

1. Select **📉 Training Manifold** in sidebar
2. Click **🚀 Start TensorBoard**
3. Wait for TensorBoard service to start
4. View curves in embedded window, or open link in new tab

TensorBoard is launched as a module by the active bundled Python, so no separate command is required on the system `PATH`. The Stop button only terminates a service verified as belonging to this project; if another program owns port 6006, it is left running and the UI reports the conflict.

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
- New training accepts a filename prefix, while `model_id` is an automatically generated UUID with no manual edit control
- Resume inherits UUID, prefix, embedded iteration, and network architecture from the checkpoint and checks directory identity conflicts

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
| Mini Batch | PPO update batch | 512 |
| Workers | Parallel processes | 4 |
| Training Device | auto / cpu / cuda | auto |

**Advanced Toggles**:
- **Central Batched Inference**: Always enabled; all workers stay on CPU
- **Disable Compile**: Recommended on Windows or when the compiler toolchain is incomplete
- **Export ONNX**: Export synchronously at every 10-iteration checkpoint for historical opponents in workers
- **Standard Core**: Enable for custom OCGCore builds without ghost bytes

**Model Action Protocol**:
- v3.5.0 introduced action semantics V2; v3.6.0 uses Model Protocol V3, binds effect-slot identity, and adds genuinely order-sensitive aggregation for the active chain and recent activation history. Checkpoints, network weights, ONNX graphs, and artifact manifests all record and validate it
- v3.6.2 uses each Lua `Effect.CreateEffect(c)` object as identity and binds the complete runtime `desc` to its existing code-semantic slot. Action candidates can consume that exact effect vector, while chain/history context and used-this-turn bits share the same mapping. Stringid generates a Core identifier but is no longer interpreted as a slot ordinal
- Action inputs include operation kind, actual response, selection constraints, target code/location/material values, and a stable semantic signature. Type 26 is decided step by step through Core's native Select/Unselect flow
- A protocol mismatch is rejected before loading so structurally different weights cannot be applied silently to the current action head

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
- **Physical Clear**: Delete the local KB, Hash map, and code-semantic vectors before a full rebuild
- **Github Sync**: Pull structured knowledge, the Hash map, code-semantic vectors, and their index from the same remote directory, then append locally missing slots automatically
- **Code Semantic Extraction**: Append only new effect slots when the local `.npy` pair is coherent; rebuild fully if its index or dimension is incompatible

#### 🧬 Custom Hash Explorer

View special effects compressed through Hash clustering.

**Features**:
- View underlying Lua logic for each hash tag
- View all cards sharing the same logic

#### 🔍 Card Semantic Viewer

Enter card code to view AI-perspective semantic features.

#### 🧪 V3 Observation Audit

- Training and Arena generate independent reports only with `--protocol-audit`; RuleBot self-check remains enabled by default
- Reports are stored under `system_logs/protocol_v3_audit/`
- Full-bundle validation cross-checks KB effect slots, code-vector rows, and index keys
- Reports can be filtered by source to inspect chain-structure anomalies and effect-slot mappings
- Lua `Stringid` is parsed offline only when the audit tab is opened, adding no script I/O to training

![Semantic KB](图片/语义知识库.png)

---

### Storage & Logs

**Overview**: Manage all files generated during system operation.

#### Managed File Types

| Tab | Directory | File Type |
|-----|-----------|-----------|
| System Logs | `./system_logs/` | `.log` |
| V3 Audit Reports | `./system_logs/protocol_v3_audit/` | `.json` |
| Thought Records | `./ai_thoughts/` | `.json` |
| Model Storage | `./models/` | `.pth`, `.onnx`, `.onnx.data`, `.artifacts.json` |
| Match Data | `./web_data/` | `.csv` |
| TensorBoard | `./runs/` | Binary |

#### Common Functions

- View/preview file contents
- Export (download) files
- Batch delete
- One-click purge (requires confirmation)

V3 audit reports have a dedicated tab for preview, download, batch deletion, and
confirmed purge. Removing these diagnostics does not affect models, checkpoints,
or semantic assets. If the corresponding audited task is still active, its next
periodic flush may recreate a deleted report.

Model storage is grouped by embedded `model_id`, then by embedded iteration. ONNX
uploads must include both the graph and every referenced `.onnx.data` file. Deleting
an iteration removes its PTH, ONNX, external data, and artifact manifest together.

The WebUI warns about the same prefix with different UUIDs, or one UUID with multiple prefixes.
Resume and overwrite authorization always use the embedded UUID rather than trusting filenames alone.

![Storage & Logs](图片/存储与日志仓库.png)

---

### Holographic Replay

**Overview**: Replay Format V2 presents both players' decisions, Core resolution events, and action-protocol semantics as one synchronized timeline.

#### Interface Components

1. **Board View**
   - True coordinate card layout
   - Dynamic display of hand, field, graveyard zones
   - Highlight both players' acting cards and every target
   - Directional arrows for chains, attacks, movement, equips, and targeting; direct attacks point to opposing LP
   - LP before/after/delta display and ghost origins for cards that have already moved away
   - Expand either player's complete initial Main and Extra Deck; duplicate cards are grouped with counts, while older recordings show a compatibility notice when this metadata is absent

2. **Decision List**
   - Show all available actions
   - Show probabilities for both P0 and P1 models and the actual RuleBot response
   - Highlight each player's final chosen action
   - Expose Select/Unselect/Finish/Cancel, selection bounds, result sets, material values, and Core prompt fields
   - Click any candidate row to preview its card images and highlight the actor, targets, and materials on the board without changing the recorded decision
   - Main Phase operations name their concrete card target; Extra Deck Special Summon entries further distinguish Link, Xyz, Synchro, and Fusion Summons

3. **Playback Controls**
   - Previous/Next step
   - Auto-play (adjustable speed)
   - Timeline slider
   - Selecting another replay stops playback and returns to frame 0

4. **Event Timeline**
   - Records moves, summons, sets, chain construction/resolution, attacks, draws, counters, and LP events
   - Board snapshots are deduplicated through a state table so abnormal long games do not copy the full board into every frame
   - Older JSON recordings with inline states and P0-only decisions remain readable

#### Vision Toggles

- 👁️ P1 Hand: Show/hide opponent hand
- 👁️ P0 Hand: Show/hide own hand
- 👁️ P1 Set: Show/hide opponent face-downs
- 👁️ P0 Set: Show/hide own face-downs
- 🔄 P1 Flip: Rotate opponent perspective 180°
- 📊 P1 Confidence: Show/hide the P1 model or RuleBot candidate-confidence table

Full deck lists are post-game audit metadata only. They are never exposed to the Arena model and do not change training observations.

![Holographic Replay](图片/全息回放.png)

---

### 📦 Model Deployment & Packaging

**Overview**: Package trained models and knowledge bases into `.gkg` deployment packages for easy sharing and import.

#### 📥 Export Package

Package models as .gkg format:

- Select an embedded `model_id` pool first, then choose `.pth`/`.onnx` primaries from that pool
- Referenced `.onnx.data` files are included automatically
- When both PTH and ONNX are selected, their embedded iteration sets must match
- Optionally include the complete runtime semantic bundle (`knowledge_base.json` + `code_embeddings.npy` + `code_embeddings_idx.json`, plus `hash_mapping_report.json` when present) and the staple pool (`meta_staples.json`)
- The three runtime semantic files cannot be imported or exported separately; the manifest cross-checks KB effect slots, vector rows, and index keys
- Always generate and validate `manifest.json`
- Custom package name support

#### 📤 Unpack & Selective Import

Import external .gkg packages into current system:

- Support direct unpack from local filesystem (no upload needed)
- Fine-grained selection of models and config files to import
- Staging area management (preview before import decision)
- Enforce path, file-type, member-count, expanded-size, and compression-ratio limits before import; revalidate UUID, prefix, and iteration before installation

The `.gkg` protocol version is maintained independently from the framework version as
`DEPLOY_PACKAGE_FORMAT_VERSION` in `model_artifacts.py`.

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
| `--target-iteration` | Absolute stopping iteration; mutually exclusive with additional iterations | - |
| `--additional-iterations` | Iterations to add from the current checkpoint | 1000 when omitted for new training |
| `--model-prefix` | New model filename prefix; inherited on resume | `galatea` |
| `--deck_dir` | Deck directory | `./decks` |
| `--d_model` | Feature dimension | 256 |
| `--n_heads` | Attention heads | 4 |
| `--n_layers` | Transformer layers | 2 |
| `--resume` | Checkpoint to resume from | - |
| `--batch_size` | Steps per collection | 4096 |
| `--mini_batch` | PPO update batch | 512 |
| `--workers` | Parallel processes | 4 |
| `--device` | Main training device (auto / cpu / cuda) | auto |
| `--timeout` | Worker collection timeout; must be greater than 30 seconds | 300 |
| `--use_onnx` | Export ONNX at checkpoints and accelerate historical opponents | - |
| `--no_compile` | Disable compilation | - |
| `--standard_core` | Disable ghost byte parsing (for custom cores) | - |
| `--protocol-audit` | Generate V3 protocol and effect-slot diagnostics | Off |
| `--gamma` | Discount factor | 0.998 |
| `--lr` | Learning rate | 1e-4 |
| `--entropy` | Entropy regularization coefficient | 0.03 |
| `--gae_lambda` | GAE smoothing coefficient | 0.95 |
| `--clip_eps` | PPO clipping threshold | 0.2 |

`--target-iteration` and `--additional-iterations` are mutually exclusive. Resume requires one
of them explicitly; new training defaults to 1,000 additional iterations when both are omitted.
`--model-prefix` accepts only letters, digits, underscores, and hyphens, and cannot override the
checkpoint prefix during resume.

**Examples**:

```bash
# Basic training
python main.py train --dir ./models --additional-iterations 1000 --model-prefix galatea --no_compile

# Advanced config
python main.py train \
  --dir ./models \
  --additional-iterations 1000 \
  --model-prefix galatea \
  --batch_size 16384 \
  --mini_batch 512 \
  --workers 6 \
  --device auto \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --no_compile

# Resume training
python main.py train \
  --resume ./models/galatea_iter_100.pth \
  --target-iteration 5000 \
  --device auto \
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

Arena loads the architecture embedded in each P0/P1 checkpoint and does not allow external
architecture overrides. Loop protection uses a complete state key covering the board, acting
player, Select/Unselect semantics, and target entities. Engine retries are hard bans, while repeated
choices are soft bans that cannot exhaust the candidate pool by themselves. If an abnormal loop
pushes every candidate past the threshold, Arena explores the least-visited legal choice in that
state instead of permanently disabling protection or reporting finite scores as numerical failures.

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
| `--sync` | Pull the complete remote semantic baseline and append missing vectors | - |
| `--remote_url` | Remote `knowledge_base.json`; sibling semantic URLs are derived automatically | Main repository Raw URL |
| `--embed` | Encode only new effect slots, rebuilding fully when necessary | - |

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
- Model files from one `model_id` pool (`.pth`/`.onnx`/`.onnx.data`)
- Knowledge base (`knowledge_base.json`)
- Hash continuation index (`hash_mapping_report.json`)
- Code-semantic matrix and index (`code_embeddings.npy`, `code_embeddings_idx.json`)
- Staple pool (`meta_staples.json`)
- Manifest file (`manifest.json`)

#### 2. Unpack & Import

Extract `.gkg` packages and import into current system:
- Model files extracted to `./models/`
- Knowledge base files overwritten to root directory
- Unsafe deserialization, out-of-bound filenames, zip bombs, and identity/iteration mismatches are rejected

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
| `mini_batch` | PPO update batch | Mainly VRAM in CUDA mode and RAM in CPU mode |
| `workers` | Parallel processes | ⬆️ **RAM** (primary) — adjust by CPU cores, typically 4-12 |
| `timeout` | Worker single-collection timeout | Prevents zombie processes, default 300s, large decks can use 600s |
| `device` | Main training device | `auto` prefers CUDA; `cpu` is CPU-only; workers always use CPU |
| `use_onnx` | Historical-opponent ONNX inference | Exports complete ONNX artifacts synchronously at checkpoints and falls back to historical PTH on failure |
| `no_compile` | Disable compilation | Recommended for Windows or legacy environments |
| `protocol_audit` | V3 observation audit | Enable manually for diagnosis; leave off for long training |

#### RL Soul Hyperparameters (Deep Tuning)

These parameters can be adjusted in WebUI or CLI for fine-grained PPO algorithm control:

| Parameter | CLI Flag | Description | Default | Tuning Advice |
|-----------|----------|-------------|---------|---------------|
| Discount Factor | `--gamma` | Future reward weighting | 0.998 | Higher = more emphasis on long-term, good for long games |
| Learning Rate | `--lr` | Neural plasticity speed | 1e-4 | Too high = unstable, too low = slow convergence |
| Exploration Coef | `--entropy` | Curiosity/exploration strength | 0.03 | Encourages trying new actions and stays at the configured value during training |
| GAE Lambda | `--gae_lambda` | Generalized Advantage Estimation λ | 0.95 | Balances bias-variance tradeoff, generally don't change |
| PPO Clip Epsilon | `--clip_eps` | Policy update clipping threshold | 0.2 | Limits single update magnitude, prevents overshooting |

---

## FAQ

### Q1: CUDA Out of Memory (VRAM insufficient)

**Solutions**:

```bash
# Option 1: Switch to CPU-only training
python main.py train --device cpu

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

On Windows, each iteration reports system commit headroom and the estimated safe requirement before
starting workers. If the preflight rejects startup, close memory-heavy applications or enlarge the
system page file in addition to reducing workers/batch size; increasing the ZMQ timeout cannot fix
commit exhaustion.
Normal training reuses the trainer's merged trajectory pool across iterations. Only a failed
headroom check releases the already-consumed old pool and retries once. Compare the staged memory
snapshots with each worker's opponent-backend line to distinguish trainer retention from a
hist/ONNX per-iteration peak.

### Q3: Windows torch.compile error

**Solution**: Add `--no_compile` flag, or check "Disable Model Compilation" in WebUI.

### Q4: Training speed is slow

**Possible causes and solutions**:

1. **No GPU**: Use `--device auto` and ensure PyTorch detects CUDA
2. **Too many workers**: Reduce `--workers`
3. **CPU thread contention**: Reduce workers to leave cores for central inference and PPO updates

### Q5: Model not converging

**Possible causes and solutions**:

1. **Learning rate too high**: Lower `--lr` or the corresponding WebUI value
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
