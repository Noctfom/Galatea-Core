# 🔧 Architecture Design

> In-depth introduction to Galatea-Core's technical architecture and core algorithms. Suitable for users who want to understand internals or contribute to development.

> 💡 **Framework's unique handling logic** (Semantic Module, 142 Announce Pool, Multi-Select Chunk Wrapper, Hand Tracker, Deck Weights, Disguise Pools) — see [Special Handling Logic Document](special_handling_en.md).

---

## 📋 Table of Contents

- [System Architecture Overview](#system-architecture-overview)
- [Neural Network Architecture](#neural-network-architecture)
- [Feature Encoding System](#feature-encoding-system)
- [Semantic Knowledge Base](#semantic-knowledge-base)
- [PPO Training Framework](#ppo-training-framework)
- [OCGCore Environment Wrapper](#ocgcore-environment-wrapper)
- [Performance Optimization Techniques](#performance-optimization-techniques)

---

## System Architecture Overview

Galatea-Core adopts a modular design consisting of the following core subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   WebUI     │    │  Trainer    │    │   Arena     │         │
│  │  (app.py)   │    │(trainer.py) │    │(model_vs.py)│         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
├─────────┴──────────────────┴──────────────────┴─────────────────┤
│                      Cognition Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GalateaNet                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │ Embedding │  │Transformer│  │  Policy   │            │   │
│  │  │   Layer   │→ │  Encoder  │→ │   Head    │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘            │   │
│  │                                 ┌───────────┐            │   │
│  │                                 │  Value    │            │   │
│  │                                 │   Head    │            │   │
│  │                                 └───────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Perception Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Feature    │    │  Semantic   │    │    Card     │         │
│  │  Encoder    │    │     KB      │    │   Reader    │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
├─────────┴──────────────────┴──────────────────┴─────────────────┤
│                      Environment Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GalateaEnv                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │  Message  │  │   Duel    │  │   Action  │            │   │
│  │  │  Parser   │  │   State   │  │  Handler  │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    OCGCore (DLL)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Core File Descriptions

| File | Layer | Function |
|------|-------|----------|
| `galatea_net.py` | Cognition | Transformer neural network definition |
| `feature_encoder.py` | Perception | Game state feature encoding |
| `semantic_kb.py` | Perception | Semantic knowledge base queries |
| `galatea_env.py` | Environment | OCGCore environment wrapper |
| `gamestate.py` | Environment | Game state parsing (core!) |
| `trainer.py` | Application | PPO trainer |
| `worker.py` | Application | Multi-process data collection |

---

## Neural Network Architecture

### GalateaNet Structure

GalateaNet is a Transformer Encoder-based policy-value network:

```python
class GalateaNet(nn.Module):
    def __init__(self, config):
        # 1. Base physical perception layer
        self.card_embed = nn.Embedding(vocab_size, d_model)      # Card ID embedding
        self.feat_proj = nn.Linear(58, d_model)                   # Numeric feature projection
        self.race_embed = nn.Embedding(30, d_model)               # Race embedding
        self.attr_embed = nn.Embedding(10, d_model)               # Attribute embedding
        self.setcode_embed = nn.Embedding(4096, d_model)          # Archetype embedding
        
        # 2. Semantic parsing cortex
        self.sem_cat_embed = nn.Embedding(4000, d_sem)            # Effect type embedding
        self.sem_req_proj = nn.Linear(128, d_sem)                 # Condition projection
        self.sem_fusion_proj = nn.Sequential(...)                 # Semantic fusion
        
        # 3. Transformer Encoder
        self.transformer = nn.TransformerEncoder(...)
        
        # 4. Output heads
        self.policy_head = nn.Sequential(...)                     # Policy head
        self.value_head = nn.Sequential(...)                      # Value head
```

### Dual-Tower Matching Mechanism

The policy head uses a Dual-Tower architecture for action evaluation:

```
Intent Tower                        Option Tower
      ↓                                   ↓
┌─────────────┐                   ┌─────────────┐
│ Global state│                   │ Target card │
│ feat vector │                   │ Action type │
│ (v_input)   │                   │ Effect desc │
└──────┬──────┘                   └──────┬──────┘
       │                                 │
       └────────────┬────────────────────┘
                    ↓
              ┌───────────┐
              │   Concat  │
              │   Fusion  │
              └─────┬─────┘
                    ↓
              ┌───────────┐
              │ Policy MLP│
              └─────┬─────┘
                    ↓
          Action probability distribution
```

### Curiosity-Driven Exploration

GalateaCore primarily drives exploration through **entropy regularization** and **historical model league matchups**, ensuring AI neither converges prematurely to local optima nor forgets previously learned strategies.

## Feature Encoding System

### Encoding Dimensions

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| Global Features | 15 | Turn count, phase, LP, zone card counts |
| Card Features | 58 | Numeric attributes + type masks + link arrows |
| Semantic Features | 128×8 | Up to 8 effect slots per card |

### Card Feature Details

```python
feat_numeric = [
    owner,              # Controller (1.0/-1.0)
    location / 100.0,   # Location
    sequence / 10.0,    # Sequence
    current_atk / 4000, # Current ATK
    current_def / 4000, # Current DEF
    base_atk / 4000,    # Base ATK
    base_def / 4000,    # Base DEF
    pos_x, pos_y,       # Field coordinates
    level / 12.0,       # Level/Rank
    lscale / 13.0,      # Left Pendulum Scale
    rscale / 13.0,      # Right Pendulum Scale
    position / 10.0,    # Battle position
    is_public,          # Whether face-up
    overlay_count / 5,  # Xyz material count
    counter_count / 10, # Counter count
    is_equipped,        # Whether equipped
]
# + 32-dim type mask (Monster/Spell/Trap/Effect/Fusion/Synchro/Xyz/Pendulum/Link...)
# + 9-dim link arrows
```

### Action Encoding

```python
act_dict = {
    'act_card_idx': [...],   # Target card indices [80, 5]
    'act_type': [...],       # Action types
    'act_desc': [...],       # Effect description Hash
    'act_mask': [...],       # Valid action mask
    'act_race': [...],       # Announced race
    'act_attr': [...],       # Announced attribute
    'act_code': [...],       # Announced card
    'act_place': [...],      # Placement position
}
```

---

## Semantic Knowledge Base

### Construction Flow

```
Lua Script (c12345678.lua)
         ↓
    ┌─────────────┐
    │ Lua Parser  │  ← Regex extraction
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Category    │  ← CATEGORY_XXX
    │ Conditions  │  ← RACE/ATTR/SETCODE
    │ Special Hash│  ← Code block clustering
    └──────┬──────┘
           ↓
    knowledge_base.json
```

### Semantic Feature Structure

Each card has up to 8 effect slots, each containing:

| Field | Dim | Description |
|-------|-----|-------------|
| category | 8 | Effect type ID |
| requirements | 128 | Activation condition multi-hot vector |
| setcode | 4 | Associated archetype |
| numbers | 4 | Magic number parameters |
| ref_codes | 4 | Associated card IDs |
| race | 4 | Associated races |
| attr | 4 | Associated attributes |

### Hash Clustering Algorithm

For effects that can't be classified with standard CATEGORY, code block hashing is used:

```python
def _hash_code_block(self, code_block):
    # 1. Lexical normalization
    clean_code = code_block
    clean_code = re.sub(r'\b1\s*-\s*tp\b', '<OPPO>', clean_code)
    clean_code = re.sub(r'\b(tp|ep|rp)\b', '<PLAYER>', clean_code)
    clean_code = re.sub(r'\b\d+\b', '<NUM>', clean_code)
    # ...
    
    # 2. Compute MD5 Hash
    hash_val = hashlib.md5(clean_code.encode()).hexdigest()[:8]
    
    return f"CUSTOM_HASH_{hash_val.upper()}"
```

---

## PPO Training Framework

### Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                      PPO Training Loop                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Multi-Process Rollout Collection                        │
│     ┌─────────┐  ┌─────────┐  ┌─────────┐                  │
│     │Worker 1 │  │Worker 2 │  │Worker N │                  │
│     │Self 60% │  │Hist 25%│  │Rule 15% │                  │
│     └────┬────┘  └────┬────┘  └────┬────┘                  │
│          │            │            │                        │
│          └────────────┼────────────┘                        │
│                       ↓                                     │
│  2. Memory Aggregation                                     │
│     ┌─────────────────────────────────────┐                │
│     │ obs, actions, rewards, log_probs    │                │
│     │ + GAE advantage estimation          │                │
│     └──────────────────┬──────────────────┘                │
│                        ↓                                    │
│  3. Policy Update                                           │
│     for epoch in range(4):                                  │
│         for mini_batch in shuffle(memory):                  │
│             ┌─────────────────────────────┐                │
│             │ PPO Clip Loss               │                │
│             │ Value Loss                  │                │
│             │ Entropy Regularization      │                │
│             └─────────────────────────────┘                │
│                        ↓                                    │
│  4. Model Save (every 10 iters)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### League Training Mechanism

To prevent strategy degradation, mixed opponent training is used:

| Opponent Type | Ratio | Purpose |
|---------------|-------|---------|
| Self-play | 60% | Pursue current optimal strategy |
| Historical Models | 25% | Prevent forgetting old strategies |
| RuleBot | 15% | Baseline anchor, ensure basic ability |

### Async Inference Server

```
Worker 1 ──┐
Worker 2 ──┼──> Request Queue ──> GPU Inference Server ──> Response Queue ─┬──> Worker 1
Worker 3 ──┘                                                                ├──> Worker 2
                                                                             └──> Worker 3
```

**Advantages**:
- Workers don't need to load model into VRAM
- Multiple workers share a single GPU inference service
- VRAM usage reduced by 70%+

---

## OCGCore Environment Wrapper

### Message Parsing

`gamestate.py` is the most critical file in the project, responsible for parsing binary messages sent by OCGCore:

```python
class MessageParser:
    def parse_message(self, msg_type, data):
        if msg_type == MSG_SELECT_IDLECMD:
            return self._parse_idle_cmd(data)
        elif msg_type == MSG_SELECT_CHAIN:
            return self._parse_chain(data)
        elif msg_type == MSG_SELECT_CARD:
            return self._parse_select_card(data)
        # ... 100+ message types
```

### State Synchronization

```python
class DuelState:
    def __init__(self):
        self.turn_count = 0
        self.phase = 0
        self.current_player = 0
        self.lp = [8000, 8000]
        self.entities = []           # All field card entities
        self.chain_stack = []        # Chain stack
        self.history_stack = []      # Action history
        self.known_hand_codes = [[], []]  # Hand tracker
```

---

## Performance Optimization Techniques

### 1. Mixed Precision Training

```python
# Auto-detect hardware capability
if torch.cuda.is_bf16_supported():
    self.amp_dtype = torch.bfloat16  # Ampere+ architecture
else:
    self.amp_dtype = torch.float16   # Older GPUs

# Use AMP context
with torch.amp.autocast('cuda', dtype=self.amp_dtype):
    logits, values, v_input = self.agent.net(batch)
```

### 2. Static Memory Pool

Pre-allocate fixed-size memory to avoid fragmentation during training:

```python
class PPOTrainer:
    def __init__(self):
        # Pre-allocated memory pool
        self.max_buffer_steps = self.update_timesteps + (self.num_workers * 1000)
        self.merged_memory = {
            'action': torch.empty(self.max_buffer_steps, ...),
            'log_prob': torch.empty(self.max_buffer_steps, ...),
            # ...
        }
```

### 3. Weight Sharing

In Windows Spawn mode, use `share_memory_()` so all child processes share the same model weights:

```python
# Main process
weights = model.state_dict()
for v in weights.values():
    v.share_memory_()

# Child processes directly use shared memory weights
```

### 4. TF32 Acceleration

Enable TF32 on Ampere+ architecture GPUs:

```python
if torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

---

## Next Steps

- 📝 View [Changelog](changelog_en.md) for version history
- 📚 Read [Feature Guide](features_en.md) for usage
- 🚀 Return to [Quick Start](quickstart_en.md) to begin training
