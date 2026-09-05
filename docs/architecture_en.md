# 🔧 Architecture Design

> In-depth introduction to Galatea-Core's technical architecture and core algorithms. Suitable for users who want to understand internals or contribute to development.

> This document applies to **Galatea-Core v3.6.2**.

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

GalateaNet is a Transformer Encoder-based policy-value network, with FiLM global modulation and SwiGLU gated FFN introduced in v3.2.0:

```python
class GalateaNet(nn.Module):
    def __init__(self, config):
        # 1. Base physical perception layer
        self.card_embed = nn.Embedding(vocab_size, d_model)      # Card ID embedding
        self.feat_proj = nn.Linear(66, d_model)                   # Numeric feature projection
        self.race_embed = nn.Embedding(30, d_model)               # Race embedding
        self.attr_embed = nn.Embedding(10, d_model)               # Attribute embedding
        self.setcode_embed = nn.Embedding(4096, d_model)          # Archetype embedding
        
        # 2. Semantic parsing cortex
        self.sem_cat_embed = nn.Embedding(4000, d_sem)            # Effect type embedding
        self.sem_req_proj = nn.Linear(128, d_sem)                 # Condition projection
        self.effect_slot_embed = nn.Embedding(8, d_model)         # Effect-slot identity
        self.sem_fusion_proj = nn.Sequential(...)                 # Semantic fusion

        # 3. FiLM global modulator
        self.film_gen = FiLMGenerator(condition_dim=15, d_model=d_model)

        # 4. Transformer Encoder (each layer with FiLM modulation + SwiGLU gating)
        self.transformer = GalateaTransformerStack(d_model, n_heads, n_layers)

        # 5. Ordered context encoding
        self.chain_context_pool = OrderedContextPool(d_model, 12) # Chain order
        self.history_context_pool = OrderedContextPool(d_model, 8)# Recent activation order
        self.place_weights = buffer([1.0, 0.8, 0.6, 0.4, 0.2])   # Sort operation weighting
        
        # 6. Output heads (SwiGLU gated)
        self.policy_head = SwiGLU(d_model*2) → Linear(1)         # Policy head
        self.value_head = SwiGLU(d_model) → Linear(1)            # Value head
```

### FiLM Global State Modulation (new in v3.2.0)

FiLM (Feature-wise Linear Modulation) dynamically adjusts the inference tendency of each Transformer layer based on global signals like current phase/turn/LP:

```python
class FiLMGenerator(nn.Module):
    def __init__(self, condition_dim, d_model):
        self.proj = nn.Linear(condition_dim, 2 * d_model)  # Outputs γ and β
        nn.init.zeros_(self.proj.weight)  # Zero-init ensures no interference early in training

    def forward(self, condition):
        out = self.proj(condition)
        gamma, beta = out.chunk(2, dim=-1)  # Split in half
        return gamma.unsqueeze(1), beta.unsqueeze(1)

# Applied in Transformer Block:
x = x * (1.0 + gamma) + beta  # LayerNorm first, then FiLM modulation
```

**Design Philosophy**: Decision logic differs drastically across game phases (opening combos vs. mid-game skirmishes vs. lethal calculations). FiLM allows a single network to automatically switch "thinking modes" based on global state, without adding separate branch networks.

### SwiGLU Gated Feed-Forward Network (new in v3.2.0)

All traditional MLPs (Linear→ReLU→Linear) have been replaced with SwiGLU gated linear units:

```python
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, multiple_of=64):
        # Auto-pad to multiples of 64, aligned to Tensor Core hardware
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        # SiLU(Gate) × Up → Down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**Key Advantages**:
- **Gating Mechanism**: SiLU-activated Gate branch performs element-wise filtering on the Up branch, letting the network decide which features pass through
- **Bias-Free Design**: All Linear layers remove bias, reducing parameter count and improving training stability
- **Tensor Core 64 Alignment**: `hidden_features` auto-pads to multiples of 64 for full GPU hardware utilization

### Dual-Tower Matching Mechanism

The policy head uses a Dual-Tower architecture for action evaluation:

```
Intent Tower                        Option Tower
      ↓                                   ↓
┌─────────────┐                   ┌─────────────┐
│ Global state│                   │ Target card │
│ feat vector │                   │ Action type │
│ (v_input)   │                   │Response/limit│
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
| Card Features | 66 | Numeric attributes + used-effect bits + type masks + link arrows |
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
    used_effect_mask[0:8], # Effect slots already activated this turn
]
# + 32-dim type mask (Monster/Spell/Trap/Effect/Fusion/Synchro/Xyz/Pendulum/Link...)
# + 9-dim link arrows
```

### Action Encoding

```python
act_dict = {
    'act_card_idx': [...],   # Visible target entity indices [120, 5]
    'act_type': [...],       # Action types
    'act_desc': [...],       # Effect description Hash
    'act_mask': [...],       # Valid action mask
    'act_race': [...],       # Announced race
    'act_attr': [...],       # Announced attribute
    'act_code': [...],       # Candidate/announced card code
    'act_place': [...],      # Placement positions [120, 5]
    'act_operation': [...],  # Yes/No/Select/Unselect/Finish/Cancel semantics
    'act_response': [...],   # Semantic response value
    'act_signature': [...],  # Four-byte stable full-action signature
    'act_context': [...],    # min/max/result count/finish/cancel [120, 6]
    'act_target_code': [...],# Hidden-zone or macro target codes [120, 5]
    'act_target_value': [...],# Tribute/dual-level/counter values [120, 5, 2]
    'act_controller': [...], # Actor-relative controller
    'act_location': [...],   # Engine zone
    'act_sequence': [...],   # Sequence within the zone
}
```

### Ordered Context Aggregation (Model Protocol V3)

Chains and recent activation history cannot be represented by adding positions and then taking a mean: `Σ(semanticᵢ + positionᵢ) = Σsemanticᵢ + Σpositionᵢ`, so swapping two events leaves the result unchanged. `OrderedContextPool` uses this path instead:

```text
semantic token + fixed-slot vector
              ↓
depthwise 1D local convolution (distinct previous/current/next weights)
              ↓
channel mixing + residual normalization
              ↓
valid-item-masked attention pooling
              ↓
one order-sensitive context vector
```

Chain slots 1–12 retain Core insertion order. Each link also encodes card ID, effect description, chain index, handler location, triggering location, and relative controllers. History slot 0 remains the most recent activation. Bidirectional local mixing only sees already-observed events and does not expose future information. Empty contexts return zero and padding cannot enter either convolution input or final attention weights.

Each card's eight effects also carry explicit slot embeddings. Bit N in `used_effect_mask` can therefore learn a direct relationship with semantic effect N instead of collapsing the Slot Attention input into an unordered effect set.

Action Protocol V2 does not treat `GameAction.index` as learned semantics. `index` and `decision_bytes` only translate the final choice back to Core; the policy sees operation, card code, location, constraints, and resulting selection set. Type 26 retains Core's native sequential Select/Unselect flow, producing a new snapshot and trajectory row at each step. Static combinatorial messages such as Types 15/20/22/23/25 first enumerate complete legal responses and then let the policy choose one.

`MODEL_PROTOCOL_VERSION` is maintained independently from both the framework release and checkpoint-container version. It is embedded in PTH top-level metadata, `net_config`, model state, ONNX metadata, and artifact manifests. A mismatch means the input tensors or action-head weights are incompatible and is rejected.

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
    knowledge_base.json + hash_mapping_report.json
    code_embeddings.npy + code_embeddings_idx.json
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

In addition to the table above, every effect has an explicit slot identity from zero through seven. GitHub sync treats the four files in the same remote directory as one semantic bundle: the structured KB provides interpretable fields, the Hash map resumes clustering, and the code-semantic matrix plus index resume existing Lua vectors. When the assets are coherent, vector generation appends only new effect slots instead of re-encoding all existing scripts.

In 3.6.1, encoder initialization began cross-checking every modeled KB slot against code-vector rows and index keys. In 3.6.2, semantic generation additionally tracks each `Effect.CreateEffect(c)` object to its `SetDescription(aux.Stringid(...))` call and binds the complete runtime `desc` to that same Lua code-semantic slot. GameState, action candidates, chain/history context, and `used_effect_mask` share this mapping; dynamic forms that cannot be proven statically fall back to whole-card semantics. Neither printed card text nor the numeric Stringid index defines slot identity.

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

Opponent types are first assigned by worker-level weights. If no worker draws RuleBot for an
iteration, one AI worker is selected in rotation and its first game is temporarily forced to
RuleBot; it then resumes its original self/hist configuration. This preserves weighted randomness
while preventing small worker pools from missing the rules baseline for several iterations. When no
historical checkpoint exists, the hist share naturally falls back to self-play.

### Central Batched Inference Service

```
Worker 1 ──┐
Worker 2 ──┼──> ZMQ Request ──> CPU/CUDA Central Inference ──> Shared Result ─┬──> Worker 1
Worker 3 ──┘                                                                ├──> Worker 2
                                                                             └──> Worker 3
```

**Advantages**:

- Workers always stay on CPU; the current policy and self opponents do not create duplicate local networks or CUDA contexts
- Multiple workers always share one central inference service
- `device=auto/cpu/cuda` only selects the central inference and PPO update device
- `auto` prefers available CUDA and otherwise falls back to CPU; explicit `cuda` fails early when unavailable
- ZMQ carries only worker/request identifiers; observations and results use shared-memory slots protected by 64-bit completion IDs
- Windows validates system commit headroom before worker startup and stops early instead of allowing a native communication-library crash

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

### 1. CUDA Mixed Precision Training

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

CPU mode stays in FP32 and does not enable CUDA autocast, GradScaler, or pinned memory.

### 2. Static Memory Pool

The first trajectory merge allocates a fixed-capacity CPU pool. Later iterations overwrite the
valid prefix from index zero, avoiding repeated multi-GiB allocation/release cycles that can make
Windows commit usage grow in steps. PPO only reads the current `total_steps` range, so stale tail
data is never trained. Under memory pressure, preflight may release the already-consumed old pool
and retry once; `close()` performs the final release:

```python
class PPOTrainer:
    def merge_rollouts(self, first_block):
        # Allocate lazily once, then reuse and overwrite the valid region
        self.merged_memory = {
            'action': torch.empty(self.max_buffer_steps, ...),
            'log_prob': torch.empty(self.max_buffer_steps, ...),
            # ...
        }
```

The trainer records commit, RSS, merged-pool, and CUDA snapshots before worker startup, after worker
reaping, after trajectory merge, and after PPO. Each worker also records one process-memory sample
after opponent initialization, separating hist/ONNX peak cost from trainer-lifetime retention.

### 3. Shared-Memory Slots and Central Weights

The current policy and self opponents do not load PyTorch weights inside workers. They request the
central model over ZMQ, while observations, action results, full logits, and completion IDs use
fixed `share_memory_()` slots:

```python
# Worker: send only request identity; observations are already in its shared slot
socket.send(encode_inference_request(worker_id, request_id))

# High-frequency observations and results use preallocated shared-memory slots
shared_tensor.share_memory_()
```

Self opponents use the central model. Hist opponents select only artifacts from the same `model_id`,
preferring a complete same-iteration ONNX artifact and lazily loading the `.pth` fallback only when
ONNX is absent or fails.

### 4. TF32 Acceleration

Enable TF32 on Ampere+ architecture GPUs:

```python
if torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 5. ZMQ Request Routing and Shared-Memory IPC

ZeroMQ ROUTER carries request routing while large tensors remain in shared memory, reducing serialization and copy overhead:

```python
context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # ROUTER mode enables routed micro-batching
socket.setsockopt(zmq.ROUTER_HANDOVER, 1)  # Auto load balancing
```

**Key Advantages**:
- **Micro-Batching**: Multiple worker requests are aggregated into one batch on the selected training device
- **Device-Aware Staging**: CUDA uses pinned memory and `non_blocking=True`; CPU uses ordinary memory and synchronous copies
- **Timeout Isolation**: Requests include the iteration and a local sequence; a timeout or mismatched response rebuilds the worker socket without reading stale results

### 6. ONNX Inference Acceleration (new in v3.2.0)

With `--use_onnx`, ONNX is exported synchronously at each 10-iteration checkpoint. Workers use
ONNX Runtime only for historical opponents; the current policy and self opponents continue through
central inference:

```python
# Main process exports ONNX synchronously
class ONNXWrapper(torch.nn.Module):
    def __init__(self, net, keys):
        super().__init__()
        self.net = net
        self.keys = keys

    def forward(self, *args):
        # Reassemble flattened inputs into dictionary, pass to original network
        batch_dict = {k: v for k, v in zip(self.keys, args)}
        logits, values, _ = self.net(batch_dict)
        return logits, values

# Worker validates graph, external data, UUID, and iteration before loading historical ONNX
```

**Key Advantages**:
- **Complete Artifact Bundle**: `.onnx`, referenced `.onnx.data`, and `.artifacts.json` are saved and authenticated together
- **Input-Type Adaptation**: FP16/FP32 and other inputs are converted from ONNX Runtime session declarations
- **Lazy Safe Fallback**: A historical PTH network is loaded for CPU inference only when ONNX is incomplete, mismatched, or fails at runtime; the normal ONNX path does not keep both history engines resident

---

## Next Steps

- 📝 View [Changelog](changelog_en.md) for version history
- 📚 Read [Feature Guide](features_en.md) for usage
- 🚀 Return to [Quick Start](quickstart_en.md) to begin training
