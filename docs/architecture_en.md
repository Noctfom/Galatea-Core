# 🔧 Architecture Design

> In-depth introduction to Galatea-Core's technical architecture and core algorithms. Suitable for users who want to understand internals or contribute to development.

> Applies to **Galatea-Core v3.15.3**, retaining Protocol4/Checkpoint3/revision14. Independent imitation provides collection, compressed caches, a BC optimizer and fixed evaluation, never automatic PPO or online-human learning. Partial player views still provide diagnostics, not complete native observations.

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

```text
WebUI / CLI
    ├─ Trainer → CPU Workers → public Core (DLL / SO)
    │                │ parsing / player-view snapshots / V4 Encoder
    │                └─ shared observation slots + ZMQ → CPU/CUDA central inference
    │                   current/self share one model; hist runs locally on CPU
    ├─ Arena → same parser / Encoder / response wrapper → model decisions
    └─ replay tools → raw Core/Link/YRP → diagnostic replay / quality gates

V4 observations + exact vocabulary + compiled semantic tables
    ├─ card backbone + Deck Encoder → global/deck layered FiLM → Transformer
    └─ chain / recent activations / 16 events / public hints → shared state
                         ↓
                  goal-planning latent
                         ↓
                  policy + value heads
                         └─ PPO: auxiliary/planning posterior targets + KL guard
```

### Core File Descriptions

| File | Layer | Function |
|------|-------|----------|
| `galatea_net.py` | Cognition | Transformer neural network definition |
| `feature_encoder.py` | Perception | Game state feature encoding |
| `semantic_kb.py` | Perception | Semantic knowledge base queries |
| `semantic_lookup.py` | Assets | Compile/load exact-token static semantics and validate their identity |
| `protocol_schema.py` | Protocol | Unified V4 fields, Model Protocol and schema revision |
| `galatea_env.py` | Environment | OCGCore environment wrapper |
| `gamestate.py` | Environment | Game state parsing (core!) |
| `deck_protocol.py` | Protocol | Full construction, single-duel deck image, and BO3 upper-layer boundary |
| `deck_trajectory.py` | Trajectory | Worker deck-profile catalogs, index remapping and lossless static-field reconstruction |
| `deck_encoder.py` | Cognition | Unordered Main/Extra encoding, deck style and upper-layer reuse interfaces |
| `goal_planner.py` | Cognition | Per-decision planning latent, policy/value gates and target decoder |
| `auxiliary_heads.py` / `auxiliary_targets.py` | Learning | Training-only prediction heads and actual-outcome posterior labels |
| `ppo_control.py` | Learning | GAE/PPO defaults, KL protection and actual-update auditing |
| `core_trajectory.py` | Trajectory | Single Core interpreter, canonical recording and replay verification |
| `trajectory_ingest.py` / `trajectory_dataset.py` | Data | External capture ingestion, quality gates and grouped isolation |
| `trainer.py` | Application | PPO trainer |
| `worker.py` | Application | Multi-process data collection |
| `model_versus.py` | Application | Normal Arena, logical/physical seat mapping, and greedy/training/deployment policies |
| `arena_benchmark.py` | Application | Fixed schedules, deck/model hashes, and benchmark statistics; never participates in training or network forward passes |

---

## Neural Network Architecture

### GalateaNet Structure

GalateaNet is a Transformer Encoder-based policy-value network, with FiLM global modulation and SwiGLU gated FFN introduced in v3.2.0:

```text
Inputs: 120 card slots / up to120 actions; 17 global values + phase/player categories
        stable Main/Extra profiles + dynamic remaining decks; no Side in duel observations
Semantics: exact tokens → static Lua slots/384D code vectors → trainable projection/Slot Attention
Backbone: numeric/categorical/public relations → Transformer (layered FiLM + SwiGLU)
Context: 12 chain entries, 8 recent activations, 16 transitions, public hints/known deck positions
Deck: 8 learned latents → deck_style → FiLM / shared state / planner
Outputs: action-matching policy + value; 128D plan_latent through independent bounded gates
Training-only: action-conditioned auxiliary heads + goal decoder (not standard ONNX outputs)
```

This is a data-flow overview, not executable constructor code. `protocol_schema.py`
maintains fields and versions; `galatea_net.py` implements the tensor paths.

### Layered FiLM: global context and deck style

FiLM applies `x * (1 + γ) + β` after LayerNorm. V4 uses two independent low-rank
generators: global context reads17 state values and a16D phase embedding, including first/second
seat, own/opponent turn, turn progress and LP; deck modulation reads the learned `deck_style`.
Each layer's attention and FFN receive separate γ/β rather than sharing one pair.

The branches combine as `0.5 * tanh(global_raw + tanh(deck_gate) * deck_raw)`, bounding total
amplitude to±0.5. Deck gates start at zero; an episode-wide10% training mask is preserved during
PPO updates. The direct deck-to-shared-state policy/value channel remains active. Modulation
**learns** conditional feature use rather than assigning a hand-authored attack/defense style
or replacing all network weights for each duel.

### SwiGLU Gated Feed-Forward Network (new in v3.2.0)

Transformer feed-forward sublayers replace the traditional MLP (Linear→ReLU→Linear) with SwiGLU gated linear units:

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
- **Bias-Free Design**: The three Linear layers in this SwiGLU module omit bias, reducing their parameter count; this does not mean every Linear layer in the network is bias-free
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

GalateaCore uses **entropy regularization** and **historical model leagues** to encourage exploration
and opponent diversity, reducing premature convergence and forgetting risks. These mechanisms
do not guarantee avoidance of local optima or improved playing strength.

## Feature Encoding System

### Encoding Dimensions

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| Continuous Global Features | 17 | Turn count, relative initiative/turn identity, per-player turns, LP, and zone counts |
| Categorical Global Features | 1 + 3 | Phase; relative decision/turn/starting-player roles |
| Card Features | 66 + 2 categories + compact public state | Numeric/effect/type/link features plus zone, position, relation, and status categories |
| Semantic Features | 128×8 | Up to 8 effect slots per card |

### Card Feature Details

```python
feat_numeric = [
    owner,              # Controller (1.0/-1.0)
    sequence / 10.0,    # Sequence
    current_atk / 4000, # Current ATK
    current_def / 4000, # Current DEF
    base_atk / 4000,    # Base ATK
    base_def / 4000,    # Base DEF
    pos_x, pos_y,       # Field coordinates
    level / 12.0,       # Level
    lscale / 13.0,      # Left Pendulum Scale
    rscale / 13.0,      # Right Pendulum Scale
    is_public,          # Whether face-up
    overlay_count / 5,  # Xyz material count
    counter_count / 10, # Counter count
    is_equipped,        # Whether equipped
    used_effect_mask[0:8], # Effect slots already activated this turn
    rank / 13.0,        # Rank
    link_rating / 8.0,  # Link Rating
]
# + separate card_zone / card_position categorical embeddings
# + 32-dim type mask (Monster/Spell/Trap/Effect/Fusion/Synchro/Xyz/Pendulum/Link...)
# + 9-dim link arrows
```

Starting with 3.8.2, standard public `query_card/query_field_count` data also supplies dynamic identity, reason bits, original owner, `DISABLED/PROC_COMPLETE/FORBIDDEN`, equip/effect-target/reason-card relations, up to 16 overlay identities, eight typed-counter slots, and the 32-bit disabled-zone mask. Reason, counter types, and zone masks are stored losslessly as little-endian bytes in trajectories and expanded only during the network forward pass, avoiding redundant bit matrices in PPO rollout storage. Player-relative visibility masking is applied before these fields enter the network, so facedown opponent cards do not leak identity or state.

Although CDB reuses one raw integer for Level, Rank, and Link Rating, Core queries expose independent `level/rank/link_rating` values. Xyz and Link monsters ordinarily have `level=0`; the model treats zero as “not applicable,” not missing, and never backfills Level from static Rank or Link Rating.

Core internally retains exact summon method, source, and player, but the current public query API does not export them. Galatea maintains no private Core fork and does not infer them from card category or movement order. Only public `STATUS_PROC_COMPLETE` currently represents “properly summoned”; the remaining fields are an upstream API item.

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
    'act_summon_method': [...], # Summon category; unknown special summons are explicit
    'act_response': [...],   # Semantic response value
    'act_signature': [...],  # Four-byte stable full-action signature
    'act_context': [...],    # min/max/result count/finish/cancel [120, 6]
    'act_target_code': [...],# Hidden-zone or macro target codes [120, 5]
    'act_target_value': [...],# Tribute/dual-level/counter values [120, 5, 2]
    'act_controller': [...], # Actor-relative controller
    'act_location': [...],   # Engine zone
    'act_sequence': [...],   # Sequence within the zone
    'act_position': [...],   # Target position category
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

Action Protocol V2 does not treat `GameAction.index` as learned semantics. `index` and `decision_bytes` only translate the final choice back to Core; the policy sees operation, card code, location, constraints, and resulting selection set. Type 26 retains Core's native sequential Select/Unselect flow, producing a new snapshot and trajectory row at each step. Static combinatorial messages such as Types 15/20/21/22/23/25 first enumerate complete legal responses and then let the policy choose one. Type 21 simultaneous-trigger sorting shares the sort response with Type 25; Type 22 exposes candidate cards, counter type, requested total, and per-card availability to Pass 1.

V4 schema revision 4 maps the six fixed `MSG_SELECT_IDLECMD` lists to normal summon, special summon, reposition, monster set, spell/trap set, and activate instead of labeling all six as activation. `act_operation` and `act_summon_method` are `[120] uint8` fields scoped to the current legal-action snapshot and consumed by the action signature, policy network, ONNX, and replay. Standard Core list ordinals prove only the broad normal/special family; they do not prove Ritual, Fusion, Synchro, Xyz, Pendulum, or Link. Special-summon candidates therefore use `SPECIAL_UNKNOWN` unless a future Core message or verified runtime context supplies direct evidence. Source zone and monster card category may be displayed separately but never become summon-method labels. The normal-summon list likewise does not prove a Tribute subtype; `TRIBUTE` remains reserved for direct evidence.

V4 schema revision 5 fixes each selection state machine against `playerop.cpp`: Type 15 enforces its count range; Type 20 enforces both maximum card count and summed `release_param`; Type 23 carries mandatory placeholders and dual values through Core's Exact/SumGreater boundary and never invents cancellation; Type 26 returns exactly one Select/Unselect index or `int32(-1)` for Finish/Cancel. Types 20/23 now expose Pass-1 material candidates so model preference participates in macro reduction. Type 26 pins its terminal action at the front so the legal exit survives the 120-action fixed shape. The `set_responseb` lifetime buffer now matches Core's fixed 512-byte read, eliminating the former 64-byte truncation and out-of-bounds read.

`MODEL_PROTOCOL_VERSION` is maintained independently from both the framework release and checkpoint-container version. It is embedded in PTH top-level metadata, `net_config`, model state, ONNX metadata, and artifact manifests. A mismatch means the input tensors or action-head weights are incompatible and is rejected.

### Exact Card Identity (Model Protocol V4)

Starting in 3.7.0, V4 no longer folds real card codes into shared embedding rows with
`code % 19990`. Root-level `card_vocab.json` assigns every real code one unique,
contiguous, append-only token. Token `0` is padding, `1` is unknown, `2/3` represent
hidden opponent Hand/Spell-Trap occupancy, and real cards begin at `10`. The current
`cards.cdb` registers 14,981 cards within the unchanged capacity of 20,000, so the
embedding parameter count does not increase.

Scene cards, top overlays, initial decks, chain cards, candidate/announcement cards,
macro-action targets, and Lua-semantic references all share this vocabulary. The structural
`protocol_schema_hash` remains stable across valid appends; PTH, `net_config`, ONNX,
artifact manifests, and `.gkg` separately store the exact training-time vocabulary hash and
count. A model loads only when its mapping is an exact prefix of the current lineage authority,
which keeps older V4 models usable while rejecting reordered or independently forked mappings.
Ordinary resource updates synchronize Galatea's authority by default and may target a custom
source. Custom cards can be appended through WebUI or `python main.py vocab`, but every machine
in one model lineage must receive the same file. A locally newer CDB only excludes affected decks.
Starting in 3.9.0, Lua static semantics have their own append-only prefix identity.
`semantic_lookup.py` compiles each card's eight Lua effect slots onto the same token row,
and PTH, ONNX, artifact manifests, and deployment packages record a logical semantic hash.
Appending a card and its effects preserves older prefixes; changing an existing card's
structured fields or code vector is rejected. Starting in 3.9.1, the Encoder emits only card
tokens and chain/history effect-slot IDs. The model reconstructs the same static semantics on
device, so observation content is unchanged while shared memory, Worker trajectories, and the
PPO merge pool no longer duplicate the matrices.
Version 3.9.2 materializes the result as `semantic_lookup_v1.npz` (model-side numeric
tables) plus `semantic_lookup_v1.json` (effect-slot runtime catalogs and integrity metadata).
Training/PTH loads the former directly, spawn Workers load only the latter, and ONNX export
freezes the same tables into the graph/`.onnx.data` so inference does not reread source assets.
These two rebuildable files stay out of Git but are included automatically in one-click and
`.gkg` packages.
`protocol_schema.py` is the single owner of V4's `MODEL_PROTOCOL_VERSION=4`, while
`checkpoint_utils.py` owns `CHECKPOINT_FORMAT_VERSION=3`. V3 checkpoints are
intentionally not migrated.

### Phase 4 deck-protocol boundary

Version 3.10.0 introduces independent `DeckSpec`, `DeckProfile`, `MatchContext`, and
`DuelSummary` contracts. `DeckSpec` retains the complete Main/Extra/Side construction and
derives content identity from section, exact card code, and copy count rather than filename
or `.ydk` ordering. `DeckProfile` is derived from Main and Extra only; its serialized form has
no Side field and is the only stable deck boundary available to the current BO1 policy.

The resource layer now parses and separately validates `.ydk` `!side` sections. A Side-only
unknown-card warning does not exclude an otherwise valid deck from current BO1 training or
Arena sampling, while Main/Extra failures still do. `DuelState` retains both an immutable
initial Main/Extra image and Core-driven remaining lists, preventing later encoders from
mistaking drawn cards for cards absent from the construction. `MatchContext` and `DuelSummary`
are external extension points for future BO3 siding and deck building; they currently emit no
policy inputs and never enter single-duel observations, the network, or PPO trajectories.

In the 3.10.1 batch, the stable image first entered training trajectories without changing model inputs at that stage.
Each Worker registers the training seat's per-duel profile by `profile_id`, stores only an
`int32 deck_profile_index` per sample, and continues storing dynamic `deck_idx/deck_mask` per
step. Because `deck_race/deck_attr/deck_setcodes` are static properties of an exact card token,
they are registered once per Worker, conflict-checked during Trainer merge, and reconstructed in
original token order immediately before each PPO mini-batch. The values match the old expanded
Encoder output element for element, so GalateaNet, central inference, PTH/ONNX, and the schema
revision remain unchanged.

The private catalog contains tensors only and remains loadable with `weights_only=True`; local
indices are remapped to Trainer-global indices during merge. In 3.10.1 the index only checked
identity and reserved an entry point; from 3.10.2 it reconstructs the Deck Encoder inputs below.
The numeric index itself is not a policy feature. Side remains excluded at the `DeckProfile` type boundary.

Version 3.10.2 adds a 128-entry model image. Each unique entry carries an exact card token,
Main/Extra section, initial and current remaining copy counts, plus the existing race, attribute,
and set-code labels. The same code in different sections is counted independently. PPO still
reconstructs stable fields from the per-duel catalog; only `deck_profile_remaining_count` is
stored per decision. The BO1 Encoder emits Main/Extra only. `DeckSection.SIDE` is reserved for a
future deck-building/BO3 caller outside `DuelState`.

The standalone `DeckEncoder` uses no positional embedding. Eight learned latents cross-attend to
the unordered entries and expose a permutation-invariant `deck_style`, fixed-shape latents, and
permutation-equivariant per-card outputs. Card inputs reuse exact tokens and static labels, plus a
masked mean of every valid Lua code vector for that card; no printed text, per-step script I/O, or
CDB lookup enters the network hot path. The current policy adds only `deck_style` to the shared
intent representation, affecting both policy and value, and skips unused per-card writeback;
explicit upper-layer calls still receive all three outputs. Layered FiLM and planning latents landed
in3.11/3.12; upper-layer deck-building consumers remain future work. Legacy dynamic `deck_idx/deck_mask` stays
as an exact fallback for cards inserted into the Deck during play but absent from the initial
profile.

The eight learned latents are not hand-authored labels such as “starter” or “end board.” They are
trainable query vectors that read every unordered deck entry and acquire distinct relational
summaries through policy and value gradients. Their meanings are not named by the framework and
they do not persist across duels. `deck_style` is an invariant aggregate, `deck_latents` retains a
fixed set of relations, and `per_card_features` remains aligned and equivariant to input cards.
The standalone API exposes all three, so a future deck-building layer need not reverse-engineer
the duel action head.

Version 3.11.0 replaces the one γ/β pair previously shared by all layers and both sublayers with
two low-rank branches. Global state/phase and `deck_style` independently produce γ/β for every
Transformer layer's attention and FFN. They combine as
`0.5 * tanh(global_raw + tanh(deck_gate) * deck_raw)`, bounding total modulation to ±0.5. The
per-layer, per-sublayer, per-channel deck gate starts at exactly zero, so deck modulation cannot
disturb the initial backbone. The deck generator itself is nonzero-initialized, allowing the main
loss to move the gate on its first gradient step instead of creating a double-zero dead path. The
direct `deck_style` policy/value path remains active, so the Deck Encoder learns before the gate
opens.

To reduce fixed-construction overfitting without violating PPO ratios, Workers disable deck FiLM
for 10% of complete training duels using a separate RNG and store the same `deck_film_mask` in
central inference and every PPO sample. Updates therefore replay the exact condition used by the
saved old log-prob, unlike ordinary train-mode Dropout. The mask never disables the direct
policy/value path; Arena, Link, and deployment default to enabled. The Deck Encoder and direct
`deck_style` path do not use ordinary Dropout: this path is the stable construction identity and
rare key-card relations should not disappear independently at each PPO evaluation. Online deck
diversity reduces exact-list memorization but does not replace held-out deck-family evaluation. If
the direct path later proves overfit, use replayable whole-duel structured ablation or consistency
between deck variants instead of resampling elementwise Dropout during PPO updates.

Version 3.11.1 adds independent `TRANSITION_EVENT_FORMAT_VERSION=1`. A record starts only after an
action is successfully submitted to Core and ends at the next decision, Retry, or terminal boundary;
the latest 16 completed transitions are retained. Each event carries actor, turn/phase, action
semantics, source/targets/effect slot/summon method, public Core message labels, and LP, eight-zone-per-
player, and chain-depth deltas. Cancel, decline, selection finish, Retry, chain negation, and disable
have deterministic result bits. Event existence, intent, source, and targets have separate player
visibility masks: fully private internal selections disappear from the opponent sequence while public
consequences remain.

Version 3.11.2 tensorizes these events into 19 fixed-shape compact fields after player-view masking.
Sources recover exact Lua effect-slot semantics, while targets use card identity and all-code semantic
summaries. Up to five targets are pooled within an event; fixed temporal positions, a depthwise local
convolution, and attention pooling retain order across events. Empty history is exactly zero. The final
vector enters shared policy/value `v_input` through a per-channel zero-initialized gate, preserving
initial behavior while allowing the first backward pass to open the path. The same contract now spans
central inference, Worker/PPO trajectories, and ONNX without changing rewards, legal actions, or GAE.

Version 3.12.0 adds the independent training-posterior contract
`AUXILIARY_TARGET_FORMAT_VERSION=1`. The network still consumes only the latest 16 visible events;
the training Worker explicitly asks the recorder to retain references to all completed events within
the current duel, aligns
training steps through the monotonically increasing sequence ID returned by action submission, and
builds four horizons at commit time: next decision, chain end, turn end, and genuine Core terminal.
Targets contain validity masks, accumulated LP and both players' eight-zone deltas, event counts, plus
immediate result bits, public message groups, boundary, summon method, chain depths, player-relative
outcome, terminal reason, and remaining event count. Aborted duels still roll back atomically. A
truncation or unfinished final action remains missing instead of receiving a fabricated zero label.
Full-event references and temporary prefix sums are released at duel end; only 187 compact bytes per
step travel with the rollout. Arena, Link, and ordinary inference do not enable complete-event
retention.

These posterior targets remain outside observations and are not consumed by central inference,
standard ONNX, PPO losses, or the model-protocol hash. The Trainer currently writes only horizon
coverage and Retry/cancel/state-change/chain-negation rates under `Auxiliary_Targets/*` in TensorBoard.
Consequently 3.12.0 changes no policy, value, reward, GAE, network parameter, or deployment behavior;
it exists to validate alignment, visibility, and label distribution before auxiliary gradients are
allowed into the model.

Version 3.12.1 advances the posterior contract to `AUXILIARY_TARGET_FORMAT_VERSION=2`, adds the
actor seat per step, and connects training-only `StructuredAuxiliaryHeads`. The heads consume the
shared state representation plus the option representation that was actually executed, then use an
independent 256-wide MLP to predict immediate result bits/public messages/boundary, immediate
resource and chain changes, chain-end/turn-end/terminal resource changes, terminal outcome, and
remaining-event length. Classification uses BCE or cross entropy and continuous targets use Smooth
L1. Every horizon is normalized through its validity mask, so missing futures contribute no loss.

The normalized auxiliary objective enters PPO with coefficient `0.1`, but predictions never feed
policy logits or value. For the first 100 mini-batch updates the shared representations are fully
detached and only the heads train as probes. During the next 900 updates, the gradient returned to
the shared backbone ramps linearly from zero to a 0.2 multiplier. This is gradient scaling, not a
guarantee that auxiliary gradient norms stay below 20% of PPO norms. Base/head clipping is separate
at 0.5/1.0, but PPO and auxiliary gradients still add on shared parameters and may agree or conflict. The schedule resumes from
checkpointed `train_step` rather than restarting.

The ordinary three-output forward does not execute the auxiliary MLP. Central inference, Arena,
Link, and historical ONNX opponents use that path. The standard ONNX wrapper still returns only
`action_logits` and `values`, so export prunes unused head weights; PTH training checkpoints retain
them for exact resume. Curves are written under `Auxiliary_Train/*`, while raw target-distribution
audits remain under `Auxiliary_Targets/*`. Schema revision 12 prevents a development checkpoint
without the head parameters from being resumed accidentally.

Version 3.12.2 advances the posterior contract to `AUXILIARY_TARGET_FORMAT_VERSION=3` and adds one
planning-consistency pair-role byte per step. Only adjacent decisions by the same actor in the same
turn and phase form non-overlapping pairs. PPO shuffles even mini-batches as two-sample units, keeping
real pairs intact without duplicating samples; incomplete random tails are omitted. Consistency constrains only the direction
of the two 128-dimensional planning vectors. Its weight is `0.1` inside the planning objective, which
itself enters PPO at `0.05`; it neither requires the same action nor introduces hand-authored tactics.

`GoalPlanner` projects the shared `state_repr`, stable `deck_style`, and global/phase `film_context`
separately and fuses them into `plan_latent` on every forward pass. A training-only target decoder
uses the existing missing-safe posteriors to predict chain-end, turn-end, and terminal resource
summaries, genuine terminal outcome, and remaining length. The first 100 updates still isolate the
old shared backbone through the gradient bridge, followed by the same 900-update ramp to 0.2, while
the planner itself remains trainable throughout. Planner and training-decoder parameters have
separate clipping from the original PPO base.

Policy and value use independent projections, zero-initialized per-channel `tanh` gates, and a ±0.5
residual bound. Zero gates preserve initial outputs bit for bit; PPO first trains the gates and then
gradually exposes decision making to the planning projections. There is no cross-decision discrete
option, action script, or tree search: any state change triggers replanning. Standard ONNX therefore
retains the planner and both bounded inference paths while still exposing only logits/value and
pruning the target decoder plus structured auxiliary heads. `Planning/*` and
`Planning_Histograms/*` diagnose losses, gate opening, pair coverage, and latent collapse only; no
latent dimension has a predefined “attack/defense/deck line” meaning.

### V4 Player and Categorical Global State

Core `MSG_NEW_TURN` independently maintains `turn_player`, the first `starting_player`, and
per-player turn counts; interaction messages alone update `decision_player`. The encoder maps
each role into unknown/self/opponent from the acting perspective, so a response during the
opponent's turn is not mislabeled as one's own turn. Phase uses an 11-class embedding that
conditions both the global token and FiLM. Card, chain, and action zones/positions use bounded
categorical embeddings instead of treating bit masks divided by constants as distances.

---

## Semantic Knowledge Base

Version 3.9.1 registers the `[card_token, effect_slot]` table as non-parameter model buffers.
Scene/deck semantics use `card_idx/deck_idx`; chain and recent-history rows additionally carry
effect-slot IDs 1–8, while 0 means an unknown binding and preserves whole-card fallback. This
path is elementwise equivalent to the 3.9.0 expanded tensors. Incomplete bundles, slot/index
mismatches, and existing-card semantic identity mismatches still stop before construction.
Starting in 3.9.2, a fully validated compiled runtime pair can construct the model without the
source trio. Included KB/vector/index files must form a complete matching group; modern code sources
also require the generation manifest. Source-less compiled deployment needs neither group.

### Construction Flow

```
Lua Script (c12345678.lua)
         ↓
    ┌─────────────┐
    │ Lua Parser  │  ← Lexical/block-stack and static-reference extraction
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
           ↓ one-time compilation
semantic_lookup_v1.npz + semantic_lookup_v1.json
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

Every effect also carries an explicit0–7 slot identity. Sync treats the KB, Hash continuation map, code matrix/index and modern `code_embeddings_meta.json` as a complete source group. Since3.13.4, source/dependency/encoder/chunking identity controls reuse: unchanged slots reuse, changed same-key content regenerates, and legacy vectors without provenance rebuild on the first local update. Key presence alone is not content validity; see [semantic assets](code_semantics_en.md).

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

The [TensorBoard Metric Reference](tensorboard_metrics_en.md) documents exact training/auxiliary/planning aggregation and source locations, separating training fit, actual updates and independent playing-strength evaluation.

Since 3.13.0, `ppo_control.py` centralizes GAE lambda=0.98, at most four epochs, and target KL=0.02.
CLI/WebUI support explicit 0.95/0.99 comparisons or target KL=0 to disable the finite-shift threshold.
Before each backward pass, sampled old/new action log-probabilities estimate
`mean(expm1(log_ratio)-log_ratio)`. Above 1.5×target, all remaining policy, value, auxiliary, and
planning updates stop for that rollout. Nonfinite shifts always stop. No extra network pass is
required; this is not a strict trust region or rollback. Full mini-batches and random-tail omission
remain unchanged to avoid dynamic-shape recompilation.

Per-iteration `PPO_Update/*` reports actual updates, completed/started epochs, effective sample
reuse, tail size, quantiles of mini-batch mean KL/clip fraction, and stop flags—not quantiles of
individual actions. `Configuration/PPO` records launch controls, requested learning rate, and
effective optimizer rates. Resume uses current launch controls rather than treating checkpoint
audit values as command-line overrides. `train_step` counts actual optimizer updates; FP16
GradScaler skips no longer advance the auxiliary ramp. Network/protocol contracts remain unchanged.

### Interpreting KL early stopping

KL measures drift from the rollout-collection policy, not win rate, percentage loss or knowledge learned.
Clipping cannot guarantee small distribution drift, so an additional guard is useful; see
[OpenAI's PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

Current mini-batch **sampled-action** log-probs estimate
`mean(expm1(new_log_prob-old_log_prob)-(new_log_prob-old_log_prob))`.
This nonnegative estimate is not exact full-rollout KL or another network call.
Default `target_kl=0.02` gives an actual `1.5×0.02=0.03` threshold.
Checks occur after forward, before backward; excessive or nonfinite drift exits all remaining epochs.

Completed optimizer steps remain: **no rollback, training termination or rejection of the whole rollout**.
Policy/value/auxiliary/planning share the network, so remaining joint updates stop together.
The next iteration collects with the updated policy. Reduced reuse trades optimization on stale data
for smaller drift/fresher sampling: not impact-free, but not disabled learning or proof of convergence.
The guard does not guarantee that regression is impossible.

For `KL=0.03670 >0.03000,165 completed updates`, protection fired normally.
With31,648 samples,512-sized batches and4 requested epochs, each pass contains61 full batches,
for244 maximum steps.165 means two passes plus43 batches of the third, or
`165×512/31648≈2.67` effective sample reuses. The416-sample tail keeps fixed-batch handling;
reshuffling means it is not necessarily the same games omitted each time.
The triggering batch is not backpropagated; its KL is not a post-update re-evaluation.

A stop on iteration3 or earlier is not inherently abnormal: the guard constrains each rollout's drift,
not elapsed training age. Open **Training Manifold → TensorBoard → Scalars** alongside fixed benchmarks:

| Metric | Interpretation |
| --- | --- |
| `PPO_Update/Effective_Epochs`, `Optimizer_Updates` | Actual sample reuse and optimizer steps |
| `PPO_Update/KL_Early_Stop`, `Approx_KL_P95`, `Approx_KL_Max` | Stop frequency and mini-batch mean drift, not playing strength |
| `PPO_Update/Clip_Fraction_P95` | Fraction of sampled ratios outside clipping bounds |
| `PPO_Update/Nonfinite_Batches` | Should remain0; distinguish numerical failures from finite-KL stopping |

Persistent first/few-batch stops, low effective epochs, nonfinite values or benchmark regression warrant
checking old log-probs, collection/update conditions and learning rates; one stop is not grounds
to relax protection. `Configuration/PPO` retains current settings. Reserved CUDA cache and reuse of
the CPU merge pool do not themselves prove leaks: compare identical stages across warmed-up iterations.
Low memory after worker cleanup cannot establish the peak during PPO backward.

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

Snapshots reconcile the field, Hand, Graveyard, and banished zones through the bundled Core's public legacy query API, while the message stream maintains turns, LP, chains, and disabled zones. Upstream Core versions using the newer API expose structured `OCG_DuelQuery`; a future upgrade must adapt the complete public ABI rather than append private fields to a local binary.

`galatea_env.py` currently binds the legacy `create_duel/process/get_message/set_response*/query_card` exports, so a modern binary exposing `OCG_DuelProcess/OCG_DuelGetMessage/OCG_DuelSetResponse/OCG_DuelQuery` cannot be dropped in directly. Modern Core also introduces 64-bit races, TLV query fields, and messages such as Types 121/122/123/180/190. Migration requires a public dual-backend adapter plus Windows/Linux regression; use the bundled public legacy Core until that work is complete.

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

#### Direct CUDA semantic aggregation (2026-10-10)

`GalateaNet.process_semantics()` still sums categories, setcodes, card references, races and
attributes separately for each effect slot. CUDA uses `embedding_bag(mode='sum')` with the
same Embedding weights instead of expanding every reference vector first. This is not a mean,
deduplication, truncation or static cache: repeated references retain their multiplicity; all
eight slots, code vectors, queries and masks remain unchanged; trainable weights are not frozen.
Nonzero padding contributions are restored with a detached padding row while its gradient
remains zero. CPU Workers/training, ONNX export and nondefault Embedding options retain the
original operator.

AMP retains the original FP32 embedding reduction. For shape512/75/8/4/512, measured forward
peak increment drops from2.9297 to0.6123 GiB, removing one2.34375 GiB reference expansion.
This is not the whole-network peak reduction. Mathematical objectives and gradient paths are
unchanged; floating-point reduction order can differ across operators/devices, so bitwise
reproduction of every training trajectory is not promised. Model Protocol4, Checkpoint3,
schema revision14, parameter names/shapes, semantic identity and PPO/GAE/auxiliary/planning
objectives remain unchanged. Compatible V4 checkpoints can resume without rebuilding assets.

Local RTX5070Ti16GiB/Torch2.9.1+cu130 tests use512/8/6, BF16, no compilation and synthetic
observations/labels containing actual semantic cards. They exercise the existing complete
PPO forward/backward/Adam path three times per case. After optimizer-state initialization:

| mini-batch | Original peak allocated | Direct-sum peak allocated | Reserved after update: original → new |
|---|---:|---:|---:|
| 128 | 7.2790 GiB | 7.3387 GiB | 8.3594 → 7.4570 GiB |
| 256 | 13.8014 GiB | 13.9213 GiB | 15.1895 → 14.2051 GiB |

Direct aggregation retains grouping/index metadata for backward. Whole-network allocated
peaks are about0.9% higher in this test, not materially lower; the benefit is avoiding large
temporary semantic allocations and reducing reserved-cache pressure. Allocated memory after
gradient release stays near0.67GiB without accumulation. Reserved memory is not live tensors
or proof of a leak. These results cannot guarantee that cloud32GiB/mini-batch512 fits the entire
PPO update. This patch does not automatically change batch sizes or add gradient accumulation
or semantic activation checkpointing.

Regression covers repeated references, nonzero padding, original reference clamping, empty
slots, AMP, finite gradients, actual CUDA PPO, ONNX/Core and input/parameter/protocol identity.
A512/8/6 BF16 fixture matches all outputs and307 gradient-bearing parameters exactly; this is
not a long-run learning-strength guarantee. `tests/benchmark_semantic_aggregation.py` reproduces
the complete PPO probe under the existing ignored-tests convention. Compilation checks cover
Dynamo eager full-graph capture only, not Linux Inductor. The user reports temporary short-run
stability on the same32GiB cloud configuration; multi-scene peaks,50-iteration trends and
production long-run/playing-strength acceptance remain required.

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
