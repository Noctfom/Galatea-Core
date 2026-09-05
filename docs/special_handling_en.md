# 🧬 Framework Special Handling Logic

> Detailed explanation of modules specially built to overcome inherent framework limitations — these are the core competitive advantages of Galatea-Core.

> This document applies to **Galatea-Core v3.6.2**.

---

## 📋 Table of Contents

- [Semantic Module (Semantic KB)](#semantic-module-semantic-kb)
- [142 Announce Pool Wrapper Logic](#142-announce-pool-wrapper-logic)
- [Multi-Select Chunk Wrapper Logic](#multi-select-chunk-wrapper-logic)
- [Simple Hand Tracker Logic](#simple-hand-tracker-logic)
- [Deck Weight Adjustment (Global Weights)](#deck-weight-adjustment-global-weights)
- [Virtual Disguise Pool Module (Virtual Mix Pools)](#virtual-disguise-pool-module-virtual-mix-pools)

---

## Semantic Module (Semantic KB)

### Why It's Needed

Yu-Gi-Oh! has over 10,000 cards, each with unique effects. If AI only identifies cards by their ID, it cannot understand that "two different cards have similar effects."

For example, "Ash Blossom & Joyous Spring" and "Effect Veiler" both negate monster effects, but their card codes are completely different. Without semantic understanding, AI must learn to deal with each card independently, unable to transfer knowledge.

### How It Works

The semantic module parses each card's Lua script to extract standardized semantic feature vectors:

```
Lua Script → Lua Parser → Effect Category (CATEGORY_XXX)
                        → Activation Conditions (RACE/ATTR/SETCODE)
                        → Special Hash (code block clustering)
                                 ↓
                        knowledge_base.json + hash_mapping_report.json
                        code_embeddings.npy + code_embeddings_idx.json
```

Each card extracts up to **8 effect slots**, each containing:

| Field | Meaning | Example |
|-------|---------|---------|
| `category` | Effect type | Destroy, negate, search, special summon... |
| `requirements` | Activation conditions | Requires specific race on field, specific phase... |
| `setcode` | Associated archetype | Shaddoll, HERO, Branded... |
| `numbers` | Magic numbers | ATK change amount, level change amount... |
| `ref_codes` | Associated cards | Search target card codes... |
| `race` | Associated race | Dragon, Spellcaster... |
| `attr` | Associated attribute | LIGHT, DARK, WATER... |

### Hash Clustering Algorithm

For effects that can't be classified with standard CATEGORY, code block hashing is used for clustering:

```python
def _hash_code_block(self, code_block):
    # 1. Lexical normalization: replace concrete values with placeholders
    clean_code = re.sub(r'\b1\s*-\s*tp\b', '<OPPO>', code_block)
    clean_code = re.sub(r'\b(tp|ep|rp)\b', '<PLAYER>', code_block)
    clean_code = re.sub(r'\b\d+\b', '<NUM>', code_block)
    
    # 2. Compute MD5 Hash
    hash_val = hashlib.md5(clean_code.encode()).hexdigest()[:8]
    
    return f"CUSTOM_HASH_{hash_val.upper()}"
```

This means: if two cards have structurally similar effect code (only differing in parameters), they are grouped under the same Hash tag, allowing AI to directly reuse experience.

### WebUI Operation

In **🧠 Semantic Knowledge Engine**:
1. Check **🌐 Sync Base KB from Github** to retrieve the complete four-file semantic baseline from the same directory
2. Sync automatically checks and appends newly parsed effect slots; enable **Extract Code Semantic Features** separately only for local updates without sync
3. Click **Start Extracting Card Semantics** and wait for completion

Hash-cluster labels are derived deterministically from normalized code through MD5. v3.6.0 fixes a different source of nondeterminism: semantic fields were deduplicated through unordered sets before fixed-slot truncation. GitHub sync now also inherits the Hash map and code-semantic vectors and automatically appends locally missing slots. If the remote Hash map is absent, continuation records are reconstructed from existing `CUSTOM_HASH_*` tags in the KB.

### V3 Observation and Effect-Slot Audit

Since 3.6.1, training workers, Arena, and RuleBot self-check collect audits without a switch. Runtime collection only aggregates Core messages and public chain fields into per-process reports under `system_logs/protocol_v3_audit/`; it never reads Lua and does not participate in action selection or reward calculation.

Version 3.6.2 no longer treats the low four bits of `desc`, or the Stringid index, as Lua effect-creation order. Semantic generation follows a statically recognizable `SetDescription(aux.Stringid(...))` on the same `Effect.CreateEffect(c)` object and stores the complete `desc` on the corresponding code-semantic slot. Dynamic expressions, ambiguous bindings, and passive effects without descriptions are never guessed; runtime falls back to whole-card semantics and does not set a false “used this turn” bit. WebUI reports unresolved observations as `binding_missing` so static-parser coverage can be expanded safely.

---

## 142 Announce Pool Wrapper Logic

### Problem Background

OCGCore sends `MSG_ANNOUNCE_CARD` (type=142) for effects like "Prohibition" or "Crossout Designator", requiring the player to announce one card from tens of thousands. Feeding tens of thousands of legal options directly to the neural network would cause instant action space explosion, VRAM overflow, and training failure.

### Solution

Galatea-Core implements a **three-layer filtering mechanism** to compress tens of thousands of candidates to a few dozen AI-understandable options:

#### Layer 1: RPN Reverse Polish Expression Filtering

The 142 message from OCGCore contains a string of RPN opcodes describing legal card conditions (e.g., "must be Dragon-type", "must be Level 4 or below").

The framework implements a complete RPN virtual machine:

```python
# Supported RPN opcodes
OP_ISCODE     = 0x40000100  # Whether matches specific card code
OP_ISSETCARD  = 0x40000101  # Whether matches archetype
OP_ISTYPE     = 0x40000102  # Whether matches card type
OP_ISRACE     = 0x40000103  # Whether matches race
OP_ISATTRIBUTE= 0x40000104  # Whether matches attribute
OP_ISLEVEL    = 0x40000105  # Whether matches level
OP_ISLINK     = 0x40000107  # Whether matches link rating
OP_AND        = 0x40000004  # Logical AND
OP_OR         = 0x40000005  # Logical OR
OP_NOT        = 0x40000007  # Logical NOT
```

#### Layer 2: Common-Sense Candidate Pool Collection

The framework collects potential legal card candidates from:

1. **Own Deck/Extra Deck**: Cards in your own deck are most likely to be announced
2. **Known Hand**: Opponent hand cards tracked by hand tracker
3. **Public Information**: Face-up cards on field, in graveyard, and banished zone
4. **Staple Pool** (meta_staples.json): Most common environment staples (Ash Blossom, Maxx "C", etc.)

#### Layer 3: Priority Sorting

Collected candidate cards are sorted by importance:

```python
def get_priority_score(c):
    score = 0
    if c in my_cards: score += 100     # Own deck highest priority
    if c in known_hand_codes: score += 50  # Known opponent hand
    if c in public_zones: score += 50      # Public zones
    if c in meta_staples: score += 10      # Environment staples
    return score
```

### Crash Prevention Fallback

If RPN parsing has flaws causing the candidate pool to be wiped out, the framework automatically falls back to the full candidate pool, delegating to RuleBot for brute-force enumeration — ensuring the program never crashes.

### WebUI Operation

In **🗃️ Assets & Deck Management → 🃏 Meta Staples (142 Cache)**:
- Add/remove commonly used environment staple cards
- Default template includes: Ash Blossom, Maxx "C", Called by the Grave, Infinite Impermanence, etc.

---

## Multi-Select and Sequential Selection Logic

### Problem Background

OCGCore exposes two easily confused protocols. Types 15/20/22/23 require the client to **return a complete combination in one response**. Type 26 (`MSG_SELECT_UNSELECT_CARD`) is the protocol where Core recomputes candidates and sends another message after each Select/Unselect. They must not share one static wrapper.

### Static Combinations: Macro Actions

For messages that expect one complete response, the legality enumerator builds Core-ready packages and the policy chooses among them:

```
Core request: choose 2 from {A, B, C}
Legal enumerator: {A,B} / {A,C} / {B,C}
Policy: choose one complete package
Response: count + original candidate indices
```

Macros retain the exact Core response while exposing codes, order, locations, and rule values to the model:

```python
action.macro_targets = [...]        # Visible entity indices
action.macro_target_codes = [...]   # Hidden-zone/deck option codes
action.macro_target_values = [...]  # Tribute, dual-level, counter values
action.macro_places = [...]         # Zone combination
action.decision_bytes = b'...'      # Complete raw Core response
```

Type 20 follows Core's summed `release_param` rule instead of approximating tribute value by card count. Multi-race/attribute Types 140/141 likewise return an integer mask with exactly `count` bits.

### Type 26: Native Sequential Decisions

Type 26 is not converted into static terminal packages. Arbitrary Lua `special_check` logic exists only inside Core, so one packet cannot reliably enumerate every terminal set; using RuleBot search would change the learning actor and may miss legal paths.

The framework preserves the native flow:

```
Snapshot N: selected {A}; Select B/C, Unselect A, or Finish
Model action: Select B
Core validates and runs Lua constraints
Snapshot N+1: selected {A,B}; candidates and finishable are recomputed
Model action: Finish
```

Every action encodes Select/Unselect/Finish/Cancel semantics, the resulting selected set, candidate code/location, min/max, and finishable/cancelable. Each step becomes its own PPO trajectory row. The network has no recurrent state, but the observation now contains the current selection state, and terminal reward propagates through GAE across the sequence, so MCTS is not required to complete it.

Training retains compact episode-wide visit counts derived from the complete state key, so an A→B→A round-trip such as “enter selection → Cancel/Unselect → return to the original state” is not lost when the intermediate state changes. The first four legal retreats in an identical full state receive no extra penalty; from the fifth Cancel/Unselect, a `-0.005` step reward applies, while ordinary actions retain the tenth-repeat threshold. The action remains legal and PPO continues stochastic sampling—there is no training-time hard mask.

#### Applicable Scenarios

| Message Type | Scenario | Handling |
|--------------|----------|----------|
| `MSG_SELECT_CARD/TRIBUTE` (15/20) | General multi-select effects | Wrap legal options/tribute combinations |
| `MSG_SELECT_PLACE/DISFIELD` (18/24) | Position selection/lock | Wrap legal position combinations |
| `MSG_SELECT_COUNTER` (22) | Counter selection | Wrap complete quantity allocations |
| `MSG_SELECT_SUM` (23) | Synchro/Ritual value selection | Wrap combinations passing Core-equivalent sum rules |
| `MSG_SORT_CARD` (25) | Sorting | Wrap legal orders |
| `MSG_ANNOUNCE_RACE/ATTRIB` (140/141) | Multi-value announcement | Wrap complete bitmasks |
| `MSG_SELECT_UNSELECT_CARD` (26) | Dynamic select/deselect | Model decides sequentially per Core message |

#### Additional Optimization

Static enumeration is capped at 5,000 legal combinations and the final action pool at 120. Pass-1 card scores drive weighted random reduction while every legal package keeps a minimum exploration weight. Equivalent off-field copies with identical code and parameters use canonical count representatives so duplicates do not consume the pool. Reduction remains stochastic and does not turn RuleBot's top score into a fixed answer.

---

## Simple Hand Tracker Logic

### Why It's Needed

In Yu-Gi-Oh!, many effects briefly expose cards in the opponent's hand (e.g., search effects). Human players remember this information, but AI by default only sees public information at the current moment and cannot leverage historically exposed hand intelligence.

### How It Works

`DuelState` maintains a `known_hand_codes` dictionary tracking known cards in the opponent's hand:

```python
class DuelState:
    def __init__(self):
        self.known_hand_codes = {0: [], 1: []}  # [self-known, opponent-known]
```

#### In-Pool (Record): When cards visibly enter opponent's hand

```python
# 1. Public search to hand (e.g. "Reinforcement of the Army")
if new_l == Zone.HAND and is_public_move and pure_code != 0:
    self.known_hand_codes[new_c].append(pure_code)

# 2. After being revealed then entering hand
if pure_code in self.recently_confirmed:
    self.known_hand_codes[new_c].append(pure_code)
```

#### Out-Pool (Forget): When cards leave hidden zones or are revealed

```python
# Leaving hand
if old_l == Zone.HAND and pure_code in self.known_hand_codes[old_c]:
    self.known_hand_codes[old_c].remove(pure_code)

# Face-down card flipped to reveal true identity
if is_from_hidden and pure_code in self.known_hand_codes[old_c]:
    self.known_hand_codes[old_c].remove(pure_code)
```

### Hand Tracker Uses

1. **142 Announce Pool Enhancement**: Tracked cards are prioritized in the 142 candidate pool
2. **Action Priority Scoring**: AI can reference known opponent hand when making decisions
3. **Situational Awareness**: Provided to neural network as part of global features

---

## Deck Weight Adjustment (Global Weights)

### Problem Background

During training, different decks/environment pools have different importance. If you want AI to train more in "competitive" environments and less in "casual" environments, you need to adjust the sampling weights of each environment pool.

### How It Works

The `global_weights.json` file in `decks/` controls sampling probabilities for each environment pool:

```json
{
    "tier1_meta": 3.0,     // Competitive weight 3x
    "tier2_rogue": 2.0,    // Tier 2 weight 2x
    "fun_decks": 0.5,      // Casual weight 0.5x
    "ygopd_MetaDecks_Latest": 1.5  // Online pool weight 1.5x
}
```

Workers sample games using `random.choices(env_choices, weights=weights)`.

### WebUI Operation

In **🗃️ Assets & Deck Management → ⚖️ Dynamic Pool Weights**:
- Set weight sliders for each pool (0.0 ~ 10.0)
- Bulk apply to all pools in a category at once
- Weights are written to `global_weights.json` in real-time, effective next game

### Dynamic Adjustment Strategy

| Training Phase | Weight Strategy |
|----------------|-----------------|
| Early (~500 iters) | Spread weights, expose AI to diverse decks |
| Mid (~2000 iters) | Increase mainstream deck weights, strengthen core ability |
| Late (2000+ iters) | Dynamic balance, auto-adjust based on win rates |

---

## Virtual Disguise Pool Module (Virtual Mix Pools)

### Problem Background

During training, you sometimes want AI to encounter cross-pool matchups like "competitive vs casual" without physically moving deck files. Traditional subfolder structure only supports "same-pool civil war."

### How It Works

Virtual disguise pools let you create a "recipe" mixing decks from multiple physical environment pools:

```json
// decks/virtual_pools.json
{
    "Meta_VS_Fun": {
        "tier1_meta": 0.7,   // 70% chance from competitive pool
        "fun_decks": 0.3     // 30% chance from casual pool
    },
    "Online_VS_Local": {
        "ygopd_MetaDecks_Latest": 0.6,
        "tier1_meta": 0.4
    }
}
```

When a Worker draws a virtual pool:
1. First, randomly select two physical pools by weight within the recipe
2. Then randomly select one deck from each physical pool
3. The two decks from different pools battle each other in a cross-environment match

### Advantages

- **No physical file movement**: Reduces disk operations
- **Flexible experimental design**: Create mixing recipes at any ratio
- **Expanded opponent diversity**: Prevents AI from only learning "civil war"

### WebUI Operation

In **🗃️ Assets & Deck Management → 🧠 Virtual Mix Pool Builder**:
- Create new virtual mix pool
- Set mixing weights for each physical pool
- After creation, virtual pools appear in the global weights panel

---

## Next Steps

- 🔧 Read [Architecture](architecture.md) for framework internals
- 📚 Read [Feature Guide](features.md) for WebUI usage
- 🚀 Return to [Quick Start](quickstart.md) to begin training
