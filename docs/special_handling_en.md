# 🧬 Framework Special Handling Logic

> Detailed explanation of modules specially built to overcome inherent framework limitations — these are the core competitive advantages of Galatea-Core.

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
                        knowledge_base.json
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
1. Check **🌐 Sync Base KB from Github** (must for first use)
2. Click **Start Extracting Card Semantics**
3. Wait for parsing to complete

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

## Multi-Select Chunk Wrapper Logic

### Problem Background

YGOPro interaction messages like `MSG_SELECT_CARD` (type=15) allow players to **multi-select / deselect** multiple cards until conditions are met. Traditional RL frameworks struggle with these "multi-step combinatorial actions" because the available action set changes dynamically each step.

### Solution: Macro Action System

Galatea-Core introduces a **Macro Action Wrapper** that encapsulates multi-step selection operations into an "atomic action":

```
Original interaction flow:
Step 1: Select Card A → options change
Step 2: Select Card B → options change  
Step 3: Deselect Card A → options change
Step 4: Select Card C → conditions met, submit

After Macro Action wrapping:
One action = Select {B, C}, place in designated zone
```

#### Core Implementation

```python
class MacroAction:
    def __init__(self):
        self.macro_targets = []    # Final selected card list
        self.decision_bytes = b''  # Original decision byte stream (for replay)
        self.macro_places = []     # Placement position list
```

#### Advantages

1. **Stable action space**: Neural network always faces fixed-dimension actions, no expansion from multi-step operations
2. **Faster training convergence**: No need to learn complex "combinatorial selection" strategies
3. **Replay consistency**: Original decisions recorded via `decision_bytes`, precisely restored on replay

#### Applicable Scenarios

| Message Type | Scenario | Macro Action Handling |
|--------------|----------|-----------------------|
| `MSG_SELECT_CARD/TRIBUTE` (15/20) | General multi-select effects | Wrap legal options/tribute combinations |
| `MSG_SELECT_PLACE/DISFIELD` (18) | Position selection/lock | Wrap legal position combination pool |
| `MSG_SELECT_COUNTER` (22) | Counter selection | Wrap legal choice pool |
| `MSG_SELECT_SUM` (23) | Sum logic (Synchro/Link) | Wrap legal choice pool |
| `MSG_SORT_CARD` (25) | Sorting logic | Wrap legal ordering combinations |

#### Additional Optimization

For `MSG_SELECT_CARD` scenarios where extremely large option counts are possible (e.g., pick 5 from 23, horrific number of combinations), a 5000-combination ceiling is set during DFS calculation to prevent computation deadlock. Before entering DFS, the framework rates individual cards by weight to form weighted combinations, and applies weight-based filtering again when passing combinations through — allowing AI to learn and select desired option groups as efficiently as possible.

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
