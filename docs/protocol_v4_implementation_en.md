# Galatea V4 Protocol Implementation Specification

> Status: in development. This is the executable specification for the V4 implementation branch. Released V3 behavior remains defined by the existing architecture documents.

## 1. Version boundaries

| Item | Current stable | V4 target | Switch point |
| --- | ---: | ---: | --- |
| Framework | 3.9.2 (Phase 3 complete) | 3.9.x; 4.0.0 after the deck-building/BO3 layer | Per release stage |
| Model Protocol | 4 | 4 | Switched when exact card identity landed |
| Checkpoint Format | 3 | 3 | Required V4 metadata has landed |
| Trajectory Schema | Not independently versioned | 1 | When the canonical trajectory recorder lands |

These versions are independent. Once Model Protocol 4 is enabled, it remains fixed throughout V4 development. Field inventories, tensor shapes, and `protocol_schema_hash` prevent intermediate artifacts from being mixed.

V4 trains from scratch. V3 weights, optimizer/scaler state, and historical opponent pools are not migrated. V3 models and Link remain supported by the archived V3 runtime.

## 2. Non-negotiable invariants

- Never expose opponent hidden information.
- Every model action must deterministically map to a legal Core response.
- Training, Arena, ONNX, and Link share observation/action builders.
- Public state remains symmetric under player-view swapping.
- Lua semantics are indexed by card code and effect slot, never printed effect numbering.
- Initial V4 stays at `512 / 8 heads / 6 layers` until protocol benefits are measured.
- Initial V4 excludes MCTS, unreliable lingering-effect tracking, and subjective behavior rewards.
- Equivalent performance optimizations require numerical parity tests.

## 3. Target data protocol

### 3.1 Exact card identity

Use an append-only `card_vocab.json` instead of modulo hashing. Token `0` is padding, `1` is unknown, and `2/3` represent hidden opponent Hand/Spell-Trap occupancy. Real cards begin at `10`, and new cards append without reordering. `protocol_schema_hash` describes the fixed structure and therefore remains stable across valid appends. Artifacts record their exact vocabulary hash/count and are accepted only when that mapping is a prefix of the model lineage's authority. The Galatea repository is the default source; custom-card users may use a private source or append from a local CDB, but every machine in that lineage must receive the same file. A locally newer CDB only isolates affected decks.

### 3.2 Global and player state

Explicitly distinguish `decision_player`, `turn_player`, `starting_player`, relative `went_first` and `is_my_turn`, absolute/self/opponent turn counts, and categorical phase/zone/position values. Categories are not represented by arbitrary numeric division.

### 3.3 Action semantics

Candidates include prompt type, operation family, source/target roles, exact card identity, effect slot, summon method, target phase, response semantics, selection bounds, finish/cancel conditions, and Core response bytes.

Operation families cover normal/tribute/special summon, reposition, monster set, spell/trap set, activate, chain, attack/direct attack, phase, select/unselect, finish, cancel, place, counter, sort, announce, and Yes/No. Ritual, Fusion, Synchro, Xyz, Link, and Pendulum are distinct summon methods.

Type 26 retains iterative Core-native Select/Unselect behavior. Static combination prompts return a complete legal combination. Their state machines remain separate.

Phase 2 batch 1 implements this field contract:

| Field | Visibility and source | Lifetime/default | dtype / shape | Consumers |
| --- | --- | --- | --- | --- |
| `act_operation` | Current public prompt; Core candidate family such as a fixed `MSG_SELECT_IDLECMD` list ordinal | Current legal-action snapshot; `DEFAULT=0` | `uint8 [120]` | Action signature, policy network, ONNX, replay |
| `act_summon_method` | Current public prompt; direct evidence from a Core message or verified runtime context | Current legal-action snapshot; `NONE=0` when not a summon and `SPECIAL_UNKNOWN=3` for generic special summons | `uint8 [120]` | Action signature, policy network, ONNX, replay |

The six standard-Core `MSG_SELECT_IDLECMD` lists prove normal summon, special summon, reposition, monster set, spell/trap set, and activate respectively. They do not carry an exact special-summon method and do not prove whether a normal summon requires tributes. Ritual, Fusion, Synchro, Xyz, Pendulum, Link, and Tribute values are reserved for future direct evidence; neither source zone nor monster card category may stand in for a summon method. The original `(candidate index << 16) | list type` response remains deterministic, so learned fields do not alter the Core reply.

Phase 2 batch 2 freezes the selection boundary: Type 26 is the only per-item Select/Unselect state machine in this batch, while Types 15, 20, and 23 return one complete combination. Type 20's `min` is a minimum release value rather than a card count, and Type 23 has no cancel response. Types 20/23 expose Pass-1 material candidates before the shared legality generator builds packages; every response is checked against Core-equivalent rules before pooling and transmission. `set_responseb` now owns the complete 512-byte lifetime buffer instead of truncating it to 64 bytes.

### 3.4 Public relations and static semantics

Public observations cover equip targets, overlays, target/material relations, control, disabled zones, and typed counters. Only facts deterministically available from Core are represented.

Version 3.8.2 consumes dynamic identity, Level/Rank, current/base ATK/DEF, reason/reason card, equip and effect-target relations, every overlay identity, typed counters, original owner, dynamic scales/link values, and `STATUS_DISABLED | STATUS_FORBIDDEN | STATUS_PROC_COMPLETE` through the bundled Core's public legacy `query_card/query_field_count` API. Graveyard and banished zones are reconciled before each decision snapshot, and opponent-hidden Core dynamic state and relations are masked before encoding. The lightweight memory may retain a previously revealed identity and its static CDB/semantic facts, but cannot use that identity to read hidden dynamic ATK/DEF, type, counters, materials, or status. Reasons, counter types, and disabled zones use lossless packed bytes in trajectories and expand only during forward execution.

Although CDB packs Level, Rank, and Link Rating into one raw `level` integer, Core exposes them as three independent runtime query fields. Normal/Fusion/Ritual/Synchro monsters ordinarily have only `level`; Xyz monsters ordinarily report `level=0, rank>0`; Link monsters ordinarily report `level=0, link_rating>0`. Zero means “not applicable,” not missing. Any effect-driven exception follows Core's query output and is never backfilled from card category.

`STATUS_PROC_COMPLETE` is Core's authoritative “proper procedure completed” flag for revive-limit rules. It is set after a successful proper summon, persists in Graveyard and banishment, and is cleared by Core when the card returns to Hand/Main Deck/Extra Deck or its summon is negated. It must become persistent state on every currently visible monster entity rather than an event-derived guess.

Core internally retains exact summon type, source, and player, and its public Lua API exposes these values to rules scripts. As of upstream v11.0, however, standard client query fields still do not export them. Galatea neither patches a local Core nor maintains a private binary, and it never infers these values from card category, source zone, or message order. The model currently receives only public `STATUS_PROC_COMPLETE`; exact summon provenance remains an upstream API/possible PR item. Current upstream uses structured `OCG_DuelQuery`, so a future upgrade must adapt its complete public ABI and TLV query format rather than replace one parser field.

The macro selector now preserves physical instances on the field, in the Graveyard, and in banishment. Hidden/reset-zone copies are folded only when card identity, prompt values, and known public state are identical. The 5,000-option cap and existing weighted reduction still bound combinatorial growth without changing legal responses.

Static Lua semantics move to model-side lookup tables indexed by exact card identity and effect slot. Per-step trajectories carry identities, slots, and required dynamic overrides instead of duplicating large vectors.

Version 3.9.0 establishes the unified compiler, logical prefix identity, and artifact validation. Version 3.9.1 registers the table on the model: Encoder/shared-memory/PPO trajectories carry existing card tokens plus compact chain/history effect-slot IDs, and the model reconstructs semantics elementwise-identically to expanded inputs on device. Unknown slots retain whole-card fallback, with no change to visibility or action logic.

Version 3.9.2 materializes the compiled result as two independent runtime assets: a pickle-free NPZ of numeric arrays and a JSON catalog carrying vocabulary/logical identity, array integrity, and lightweight effect-slot bindings. PTH/training restores the NPZ directly, spawn Workers register only the JSON catalog, and ONNX freezes the same constants into `.onnx.data`. GKG V4 requires the runtime pair while keeping the source KB/vector trio optional for continued semantic generation.

### 3.5 Deck protocol

- Initial Main, Extra, and Side sections have distinct section and copy-count labels.
- Duel policy receives only the controller's knowable deck information.
- Stable initial deck identity is separated from remaining-deck counts/masks.
- Workers register one profile per duel; samples reference `deck_profile_index`.
- Side Deck is reserved but does not condition BO1 duel actions.

## 4. Target network

### 4.1 Reusable Deck Encoder

A permutation-invariant Deck Encoder initially uses 8 learnable latent queries and emits `deck_style`, `deck_latents`, and `per_card_features`. It exposes an independent API and must pass permutation-invariance tests.

### 4.2 Layered FiLM

Global state and deck style use separate branches. Every Transformer layer receives independent attention- and FFN-side modulation:

`global_film + deck_gate * deck_film`

`deck_gate` starts at zero, modulation is bounded, and light deck-condition dropout limits early over-reliance. Deck summaries also enter policy/value heads directly.

### 4.3 Compact event history

Store 16 state transitions between model decisions, including actor, turn/phase, prompt/operation, source/target/effect slot/summon method/chain, LP and zone deltas, and cancellation/retry/invalid/resolution outcomes. A small event encoder preserves board-token capacity.

### 4.4 Auxiliary heads and planning

State-, action-, and deck-level auxiliary heads require verifiable labels and missing-label masks. Initial auxiliary gradients target roughly 10%–20% of main-task gradients; unreliable targets remain detached probes.

Future summaries cover next decision, chain end, turn end, and terminal state. A Goal Planner emits `plan_latent` for `policy(action | state, deck_style, plan_latent)`. The first implementation recomputes it at each decision and applies within-turn/phase consistency loss before considering persistent Options or search.

## 5. Training and data

- Keep current terminal rewards and confirmed extreme-deadlock safeguards.
- Keep `gamma=0.998`; test GAE `lambda=0.98` against `0.95` and `0.99`.
- Use at most four PPO epochs with target-KL early stopping and detailed optimizer metrics.
- Core owns one canonical trajectory interpreter; Link and YRP provide raw messages, responses, and source metadata.
- YRP enters behavior cloning only after deterministic replay and unique legal-candidate mapping.
- Split data by whole duel, deck, and player.

Canonical trajectories record decks/Side Deck, seed, messages, responses, roles, result, Core/script/CDB/semantic/vocabulary hashes, source, and quality.

## 6. Deck-building and BO3 interfaces

V4 Core exposes independent `DeckSpec`, `MatchContext`, `DuelSummary`, Deck Encoder, and fast-evaluation interfaces. A later layer handles legal deck edits, BO3 siding/turn order, opponent-deck posterior, counterfactual evaluation, and context-aware Type 142 generation. Match control remains outside single-duel `DuelState`.

## 7. Implementation phases

- [x] **Phase 0: specification freeze**—boundaries, non-goals, contracts, and gates.
- [x] **Phase 1: identity/global state**—exact vocabulary; decision/turn/starting player; categorical state.
  - [x] Review batch 1: append-only exact vocabulary, V4 schema hash, and PTH/ONNX/artifact/package identity closure.
  - [x] Review batch 2: `decision_player`, `turn_player`, `starting_player`, and relative perspective.
  - [x] Review batch 3: categorical phase, zone, and position representations.
- [x] **Phase 2: actions/public relations**—operations, summon methods, selections, relations, audit.
  - [x] Review batch 1 (3.8.0 / schema revision 4): correct the six `MSG_SELECT_IDLECMD` operation families and add evidence-backed summon categories. When standard Core does not expose an exact method, encode unknown instead of guessing from card type.
    - Acceptance evidence: of 188 default tests, 187 pass and one gated real-Core case is skipped; the gated case passes when enabled separately, PyTorch/ONNX agree within `1e-3`, and empty semantic rows remain finite.
  - [x] Review batch 2 (3.8.1 / schema revision 5): retain Core-native iterative Type 26 and complete-combination Types 15/20/23; add Type 20/23 Pass-1 material semantics, Core-equivalent response validation, retained finish/cancel exits, and a 512-byte response buffer.
  - [x] Review batch 3 (3.8.2 / schema revision 6): use the public legacy Core query API for equip source-target, effect target, reason card, all overlay identities, typed counters, original owner, dynamic Level/Rank/Scale/Link values, disabled zones, and `STATUS_DISABLED | STATUS_FORBIDDEN | STATUS_PROC_COMPLETE`; preserve Graveyard/banished physical copies, classify audited messages as explicitly applied, query-reconciled, or known observation gaps, and connect Type 21 sorting plus Type 22 Pass-1 semantics. Exact summon provenance remains an explicit upstream item because public Core does not export it; this batch has no private-Core dependency.
- [x] **Phase 3: static semantic lookup**—deduplication, asset hashes, ONNX parity.
  - [x] Review batch 1 (3.9.0 / schema revision 7): compile one table aligned to exact tokens and Lua effect slots, share it across Encoder/network consumers, and carry a prefix-compatible semantic hash through PTH, ONNX, artifacts, and deployment packages. Legacy network inputs remain as a lossless bridge.
  - [x] Review batch 2 (3.9.1 / schema revision 8): perform model-side lookup from card/effect-slot identities and remove repeated structured-semantic matrices from Encoder/shared memory/PPO trajectories. Compact and legacy-expanded full forwards match with zero elementwise error; net storage drops by 152,996 bytes per step, or about 4.67 GiB for one 32,768-step trajectory pool.
  - [x] Review batch 3 (3.9.2 / schema revision 8): materialize and validate independent static-semantic runtime assets; restore effect-slot catalog registration in spawn Workers; keep compact ONNX inputs while freezing static tables into external data; bump GKG to format 4 and require the vocabulary plus compiled runtime pair cross-machine while source semantics remain optional maintenance assets. The default suite passes 207 of 208 tests with one gated skip; explicit real-Core and external ONNX Runtime smoke gates pass.
- [ ] **Phase 4: deck protocol/encoder**—profile deduplication, labels, reusable encoder.
- [ ] **Phase 5: layered FiLM/history**—separate modulation and 16 compact events.
- [ ] **Phase 6: auxiliary heads/planning**—verifiable tasks, future summaries, `plan_latent`.
- [ ] **Phase 7: PPO/canonical trajectories**—GAE/KL controls and Link/YRP inputs.
- [ ] **Phase 8: V4 scratch-training validation**—short, medium, and long gates.
- [ ] **Phase 9: deck-building/BO3 layer**—framework becomes 4.0.0 after completion.

Each phase requires static audit, unit tests, real-Core smoke, numerical/performance checks, and bilingual docs.

## 8. Acceptance gates

Every incompatible change needs field/dtype/shape/mask/enum tests, Core-response round trips, hidden-information checks, player symmetry, applicable permutation/replay checks, PyTorch/ONNX parity, and early rejection of old checkpoints.

Before short training: 1,000 RuleBot self-check games, V4 self/Rule/fixed-V3 external baselines, 50-iteration memory/handle stability, Windows/Linux real-Core smoke, and no unexplained inference regression above 15%.

Before long training: 5–10 iterations at 4K/8K steps, 50 iterations at 32,768 steps, then 100–200 iterations with healthy optimizer, auxiliary, and fixed-Arena metrics.

## 9. Change control

- “Complete” requires code, tests, and runtime evidence.
- New fields document visibility, source, lifecycle, missing value, dtype, shape, and consumers.
- Auxiliary heads document label generation, masking, weight, and trunk-gradient behavior.
- Optimizations document mathematical-equivalence boundaries and comparison results.
- Preserve an audit fixture before changing the specification for unexpected Core behavior.

## 10. Parameters requiring measurement

Macro target capacity, rare-message/relation capacity, 4 versus 8 deck latents, target KL, auxiliary weights, GAE `0.98` versus `0.99`, and whether `plan_latent` should become persistent remain measurement-driven decisions.
