# Galatea V4 Protocol Implementation Specification

> Status: in development. This is the executable specification for the V4 implementation branch. Released V3 behavior remains defined by the existing architecture documents.

## 1. Version boundaries

| Item | Current stable | V4 target | Switch point |
| --- | ---: | ---: | --- |
| Framework | 3.7.0 (Phase 1 complete) | 3.7.x–3.9.x; 4.0.0 after the deck-building/BO3 layer | Per release stage |
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

### 3.4 Public relations and static semantics

Public observations cover equip targets, overlays, target/material relations, control, disabled zones, and typed counters. Only facts deterministically available from Core are represented.

Static Lua semantics move to model-side lookup tables indexed by exact card identity and effect slot. Per-step trajectories carry identities, slots, and required dynamic overrides instead of duplicating large vectors.

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
- [ ] **Phase 2: actions/public relations**—operations, summon methods, selections, relations, audit.
  - [ ] Review batch 1: correct the six `MSG_SELECT_IDLECMD` operation families and add evidence-backed summon categories. When standard Core does not expose an exact method, encode unknown instead of guessing from card type.
  - [ ] Review batch 2: audit Type 15/20/23/26 and macro-action state machines, response round trips, finish/cancel boundaries, and fixed-shape encoding message by message.
  - [ ] Review batch 3: add public equip source-target, card-target, full overlay-material, typed-counter, and disabled-zone relations plus message-coverage auditing.
- [ ] **Phase 3: static semantic lookup**—deduplication, asset hashes, ONNX parity.
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
