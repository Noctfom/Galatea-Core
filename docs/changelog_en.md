# 📝 Changelog

> This document records the version update history of Galatea-Core.

---

## [v3.5.1] - 2026-09-02

### 🎞️ Holographic Replay V2

- **Complete two-sided timeline**: Replays no longer capture only P0 model decisions. P0/P1 model probabilities, RuleBot responses, and Core move, summon, set, chain, attack, targeting, counter, draw, and LP events are recorded in their actual order
- **Correct arrow semantics and direction**: A chain response points from the newly activated card to the previous chain link; an attack points from attacker to target or opposing LP; card movement points from source zone to destination. Equip, targeting, and multi-target choices support one-to-many arrows, with a ghost source retained after a card leaves its old zone
- **Action Protocol V2 visualization**: The option panel exposes Select/Unselect/Finish/Cancel, selection bounds and result count, target codes/material values, finish/cancel conditions, and prompt fields, making the information introduced in 3.5.0 directly inspectable
- **Interactive candidate preview**: Clicking any row in the confidence table highlights its actor, targets, and materials on the board and shows related card images and protocol semantics above the table. Previewing never rewrites the action actually recorded in the replay
- **P1 confidence visibility toggle**: The P1 model or RuleBot candidate table can be hidden while retaining its final decision, board events, and timeline frames
- **Full two-sided deck quick view**: New recordings store each player's deck name, Main Deck, and Extra Deck once in replay metadata. The UI groups duplicate cards and displays all four lists. This post-game audit data never enters model observations, and older recordings remain readable when the optional field is absent
- **Detailed Special Summon descriptions**: Type 11 now uses the candidate code and raw location to recover actors not yet present in the entity table, naming the target and identifying Link, Xyz, Synchro, Fusion, or other Special Summon entries. When Core reports only the result without a proper-summon reason, the replay conservatively labels the monster category instead of misreporting a revival as a proper summon
- **Synchronized playback controls**: Previous/next buttons and the timeline now share one cursor. Selecting another replay stops autoplay and returns to frame 0; single-frame recordings no longer create an invalid slider range
- **Independent replay format**: Added `REPLAY_FORMAT_VERSION = 2`. Full boards are deduplicated into a state table and frames reference `state_id`, with compact JSON output to prevent abnormal long games from copying the board into every frame. A real 574-frame two-sided recording measured about 1.09 MB. The UI still reads older inline-state recordings
- Replay widgets use Streamlit's `width="stretch"` API, removing the `use_container_width` deprecation warnings from this page

### 🧩 Core and Arena Fixes

- **Fixed Type 16 byte alignment**: Restored the bundled Core header `spe_count + global forced + hint_timing[2]`, including separator bytes before candidates after the first. Standard Core uses the no-separator layout, and Cancel is offered only when global forced is zero
- **Fixed missing LP-cost state updates**: `MSG_PAY_LPCOST` (Type 100) now deducts P0/P1 LP in DuelState, so model observations and replays do not retain stale LP after paying a cost
- Chain-stack entries retain the Core chain index, so replay and diagnostics no longer show an unknown link number
- **Expanded loop diagnostics**: When soft bans cover the entire pool, Arena prints MsgType, action descriptions, and repeat counts. Real runs confirmed the warnings were genuine Type 26 Select/Unselect oscillation rather than merged state keys
- **Fixed exhausted Arena soft bans**: Once every candidate reaches the repeat threshold, Arena no longer disables loop suppression permanently for that state. Only abnormal loop states rotate to the least-visited legal candidate, giving Cancel, Select, Unselect, and Finish a chance while ordinary states keep greedy model decisions
- **Added training feedback for cancel round-trips**: Full-state visit counts now survive A→B→A transitions. The first four Cancel/Unselect choices in an identical full state remain untouched; from the fifth visit they receive the existing `-0.005` step reward, while ordinary actions retain the tenth-visit threshold. Training remains stochastic and never masks a legal Cancel, preserving normal retreats and PPO behavior-policy consistency
- **Further relaxed long-game shaping**: Based on a 50-iteration run that still converged normally, the tiny per-step turn penalty now starts only after turn 40. The 1,500-step hard truncation, turn-40 slow-win reward of `0.05`, and 300-decisions-per-turn threshold remain unchanged, keeping the pressure focused on extreme long games

### 🖥️ WebUI Stability Fixes

- **Fixed TensorBoard startup**: The WebUI now launches `sys.executable -m tensorboard.main` from the active bundled `python_env` with an absolute log path, instead of requiring `tensorboard.exe` on the system `PATH`. Startup failures are reported in-page rather than as a full traceback
- **Safe TensorBoard process ownership**: PID and creation time are registered, and a service previously started by this project's WebUI can be safely re-adopted after a page reconnect. Global process-name killing was removed; an unrelated service on port 6006 is explicitly left untouched
- **Fixed replay autoplay traceback**: After the slider widget is instantiated, autoplay advances only the logical cursor and synchronizes the widget key on the next rerun, avoiding Streamlit's post-instantiation state mutation error

### ✅ Validation and Compatibility

- 127 automated tests completed (one environment-dependent skip), covering replay cursor state, candidate previews, initial deck lists, Special Summon categories, P1 decisions, RuleBot response mapping, chain/attack direction, move and LP events, state deduplication, cancel round-trip shaping, and Arena candidate rotation
- A real model-vs-RuleBot replay completed with 574 frames, both players' decisions, and 393 Core events. A dual-model stress game confirmed P1 probability distributions and Type 26 semantics are recorded
- Bundled TensorBoard 2.20.0 completed a real start/stop smoke test on an isolated port without any system-level Streamlit or TensorBoard executable
- Network input shapes, PPO equations, and checkpoint structures are unchanged; only the repeated-action accounting and localized Cancel/Unselect threshold receive the shaping changes described above. `MODEL_PROTOCOL_VERSION` and `CHECKPOINT_FORMAT_VERSION` remain 2. Deck lists are optional Replay V2 metadata written once per recording, and Arena games without recording pay no replay-serialization cost

---

## [v3.5.0] - 2026-09-02

### 🧩 Model Action Protocol V2

- **Unified action semantics**: `GameAction` now formally carries operation kind, response value, raw location, selection limits, resulting count, finish/cancel conditions, prompt fields, macro target codes, and material values. Critical meaning no longer exists only in model-invisible indices, debug text, or raw response bytes
- **Refactored action-head inputs**: Added operation, response, full semantic signature, constraint context, target code/value, and actor-relative location tensors. A stable 32-bit semantic signature continues to distinguish the remainder of combinations that exceed the five explicit target slots
- **Fixed deterministic but unlearnable choices**: Type 12/13 Yes/No, Type 14 option descriptions, Type 19 battle positions and card code now produce distinct model features. Type 10 direct-attack flags and candidate codes are also preserved
- **Exposed Type 11 shuffle**: `can_shuffle` now creates Core action type 8, so this legal command can enter the model action pool

### 🔁 Sequential Selection and Combination Legality

- **Native sequential Type 26 decisions**: Every Select/Unselect action encodes the resulting selected set, candidate code/location, min/max, and finishable/cancelable state; Finish and Cancel are distinct semantic operations. The model interacts with Core one step at a time, each decision enters the PPO trajectory, and terminal reward propagates through GAE without requiring RuleBot control or MCTS
- **Preserved the static-macro boundary**: Types 15/18/20/22/23/24/25 still use the legal enumerator to produce complete responses. Candidates now retain codes, ordering, tribute values, dual level values, and counter allocations so legal packages no longer collapse in the action head
- **Fixed Type 20 tribute legality**: Matches Core's rule that selected count must not exceed `max` and summed `release_param` must reach `min`, supporting one double-tribute card while excluding count-only invalid packages
- **Fixed Type 140/141 multi-value announcements**: When Core requests multiple races or attributes, the pool returns one OR mask containing exactly `count` bits instead of submitting an invalid single bit
- **Bounded combination cost**: Equivalent off-field copies with the same code and parameters generate canonical count representatives. The existing 5,000-option enumeration cap, 120-option weighted sampling, and minimum exploration weight remain in place

### 📨 Model Artifact Protocol

- **Independent model protocol version**: Added `MODEL_PROTOCOL_VERSION = 2` and raised `CHECKPOINT_FORMAT_VERSION` to 2. PTH top-level metadata, `net_config`, model `state_dict`, ONNX metadata, and artifact manifests all record and validate the model protocol; WebUI displays and rejects mismatches during resume
- Because network inputs and the action head changed, 3.5.0 accepts only Model Protocol V2 weights and never silently applies an older action protocol to the new tensors
- **Resource impact**: New trajectory fields add about 6.1 KiB per step, or roughly 133 MiB for a 22,384-step pool. Action embeddings and temporary tensors for one six-worker inference batch remain small relative to the existing model and commit budget; rewards, commit boundaries, GAE, and PPO equations are unchanged
- **Validation**: 114 automated tests cover message parsing, Type 26 transitions, Type 20 legality, counter allocation, multi-bit announcements, network forward, checkpoint/ONNX protocol checks, and a real ONNXRuntime run. A temporary V2 model completed a real Core + Lua + deck smoke duel against RuleBot

### 📚 Version and Documentation

- Updated the displayed framework version, `version.txt`, README badges, and all bilingual document applicability markers to `3.5.0`
- Architecture and special-handling guides now distinguish static legal macro actions from native sequential Type 26 decisions and document Action Protocol V2 tensors and version boundaries

---

## [v3.4.2] - 2026-08-31

### 🐛 Training Hotfix

- **Fixed the post-collection garbage-collection crash**: Removed the duplicate `import gc` inside `collect_rollouts()`, which shadowed the module-level dependency and caused the first `gc.collect()` after worker completion to raise `UnboundLocalError`
- The failure occurred before trajectory merging and PPO updates, so it could not write a partial optimizer state or contaminate an existing model. The fix only restores the original garbage-collection call and does not change sampling or learning behavior
- **Fixed stepwise commit loss across iterations**: The trainer's merged trajectory pool now overwrites and reuses its valid prefix instead of being destroyed and reallocated every iteration. A low-headroom preflight may safely release the already-consumed old pool and retry once; trainer shutdown performs the final release
- **Added staged memory auditing**: Logs now capture system commit, trainer private commit/RSS, merged-pool size, and CUDA usage around worker startup, worker reaping, trajectory merge, and PPO. Workers record process usage after opponent-backend initialization, allowing hist/ONNX peak cost to be distinguished from trainer-lifetime retention
- The final gradient references are explicitly cleared after PPO; sampling, rewards, GAE, valid-sample bounds, and parameter-update results are unchanged
- **Fixed Arena MSG 26 false aborts**: Loop detection now uses a complete state key covering the board, acting player, Select/Unselect semantics, target entities, and macro responses, preventing different selection phases with matching slots from sharing bans
- **Separated Arena hard and soft bans**: Engine retries remain authoritative hard bans. Repeated model choices use soft loop bans that are withdrawn if they would exhaust the pool, so normal finite logits are no longer conflated with `all model actions are banned or non-finite`
- **Made checkpoint architecture authoritative in Arena**: P0/P1 no longer allocate temporary networks from CLI defaults before loading. Each network is built directly from its checkpoint metadata, removing misleading logs and one redundant allocation. A 20-game AI-vs-RuleBot run with the same model completed all games normally
- **Relaxed long-game reward shaping**: The tiny turn penalty now starts after turn 30 instead of 20, and the per-turn model-decision penalty starts after 300 decisions instead of 200. The 1,500-step truncation, `0.05` reward for wins after turn 40, and existing safety breakers remain intact
- **Fixed false training-loop penalties**: Training and Arena now share a complete state key covering the board, acting player, action semantics, targets, and macro responses. This replaces the `action_type + index` shortcut that could penalize valid progress as repetition. A maximum-shape microbenchmark measured about `39.6 µs/call`, or `0.65s` for 16K calls

### 📦 Portable Environment and Release Packaging

- **Restored the GPU runtime**: Replaced the portable environment's CPU-only Torch with `PyTorch 2.9.1+cu130`, verified by an actual CUDA tensor operation, restoring CUDA training when WebUI or CLI is launched from `python_env`
- **Complete runtime-resource checks**: Environment validation now covers the card database, Lua scripts, decks, semantic-knowledge files, portable interpreter, and the presence and loadability of the `ocgcore` dynamic library in addition to Python packages
- **Separated launch and release requirements**: Ordinary one-click startup still allows systems without NVIDIA GPUs to fall back to CPU under `auto`; release packaging requires the bundled CUDA runtime to execute successfully, preventing another mislabeled CPU-only bundle
- **Added one-click packaging**: `构建一键包.bat` invokes `build_portable_package.py` to run release validation and package source, `python_env`, `cards.cdb`, `script`, and `decks` as `Galatea_Core_Vx.x.x.zip`
- **Excluded user and development data**: Git metadata, models, training logs, replays, caches, tests, and engine build sources are omitted. Archives use one `Galatea_Core/` root, ZIP64, and atomic temporary-file replacement
- **Compatibility boundary**: The current CUDA Wheel targets modern NVIDIA architectures and supports RTX 20/30/40/50 and GTX 16 series. GTX 10 and older GPUs require an older PyTorch environment or the framework falls back to CPU
- Updated the displayed framework version, bilingual documentation, README badges, and `version.txt` to `3.4.2`; all 102 tests pass

---

## [v3.4.1] - 2026-08-31

### 🛡️ Windows Collection Stability and Memory Optimization

- **Identified the WinSock 10055 root cause**: Confirmed that `No buffer space available` and the native libzmq assertions were not caused by a forced Rule game conflicting with a hist opponent. They were a cascade triggered when multiple workers, large rollout pools, and duplicate model copies exhausted the Windows commit limit
- **Lightweight worker inference frontends**: The current policy and self opponents now retain only feature encoding and action-packing support inside workers. Full PyTorch networks that never performed local forwards are no longer created; central requests, opponent weights, and sampling distributions remain unchanged
- **Removed redundant temporary weights**: Eliminated per-iteration creation, loading, and cleanup of `tmp_weights_iter_*.pt`, removing duplicate self-worker weight copies and unnecessary disk I/O without changing the formal PTH checkpoint protocol
- **ONNX-first historical inference**: Hist workers first validate and mount ONNX artifacts carrying the model UUID, prefix, and iteration. A formal PTH fallback network is loaded only if ONNX is missing, cannot be initialized, or fails at runtime, so the normal path no longer keeps two historical models resident
- **Safe fallback and resource cleanup**: An ONNX runtime failure changes only that worker's historical-opponent backend, while the affected episode follows the existing rollback rules. ZMQ contexts are created after large rollout-pool allocation and are safely released after partial initialization failures
- **Windows commit-memory preflight**: Before workers start each iteration, the trainer reads current commit usage and limit, then estimates rollout-pool, worker-process, and trainer safety requirements. Insufficient headroom now produces a clear error before child-process creation instead of escalating into PyTorch OOM, WinSock 10055, or a libzmq assertion
- **Regression coverage**: Added coverage for networkless workers, ONNX-first and lazy PTH fallback, commit-memory estimation, and removal of temporary weights; all 87 tests pass

### 📚 Version and Documentation

- Updated the bilingual Feature Guide, Architecture Guide, and changelog with the lightweight-worker, historical-model fallback, and Windows commit-limit protection behavior
- Updated the displayed framework version, README badges, and `version.txt` to `3.4.1`

---

## [v3.4.0] - 2026-08-31

### 🧠 Central Inference Architecture Refactor

- **Always-on central batched inference**: Removed the obsolete async-inference toggle. The current policy and self opponents now share one central inference service through ZMQ ROUTER routing, shared-memory slots, and request completion IDs
- **CPU-only workers**: Collection workers no longer accept a pseudo device option and never create CUDA contexts. The current policy and self opponents are both evaluated by central inference
- **Real CPU/CUDA training modes**: Replaced the old device controls with `--device auto|cpu|cuda`, which selects the device for both central inference and PPO updates. `auto` prefers CUDA and falls back to CPU
- **Device-specific optimization paths**: CUDA retains pinned memory, non-blocking transfers, TF32, BF16/FP16, and optional `torch.compile`; CPU uses ordinary memory, FP32, and phase-aware thread budgets
- **Correct self/hist opponent split**: New-training self opponents use the current iteration's policy weights instead of sending internal temporary `.pt` files through the formal `.pth` checkpoint loader. Hist opponents continue to use UUID-authenticated formal checkpoints

### 📚 Interfaces and Documentation

- Removed legacy `async_infer` and worker-device controls from both CLI and WebUI; training device choices are now `auto`, `cpu`, and `cuda`
- Updated the bilingual READMEs, Quick Start, Feature Guide, Architecture, and Special Handling documents
- Updated the displayed framework version and `version.txt` to `3.4.0`

---

## [v3.3.1] ~ [v3.3.10] - 2026-06 to 2026-08

The development releases below follow Git commit order; higher versions are closer to v3.4.0.

### [v3.3.10dev] - 2026-08-30 · `85a244e`

- Unified AI, RuleBot, environment, and operating-system failures as abnormal episodes and rolled back all uncommitted trajectories from the affected game so partial samples cannot enter PPO updates
- Added a cross-platform single-Trainer file lock with PID/run_id ownership metadata, preventing multiple trainers in the same project from competing for models, ports, and temporary files
- Refactored WebUI background-process registration and shutdown. Zombie cleanup now targets only processes carrying this project's identity marker instead of scanning and killing every `python.exe` on the system

### [v3.3.9dev] - 2026-08-27 · `3a9c35a`

- Switched external PTH loading to restricted `weights_only` deserialization and reject unsafe globals, symlinks, non-regular files, invalid suffixes, and oversized files before loading. WebUI metadata inspection uses FakeTensor to avoid materializing full weights
- Organized the model repository into embedded-`model_id` pools. Import, overwrite, and packaging paths verify UUID, prefix, iteration, and artifact manifests; deployment packaging rejects mismatched PTH and ONNX iterations
- Added `.onnx.data` and `.artifacts.json` to complete ONNX artifact management, validating external-data paths, graph identity, and manifest consistency so real weights or files from another iteration cannot be omitted or mixed
- Introduced an independent `.gkg` package protocol version plus safe-filename, reserved-name, path-traversal, symlink, duplicate-member, compression-bomb, member-count, and size limits. Imports use staging and atomic installation

### [v3.3.8dev] - 2026-08-26 · `85b0027`

- Added configurable model prefixes while retaining `galatea` as the default. CLI accepts a prefix for new training, WebUI exposes it in the model-architecture area, and resumed runs inherit a read-only value from the checkpoint
- New training generates a random UUID `model_id`, while resumed training inherits its original identity. PTH, ONNX, and artifact manifests embed and cross-check UUID, prefix, and iteration; same-prefix/different-UUID files warn and cannot enter the historical pool
- Added `CHECKPOINT_FORMAT_VERSION`, maintained independently from the framework version. WebUI warns about protocol mismatches and training entry points reject incompatible checkpoints
- Split resume targets into absolute `--target-iteration` and relative `--additional-iterations` semantics in both CLI and WebUI, removing filename suffixes as an implicit source of the resume target
- Fixed historical ONNX input dtype adaptation by following the runtime-declared FP16/FP32 types; historical opponents are selected by the same model UUID and the embedded iteration

### [v3.3.7dev] - 2026-08-25 · `92f9a16`

- Explicitly switches to `train()` for PPO updates and restores `eval()` whether the update succeeds or raises, fixing retained mode state that caused unexpected VRAM usage and later inference behavior
- Standardized `deck_utils.get_random_deck_pair()`: success always returns five values and failure returns `None`, with training, Arena, and test callers updated together
- Made ONNX export a complete-artifact operation: graph and `.onnx.data` are saved, validated, and tagged together. Every checkpoint records export-in-progress, complete, failed, or disabled status, and historical matches use only complete artifacts
- Added centralized training-configuration validation for model/head divisibility, batch sizes, worker counts, timeouts, and PPO numeric parameters so invalid configurations fail before collection starts

### [v3.3.6dev] - 2026-08-25 · `f983d2d`

- Added iteration-scoped unique request IDs, shared completion IDs, and explicit success/error replies to central inference, preventing a timed-out request from reading stale results left by a previous request or iteration
- On ZMQ send/receive timeout or protocol mismatch, workers close the old REQ socket, reconnect under a new identity, and discard results with uncertain ownership. The ROUTER rejects duplicate, stale, and superseded messages
- Added one-click environment checking and repair to parse requirements, verify real imports, install missing packages, and configure portable-Python project paths; both the Windows bundle launcher and Linux setup flow use it

### [v3.3.5dev] - 2026-08-24 · `81dda52`

- Reordered resume initialization to load and validate the full checkpoint and network configuration first, strictly restore canonical weights before `torch.compile`, then restore optimizer, mixed-precision scaler, iteration, and training-step state
- Canonicalized compiled-model `_orig_mod.` keys only during export and reject missing, unexpected, or compiler-private keys in stored checkpoints
- Unified Arena action-candidate construction and response packing with training, fixing complex macro actions, padding-sentinel indices, empty legal-action sets, failed model loads, and inconsistent initialization return arity

### [v3.3.4dev] - 2026-08-24 · `641ec81`

- Added a rollout cursor that separates tentative from committed episode rows. Engine, parser, state-update, or AI failures roll back the entire episode; only terminal or validly truncated episodes commit, with observation/trajectory lengths checked
- Made feature encoding explicitly actor-relative. P1 perspective now swaps LP, hand, graveyard, banished-zone, and other global resources, removing P0/P1 cognition asymmetry
- Synchronized field snapshots from engine queries, prevented replacement cards from inheriting stale effect masks, and kept opponent face-down Extra Deck and banished cards hidden to block information leakage

### [v3.3.3dev] - 2026-06-27 · `175893f`

- Reworked semantic-knowledge mounting and caching to reuse loaded data, reduce duplicate concurrent Worker reads, and eliminate concurrent-read failures

### [v3.3.2dev] - 2026-06-25 · `434ab39`

- Merged the external card-effect-code semantic model into the existing hash-deduplication system and training features, avoiding duplicated semantics and split sources

### [v3.3.1dev] - 2026-06-02 · `d072b67`

- Added tracking for card effects activated during the current turn, corrected card-snapshot cleanup, and added location features for unknown cards to preserve necessary board context

---

## [v3.3.0] - 2026-06

### 🔧 Linux Platform Compatibility

- **🖥️ Linux build support**: Recompiled `ocgcore.so` with the latest YGOPro kernel, fully fixing GLIBC version mismatch load crashes
- **🐚 One-click setup script**: Added `setup.sh` — auto-detects CUDA, creates venv, installs deps. Supports `--train`, `--duel` modes
- **⬇️ Enhanced update tool**: `update_core_code()` now falls back to GitHub ZIP Archive when `.git` is absent (one-click package scenario)
- **🔧 GLIBC compatibility check**: `setup.sh` auto-detects the GLIBC version required by `ocgcore.so` and prints upgrade instructions on mismatch

### ✨ New Features

- **👻 Ghost byte parsing toggle (`--standard_core`)**: Both WebUI and CLI now have a "Disable Ghost Byte" option, unchecked by default (ghost byte parsing enabled). For self-compiled standard cores that experience parsing errors/corruption on messages 16/31, toggle this on to fix

### 🐛 Fixes

- Fixed `update_core_code()` failing on Linux when `.git` directory is missing
- Fixed default repo URL in `update_tools.py` pointing to wrong repository

---

## [v3.2.0] - 2026-05

### ✨ Major Architecture Upgrades

- **🧬 FiLM Global State Modulation**: New FiLM Generator dynamically produces per-layer scale (γ) and shift (β) parameters from global signals like current phase/turn/LP, enabling the network to have different inference tendencies at different game stages
- **🔀 SwiGLU Gated Feed-Forward Network**: Replaced all traditional MLPs (Linear→ReLU→Linear) with SwiGLU gated linear units, using bias-free design with Tensor Core 64 alignment, significantly improving non-linear modeling capability
- **⚡ ZMQ Zero-Copy IPC**: Replaced old pipe-based communication with ZeroMQ ROUTER micro-batching architecture, combined with pinned memory for extreme-speed DMA async transfers, cutting collection time in half
- **📦 ONNX Inference Acceleration**: Added `--use_onnx` option, automatically exports ONNX computation graph during training sync, Workers use ONNX Runtime for high-speed inference, drastically accelerating data collection

### 🔧 Functional Enhancements

- **🧵 Worker Thread Auto-Adaptation**: Dynamically adjusts worker behavior based on real-time system CPU/memory status
- **🔗 Chain Stack Learning**: Chain stack ordering and semantic information now included in training (chain_pos_embed + chain semantic slots), enabling AI to truly understand reverse chain resolution
- **📍 Position Sorting Learning**: Fixed the issue where card position sorting was not included in learning for #25 operations; now uses weighted position embeddings

### 🔧 Optimizations

- **⏱️ Worker Timing Audit Report**: Added detailed per-step timing statistics (inference/encoding/communication) for each Worker, with real-time performance bottleneck diagnostics in terminal output
- **📡 Read/Write Staggered Scheduling**: Data collection and storage phases are now stagger-scheduled to avoid I/O congestion causing frame drops
- **📚 KB Single-Injection**: Knowledge base data is injected only on first load, with subsequent iterations reusing cache directly to reduce memory thrashing
- Other underlying stability and detail optimizations

### 📦 New Dependencies

- `pyzmq` - ZeroMQ process communication
- `onnxruntime` - ONNX inference acceleration

---

## [v3.1.1] ~ [v3.1.3] - 2026-05

### 🐛 Fixes

- Fixed incompatibility where some cards "treated as a certain Level" failed in #23 message's level-matching logic
- Added weight pre-filtering logic for #23 multi-select packaging
- Fixed minor WebUI layout errors
- Fixed multiple hardcoded `"python"` calls in app.py causing one-click package environment failures, unified to use `sys.executable`

### ✨ New Features

- Auto version check: WebUI automatically compares remote version on startup and shows notification when a new version is available

### 🔧 Optimizations

- Revised project documentation (README CN/EN, acknowledgments section completed)
- Optimized Resource Sync Hub module layout

---

## [v3.1.0] - 2026-05

### ✨ New Features

- **🆕 Auto Version Check**: WebUI automatically checks for new versions on startup and displays a notification when one is available

### 🐛 Fixes

- Fixed incompatibility where some cards "treated as a certain Level" failed in #23 message's level-matching logic
- Added weight pre-filtering logic for #23 multi-select packaging
- Fixed minor WebUI layout errors

### 🔧 Optimizations

- Revised project documentation
- Improved Resource Sync Hub module layout

---

## [v3.0] - 2026-05 (Major Release)

### ✨ Major Additions

- **📦 One-Click Package & Portable Environment**: Built-in Python 3.11 portable environment, zero configuration, download and extract to use
- **🧪 Complete Documentation System**: Full Chinese illustrated documentation (Quick Start, Feature Guide, Architecture, Special Handling, Changelog) + English README
- **🌐 Online Dynamic Environment Builder**: YGOProDeck API integration, supports batch auto-fetching decks by tournament/format type, with background daemon for scheduled auto-updates
- **🧠 Virtual Mix Pool Builder**: Cross-pool mixing recipe system, creates arbitrary-ratio deck mix pools, cross-pool battles without physical file movement
- **📦 Model Deployment & Packaging Module**: `.gkg` format model package export and selective import, including models, knowledge base, and staple pool in one-click share deployment

### 🔧 Feature Enhancements

- **WebUI Comprehensive Upgrade**:
  - 🗃️ Assets & Deck Management: New staple pool config, holographic deck viewer, dynamic weight scheduler, online fetch module
  - ⚔️ Launch & Monitor Hub: New process management (PID tracking/emergency abort/zombie purge), live terminal dual-column view (full logs + alert extraction)
  - 👁️ Holographic Replay: SVG board rendering refactored, supports ghost X-ray effects, action arrow connections, real-time coordinate mapping
  - 📉 Training Manifold: Embedded TensorBoard directly viewable
  - 📈 Meta Dashboard: Multi-source merged analysis, dual deck PK comparison
- **142 Announce Pool complete rewrite**: RPN reverse Polish expression virtual machine rewritten, added three-layer filtering (RPN filtering + common-sense candidate pool + weighted priority sorting), crash prevention fallback
- **Multi-select & 142 Weighted Priority Sorting**: Intelligent priority ranking by card source (own deck > known hand > public zones > staple pool), greatly improving announcement accuracy

### 🚀 Performance Optimizations

- **Data processing optimization**: Significantly optimized CPU usage for message parsing and feature encoding
- **Holographic replay rendering optimization**: SVG layer rendering refactored, large file loading speed significantly improved
- **WebUI responsiveness**: Cache strategy and log truncation logic improved, no lag during long-running operations
- **Memory management**: Multi-process worker memory reuse improvements

### 🔧 Under-the-Hood

- **Algorithm logic refinements**: PPO training stability improvements, reward function fine-tuning
- **Linux environment support**: Provided `ocgcore.so` (not fully tested), expanding cross-platform capability

---

## [v2.8] - 2026-04

### ✨ New Features
- **WebUI Console**: Brand new Streamlit Web interface with complete training and management functions
  - 📈 Meta Dashboard: Real-time win rate stats and counter matrix
  - 📉 Training Manifold: Embedded TensorBoard monitoring
  - ⚔️ Launch & Monitor Hub: One-click training/arena
  - 🔄 Resource Sync Hub: Auto-update card DB and scripts
  - 🧠 Semantic Knowledge Engine: Lua script semantic parsing
  - 📁 Storage & Logs: File management
- **142 Announce Handling**: Staple pool fallback mechanism to resolve card announcement effects

### 🔧 Optimizations
- Bilingual Chinese/English UI support
- Real-time terminal log mapping

---

## [v2.7] - 2026-04

### 🔧 Optimizations
- **Training reward stability optimization**: Improved reward function, reduced training fluctuation
- **Reduced game step limit**: Lowered from 3000 to 2000 steps, faster training speed

### 🐛 Fixes
- Fixed memory leak after prolonged training

---

## [v2.6] - 2026-04

### ✨ New Features
- **League training mechanism**: Mixed opponent training (Self-play 60% + Historical models 25% + RuleBot 15%)
- **Async inference server**: Centralized GPU inference, VRAM usage reduced by 70%+

### 🔧 Optimizations
- Complete training framework refactoring
- Self-play matching logic upgraded

---

## [v2.5] - 2026-03

### ✨ New Features
- **Hand tracker logic**: Track known cards in opponent's hand

### 🔧 Optimizations
- Field reading processing optimization
- Card position encoding improvements

---

## [v2.4] - 2026-03

### ✨ New Features
- **Auto-update tool**: `python main.py update` command support
- **Semantic effect parsing**: Improved Lua script parsing capability

### 🔧 Optimizations
- Effect semantic capability enhanced
- Knowledge base structure optimized

---

## [v2.3] - 2026-02

### 🐛 Fixes
- Fixed card reading errors
- Fixed field cognition issues

### 🔧 Optimizations
- Reward distribution mechanism optimization
- Feature encoding stability improvements

---

## [v2.2] - 2026-02

### ✨ New Features
- **Card selection effect learning**: SELECT_CARD type effect learning support
- **Chain cognition**: Improved chain stack understanding

### 🔧 Optimizations
- Action space encoding optimization

---

## [v2.1] - 2026-01

### ✨ New Features
- **Material selection learning**: DFS algorithm for Synchro/Link/Xyz material selection

### 🔧 Optimizations
- Memory management optimization
- Multi-process stability improvements

### 🐛 Fixes
- Fixed various training bugs

---

## [v2.0] - 2026-01

### ✨ New Features
- **Card effect reading & parsing**: Lua script semantic parsing system
- **Universal model architecture**: No longer deck-dependent, truly universal AI

### 🔧 Optimizations
- Neural network architecture refactored
- Feature encoding system redesigned

---

## [v1.x] - 2025

### Initial Version
- Basic PPO training framework
- OCGCore environment wrapper
- Simple RuleBot battles

---

## Version Naming Convention

- **Major version**: Significant architecture changes
- **Minor version**: New feature additions
- **Patch number**: Bug fixes and minor optimizations

---

## Next Steps

- 🚀 Return to [Quick Start](quickstart_en.md) to begin training
- 📚 Read [Feature Guide](features_en.md) for usage
- 🔧 Read [Architecture](architecture_en.md) for framework internals
