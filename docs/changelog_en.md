# 📝 Changelog

> This document records the version update history of Galatea-Core.

---

## [v3.15.8] - 2026-10-31

### WindBot high-descriptor fix for Linux training Workers

- Replace direct `select.select()` in the Bot idle-connection check. On Linux it only accepts descriptor numbers below1024; numerous shared tensor storages can push new Bot sockets past that range. This explains independent duels working while multi-Worker training immediately rolls games back, and is distinct from exhausting the process's descriptor limit
- Use `selectors.DefaultSelector` for automatic backend selection, including epoll on Linux. Create/reuse one selector per connection and release it and sockets on registration failure, normal closure and errors rather than creating per-decision resources
- Preserve zero-wait readiness checks, packet bounds, receive deadlines, disconnect/unsolicited-response rejection and whole-game rollback. Public Core, WindBot executable/executors, shared-memory layout, network, rewards and PPO remain unchanged, as do Protocol4/Checkpoint3/revision14
- No Torch sharing-strategy switch, system-ulimit change or platform option. Existing Bot/Core artifacts and V4 checkpoints remain usable; restart training after updating
- Validation:115 related baseline checks, two real Windows both-seat Bot games with independent replay, and one real Worker collection/PPO update passed,118 total. Coverage includes idle-control draining, unsolicited responses, partial-frame deadlines, disconnects, floods, selector reuse and registration/close failures. A real Linux socket≥1024 regression was added but skipped on this Windows host. No WSL testing or claim of cloud12-Worker acceptance

See [WindBot descriptor limits](../integrations/windbot/README.md#linux-workers-and-high-descriptor-numbers).

## [v3.15.7] - 2026-10-31

### Bundled source-runtime files and Linux initialization fixes

- Explicitly allowlist the shared AnyCPU WindBot program/configuration, decks/dialogs, native Windows SQLite libraries, provenance, notices and manifest in Git. Build/debug outputs and old platform copies remain excluded. Runtime attributes preserve original bytes so checkout newline conversion cannot invalidate hashes
- Reuse the existing3.15.5 runtime and pinned upstream source without recompilation, executor changes or another private Core. Repository checkouts and source archives now contain the Bot program
- Linux setup reuses current Python3.10+/Conda or an explicit interpreter. It creates no venv, upgrades no pip and no longer hardcodes cu121. Existing Torch is pinned to its full version during requirements installation; broken imports or dependency conflicts stop instead of replacing it
- Ubuntu/Debian installs only missing Mono/class-library/system-SQLite runtime packages, using sudo for non-root users and no build tools. Other distributions reuse prepared prerequisites. `--skip-windbot` bypasses the optional module; training/Arena entry points never install system packages automatically
- `--check` is read-only and `--install-only` initializes without launching. pip and launch use the same interpreter. Only installation mode fetches missing card/script resources. Core load errors retain their diagnostics without upgrading glibc or compiling replacements
- No Linux ZIP, Windows portable-environment change or change to `--with-windbot` packaging. Bilingual setup/provenance guides updated; network, rewards, PPO, Protocol4/Checkpoint3/revision14 unchanged
- Validation:68 related checks passed; one real-Bot case was not enabled. Git Bash syntax/command-stub checks cover read-only behavior, current/explicit interpreters, Torch retention/temporary cleanup, missing runtime prerequisites, resource updates and argument forwarding. File hashes, Git allowlists/byte-preserving attributes, bilingual links/versions and existing packaging rules pass. No WSL, actual Linux initialization, apt installation or compilation was performed; Ubuntu22.04 needs target-machine acceptance

See [Linux setup](../README_EN.md#linux-users) and [Quick Start](quickstart_en.md#linux-users).

## [v3.15.6] - 2026-10-31

### Central inference transfers and separate WindBot statistics

- Pack active requests in arrival order into contiguous staging rows. Basic slicing retains pinned memory and avoids advanced-index allocations across108 fields. Transfer action/log-probability/value results together, and logits as one batch, instead of synchronizing individual Worker outputs. Request/completion IDs, sampling order and effective batch sizes remain unchanged
- Correct audit attribution: both self-play sides use central inference; historical CPU inference and WindBot refresh/reply waiting have separate timers. Central round trips include queues, batch waiting, transfers and forward computation, not just ZMQ transport. Timer totals are not full-game wall time
- Write WindBot games to `web_data/windbot_history_<run_id>.csv`, preserving mode, executor/level, model deck/seat, opponent build hash and genuine W/D/L. Aborts/truncations are separate. Exclude WindBot from ordinary pool rankings, matchup matrices and `Deck_WinRates`; existing CSV storage controls manage the files
- Add separate mirror/pool dashboard charts, executor and seat denominators, and pool-deck details. TensorBoard adds `WindBot_Mirror`, `WindBot_Pool` and mode-specific executor tags. No completed games means no win-rate zero point. Overall league metrics still describe the actual opponent mixture
- Show identifiable legacy `windbot:` mirror rows separately without inventing draws or build metadata. Legacy pool rows lacking opponent provenance cannot be reliably separated. Historical CSV/event files remain intact; appending ordinary legacy CSV preserves its column order
- Use existing version-aware width arguments for remaining WebUI calls, removing current Streamlit deprecation warnings
- No network, observation, reward, PPO, Protocol4, Checkpoint3 or revision14 change. No WSL build or test in this batch
- Validation:209 and30 related checks passed in separate, overlapping batches, plus real Windows WindBot collection/PPO. All108 fields, forward outputs and fixed-seed samples match exactly on CPU/CUDA. Bilingual dashboard checks cover WindBot-only/mixed legacy data, empty filters and startup failures without pool provenance. A warmed512/8/6 synthetic6-client ZMQ probe on5070 Ti improved from about143 to173–185 requests/s; this is not full-rollout throughput, learning improvement or a cold-start comparison

See [metric definitions](tensorboard_metrics_en.md#11-windbot-main-training-opponent-metrics) and [dashboard usage](features_en.md#meta-dashboard).

## [v3.15.5] - 2026-10-30

### League probabilities and cross-platform WindBot

- Review adds linked training/holdout selectors: removal moves to the other side, addition removes from it, and identical-build aliases move together; initially excluded unfinished/test scripts stay unassigned. An independent `windbot_build` source kit provides maintainer recipes. Runtime distribution uses one shared managed executable, with old platform subdirectories only as fallback; portable packages always exclude the build kit. The new common binary needs Ubuntu22.04 target-machine acceptance; this batch performs no Linux builds or tests.
- October31 review fixes false-empty WebUI executor lists: scans own and close a SQLite connection in the calling thread instead of reusing a global cross-thread connection whose failures were cached as missing cards. Recovering a formerly empty catalog restores suggested selections; ordinary refresh preserves manual choices. Scan counts and skipped-build reasons are visible without bypassing actual card/vocabulary checks or changing network/training actions. All102 related regressions pass, plus a real Windows CPU collection/PPO-update check.
- Exposes absolute RuleBot/WindBot/history/self probabilities: default `0.15/0/0.25/0.60`, enabled WindBot `0/0.15/0.25/0.60`, replacing only the base RuleBot share. Optional UI customization or CLI `--league-weights` requires a total of1 without implicit normalization. Each Worker draws once; empty history still falls back to self-play.
- Extra RuleBot coverage stays separate and prefers ordinary AI Workers in mixed mode; 100% WindBot adds no Rule games. Explicit legacy `--windbot-rate` retains remaining-share scaling and cannot be combined with the new argument. The UI emits only the four new values.
- Optional checkpoint `league_weights` and existing TensorBoard `Configuration/PPO` record the configuration; UI custom mode can read it. No new scalar tags or checkpoint format changes. Resume still requires explicit WindBot enablement and current local assets.
- Controlled games, independent imitation and main training automatically adapt to Windows/Linux without an OS selector or user build workflow. A companion prebuilt runtime package runs directly on Windows and through Mono/system SQLite on Linux. Users need no compiler; upstream project and decision sources remain unchanged.
- Linux uses Mono/system SQLite with a private per-game native-library map, cleaned on exit without changing Windows files. Legacy common `runtime/WindBot.exe` and advanced CLI path overrides remain supported. Missing dependencies are reported without automatic installation or opponent replacement; `--inspect` checks dependencies only, not game acceptance.
- Bilingual usage and release entries updated. Protocol4/Checkpoint3/revision14, network, rewards and PPO formulas remain unchanged; imitation and online-human learning still start independently.
- Validation: all218 final related regressions pass; the nine real-Bot gates skipped in that run were verified separately. Windows and Ubuntu24.04/WSL2 x86_64 pass both-seat Blue-Eyes, Blue-Eyes/Albaz mirrors, raw/observation replay, three model policies and abnormal cleanup. Both platforms complete central-inference single-Worker collection and PPO weight updates. Runtime-file hashes, default/optional portable inventories, bilingual links, syntax and version boundaries pass. No acceptance of every executor, other distributions/CPU architectures or cloud-GPU long training, and no claim of learning gains.

See [dependencies and runtime installation](../integrations/windbot/README.md#运行包与使用) and [metric definitions](tensorboard_metrics_en.md#11-windbot-main-training-opponent-metrics).

## [v3.15.4] - 2026-10-28

### Optional WindBot main-training opponents

- Adds `train --windbot` and a WebUI panel, reusing Workers, central inference, model candidates and ordinary PPO. Only model actions train; Bot replies are environmental transitions. No BC, expert demonstrations or human games are mixed in.
- Defaults to 20% Worker allocation, random model seats and mirror builds; regular model-side deck sampling is optional. Remaining Workers keep the original league. Mixed mode prefers regular AI Workers for Rule coverage; 100% uses only WindBot so small budgets are not consumed by Rule games.
- Suggests CDB/vocabulary-compatible Normal/Easy scripts, reserving one per multi-executor level and aliases sharing identical Main/Extra builds. Side differences do not create an independent BO1 build. Lists remain editable, with unfinished/test scripts excluded by default. Holdouts constrain WindBot training opponents only, not other deck pools or BC exposure.
- Shares each Worker's existing Core without extra GPU policies or experience pools; full replay recording stays off by default. Bots close per game. Disconnects, external Retry or parse failures roll back games without replacement opponents or invented outcomes. Forced cleanup checks registered child PID/creation time. Diagnostic-only results allocate no large merge pool and invoke no PPO.
- Checkpoints/TensorBoard retain active/reserved rosters, builds, levels, file/runtime hashes and launch settings. Adds trustworthy-completed-game win rates, completion/failure/truncation counts, accepted model rows, reply timing and per-executor metrics. Resume requires explicit opt-in; disabled paths do not import the optional module.
- Aligns bilingual guides, metric definitions and release entry points. Network/Protocol4/Checkpoint3/revision14, rewards and PPO remain unchanged; public Core and upstream WindBot source are unmodified.
- Acceptance: 672 regressions and 50 subtests pass in the full run. After updating four metric-count assertions, all 147 final related tests pass; four independent semantic rebuild gates remain disabled. A real 32/4/1 CPU model completes single-Worker pure-WindBot collection and PPO updates. Disabled defaults, bilingual UI, mixed Rule coverage, build holdouts, diagnostic-only handling without merge allocation and owned-child cleanup pass. No Linux/cloud-GPU long run or strength validation across all executors was performed; these checks do not establish learning gains.

See [configuration and boundaries](../integrations/windbot/README.md#optional-main-training-opponents)
and [training-opponent metrics](tensorboard_metrics_en.md#11-windbot-main-training-opponent-metrics).

---

## [v3.15.3] - 2026-10-28

### Independent imitation learning and fixed-opponent evaluation

- UI review: defaults now split different complete games from shared teachers; executor holdouts move to optional advanced settings, with a train-all option. Collection/ordinary WindBot/evaluation randomize Core starts by default, optionally pinned and always recorded; split seeds are separate. Existing replays require no executor import. Teacher seats, split goals, BC resume consequences and evaluation controls gain explanations/help.
- Imitation tab1 unifies checks, reconstruction, label extraction and splitting into one task; standalone expert-read is removed from everyday UI but retains its developer CLI/API. YRP diagnostics avoid redundant independent self-replay, producing reports without datasets when ineligible. Storage only manages files, adding source/batch/category selection for caches, captures, traces, reports, logs, BC models and TensorBoard, never following manifests into external-source deletion.

- Adds optional `integrations/imitation/` and `imitate collect/prepare/train/evaluate` CLI commands. Launch & Monitor Hub exposes separate teacher/data, BC training and evaluation tabs. Every stage is explicit; ordinary duels, main PPO, the league and online-human learning are not automatically connected.
- Freezes teacher executors/builds and split settings before collection. Supports episode, exact-build and executor isolation while retaining human/unknown identity associations. Explicit executor holdouts keep identical builds/declared families within roles, and duplicate openings always stay together. Upstream levels are declarations, not measured ratings; final test data is not monitored each epoch or used for checkpoint selection.
- Native WindBot mirrors directly cache both teachers' actual observations and unique labels using one Core per game. Complete legacy Link captures generate examples during one original-output reconstruction; epochs never replay Core. Canonical traces reuse strict expert reading. Unmapped actions do not inject answers; failed/nonterminal games do not publish partial caches.
- Compressed numeric NPZ blocks deduplicate static deck profiles and verify digests without pickle. Decompression is capped at32MiB/block and compressed blocks at2GiB/duel. Relative dataset references permit workspace relocation. BC requires matching vocabulary/semantic input protocols, not identical Core binaries across operating systems.
- Teacher-balanced cross-entropy supervises policy; forced choices contribute neither loss nor accuracy. Defaults are5 epochs, batch64 and learning rate1e-5. Critic, auxiliary and planning-target decoders receive no direct BC supervision, while shared policy representations can change. Optimizer, new UUID, experiment outputs and10 TensorBoard metrics are separate; action imitation is distinct from playing strength.
- Retains checkpoint format3 with `training_stage=imitation`, dataset/provenance and epoch counters. Explicit BC resume restores its optimizer/scaler/RNG. BC cannot use PPO `--resume` but explicitly initializes a fresh PPO weight branch via `train --init-from`. Online WindBot Workers remain a later batch.
- Naming/continuation review: `galatea_iter_<cumulative PPO>_bc_<cumulative BC>.pth` retains independently advancing counters and parent provenance. Main training, Arena, model storage and packaging include BC/branch directories. PPO initialization creates a new UUID/directory and optimizer/update/rollout counters without modifying BC. PTH/ONNX artifacts retain/check lineage. New replays display full names; independent evaluation recordings default to model-side state predictions, separate candidate/baseline directories and unified replay discovery. Old recordings are not recomputed.
- Evaluation freezes actual held-out builds, Core seeds, model policy and paired seats for candidate/baseline comparisons. Reports outcomes, failures and descriptive Wilson intervals. WindBot RNG remains uncontrolled and paired games are not IID evidence. Evaluation-only captures are excluded from learning.
- Aligns bilingual documentation and release entry points to3.15.3. Model Protocol4/Checkpoint3/revision14, network, rewards and PPO remain unchanged; public Core and upstream WindBot source are unmodified.
- Acceptance: 655 regressions and 50 subtests pass in the full check. After adapting four packaging test fixtures to the new model-directory argument, all 131 final related tests pass with real Core/WindBot enabled. Coverage includes BC→PPO→BC counter round trips, exact initial PPO weights, fresh optimizers, unchanged BC files, PTH/ONNX lineage pairing, bilingual UI and prediction recordings in real evaluations. Native/reconstructed input parity, BC resume, best-checkpoint selection, batch cleanup and external-source preservation pass, along with syntax, documentation links and default portable-package scanning. Four independent semantic rebuild gates remain disabled. No Linux/cloud-GPU long training or human-online wiring was performed; small engineering checks do not establish stronger play.

See [Independent imitation learning](core_trajectories_en.md#3153-independent-imitation-workflow) and the [imitation metric reference](tensorboard_metrics_en.md#10-independent-imitation-metrics).

---

## [v3.15.2] - 2026-10-21

### Expert samples from complete traces and identity association audit

- Adds a context-managed read-only expert API in existing `trajectory_dataset.py`. One strict Core replay reuses the canonical interpreter, original candidate labels and V4 Encoder to expose CPU observations, unique action indices and physical-seat/provenance/digest metadata. Defaults to explicitly declared WindBot seats; human/model/rule examples require explicit selection. Losses are not automatically excluded.
- Freezes a bounded source copy and optionally checks expected file SHA-256. Provenance, assets, terminal, raw output, parse coverage, unique mapping and observation-digest gates all precede iteration. No expert-answer injection, examples from automatic empty chains or invented PPO probabilities/advantages.
- Anonymous temporary numeric storage is capped at8MiB/sample and512MiB/duel, read without pickle into individual CPU observations and deleted on context exit. Exhaustion rejects access rather than releasing a staged prefix. Normal PPO/Arena/WindBot execution uses neither the cache nor extra inference.
- Adds `expert-read` CLI and WebUI section4 under Trajectory Ingress & Quality: expert-seat selection, separate offline verification, sample/prompt counts, input specifications and label previews. Section5 manages JSON expert reports, not persistent tensor archives; deletion does not affect source traces/models.
- Dataset manifests add declared participant kinds, per-seat original decision counts and program-identity association audits. Actual splits still isolate all stable identities. Hypothetical counts without program edges are diagnostic only; human/unknown/conflicting declarations remain conservative, without manufactured independent validation.
- Reads322 WindBot examples out of520 decisions in5 existing native-model/Blue-Eyes duels and471 examples from an Albaz mirror. All6 traces pass strict replay with byte-exact original output. Even without program edges, the5 Blue-Eyes duels remain1 group through the shared build; isolation is not relaxed.
- Bilingual public documentation and release entry points move to3.15.2. Protocol4/Checkpoint3/revision14, trajectory2, GKG4 and static semantic assets1 stay unchanged. No network, reward, PPO/league, public Core or upstream WindBot changes; no imitation training or complete single-player online Link inference is claimed.
- Complete regression passes629 tests and50 subtests, with4 independent semantic-asset gates disabled. A subsequent41-test pass covers added boundaries, real expert reading, source isolation, bilingual WebUI and documentation: wrong digests, absent terminals/default experts, masked labels, non-finite inputs, cache limits, withheld failed prefixes, interrupted-reader cleanup and mixed program/human declarations.

See [expert reader usage, cache lifecycle and audit semantics](core_trajectories_en.md#3152-expert-samples-from-complete-verified-traces).

---

## [v3.15.1] - 2026-10-21

### Core-owned player-view ingress and diagnostics

- Adds the independent `integrations/player_view.py`: BO1 physical seats, Start4, full-zone6/single-card7 refreshes and bounded YGOPro framing. Host refreshes remain separate from original Core events/traces. Future Link reuses Core's interface without being a runtime prerequisite.
- Explicit query-cache modes reuse the existing binary decoder. Sparse fields/known bits distinguish unknown from zero. Entire refreshes apply atomically; unsupported flags, invalid slots, truncation and reordering fail closed. Movement/shuffle/reload invalidates instance caches without same-code state resurrection.
- Queries never establish hidden Main Deck order; opposing hand/face-down identities and dynamic values are masked again. Grave restrictions apply to the actual viewer. Only the own stable Main/Extra profile is accepted, never opponent builds/Side.
- Diagnostic projections reuse existing DuelState/V4 data structures without finalizing live transitions; empty-chain automatic replies create no decisions. Default V4 encoding remains gated: this single-player profile cannot prove all opponent response boundaries, Type160 hints or attribute coverage. Reports never claim complete observations or learning eligibility.
- WindBot reuses Core's builders/visibility filter without extra player-state adapters, networks or inference. Native training, weights, Protocol4/Checkpoint3/revision14, PPO and full-trace gates remain unchanged.
- New/old visibility bytes match in8,980 seat-message comparisons across6 existing captures. Adds framing, malformed input, cached/uncached fields, privacy, lifecycle, diagnostic encoding and real Core both-seat query tests.
- Complete regression passes613 tests and50 subtests;4 independent semantic-asset gates are disabled. Covers real Core/WindBot, Blue-Eyes/Albaz mirrors, all three model policies, script-initialization counts, trace replay, source isolation, ingress privacy and abnormal cleanup. Protocol boundaries, syntax and bilingual documentation links pass. No actual Linux networking, complete single-player online V4 inference or imitation training was performed.

See the [player-view SDK](core_trajectories_en.md#3151-core-owned-player-view-ingress) for usage, completeness and cost.

---

## [v3.15.0] - 2026-10-13

### Optional external sessions and controlled WindBot duels

- Core owns contract v1: independent UUID, monotonic request ID, physical seat and actual response submission. No Link prerequisite or change to Model Protocol4 / Checkpoint3 / revision14 or existing trace formats.
- Vendors the unmodified IceYGO/windbot `master` snapshot `dec0535f8669dca01457a898890a968139711b19` under `integrations/windbot/`, preserving MIT/dependency notices and `UPSTREAM.json`. Original adapters are separate; WindBot decisions and public ygocore remain unmodified.
- Explicit CLI/WebUI activation supports native RuleBot/V4 PTH versus WindBot, two Bots and swapped seats in controlled BO1. One native Core is authoritative; derived masked client messages are separate from full raw capture. Loopback peers use room tokens, deck checks, serialized responses and bounded logs. Disconnect/timeout/Retry/unknown external messages preserve an abnormal prefix without replacement opponents or invented outcomes.
- Reuses the Link SDK and optional independent conversion/replay with participant provenance retained. No PPO samples, league/reward/network changes, imitation optimizer, human wiring or complete YRP sidecar ingestion.
- Default portable packages exclude vendor/runtime/build; `--with-windbot` includes installed runtime and source/licenses without implicit downloading/building. Ordinary features do not depend on the optional directory.
- Anchors the root deck ignore to `/decks/`, preventing Windows from excluding upstream `vendor/Decks/` and `Game/AI/Decks/`. All 294 snapshot files remain trackable; build outputs remain ignored.
- Scope: P0 starts; Core seeds do not control Bot randomness. Native models use CPU and uniform canonical macro pools, not full training Pass1 equivalence or fixed benchmark semantics. Initial Windows closed loops pass; Linux, every executor and long sessions need separate acceptance.
- Acceptance (October15): the real Core/WindBot full regression passes554 tests and50 subtests, with4 independent semantic-asset gates disabled. Coverage includes real Blue-Eyes/RuleBot both-seat sessions, Blue-Eyes/Albaz mirrors, response-label contracts, UI reactivity, deck sources, Holographic Replay/prediction provenance, bilingual displays and abnormal cleanup. An actual mirror reaches mode0/P1 Type23; original output/decision observations replay independently, abnormal prefixes remain ineligible and Bot processes are reclaimed. Default/optional file scans, syntax and local documentation links pass. No Linux live run, cloud long training or imitation optimization.
- The WebUI reads executor/compatible V4 model lists and reacts immediately to side-type changes. Native opponents reuse weighted/physical/virtual/single-deck Arena sources, defaulting to the Bot's exact build. Independent Holographic Replay and native-model state-evaluation options supplement always-on raw capture; external actions show actual selections without fabricated confidence.
- October15 adds WindBot-side Type15 label equivalence: the two Albaz end-phase discard misses are index permutations, not pool truncation; all28 combinations already exist. Validate first, compare physical-index sets, and let Core define an explicit versioned per-seat comparison contract for future Link reuse. Actual responses/default matching are not reordered; ordered prompts are not relaxed and unknown contracts fail closed.
- Re-converting the original Albaz capture now uniquely maps all471 decisions, fixing the two labels to pool slots6 and26. Real terminal/exact raw output pass; actual responses and every observation digest are unchanged. Existing canonical files retain their old rule: re-convert raw capture and rebuild quality/split manifests. Unique matching, the120-slot boundary and all other whole-duel gates remain; no expert-answer injection or training is started. See [interface and label boundaries](../integrations/windbot/README.md#原始重放与学习标签等价性).
- October20 exposes native V4 policy/deployment-temperature controls in WindBot WebUI, reusing the existing CLI/sampler and preserving the training-style T=1.0 default. WindBot and normal Arena settings are unaffected. Results add effective `native_policy`; replays retain existing policy metadata. Documents the three play/conversion/replay script initializations, adds scoped stage logs and separate `verification_seconds`; script caching, public Core, network and training are unchanged.
- October20 related regressions pass112 tests: bilingual policy/temperature reactivity and CLI forwarding, all three real-model policies, three verified/one unverified script-preload counts, Blue-Eyes/Albaz controlled sessions and cleanup, ordinary trajectory/Arena regression and documentation checks. This is separate from the October15 full-suite result above.

### Type23 responding-player correction

- Sum-selection payload begins with mode, then player. Mode-derived seats could give P1 material choices the P0 view or mode1/P0 the P1 view.
- State, Worker rule dispatch, self-check, Arena, Link/YRP ingestion and canonical replay now share strict player decoding. Both modes/seats are tested. Shapes, parameters and PPO are unchanged; actual decision ownership/observations are corrected. Old wrong-actor labels are not silently rewritten.

See [module build and usage](../integrations/windbot/README.md).

---

## [v3.14.1] - 2026-10-11

### 🧩 Release and compatibility boundaries

- Groups the still-uncommitted decoding/export, UI and documentation fixes following 3.14.0 deployment into **3.14.1**.
- Aligns framework, WebUI, README, dependency and setup versions. Model Protocol4 / Checkpoint3 / revision14 / static assets1 / GKG4 / raw traces2 / holographic format3 remain unchanged. Networks, inputs, asset identities, training settings and Arena policies are not changed.

### 2026-10-13 Optional external-module boundaries

- Adds `integrations/windbot/README.md` with IceYGO/windbot provenance, MIT licensing and requirements for source snapshots, pinned commits and local patches. Git excludes local runtime/build outputs. No third-party source, dependency or working integration has been introduced.
- Bilingual roadmaps and V4 change control establish Core-owned project interfaces with later Link reuse. WindBot is independent and optional; controlled capture, provenance evidence and isolated imitation experiments precede training/league integration. Small UI/chart changes remain in existing files. Framework version and all protocol identities are unchanged.

### 🩹 Cloud-feedback parser and export hotfix

- **Type160 legal boundaries**: subtype0 clears only scalar hints, leaving independent description references. Public-Core initialization at `PLAYER_NONE(2)/location0`, and other legal empty zones, creates no visible host/history/holographic event. Real20989253 initialization and `SetHint(0,0)` reproduce the old rejection; this is not a missing dependency or broken deck. Strict nine-byte/subtype/controller validation remains, with field-specific diagnostics. Raw packets are retained.
- **Type142 alignment**: pure interpreter `card_announcement.py` treats `[1]` as constant true, not a whitelist. Adds arithmetic/negation/division toward zero and preserves raw Core bit-intersection results for type/race/attribute. Removes invented Level/Link opcode semantics; validates stack shape and packet length. Cached real metadata excludes missing cards, aliases and Tokens without removing legal low codes or truncating complete identities.
- **Candidates and visibility**: legal small-pool entries retain sources, ordering and existing caps/weighted reduction. An empty pool searches the database under the same predicate; legality is never waived. The actual responding player supplies the own deck; hidden opponent facedown-banished identities cannot populate/prioritize candidates. No legal match fails through existing abnormal-game rollback rather than inventing valid options.
- **ONNX tracing warnings**: only export skips Python shape assertions in Deck/Event Encoders; normal PyTorch inference/training retains validation. Empty-deck/event masks remain live tensor computations, not trace-time constants. ONNXRuntime `CanUpdateImplicitInputNameInSubgraphs` warnings remain: unsafe name-collision rewrites are skipped without disabling all optimization, changing graph semantics or suppressing warnings globally. Existing ONNX files need not be re-exported solely for these warnings; subsequent normal checkpoint exports use the fix.
- **Learning and resumption**: PPO/GAE/rewards, network parameters/input shapes, Model Protocol4/Checkpoint3/revision14 and asset identities are unchanged. Correcting formerly invalid candidates can change choices without migrating weights. Database expansion runs only when a142 small pool has no match, with no new per-step large semantic/trajectory allocations. Restart processes and resume the last saved compatible checkpoint; already-running cloud Workers are not patched live, and unsaved iterations are not automatically retained.
- **Hotfix acceptance**:448 tests and50 subtests pass with real public-Core gates enabled;4 independent rebuilt-asset gates remain disabled. An additional74 targeted hotfix/bilingual-document tests pass. Covers real160 initialization/clearing,9 real142 response/rejection cases, arithmetic/setcodes/masks, invalid packets/constrained fallback, legacy/default ONNX export, empty/nonempty deck/public-event CPU parity, normal shape rejection and PPO/trajectory/deployment paths. Comparing pre-fix encoders separately at512/8/6 gives bitwise-identical outputs and gradients across all313 parameter entries for empty/nonempty contexts. First-time metadata caching for14,981 cards measures0.103s including tracemalloc,4.61MiB retained/5.95MiB peak per process; this is not a whole-training benchmark. Portable scanning includes the new module; syntax/whitespace checks pass. No Core modification, asset rebuild or cloud long-run test.

### 📖 Public documentation and deployment UI corrections

- **142 announcement add-on pool**: aligns WebUI, quickstart, features and special-handling names, explaining how it extends known candidates and distinguishing it from constrained database expansion on an empty pool. Candidate generation, priorities, reduction, caching and file formats are unchanged.
- **Package names and versions**: WebUI defaults and examples derive the V4 label from the current model-protocol constant; the interface separately displays framework, model protocol and GKG format. CLI removes its stale V3.0.0 title. Embedded identity and validation rules are unchanged.
- **Public metric reference**: both languages retain all 80 fixed scalars, 3 distributions and configuration text, with clearer purposes, units, formulas and trend interpretation. Single-run statistics and attribution move to a separate development analysis record; README does not add development-plan navigation. Public architecture/functionality and source links remain, without changes to networks, training settings, protocols or checkpoint versions.
- **UI/documentation acceptance**:189 tests pass across bilingual add-on editing, protocol-derived default/custom names, unsafe-path rejection, documentation links/anchors/metric coverage, deployment artifacts and the preceding real-Core/ONNX hotfix. Syntax/whitespace checks pass; no additional cloud training was run.

Chart definitions: [TensorBoard Metric Reference](tensorboard_metrics_en.md). That experiment's statistics: [Initial V4 cloud training analysis](training_analysis_v4_initial_en.md).

### 📈 Holographic prediction display fixes

- **Persistent visibility**: replaces the default-collapsed prediction panel with a stateful toggle that persists across pauses, stepping, seeks and replay changes. Axes, legends, hover labels and terminal references follow the interface language.
- **Playback position**: both charts use the slider’s zero-based steps. A dashed cursor follows playback; distinct colored dots mark each side’s latest prediction source. Win probability uses0–100%; remaining events use a separate nonnegative numerical axis.
- **Separate figures**: P0/P1 retain independent latest predictions, source steps and frame ages through events/pauses. No future/opponent values are substituted, no model is rerun, and retained historical predictions are not presented as fresh inference.
- **Coverage diagnostics**: shows counts/spans, explicitly model-sourced decisions without predictions and rejection reasons. Normal gaps are not missing-data failures; RuleBot/old recordings are explained. Actual outcomes remain separate, and recordings beyond5,000 points avoid chart row limits.
- **Compatibility/scope**: prediction/recording formats stay unchanged, including early decisions-list support; only replay display and read-only validation change. Secondary-stage greedy Cancel improvements remain under discussion; masks, cancellation, soft bans, rewards and training are not modified.
- **Regression acceptance**:468 tests and50 subtests pass with real public-Core gates enabled;4 independent rebuilt-asset gates remain disabled. A subsequent22-test pass covers legacy chart API compatibility and final documentation/UI checks. Coverage includes localized axes/tooltips/cursors, retaining past values on events, no future leakage on backward seeks, pause/step/replay changes, rule/malformed records,5,500-point charts and existing PPO/ONNX/trajectory/deployment paths. Actual Game4 (2,205 frames/1,090 predictions) and Game6 (502 frames/221 predictions) contain no missing or invalid predictions; Game6 renders successfully in the patched real Streamlit panel. New modules enter portable scanning; syntax/whitespace checks pass. No additional cloud training or manual browser visual acceptance.

See [prediction curves and playback](core_trajectories_en.md#prediction-curves-and-playback-position).

### 📈 2026-10-12 Arena per-frame state predictions

- New recordings evaluate each model side on visible-event/decision snapshots, including opponent actions and terminal events. No invented RuleBot curve or legacy recomputation. Optional same-frame multi-view fields preserve individual format1/holographic format3 compatibility.
- Reuses existing state/planner weights through an entry that skips the action tower. Existing actor predictions are reused and non-actor candidate lists excluded. No new parameters or changes to sampling, rewards, PPO/GAE, standard ONNX or model/checkpoint identities.
- Events use consumed-message mirrors; decisions retain original query correction. No extra Core queries or future-state reads. Hover identifies snapshot sources and coverage accounting follows recording mode. Actual outcomes are separate per physical perspective, never replacing predictions or forcing final points to100% win/loss.
- Extra state-backbone inference and prediction storage affect only logged games with evaluations enabled. Disabled recording adds no inference; training Workers stay unchanged. Replay viewing needs no network; portable scanning includes the new module.
- Acceptance on2026-10-13:493 tests and50 subtests pass with real public-Core gates enabled;4 independent rebuilt-asset gates remain disabled. Fixed seeds/swapped seats preserve every decision, outcome, step count and Core query count compared with recording off; canonical traces replay completely. Two models produce two predictions per frame; rule-opponent games produce only the model-side prediction. At512/8/6, CUDA FP32/BF16 state representations, values and terminal predictions are bitwise-identical to the original full forward; parameter keys/shapes stay unchanged. Tests cover absent candidate inputs, unchanged random streams, nonfatal evaluation failures, separate terminal references, legacy compatibility and real Streamlit rendering. Syntax/whitespace and portable-module checks pass. No WindBot integration, Core modification or cloud long-run training.

### 📊 2026-10-12 dashboard matrix display hotfix

- **Cause/fix**: growing deck counts exceed the Pandas Styler cell limit, not training-data/VRAM capacity. Keeps the full pivot statistics and paginates styling at up to50×50 by default, respecting any lower environment budget without raising global Pandas limits.
- **Data/controls**: independent own/opponent pages, stale-page clamping after filters, full CSV export, original0–1 rates and unknown-matchup blanks. A fixed0–100% color scale keeps rates comparable across pages. Small matrices retain full percentage views; a disabled Styler uses plain raw-rate tables.
- **Scope/verification**: WebUI-only fix with bilingual descriptions; release remains3.14.1. Training, networks, protocols, evaluation sampling and greedy decisions stay unchanged.14 targeted tests pass, reproducing560,250 cells, bilingual paging, low/zero limits, changed filters, full CSV and unchanged source data.42 tests pass with prediction replay, deployment forms and bilingual documentation included. Portable scanning includes the new module; syntax/whitespace checks pass. Auxiliary heads and secondary-choice mechanisms are not modified.

---

## [v3.14.0] - 2026-10-11

### 🧭 Full V4 main-framework deployment and CUDA semantic aggregation

- **Deployment milestone**: consolidates V4 development across 3.7–3.13 and enters full main-framework deployment/training validation. Exact vocabulary, visible state/actions, static Lua semantics, Main/Extra profiles, Deck Encoder, layered FiLM, ordered events, structured auxiliary heads, goal planning and PPO update guards are connected. This release does not redesign the network again; deck-building/BO3, imitation optimization and MCTS remain future work.

- **Large temporary allocations**: CUDA categories/setcodes/card references/races/attributes use direct sums over the original shared weights instead of expanding the reference axis. For512 samples/75cards, measured forward peak increment changes2.9297 →0.6123GiB; output remains FP32.
- **Learning contract preserved**: no means, deduplication, slot removal or frozen weights. Repetition counts, nonzero padding forward/zero padding gradients, original reference clamping and masks remain intact. CPU/ONNX/nondefault Embedding configurations retain the original operator. Inputs, parameters, asset identity and PPO/GAE/auxiliary/planning objectives do not change; compatible V4 checkpoints still resume.
- **Capacity boundaries**: complete512/8/6 PPO repeats three updates each at128/256 without live-memory accumulation. Whole-network allocated peaks are about0.9% higher, not reduced by the isolated-operator ratio; post-update reserved memory is about0.9–1.0GiB lower. Eliminating this allocation does not guarantee32GiB/batch512 capacity. No automatic batch reduction, gradient accumulation or semantic recomputation; cloud long-run acceptance is not claimed.
- **Memory-optimization acceptance**:33 targeted tests pass; the full suite with real-Core gates enabled passes375 tests and50 subtests, with4 independent rebuilt-asset gates not enabled. Coverage includes512/8/6 BF16 outputs/all307 parameter gradients, real CUDA PPO, padding/repeats/AMP/input and protocol identity, ONNX and raw-trajectory replay. Dynamo eager full-graph capture passes; Linux Inductor is not accepted. Compilation/whitespace checks pass. Bilingual architecture documents record measurement scope and limits; tests retain their existing ignored-directory convention.
- **Bilingual documentation audit**: preserves the user's revised Chinese wording and aligns English. Updates applicability, V4 stage status, network/layered-FiLM diagrams, prediction replay, categorized trajectory management, semantic continuation/provenance and deployment instructions. README separates release and model-protocol badges while retaining concise navigation and consistent icons. Removes stale claims that predictions are not recorded and that the old broken semantic library is the current asset state; preserves historical evidence and rare-message limitations.
- **Training/environment guidance**: documents pre-backward KL checks, retention of completed updates, stopping the remaining joint updates and normal collection on the next iteration; adds effective-epoch examples and TensorBoard metrics. Installation examples use `python -m streamlit` and explain Python/PyTorch support, the Linux helper's fixed cu121 path and large V4 batch capacity. Learning settings and installation behavior are not changed.
- **Cloud feedback scope**: the user reports temporary short-run stability with the same 32 GiB cloud configuration after the patch. This is not long-run capacity or playing-strength acceptance; peak/cross-iteration monitoring and fixed Arena benchmarks remain required.
- **Independent TensorBoard reference and cloud analysis**: documents axes, frequencies, denominators, source/design links for all 80 fixed scalars, 3 planning distributions and configuration text, excluding individual deck curves. Reads the supplied events through 79 collections/78 PPO phases and compares win rates, rewards, future prediction, KL and actual collection budgets. Explains possible semantic-asset, observation, auxiliary/planning and GAE contributions without equating training progress with generalization acceptance. Records duplicate timeout counting, unweighted Worker means and incomplete abort-counter coverage; documentation only, no training changes.
- **Release recheck**:50 tests and7 subtests pass across document links/heading anchors, release/protocol versions, PPO/KL, semantic aggregation and portable packaging; syntax and whitespace checks pass. Existing user edits are preserved, with no learning-setting, network or environment-installation behavior changes.
- **Versions**: framework, WebUI, README and portable-package version source move to3.14.0. Model Protocol4/Checkpoint3/revision14/static1/GKG4/raw trajectory2/holographic replay3 remain unchanged. Compatible V4 checkpoints resume with matching vocabulary/semantic identities; V3 is not migrated.

See [CUDA semantic allocation boundaries](architecture_en.md#direct-cuda-semantic-aggregation-2026-10-10).

---

## [v3.13.5] - 2026-10-09

### 🎞️ External trajectory ingress and isolated quality gates

- **Link capture SDK**: independent capture format1 separates full server and player-view data, preserving raw Core chunks, actually submitted responses, physical seats and stable anonymized IDs. Full ingestion verifies original assets/output/response/terminal alignment; partial views never invent seeds or hidden decks. Core-side implementation is complete; external Link wiring/live acceptance remains a separate integration
- **Bounded YRP reader**: standard YRP1, extended-header-v1 YRP2 and strictly framed YRP3D; no magic scanning or ignored residue. File/decompression/dictionary/response limits reject malformed input. Tag, single-script, unknown extensions and ambiguous containers fail explicitly. Non-UNIFORM legacy files remain inspectable without switching Core modes
- **Public Core reconstruction**: YRP1 transforms the header seed through the first MT19937 output; YRP2 preserves eight seed words via existing public create_duel_v2. Stored array order, initial positions and rules are retained. No Core source/binary edits, global reset monkey-patch or parked Replay Worker activation
- **One interpreter/label boundaries**: existing MessageParser/DuelState/Encoder/response packing is reused; no skipped cancels, guessed Retry recovery or expert-answer injection. Offline uniform-prior macro pools use local fixed RNG and existing120-candidate limits. Unmatched/ambiguous responses remain diagnostic, not labels. Ordinary YRP cannot gain imitation eligibility from reconstruction alone without historical assets/raw-output evidence; no fabricated PPO statistics or optimization
- **Dataset gate**: independent manifest format1 deduplicates complete duels and transitively groups exact Main/Extra/Side compositions and stable players. Never splits adjacent states, uses filenames/nicknames as identities or forces shared groups into separate holdouts. Invalid/truncated/nonterminal/unverified/divergent data is quarantined; insufficient groups/empty holdouts warn. Similar archetype-family clustering is not claimed
- **CLI/WebUI**: replay-ingest source/inspect/conversion/report commands plus dataset-build. Storage & Logs → Trajectory Ingress & Quality offers read-only checks, managed background tasks, quality counts/rejections and capture/trace/report/manifest management. Existing managed tasks are not replaced; outputs do not overwrite. Failed diagnosis seals a prefix and returns CLI status2
- **UI cleanup/retired code**: four numbered sections separate raw inspection/conversion, canonical audit/splitting, results and categorized file management. Input/output explanations, rejection-code help and question-mark tooltips accompany controls. Four collapsible file categories and the shared manager now display the previously omitted title, directory, suffix and count; deletion/clearing labels identify category and nonrecursive scope, correcting the misleading all-files warning. YRP/Link selection state is separate to avoid empty-source selection carryover; downloads use compatible width arguments. Unreferenced `yrp_parser.py`/`replay_worker.py` are removed; new ingestion, training and Core remain unchanged
- **Versions/cost**: framework3.13.5; Model Protocol4/Checkpoint3/revision14/static1/GKG4 unchanged. Network, observations, rewards/PPO/GAE, deployment outputs and normal training remain unchanged. Manual offline tools carry their own CPU/I/O cost without enlarging normal rollout storage, central inference or parameters. dev/main release policy is untouched
- **Real-sample checks**:13 local files parse;4 non-UNIFORM samples remain diagnostic-only. YRP1/YRP2 samples consume1,137/809 responses through genuine terminals, with zero Retry/full parsing and724/389 observation digests plus413/420 automatic responses passing independent replay. Incomplete macro mappings and historical evidence keep eligibility false. Synthetic real-Core full Link capture passes conversion/gates; ordinary Arena recording-on/off paired regressions pass
- **Regression evidence**: full suite334 tests and50 subtests pass; four pre-existing new-semantic-asset gates remain default-skipped. After adding corrupt/duplicate-file and real Streamlit render checks,25 ingestion/canonical-trajectory tests pass again. Portable collection includes all new modules; compilation/whitespace checks pass. Tests retain the repository's existing ignored-tests development convention
- **UI patch checks**: seven real Streamlit interaction tests pass across Chinese/English, covering sections/help, source switching, reports, background commands and busy-task guards. Actual batch deletion/clearing in test-owned directories preserves other categories, suffixes and subdirectories.29 ingestion/canonical-trajectory tests pass, including real-Core conversion/replay. Portable collection retains new ingress modules and excludes retired ones; compilation/whitespace checks pass. Manual browser visual acceptance is not claimed

See [ingress, Link contract and isolation](core_trajectories_en.md#3135-external-ingestion-and-isolated-quality-gates). No imitation optimizer, YRP prediction backfill or counterfactual search; live external Link end-to-end and production long training are not claimed.

---

## [v3.13.4] - 2026-10-09

### 🧬 Code-source provenance and trusted incremental semantics

- **Confirmed cause**: old27,647 vectors had530 distinct values and27,007 identical rows;26,942 lacked raw code and65 were empty. The cached encoder's empty-text output matched that group. Empty-source fallback plus key-only reuse perpetuated the defect; coarse hashes did not directly merge embeddings
- **Source extraction**: token/block parsing covers registration-only effects, SetValue, Clone overrides and uniquely resolvable local/cross-card/utility/procedure callback closure without duplicated Operation. Actual literals remain intact; source/block/dependency identities and unresolved references are recorded. Lua is never executed and descriptions do not define effect order
- **Independent coarse classification**: normalization rules are unchanged; only duplicate/stale registry members are cleaned. Full comparison with the prior committed parser found zero slot/category/requirement differences and2,156 classes. Cards without independent effects receive no artificial slots
- **Content-aware reuse**: changed same-key code/dependencies regenerate; unmanifested legacy vectors fully rebuild, missing/empty sources cannot generate. Actual encoder weights/config/tokenizer and pinned revisions establish identity; temporary padding/truncation does not trigger rebuilds. Source-cleaned bundles may only reuse certified identical-source vectors
- **Complete long-code coverage**:256-token windows/32-token overlap with role/block/part prefixes; new-token-weighted mean and L2 retain384-dimensional output. Bounded generation groups avoid retaining every expanded long input; no new network or per-step generator
- **Maintenance/asset chain**: read-only CLI `semantic-check --directory` and WebUI Code Semantic Quality inspect source hashes, duplicates and coverage on request. Independent output directories are supported. New generation manifest format1 participates in sync/GKG/source import and compiled checks; modern missing/mismatched provenance fails, legacy boundaries remain supported
- **Rebuild**:13,400 cards with independent effects,27,647 nonempty-source slots,26,092 distinct vectors, largest identical group657;25,748 multi-chunk slots and172,018 chunks.27 sources remain explicitly incomplete. An assets-only GKG of about87.32MiB passes extraction validation; original root assets are preserved and source commits do not automatically publish new assets
- **Versions/cost**: framework3.13.4; Model Protocol4/Checkpoint3/revision14/static1/GKG4 unchanged. Network, inputs, rewards/PPO, legality and public Core remain unchanged. Offline first build and large-source startup checks increase; parameters/per-step storage/dictionary row count do not. Changed semantic hashes require fresh training, not migration or an untested strength claim
- **Lossless size reduction**: compact JSON removes formatting whitespace, reducing the KB from about104.71MiB to73.70MiB; source/structured data and logical semantic identity remain unchanged
- **Verification**:324 tests/50 subtests pass; four new-asset gates all pass separately. Coverage includes extraction/reuse/chunking/tampering/package/sync, network shapes, PPO updates, standard ONNX inputs/numerical parity and real-Core duel/raw-trajectory replay. Production-scale long training and manual browser visual acceptance are not claimed

See [3.13.4 code-semantic asset guide](code_semantics_en.md). Next is Phase7 batch3: Link/YRP ingestion and canonical quality gates before imitation learning.

---

## [v3.13.3] - 2026-10-09

### 🧩 Public hints and state-observation completion

- **Type120**: independent ordered missed-timing notifications enter16-step event history/replay, with code/location/message order/last processing-chain context and an auxiliary result output. No action-source overwrite, invented slot/cause/target or manual penalty
- **Type160/165 lifetimes**: latest scalar hints and reference-counted instance/player descriptions remain separate. Known movement, control swaps, continuous-zone/material reindexing and hand-shuffle re-emission preserve correct mappings; untrackable shuffles/unmatched removes mark incompleteness. No invented expiry at turn/history boundaries; CARD_QUESTION viewing restrictions apply to the receiving player, without replacing actual stats
- **Existing Lua semantics**: a small catalog distinguishes exact slot, explicit parent-code context and unknown, preserving full desc identity and projecting only one code row. Lua extraction retains`public_hint_parent_desc_ids`; existing semantic sync/GKG paths carry it and catalog identity participates in hashes. Many old local entries lack raw/source metadata, including current Magic Eye examples: raw identities remain learnable, but no current-script-to-old-vector forced binding or unsolicited asset rewriting occurs
- **Types37/38**: both-player reversal and legally known positions/code/face-up flags;38 offsets use logical size when consumed, including before DRAW. Drawing/moving/shuffling maintain or invalidate order without revealing hidden draws; full8-bit public coordinates are retained
- **Type162**: strict shared fixed7/8-slot decoder, atomic LP/count/occupancy/position/overlay/chain rebuild followed by public queries. Missing identities/hints/order remain unknown, old context is invalidated and stable construction is preserved. Type35 bulk Grave/Deck exchange remains deferred; Tag161 is not added
- **Shared paths/replay**:23 compact inputs span Encoder/shared memory/Worker/PPO/central inference/ONNX history opponents; standard outputs remain logits/value. Holographic JSON records public hints/known positions/frozen semantic bindings, with WebUI tables and new event labels; old JSON remains readable
- **Lightweight model/cost**: at most128-dimensional pooling and zero-initialized bounded residuals preserve the120-card backbone, stable DeckProfile, Side isolation, legality, rewards and PPO/GAE. Adds8,710bytes/step (272.19MiB at32,768),665,986 parameters for512/8/6; local CPU forward+2.2%, CUDA batch6 forward+7.5%, batch128 BF16 forward/backward+4.2% and ~250.69MiB additional peak allocation. These are not whole-run guarantees
- **Versions**: framework3.13.3; Model Protocol4/Checkpoint3 unchanged, schema revision14 rejects older development artifacts. Event2/public observation1/hint catalog extension1 are separate. Raw Trajectory2/holographic3 unchanged; external Link adapters must adopt new observation identity
- **Validation**: 303 full tests and 50 subtests pass with real-Core gates enabled, covering real snapshots/hint references, 3 model duels and raw response/observation replay, new ONNX inputs/numerical agreement, finite gradients/initial bitwise equivalence, visibility and lifetimes. A CPU32/4/1 single-Worker centralized-inference/PPO run completes 1 iteration with 438 samples. No production-scale long-run or manual browser visual acceptance is claimed
- **Additional asset risk**: a read-only audit found that 27,007 of 27,647 local code vectors are identical. Verify generation provenance, missing code and incremental reuse before concluding why. Source vectors are not overwritten; prioritize investigation before V4 long training

See [Public observations](public_observations_en.md) and the [message audit status](core_message_audit_en.md).

---

## [v3.13.2] - 2026-10-08

### 🛠️ Fixed Core parsing stability and AI prediction labels

- **Type 16 root cause**: shared `core_message_protocol.py` decodes the bundled public Core's 11-byte header and 14-byte entries. Forced flags are per candidate, not header fields; supposed separators were subsequent effect modes. Correct timings, forced/mode fields and legal Cancel; empty chain prompts are no longer discarded
- **No invented learning actions**: Worker/Arena/RuleBot self-check automatically send `-1` for empty chains without RNG, central inference, PPO samples, decision steps or transition events. Truncated/invalid fields fail explicitly and training retains invalid-episode rollback. RuleBot no longer guesses another layout or falls back to illegal cancellation during forced chains
- **Type 31 and entry-point consistency**: the extra confirmation byte is explicitly `skip_panel`; preserve card/position decoding. Remove CLI `--standard_core`, three WebUI controls and all internal ghost-byte flags/arguments. Only the bundled layout is supported, with no public Core modifications. Connect Type 21 sorting to full `DuelState.update()` dispatch
- **Trajectory boundary**: independent Trajectory Schema 1 becomes 2 with fixed `core_message_protocol` identity and separate validated/replayed `automatic_response` records, never behavior-cloning action labels. Raw Schema 1 traces are rejected. Model Protocol 4 / Checkpoint Format 3 / network schema revision 13 / holographic JSON 3 / standard ONNX remain unchanged
- **AI predictions**: UI/CLI label model assessments as AI predictions, not actual win rates; no “uncalibrated” warning titles or changed probabilities, weights or targets
- **Audit follow-up**: bilingual documentation explains full functionality, cognition gaps and proposals for missed timing 120, card/player hints 160/165, bulk swap 35, reversal/top 37/38 and snapshot 162. These additions are not implemented. CLIENT_HINT does not cover every ongoing effect; it cannot guarantee tracking Maxx “C” or Droll
- **Verification**: 284 regression tests and 50 subtests pass; two native Core gates skip by default and both pass explicitly, including three model duels, recording-toggle/seat-swap comparisons, raw responses and observation replay. A real small CPU configuration completes one training iteration with 468 samples, single Worker, forced Rule game returning to self, central inference and PPO update. No production-scale long-run validation or overall speedup is claimed. Networks/rewards/PPO algorithms are unchanged; corrected candidate/features can appropriately change previously faulty decisions

See the [fixed Core message/cognition audit](core_message_audit_en.md) and [trajectory/AI prediction replay](core_trajectories_en.md).

---

## [v3.13.1] - 2026-10-02

### 🎞️ V4 Phase 7 batch 2: canonical trajectories and state-evaluation replay

- **One semantic path**: `core_trajectory.py` reuses MessageParser/DuelState/the existing response packer; records raw Core chunks, consumed messages, query timing, raw macro pools, responses and observation digests without macro resampling or full tensors
- **Determinism and identity**: preserve Core seed, actual Python-shuffled injection order and original stable builds, plus UUID/internal iteration/checkpoint digest and public Core/CDB/loaded Lua/card-cache/vocabulary/semantic/protocol identities. New read-only `trajectory-check` and `--replay` verify records and restore callbacks/dialect afterward. Explicitly distinguish byte-exact output from same-card consecutive client-hint-removal reordering: diagnostic observation reproduction never grants the latter behavior-cloning eligibility
- **Bounded quality gates**: UUID gzip streams, 64 MiB uncompressed per-duel cap, bounded JSON readers; real terminals, no quality flags, unique response mappings and complete observation checks are necessary for behavior-cloning eligibility. Retry, ambiguous rule macros, unconsumed bytes and truncation never fabricate labels; checksums are not authentication
- **Optional state curves**: synchronized CLI/WebUI switches only affect logged games. Decode existing state terminal/length heads without new weights or changed selection. Replay JSON format 3 remains backward viewable, adding independent win/remaining-event curves, draw probabilities and separate actual-terminal references. Probabilities are uncalibrated; events are not turns and PPO value is not win probability
- **Storage**: Storage & Logs manages `.core.jsonl.gz`; Side is upper-layer metadata only, never single-duel input
- **Evidence and follow-up**: 276 regression tests pass with two explicit Core gates skipped by default; both pass when enabled, with three games in the original gate. The new gate covers two recording-switch comparisons/replays with swapped seats, plus ordinary shuffled injection and rule-opponent observations. New/old replay curve blocks pass actual Streamlit widget tests. CPU and 512/8/6 CUDA policy/value outputs are bitwise identical without extra RNG; the diagnostic branch did not increase forward peak allocated memory in the short CUDA check. Existing partial Type 16 parsing retains per-chunk coverage metadata/quality flags and is excluded from automatic imitation eligibility for a dedicated public-dialect audit next batch
- **Version boundaries**: framework 3.13.1; Model Protocol 4 / Checkpoint Format 3 / schema revision 13 unchanged; independent Trajectory Schema 1, Evaluation Format 1 and Replay Format 3. Training/rewards/shared-memory/standard ONNX remain unchanged; Link/YRP adapters and imitation optimization are not implemented in this batch

See [Canonical trajectories and state-evaluation replay](core_trajectories_en.md).

---

## [v3.13.0] - 2026-10-01

### 🛡️ V4 Phase 7 batch 1: PPO update boundaries and GAE comparisons

- **Unified controls**: Added `ppo_control.py`. Trainer/Worker/CLI/WebUI default to GAE lambda `0.98`, retaining explicit `0.95/0.99` comparisons. Gamma stays `0.998`; rewards and network remain unchanged. Added `--ppo-epochs` (1–4, default 4) and `--target-kl` (default 0.02; 0 disables the finite-KL threshold), mirrored in WebUI advanced RL settings
- **Shared-update KL guard**: The sampled-action nonnegative estimate is `mean(expm1(log_ratio)-log_ratio)`. Before each backward pass, an estimate above 1.5 times target stops all remaining epochs for this rollout, including policy, value, auxiliary, and planning updates. Nonfinite shifts always stop. This is not a strict trust region, does not roll back earlier updates, and does not promise four complete epochs
- **Actual-update audit**: `train_step` advances only after an optimizer update; FP16 GradScaler overflow skips do not advance the auxiliary schedule. Per-iteration `PPO_Update/*` metrics report completed/started epochs, effective sample reuse, optimizer updates, early stopping, quantiles of mini-batch mean KL/clip fraction, and nonfinite batches. `Configuration/PPO` records current controls and effective optimizer learning rates
- **Fixed batches and validation**: Existing full-mini-batch updates and random tail omission remain unchanged to avoid dynamic-shape recompilation; omitted counts are now explicit. Shared CLI/WebUI/Trainer validation rejects odd mini-batches, invalid epoch counts, and negative/nonfinite KL. Pair-preserving shuffling never duplicates samples, but incomplete tails are not guaranteed updates
- **Cost and compatibility**: Checks reuse the existing forward probabilities without another model pass or saved graph. Audit storage is a few Python scalars per mini-batch, with no new per-step trajectory fields. Model Protocol 4, Checkpoint Format 3, V4 schema revision 13, and standard ONNX contracts remain unchanged. Existing development checkpoints resume; learning controls are selected explicitly for the current launch
- **Prediction-display boundary**: `Auxiliary_Train/*` and `Planning/*` currently summarize training batches; replay JSON does not yet contain per-decision outcome/length predictions. Later canonical-trajectory work reserves optional prediction records, separating state evaluation from chosen-action prediction and labeling event-count length. Probability calibration requires validation split by complete duel before treating outputs as a win-rate curve
- **Validation**: The project suite passes 265 tests with one real-Core gate skipped; explicitly enabling it passes three consecutive decision loops with zero query parse errors and zero response fallbacks. Added real CUDA GradScaler-overflow skips, first-batch freezing of all auxiliary/planning parameters, CLI forwarding, and tail audits; standard ONNX parity/pruning regressions remain passing. An isolated 512-sample guard benchmark measured old/new CPU checks at about 4.7/24.7 μs and CUDA at 133.5/129.0 μs, indicating check cost only—not end-to-end throughput or long-run playing strength

## [v3.12.2] - 2026-09-23

### 🧭 V4 Phase 6 batch 3: goal-planning latent and safe policy integration

- **Per-decision goal planner**: Added `goal_planner.py`. At every decision, it reads the shared state representation, stable `deck_style`, and global/phase context to produce a 128-dimensional `plan_latent`. This is neither a persistent option, a hand-authored tactical class, an action script, nor MCTS; it is a continuous goal summary re-estimated from the complete currently visible state
- **Verifiable self-supervision**: A training-only target decoder predicts player-relative public-resource changes at chain end, turn end, and terminal, plus genuine Core-terminal outcome and remaining event length. All targets retain validity masks, so aborted/truncated duels and missing futures cannot become fabricated labels. Planning enters PPO with coefficient `0.05` without changing rewards, GAE, advantages, clipping, entropy, or legal actions
- **Low-weight temporal consistency**: The posterior contract advances to `AUXILIARY_TARGET_FORMAT_VERSION=3` with one pair-role byte per step. Adjacent decisions by the same actor in the same turn and phase may form non-overlapping pairs whose goal directions receive a low-weight cosine constraint. Even mini-batches shuffle two-sample units so pairs do not cross batch boundaries or duplicate samples; incomplete random tails retain the original omission behavior
- **Gradual bounded policy/value feedback**: Separate projections and per-channel zero-initialized gates inject the latent into policy intent and value, with residual amplitude bounded to ±0.5. Initial policy/value outputs are bitwise identical to the ungated path, while gates receive PPO gradients from the first update. Planner, original PPO base, and training decoders are clipped separately at `1.0`, `0.5`, and `1.0`
- **Deployment contract**: Ordinary three-output inference, central rollout inference, Arena, Link, and historical opponents execute the learned planning path. Standard ONNX still exposes only `action_logits` and `values`; it retains the inference-required planner/projections/gates while pruning the target decoder and the 3.12.1 structured auxiliary heads. PTH training checkpoints retain all parameters for exact resume
- **Visualization and diagnostics**: TensorBoard adds `Planning/*` total/component losses, latent RMS/mean dimension standard deviation/active-dimension fraction, pair cosine and coverage, terminal accuracy, policy/value gate RMS, and a separate gradient norm. Every 200 updates, `Planning_Histograms/*` records latent and both gate distributions. These diagnose learning, gate opening, and representation collapse; they are not card-level or hand-authored tactical explanations
- **Size and performance**: The posterior now uses 189 bytes/step, about 5.90625 MiB at 32,768 steps—only 32 KiB above 3.12.1. A 512/8/6 model adds 426,620 parameters: 402,370 on the deployment path and 24,250 in the training decoder, for 52,995,987 total. On the local RTX 5070 Ti, batch-128 CUDA BF16 eager forward/backward rose about 1.1%–1.3% with roughly 3.79 MiB more peak allocated memory; standard-forward differences were near short-run timing noise, so these numbers are directional only
- **Version and validation boundary**: Framework advances to 3.12.2; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision advances to 13. Coverage includes target shapes/masks, probe-gradient isolation, pair construction/shuffling, bitwise zero-gate equivalence, gate gradients, a real lightweight PPO update, ONNX planner retention, and training-decoder pruning. The project suite passes 255 tests with one gated real-Core skip, and the explicitly enabled real-Core decision loop passes. Older development structures cannot be mixed silently

## [v3.12.1] - 2026-09-22

### 🧠 V4 Phase 6 batch 2: structured auxiliary heads and controlled shared gradients

- **Training-only structured heads**: Added `auxiliary_heads.py`. The branch reads only the shared policy/value state representation and the actually selected option representation, then predicts immediate result bits, ten public Core-message groups, transition boundary, immediate LP/both-player eight-zone/chain changes, chain-end/turn-end/terminal resource changes, terminal outcome, and remaining-event length. Posterior targets never become observations, and neither RuleBot actions nor next-action imitation are treated as truth
- **Missing-safe multitask loss**: Binary results use BCE, boundary/outcome use cross entropy, and resource/remaining-length regression uses Smooth L1. Every horizon is independently normalized through the 3.12.0 validity mask, so incomplete chains, truncations, and non-Core stops cannot become fake zero labels. The normalized seven-task objective enters total PPO loss with coefficient `0.1`; rewards, GAE, advantages, PPO clipping, and entropy equations are unchanged
- **Controlled instead of dominant backbone gradients**: The first 100 resumable mini-batch updates train the heads as probes only. The following 900 updates linearly open shared gradients to a maximum factor of `0.2`, without changing forward values. PPO-base and auxiliary-head parameters are clipped separately at `0.5` and `1.0`, preventing head gradients from consuming the original PPO clipping budget
- **Action/deployment isolation**: Predictions never feed policy logits or value, and ordinary `GalateaNet(batch)` keeps its three-output contract. Central rollout inference, Arena, Link, and historical opponents do not execute the branch. Standard ONNX still emits only `action_logits` and `values` and contains no auxiliary-head weights. PTH training checkpoints retain the parameters for exact resume
- **Audit metrics**: TensorBoard adds `Auxiliary_Train/Total_Loss`, seven component losses, result/message bit accuracy, boundary/terminal accuracy, future-target validity, backbone-gradient scale, and auxiliary gradient norm. Existing `Auxiliary_Targets/*` distribution audits remain available
- **Size and performance**: The posterior contract becomes `AUXILIARY_TARGET_FORMAT_VERSION=2`, adding one actor-seat byte for strict player-relative reordering: 188 bytes/step, about 5.875 MiB for 32,768 steps. A 512/8/6 model adds 356,709 parameters (about 1.36 MiB of FP32 weights) for 52,569,367 total. On the local RTX 5070 Ti with batch 128 and CUDA BF16 eager forward/backward, median time rose about 1.09% and peak allocated memory about 2.86 MiB; per-decision central inference adds no computation
- **Version boundary**: Framework advances to 3.12.1; Model Protocol remains 4 and Checkpoint Format remains 3, while V4 schema revision advances to 12. The new training parameters prevent revision-11 development checkpoints from being resumed as current training checkpoints. The standard deployment ONNX interface is unchanged, though schema identity still rejects mixed development artifacts
- **Validation**: Covers output shapes, missing masks, player-seat symmetry, probe/ramped gradients, bitwise-identical default forward outputs, a real lightweight PPO update, separate gradient clipping, and exclusion of auxiliary weights from ONNX. The project `tests/` suite passes 250 tests with one real-Core gate skipped, and the explicitly enabled real-Core decision loop passes

## [v3.12.0] - 2026-09-22

### 🧭 V4 Phase 6 batch 1: multi-horizon auxiliary-target contract and audit

- **Independent posterior contract**: Added `auxiliary_targets.py` and `AUXILIARY_TARGET_FORMAT_VERSION=1`. Every submitted action receives a strictly increasing event sequence ID. The latest-16 rolling history consumed by the network is unchanged. Only a training Worker explicitly enables complete finalized-event references for the current duel lifetime; Arena, Link, and ordinary inference pay no such resident-memory cost
- **Four verifiable horizons**: Duel commit now builds next-decision, chain-end, turn-end, and genuine-Core-terminal targets. They contain validity masks, accumulated LP and both players' Deck/Hand/MZone/SZone/Grave/Removed/Extra/Overlay deltas, event counts, and immediate result bits, ten public message groups, boundary, summon method, chain depths, player-relative outcome, terminal reason, and remaining-event count
- **Strict alignment and missingness**: A training step maps only to its submitted action by sequence ID. One reverse scan produces future boundaries and prefix sums produce resource deltas. Aborted duels still roll back atomically; a timeout truncation or unfinished final action remains invalid instead of receiving a fabricated all-zero future. Framework-forced stops outside a genuine Core terminal do not produce terminal supervision
- **Hidden-information boundary**: Immediate events pass through existing player-visibility filtering, while multi-horizon aggregation uses only public LP, zone counts, and boundaries. Opponent-private internal selections do not become intent/source/target labels. Future information enters only the post-duel training target and never the sampling-time observation
- **Audit, not learning**: The Trainer writes four horizon-coverage rates plus Retry, Cancel, state-change, chain-negated, and chain-disabled rates under `Auxiliary_Targets/*` in TensorBoard. This batch adds no auxiliary loss, Planner, reward, or gradient and changes no policy logits, value, GAE, central inference, shared observation, or standard ONNX graph
- **Size and performance boundary**: Compact targets use exactly 187 bytes/step, about 5.84 MiB for 32,768 steps. Complete events are retained only as object references and released at duel end. Boundary indexes and resource prefixes are linear; a local synthetic maximum of 1,500 events/750 training steps took about 186 ms, while normal 300–400-step duels are far smaller and no work enters the per-decision inference hot path
- **Version boundary**: Framework advances to 3.12.0; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision remains 11. `AUXILIARY_TARGET_FORMAT_VERSION` describes only Worker/Trainer training posterior data and is excluded from the model-input hash, so 3.11.2 network/deployment graphs are not invalidated by this batch
- **Validation**: Added tests for multi-horizon accumulation, terminal preemption, unfinished final actions, internal-gap rejection, actor validation, truncation without terminal supervision, storage budget, and audit statistics. The project `tests/` suite passes 244 tests with one real-Core gate skipped, and the explicitly enabled real-Core decision loop passes

## [v3.11.2] - 2026-09-22

### 🧠 V4 Phase 5 batch 3: transition history enters the policy network

- **Compact event tensors**: The latest 16 visible 3.11.1 events now use fixed-shape inputs for source identity and exact Lua effect slot, up to five target identities/locations, actor, prompt/operation/summon method, phase/boundary, result bits, ten public Core-message groups, relative LP/eight-zone deltas, and chain depth. Private events and fields are removed or masked for the observing player before tensorization
- **Independent ordered encoder**: New `event_encoder.py` reuses model-side exact card-by-effect-slot semantics for the source and card identity plus all-Lua-code summaries for targets. It performs masked slot-aware target pooling, then chronological positional mixing, depthwise local convolution, and attention pooling. An empty history produces an exact zero vector
- **Safe policy/value fusion**: A per-channel zero-initialized gate adds the event vector to shared `v_input`. At initialization, policy/value outputs are bitwise identical to the same network without event inputs; the gate receives gradients on the first update and subsequently opens normal learning through the event encoder. Legality, reward, GAE, PPO loss, and duel state are unchanged
- **End-to-end transport**: Nineteen compact fields now pass through the Encoder, central-inference shared memory, Worker temporary trajectories, Trainer merge storage, PPO mini-batches, and ONNX. ONNX Runtime consumes the same observation contract and remains numerically aligned with PyTorch, including historical-opponent inference
- **Size and performance**: Event inputs use exactly 1,440 bytes/step, or 45.0 MiB for 32,768 steps. A 512/8/6 model adds 589,056 parameters for 52,212,658 total. Local isolated measurements showed about +2.72% single-sample CPU forward overhead and +10.65% for batch-6 CUDA BF16; tensorization took about 0.031/0.252 ms for empty/full-16 history, remaining below the phase's 15% gate. CUDA BF16 forward/backward produced finite gradients for every event parameter
- **Version boundary**: Framework advances to 3.11.2; Model Protocol remains 4 and Checkpoint Format remains 3, while V4 schema revision advances to 11. New inputs and weights prevent revision-10 development checkpoints/ONNX graphs from being mixed with the current structure; V4 still trains from scratch
- **Validation**: Covers field shapes/types, player-relative symmetry, private-event isolation, empty history, order sensitivity, zero-gate equivalence, gate gradients, and complete ONNX inputs. The project suite passes 238 tests with one real-Core gate skipped, and the explicitly enabled real-Core decision loop passes

## [v3.11.1] - 2026-09-22

### 🧭 V4 Phase 5 Batch 2: Between-decision transition-event protocol

- **Strict lifecycle**: Added independent `event_history.py` and `TRANSITION_EVENT_FORMAT_VERSION=1`. An event starts only after an action is successfully sent to Core and ends at the next decision, Retry, or terminal boundary. LP/zone/chain deltas are frozen after query reconciliation, and the latest 16 events are attached oldest-first to `GameSnapshot.transition_history`
- **Deterministic labels**: Each event carries actor, turn/phase, prompt/operation, summon method, source card, effect slot, up to five target locations, public Core message types, both LP deltas, Deck/Hand/MZone/SZone/Grave/Removed/Extra/Overlay count deltas, and cancel/decline/finish/Retry/chain-negated/disabled/terminal flags. RuleBot and external responses are matched back to the same legal-action semantics when possible
- **Hidden-information isolation**: Event existence, intent, source, and targets have separate two-player bitmasks. Fully private hand/deck selections disappear from the opponent view. Summon, chain, and public board changes may promote visibility without exposing card identity or location that remains hidden
- **All decision paths**: Training Workers, history/self play, Arena, RuleBot self-check, and the simple AI test register events after successful submission. The dual-mirror self-check updates both states so one side cannot lose opponent history
- **Cost boundary**: Ordinary Core messages update only compact labels and chain depth; the complete 16-zone resource digest is produced once at action start and once at boundary finalization. A local 100,000-event begin/finalize microbenchmark measured about 6.64 µs/event. Version 3.11.1 adds no network parameter, input tensor, shared-memory field, PPO trajectory field, ONNX input, or reward
- **Deck Encoder Dropout decision**: Ordinary elementwise Dropout is not added to the Deck Encoder or direct `deck_style` policy/value path. That path preserves stable construction identity and sparse key-card relations; the replayable 10% whole-duel mask remains limited to optional Deck FiLM. If held-out deck-family evaluation later shows overfitting, use whole-duel structured ablation or deck-variant consistency instead
- **Version boundary**: Framework advances to 3.11.1; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision remains 10. Because the event protocol is not yet consumed by the network, 3.11.0 development checkpoints and ONNX structure are not invalidated by this batch
- **Validation**: Added six tests for lifecycle, Retry, visibility, hidden-selection isolation, fixed capacity, and RuleBot response recovery. The project suite passes 234 tests with one real-Core gate skipped, and the explicit real-Core decision loop passes

## [v3.11.0] - 2026-09-22

### 🎛️ V4 Phase 5 batch 1: safe layered FiLM and deck-style modulation

- **Layered rather than globally shared**: The previous global FiLM reused one γ/β pair across every Transformer layer and both attention/FFN. A low-rank `LayeredFiLMGenerator` now gives each layer and sublayer independent γ/β, allowing phase, turn order, current resources, and deck style to play different roles in shallow representation and deeper reasoning
- **Separated global and deck branches**: The global branch retains the 17-value relative state plus phase embedding, while the deck branch consumes 3.10.2's `deck_style`. `0.5 * tanh(global_raw + tanh(deck_gate) * deck_raw)` hard-bounds total modulation to ±0.5 so extreme scales cannot destabilize the post-LayerNorm backbone
- **Zero gate without a gradient dead path**: Deck gates start at exactly zero per layer, attention/FFN, γ/β, and channel, making initial deck modulation identically zero. The deck generator remains nonzero-initialized, so the policy/value loss can move the gate on its first step. The direct policy/value `deck_style` path stays active, letting the Deck Encoder learn before the gate opens
- **PPO-consistent condition dropping**: Ordinary train-mode Dropout would change the conditions behind saved old and recomputed new log-probs, so it is not used. A separate Worker RNG disables deck FiLM for 10% of complete training duels and writes the one-byte `deck_film_mask` to both Pass1/Pass2 central inference and every trajectory sample. PPO reuses that mask. Arena, Link, historical ONNX, and deployment default to enabled, and the direct policy/value path is never masked
- **Boundary and non-goals**: No Core query, reward, action logic, or visible deck information changes; Side remains outside BO1. Version 3.11.0 completes only Phase 5's layered FiLM. The 16 state-transition events between model decisions remain for 3.11.1/3.11.2
- **Size and performance**: Adds a net 315,008 parameters over 3.10.2 (+0.61%) for 51,623,602 total; trajectories add one byte per step. In isolated 512/8/6 microbenchmarks, layered FiLM added about 0.42% to single-sample CPU forward and 2.91% to batch-6 CUDA BF16; CUDA BF16 forward/backward passes
- **Version boundary**: Framework advances to 3.11.0; Model Protocol remains 4 and Checkpoint Format remains 3, while V4 schema revision advances to 10. New inputs and parameters make revision-9 development models incompatible under the established V4 scratch-training policy
- **Validation**: Covers layer/sublayer shapes, exactly zero initial modulation, the total amplitude bound, zero-gate gradients, per-duel masking limited to deck FiLM, the existing Deck Encoder gradient path, and PyTorch/ONNX Runtime. Of 229 default tests, 228 pass with one gated real-Core skip; the explicitly enabled real-Core decision loop passes

## [v3.10.2] - 2026-09-21

### 🧠 V4 Phase 4 batch 3: section-aware deck images and reusable Deck Encoder

- **Complete profile labels reach the model**: Adds up to 128 `deck_profile_*` entries. Every unique card entry explicitly carries an exact token, Main/Extra section, initial and current remaining counts, plus race, attribute, and set-code labels. Main and Extra counts remain independent, and the current BO1 Encoder emits no Side entry
- **Permutation-invariant Deck Encoder**: Adds standalone `deck_encoder.py`. Eight learned latents cross-attend to unordered entries without positional embeddings. Reordering leaves `deck_style/deck_latents` invariant while `per_card_features` follows the same permutation, providing a reusable boundary for later layered FiLM, auxiliary heads, and the upper deck-building module
- **Lua semantics without printed-text guesses**: Each profile entry receives the masked mean of all valid Lua code vectors for that card from the compiled static assets. There is no per-step script/CDB I/O and printed-effect numbering is never treated as a code slot. `deck_style` now enters shared `v_input`, conditioning both the policy intent tower and value head; layered FiLM remains Phase 5
- **Exact dynamic fallback retained**: Existing per-step `deck_idx/deck_mask` remains for effects that insert cards absent from the initial construction. Stable profile fields are still reconstructed from the per-duel catalog before PPO mini-batches; only the 128-byte `deck_profile_remaining_count` is added per sample. A stable image uses about 7,552 bytes per unique profile, and private `DECK_TRAJECTORY_FORMAT_VERSION` advances to 2
- **Unchanged BO3/deck-building boundary**: A future external Deck Encoder caller may use `DeckSection.SIDE`, but the current `DeckProfile`, `DuelState`, training, Arena, and ONNX BO1 path emits Main/Extra only. Match score, siding, and complete opponent construction remain hidden
- **Performance and size**: The current policy skips unused per-card writeback while explicit upper-layer calls still receive it. In local 512/8/6 microbenchmarks, single-sample single-thread CPU moved from about 74.61 ms to 76.42 ms (+2.42%), while batch-6 CUDA BF16 moved from about 7.50 ms to 8.56 ms (+14.03%) and completed BF16 forward/backward. The Deck Encoder adds 1,582,592 parameters, about 3.08% of the current 51,308,594. These are directional; real short training remains an acceptance gate
- **Version boundary**: Framework advances to 3.10.2; Model Protocol remains 4 and Checkpoint Format remains 3, while V4 schema revision advances to 9. New inputs and weights make revision-8 development models incompatible; the established V4 scratch-training policy provides no migration
- **Validation**: Covers field shapes/types, section/count alignment, Side isolation, permutation invariance/per-card equivariance, fast/full-path numerical identity, finite all-empty profiles, policy/value gradients, zero-error compressed reconstruction, and complete PyTorch/ONNX Runtime forwards. The default suite passes 224 of 225 tests with one gated real-Core skip; the explicitly enabled real-Core decision loop passes

## [v3.10.1] - 2026-09-21

### 🗂️ V4 Phase 4 batch 2: deck-profile indices and lossless PPO trajectory deduplication

- **Per-duel profile registration**: Adds `deck_trajectory.py`. Workers deduplicate the stable single-duel image with 3.10.0's Side-free `DeckProfile.profile_id`, retaining exact card tokens, Main/Extra sections, and initial copy counts. Each training sample appends only one `int32 deck_profile_index` instead of another complete image
- **Static-field deduplication**: Worker temporary trajectories and the Trainer merge pool no longer repeat `deck_race`, `deck_attr`, and `deck_setcodes` at every step. Dynamic remaining-card order in `deck_idx` and `deck_mask` is preserved exactly, so draws, searches, and returns to Deck remain visible per step
- **Lossless legacy-network reconstruction**: The Trainer merges Worker catalogs, remaps local indices, and builds one small exact-token metadata table per PPO update. The three legacy fields are reconstructed immediately before every mini-batch enters the unchanged network. Central-inference shared buffers, GalateaNet input keys, weight shapes, PPO objectives, rewards, ONNX, and duel actions are unchanged
- **Corruption barriers**: Temporary catalogs remain `weights_only=True` safe and validate format version, tensor shapes, profile hashes, index ranges, cross-Worker token-metadata consistency, and coverage of every active token. Invalid data is rejected rather than silently training with zero-filled fields
- **Space and hot path**: The three old fields used 3,600 bytes per step; the new index uses 4 bytes, a net reduction of 3,596 bytes or about 112.4 MiB for each 32,768-step Worker/merge trajectory. Repeated deck observations hit a small tensor snapshot instead of repeating Python card iteration; a local 32,768-call identical-observation microbenchmark fell from roughly 13.6 s to 0.4 s. PPO also replaces three CPU-to-device field transfers with device-side indexing
- **Version boundary**: Framework advances to 3.10.1 while Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision remains 8. `DECK_TRAJECTORY_FORMAT_VERSION=1` covers only the private Worker transport and is not the future public canonical-trajectory format
- **Validation**: Adds profile-order deduplication, `weights_only` round trips, cross-Worker merge, exact field reconstruction, zero-error full GalateaNet outputs, metadata-conflict rejection, commit-memory accounting, and a real CPU PPO forward/backward test. The default suite passes 217 of 218 tests with one gated real-Core test skipped; the explicitly enabled real-Core duel passes

## [v3.10.0] - 2026-09-21

### 🃏 V4 Phase 4 batch 1: deck-protocol boundary and BO3 extension points

- **Full construction vs single-duel image**: Adds `deck_protocol.py`. `DeckSpec` retains Main, Extra, and Side, while `DeckProfile` contains Main/Extra only and excludes Side both by type and serialized payload, making it the sole deck-image boundary for the current BO1 policy
- **Stable content identities**: Full `deck_id` and single-duel `profile_id` are derived from section, exact card code, and copy count, independent of `.ydk` order, filename, or display name. These keys prepare per-duel Worker registration and profile deduplication
- **Initial/remaining state separation**: `DuelState` now retains immutable initial Main/Extra copies while the existing `p0/p1_deck/extra` lists continue following Core-driven remaining cards. `GameSnapshot` exposes both, preventing later encoders from treating drawn or summoned cards as absent from the original construction
- **Side resource closure without policy exposure**: `.ydk` `!side`/`#side` sections are parsed and unknown Side cards are tracked separately. A Main/Extra-valid deck with incomplete Side remains usable in current BO1 training and Arena; WebUI preflight reports a separate Side warning instead of marking it BO1-incompatible
- **BO3/deck-building extension points**: Adds external `MatchContext` and `DuelSummary` contracts outside `DuelState`. `policy_inputs()` is intentionally empty, so score, siding, and Side data do not enter current observations, the network, ONNX, or PPO trajectories
- **Compatibility/version boundary**: Framework advances to 3.10.0 while Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision remains 8. This batch establishes resource/state contracts and a stable profile source without changing network inputs, weight shapes, rewards, or duel-action behavior
- **Validation**: Adds tests for permutation-invariant identities, Side isolation, Side-only BO1 compatibility, section/copy labels, and restoration of initial images through `DuelState.reset()`. The default suite passes 211 of 212 tests with one real-Core gate skipped; the explicit real-Core duel passes with zero query parse errors

## [v3.9.2] - 2026-09-21

### 📦 V4 Phase 3 Batch 3: Static-Semantic Runtime Assets and Cross-Machine ONNX/GKG Closure

- **Independent compiled runtime assets**: Added derived `semantic_lookup_v1.npz` and `semantic_lookup_v1.json`. The former stores model-side structured tables and the shared code-vector dictionary as pickle-free validated arrays; the latter stores exact-vocabulary identity, logical semantic hash, array integrity, valid effect slots, and the full `desc → Lua slot` catalog. Both are deterministically rebuildable from the semantic source trio and are not committed to Git
- **No stale cache after dynamic updates**: Remote semantic sync, local Lua extraction, code-vector generation, and authoritative-vocabulary appends explicitly invalidate the previous compiled pair; semantic management eagerly rebuilds it afterward. Training, Arena, and one-click packaging also perform a one-time pre-network build when missing—never per duel or per step
- **Fixed spawn-Worker effect-slot degradation**: After 3.9.1 removed the Encoder semantic KB, Windows spawn Workers could lack the main process's runtime effect bindings and downgrade chain/history slots to unknown. Encoder now registers only the roughly 0.6 MiB on-disk catalog rather than loading the 53.47 MiB numeric table. Local `tracemalloc` retained about 6.8 MiB of Python catalog memory, still about 46.7 MiB less per Worker than the complete table
- **Clear PTH/ONNX responsibilities**: Training/PTH loads NPZ once during construction and registers non-parameter buffers. ONNX export freezes the same tables into `.onnx.data`, so inference needs neither NPZ nor JSON and exposes no `sem_*` input. The external-data smoke model retained 57 compact inputs and produced the expected ONNX Runtime output shapes
- **GKG deployment format V4**: `.gkg` now requires `card_vocab.json` plus the compiled runtime pair and cross-checks file digests, array dtype/shape, logical semantics, model UUID/iteration, and ONNX external data. `knowledge_base.json + code_embeddings.npy + code_embeddings_idx.json` become optional continuation sources while remaining indivisible. WebUI/CLI installs runtime assets automatically and rejects an older package's semantic sources from overwriting a locally newer vocabulary
- **Space and startup cost**: Across five local runs, source compilation averaged about 0.528 s versus about 0.140 s to restore materialized tables while rechecking source digests, roughly 3.8× faster at startup. Expanded runtime assets occupy about 54.06 MiB, 18.7% below the approximately 66.49 MiB maintenance sources including the Hash map. A runtime-only sample GKG was about 1.54 MiB versus 3.93 MiB with maintenance sources; real model packages remain dominated by PTH/ONNX weights. Keeping both sources and rebuildable cache locally adds about 54.06 MiB
- **Unchanged hot path**: No per-step I/O, Core query, network parameter, or PPO-objective change was introduced; 3.9.1's 152,996-byte-per-step and roughly 4.67-GiB-per-32,768-step trajectory savings remain intact. Framework becomes 3.9.2 while Model Protocol stays 4, Checkpoint Format stays 3, and V4 schema revision stays 8, so 3.9.1 checkpoints are not invalidated by a deployment-asset format change. GKG independently advances to format 4
- **Acceptance**: The default suite passed 207 of 208 tests with one real-Core gate skipped; the explicitly enabled real-Core decision loop passed. External `.onnx.data` export, identity tagging, dependency enumeration, CPU ONNX Runtime inference, and absence of `sem_*` inputs also passed

## [v3.9.1] - 2026-09-21

### 🧠 V4 Phase 3 Batch 2: Model-Side Semantic Lookup and Trajectory Deduplication

- **Lossless observation switch**: Scene/deck rows keep the existing `card_idx/deck_idx`; chain adds `c_effect_slot`, while recent history adds `h_card_idx/h_effect_slot`. Values 1–8 select exact Lua effect slots, and 0 preserves whole-card fallback for unknown bindings. Hidden-information masking, public relations, action candidates, and Core responses are unchanged
- **Unified model-side lookup**: `GalateaNet` registers the 3.9.0 structured tables as non-parameter buffers and reconstructs the former nine semantic tensors from card tokens/effect slots on the active device. These buffers are not trainable parameters or checkpoint weights and do not alter gradients or PPO objectives
- **Removed per-step matrix copies**: Encoder no longer loads/compiles semantics or emits `sem_* / d_sem_* / c_sem_* / h_sem_*`. Trainer shared memory, pinned staging, Worker trajectories, and the PPO merge pool remove them as well. Central-inference Workers therefore no longer retain roughly 53.47 MiB of static semantic assets each
- **Memory result**: Expanded fields used 153,080 bytes per step; replacement identities use 84 bytes, a net reduction of 152,996 bytes (about 149.4 KiB/step), or about 4.67 GiB for one 32,768-step trajectory pool. A default 512-sample PPO mini-batch also replaces roughly 74.75 MiB of repeated semantics with one roughly 12.97 MiB resident structured table
- **Performance boundary**: On-device lookup has a fixed indexing cost. On the local RTX 5070 Ti with a 512/8/6 single-sample benchmark, pure forward rose by about 1.59 ms, but removing CPU-to-CUDA expanded-matrix transfer reduced total time from about 13.11 ms to 12.16 ms. Real batching/async overlap varies, so this is directional evidence only; no extra Lua/CDB lookup or per-step disk I/O is introduced
- **Elementwise parity gate**: A legacy-expanded branch remains solely for tests. Full compact and expanded forwards under identical weights match `logits/value/v_input` at `rtol=0, atol=0`. ONNX Runtime also validates the compact signature, which contains no expanded `sem_*` input
- **Schema boundary**: Framework becomes 3.9.1; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision becomes 8. Trainable parameter shapes are unchanged, but revision-7 development artifacts remain rejected under the V4 scratch-training policy
- **Regression result**: The default suite runs 207 tests (206 passed and one real-Core environment-gated skip). The explicitly enabled public-Core decision loop passes separately, and ONNX/PyTorch outputs remain within the existing tolerance

## [v3.9.0] - 2026-09-21

### 🧠 V4 Phase 3 Batch 1: Model-Side Static Semantics and Asset Identity

- **Unified static-semantic compilation**: Added `semantic_lookup.py`, which compiles structured semantics, requirements, referenced cards, code-vector indices, and valid effect slots once into dense tables aligned to exact `card_vocab.json` tokens and Lua effect slots. Lookup never uses printed-effect numbering or modulo hashes
- **Compatibility bridge without input changes**: The existing Encoder temporarily reads the unified tables through the `SemanticKnowledgeBase` compatibility interface. Field shapes, numeric IDs, action protocol, network parameters, rewards, and PPO remain unchanged; removal of duplicated trajectory tensors is reserved for 3.9.1
- **Removed duplicate in-process assets**: Encoder and `GalateaNet` now share one code-vector/structured-semantic cache. The former random KB startup delay and the network's separate `.npy` copy are gone. Missing or incoherent semantic assets remain fatal instead of silently falling back to zero semantics
- **Semantic prefix identity**: Added `semantic_lookup_format_version`, `semantic_lookup_hash`, and `semantic_lookup_card_count`. The hash covers exact card tokens, effect slots, structured fields, and logical code vectors rather than physical `.npy` row numbers. Pure card/semantic appends preserve older prefixes, while any existing-card semantic change is rejected before network construction
- **Artifact identity closure**: PTH top-level/config metadata, ONNX metadata, per-iteration manifests, and `.gkg` manifests now carry the semantic identity. Package creation and extraction recompute it from the semantic bundle actually being packaged, so a shape-valid but content-replaced vector matrix is rejected
- **Live-update cache invalidation**: Cache keys track timestamp/size signatures for the KB, code vectors, and index. A new Encoder switches to the rebuilt table after remote sync or local continuation instead of retaining stale WebUI-process semantics
- **Resource benchmark**: The current 14,981-card vocabulary and 27,647 effect slots compile in about 0.35 seconds. Structured tables use about 12.97 MiB and the shared code dictionary 40.50 MiB, for 53.47 MiB total. The former two code-vector copies alone used roughly 81 MiB before per-card Python caches; 100,000 card lookups take about 0.095 seconds
- **Schema boundary**: Framework version is 3.9.0; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision becomes 7. Revision-6 development checkpoints lack the semantic identity and are explicitly rejected under the scratch-trained V4 policy
- **Regression scope**: Added exact token/slot alignment, append-only semantic-prefix, physical-row-reorder equivalence, existing-vector tamper, and deployment-package tamper coverage. The full suite runs 206 tests (205 passed and one environment-gated skip), with the separately enabled real-Core decision loop also passing. Automated coverage includes ONNX export/runtime parity. This batch deliberately retains the legacy semantic tensors in shared memory/trajectories and preserves current ONNX inputs as an independently reviewable bridge before 3.9.1

## [v3.8.2] - 2026-09-20

### 🔗 V4 Phase 2 Batch 3: Public Card Relations and Core-State Closure

- **Public Core only**: Added a dedicated legacy `query_card` parser for every standard public field and use `query_field_count` to reconcile Hand, Graveyard, and banished zones. Neither `ocgcore.dll/.so` nor a private Galatea Core fork is modified
- **Fixed real query-buffer failures**: Legacy Core writes query results into a `ctypes.c_byte` buffer. Converting its signed Python list directly with `bytes()` failed whenever a byte was at least `0x80`. The implementation now copies the underlying memory as unsigned raw bytes, and the real-Core gate requires at least one successful query and zero query parse errors so message-mirror fallback cannot hide the defect
- **Completed dynamic per-card state**: The model now receives dynamic identity, Level/Rank, current/base ATK/DEF, Pendulum Scales, Link Rating/markers, reason bits, original owner, and `STATUS_DISABLED / STATUS_PROC_COMPLETE / STATUS_FORBIDDEN`. A zero Level is no longer overwritten by a static Rank or Link value
- **Completed public relations**: Equip source-target, effect target, reason card, and inverse equip edges map to entity indices. Xyz material identity expands from only the first material to at most 16, and counters expand from one total to eight typed amount slots
- **Proper-summon awareness**: `STATUS_PROC_COMPLETE` follows visible field, Graveyard, and banished entities so revive-limit behavior is learnable. Exact summon method/source/player remains unknown because the public query API does not export it; card type, zone, and message order are never used as guesses
- **Preserved the public-Core boundary**: Upstream v11.0 still exposes summon provenance to rule scripts internally but not through standard client query fields. It remains an upstream API/possible PR item. A future upgrade will adapt the complete public `OCG_DuelQuery` ABI instead of adding private binary fields
- **Fixed false macro equivalence**: Physical copies are always preserved on the field, in the Graveyard, and in banishment. Hidden/reset-zone copies fold only when identity, prompt values, and known state match; the existing 5,000-combination cap and weighted reduction remain intact
- **Completed public position/relation messages**: Added `FIELD_DISABLED`, `UNEQUIP`, and `CARD_TARGET/CANCEL_TARGET` mirrors, recognize Core's `POS_REVEAL` visibility bit, and admit message Types 95/161/162 through the parser whitelist
- **Message-coverage audit**: Audit schema 3 classifies every seen message as `decision_parsed`, `state_applied`, `query_reconciled`, or `known_observation_gap`; WebUI surfaces rare gaps rather than leaving public information silently unused
- **Closed residual action-message gaps**: Type 21 chain sorting now crosses parsing, RuleBot, macro pooling, Worker, Arena, and both model passes. Type 22 counter allocation exposes each card and available amount to Pass 1. Coin/dice and hand-result notifications are no longer mistaken for response-waiting decisions by Worker, Arena, or RuleBot self-check, preventing them from discarding later messages in the same batch
- **Corrected self-check diagnostics**: Idle Type 0/2/4 and the other fixed families now have readable operation labels. Holographic inspection reads independent `rank` and `link_rating` fields instead of the obsolete V2 convention of displaying both through `level`
- **Closed a memory-based hidden-state leak**: A previously revealed card may retain its identity and static card knowledge after entering an opponent-hidden zone, but no longer reuses Core-query dynamic ATK/DEF, type, relations, materials, counters, or status
- **Known remaining boundary**: Exact summon provenance awaits an upstream public API. Types 35/38/120/160/161/162/165 (grave/deck swap, top-card/hint, and nonstandard-mode traffic) are retained as audit evidence instead of fabricated normal-BO1 observations
- **Compact trajectory storage**: Reasons, counter types, and disabled zones are losslessly packed as little-endian bytes; bounded vocabulary indices use `int16` and relation indices use `uint8`. The baseline adds about 13.1 KiB per step, or roughly 420 MiB per 32,768 steps, avoiding a much larger eagerly expanded bit representation
- **Schema boundary**: Framework version is 3.8.2; Model Protocol stays 4, Checkpoint Format stays 3, and V4 schema revision becomes 6. Revision-5 checkpoints are rejected before network construction because this batch adds inputs and parameters
- **Regression**: The portable UTF-8 suite runs 200 tests: 199 pass and one real-Core gated case skips as expected. Enabling the gate separately completes a real-Core duel and verifies that successful `query_card` calls are nonzero while parse errors remain zero. Coverage includes complete public-query payloads, Type 21/22 action paths, truncation rejection, hidden-state masking, relation/status encoding, distinct same-name Graveyard copies, shared memory, PyTorch/ONNXRuntime parity, and actual Core responses

---

## [v3.8.1] - 2026-09-20

### 🧩 V4 Phase 2 Batch 2: Selection State Machines and Response Closure

- **Preserved native Core boundaries**: Type 26 remains iterative Select/Unselect, while ordinary Type 15 selection, Type 20 tribute selection, and Type 23 sum selection return one complete legal combination. No synthetic intermediate state is introduced
- **Completed material-intent inputs**: Types 20/23 now expose one Pass-1 action per candidate with release value, single/dual sum values, target, mandatory count, and Exact/SumGreater mode, allowing model preference to influence weighted macro-pool reduction
- **Mirrored Core checks message by message**: A shared selection validator now covers index bounds, duplicate indices, count ranges, summed release value, mandatory placeholders, dual-value sums, one-index Type 26 replies, and finish/cancel permissions
- **Removed fake Type 23 cancellation**: `MSG_SELECT_SUM` has no cancel flag. The old unconditional `int32(-1)` package always led to `MSG_RETRY`; it is now absent and rejected both at pool construction and transmission
- **Fixed Type 26 response transport**: `response_value` previously caused iterative actions to return an integer through `set_responsei`. Select/Unselect now always emits `count=1 + index` bytes, and only Finish/Cancel uses `int32(-1)`
- **Retained exits under fixed shapes**: Type 26 Finish/Cancel is pinned to the first candidate slot, while genuine Type 15/20 cancel packages are pinned during macro reduction, so an action count above 120 cannot remove the legal exit
- **Fixed response-buffer over-read**: Core's `set_responseb` copies the fixed `SIZE_RETURN_VALUE=512` bytes. The former 64-byte Python lifetime buffer caused truncation and an out-of-bounds read; responses are now held in a zero-padded 512-byte buffer and oversized payloads are rejected
- **Unified RuleBot and model legality**: RuleBot no longer maintains approximate random Type 15/20/23 implementations. It reuses the same complete-combination generator and validator, failing the episode explicitly rather than fabricating a response Core may reject
- **Schema boundary**: Framework version is 3.8.1; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision becomes 5. This batch adds no parameters or tensors; the revision locks state-machine semantics so development checkpoints cannot cross the boundary silently
- **Per-card Core-state audit**: `STATUS_PROC_COMPLETE` is the authoritative proper-summon bit used by revive-limit rules and persists through Graveyard/banishment. Core internally retains summon method/source/player, but standard client queries do not export them. They must never be guessed from card category or movement messages and may be added only through a public upstream API
- **Other public fields queued**: Dynamic alias, rank, base ATK/DEF, reason/reason card, equip target, target-card relations, all overlay materials, typed counters, original owner, disabled/forbidden/proper status, and dynamic scales/link values are now explicit later-batch items
- **Regression**: The portable UTF-8 Python suite runs 194 tests: 193 pass and one real-Core gated test is skipped as expected. Coverage includes Type 20/23 Pass 1, exact dual-value and SumGreater checks, no fake cancellation, Type 26 response round trips and index bounds, exits beyond 120 candidates, and the 512-byte buffer

---

## [v3.8.0] - 2026-09-20

### 🎯 V4 Phase 2 Batch 1: Action Families and Summon Evidence

- **Corrected the six Main Phase candidate families**: Normal Summon, Special Summon, reposition, Monster Set, Spell/Trap Set, and activate candidates from `MSG_SELECT_IDLECMD` no longer all share the `ACTIVATE` label; the policy can now distinguish their actual operation family
- **Added a summon-method input**: A `[120] uint8` `act_summon_method` field, dedicated embedding, shared-memory path, and ONNX input now distinguish non-summons, normal summons, and special summons whose exact method is unknown. Exact Ritual/Fusion/Synchro/Xyz/Pendulum/Link values are reserved for future direct evidence
- **Stopped inferring methods from card types**: Standard Core exposes a generic Special Summon candidate but no exact method. The framework therefore emits `SPECIAL_UNKNOWN`; Extra Deck source and Fusion/Synchro/etc. monster category remain separate replay facts and no longer masquerade as summon-method labels
- **Preserved Core replies**: Candidates still round-trip deterministically as `(index << 16) | action_type`. The new fields affect learning and auditing only, not legal actions, candidate order, or engine responses
- **Separated the field-only marker**: The Core `0x80000000` marker on activation candidate codes is moved into prompt flags before exact-vocabulary lookup, preventing field-only effects from appearing as nonexistent oversized card IDs
- **Corrected replay semantics**: Holographic replay records operation family, summon evidence, source zone, and monster category separately. Legacy/manual actions without the field safely fall back to “Special Summon (exact method unknown)”
- **Stabilized empty semantics in ONNX**: Fully empty semantic rows use a zero-valued attention sentinel and are forced back to zero after aggregation, preventing all-masked Softmax NaNs in ONNX backends while leaving every row with a valid semantic slot unchanged
- **Schema boundary**: Framework version is 3.8.0; Model Protocol remains 4, Checkpoint Format remains 3, and V4 schema revision becomes 4. Development checkpoints from schema revision 3 are rejected before network construction rather than partially loaded
- **Resource cost**: The only new parameters are `11 × d_model` (5,632 parameters at the baseline `d_model=512`, about 0.0113% of the current 49,653,298 parameters), and each Worker gains 120 bytes of shared input. An isolated local CPU lookup-and-add microbenchmark measured about 57.6 microseconds; board-token count, action capacity, Core interactions, and inference round trips do not increase
- **Batch boundary**: This release does not change Type 15/20/23/26 selection state machines or add equip, target, full-overlay, typed-counter, or disabled-zone relations; those remain in the next two Phase 2 batches
- **Regression**: The UTF-8 Windows suite runs 188 default tests: 187 pass and one explicitly gated real-Core case is skipped as expected; enabling that case separately completes a real-Core duel. Coverage includes all six operation mappings, enum-bound rejection, response round trips, field contracts, PyTorch/ONNX parity at `1e-3`, and ONNXRuntime execution

---

## [v3.7.0] - 2026-09-15

### 🆔 V4 Phase 1: Identity and Global State (All Three Batches)

- **Removed real-card identity collisions**: Added append-only `card_vocab.json` and replaced modulo hashing with one unique token per card. It currently covers all 14,981 entries in `cards.cdb`; real-card tokens begin at 10 and the fixed capacity remains 20,000
- **Unified every card-identity channel**: Scene cards, overlays, initial decks, chains, candidates/announcements, macro targets, and Lua-semantic references all use the same exact mapping. Missing cards explicitly map to `UNK` instead of silently colliding with known cards
- **Cross-machine append-only vocabulary**: Repository `card_vocab.json` is the default numbering authority. Ordinary WebUI/CLI data updates synchronize it and may target a custom vocabulary repository/file source. Custom-card users can append from a local CDB through a dedicated Resource Sync page or `main.py vocab`, with explicit fork and distribution warnings; resident processes invalidate their vocabulary cache when the file changes. Older V4 artifacts remain valid when their mapping is an exact prefix, while reordering or forks are rejected
- **Soft isolation and deck preflight**: A CDB that contains cards newer than the authoritative vocabulary no longer blocks the runtime. Affected decks remain on disk but leave training/Arena random pools, and WebUI preflight reports their unsupported codes
- **Fixed false failures in a resident WebUI preflight**: Every deck scan reopens the current project-root `cards.cdb` and authoritative vocabulary. If the CDB is downloaded or atomically replaced after WebUI startup, an empty/stale SQLite connection can no longer mark every deck incompatible. Successful resource updates also clear stale scan results from the page
- **Bounded online deck-pool updates**: YGOProDeck tasks now default to a 100-deck per-pool cap. Each manual or automatic update removes only that online pool's oldest excess `.ydk` files. The WebUI subscription list exposes and edits Latest/Random mode, batch size, pool cap, and current count; legacy tasks migrate to the 100-deck default
- **Configurable resource sources**: Resource Sync advanced options now expose the CDB file source and vocabulary repository/file source while retaining the MyCard Chinese CDB and Galatea raw vocabulary defaults. CLI accepts the same settings and rejects local-file schemes or embedded credentials
- **Separated semantic sync and extraction**: The remote base now defaults to the Galatea repository URL. `--sync` only downloads remote semantic assets, while `--local-update` scans local Lua and automatically continues both structured semantics and code vectors. Clear + local update without sync rebuilds from scratch; every other local-update combination continues from existing or freshly synchronized assets
- **Separated player state**: `decision_player`, `turn_player`, `starting_player`, and P0/P1 turn counts are maintained independently from Core messages. Responses during the opponent's turn are no longer mislabeled as one's own turn; the network receives relative player-role classes, `is_my_turn`, and `went_first`
- **Categorical phase and location state**: Phase, card zone, battle position, chain handler/trigger zones, and action target positions now use bounded embeddings. The old `/10` and `/100` pseudo-continuous encodings are removed, and phase categories condition both the global token and FiLM
- **Closed V4 artifact identity**: Vocabulary SHA-256, count, capacity, schema revision, and `protocol_schema_hash` are embedded throughout PTH, `net_config`, ONNX, artifact manifests, and `.gkg`. Structural hashes stay stable across valid appends while each model retains its exact training-prefix identity
- **Deployment and environment integration**: `card_vocab.json` is mandatory in portable environments and `.gkg`. Import validates models against the packaged vocabulary before append-only synchronization, allowing packages with newer cards to move safely onto older installations
- **Version boundary**: Framework version is 3.7.0, `MODEL_PROTOCOL_VERSION` is 4, and Checkpoint Format, artifact-manifest format, and deployment-package format are all 3. Per the scratch-trained V4 boundary, V3 models are not migrated
- **Implementation status**: All three Phase 1 review batches are complete. V4 schema revision is 3; Phase 2 proceeds to action semantics and public relations
- **Regression**: The default suite passes 184 tests with one real-Core case skipped as expected; that case completes a real duel when enabled separately. Coverage includes append/fork validation, configurable resource sources, separated remote/local semantic flows, soft deck isolation, refreshed preflight assets, online-pool caps, player perspectives, categorical bounds, checkpoint/ONNX/package round trips, and ONNXRuntime execution

---

## [v3.6.5] - 2026-09-07

### 🎛️ Arena Inference Policies and Replay Fix

- **Training-distribution mode**: Arena can sample from the same `Categorical(logits)` distribution used during training at a fixed temperature of 1.0, exposing the model's actual training-time policy without changing training code or model parameters
- **Deployment temperature sampling**: Temperature is adjustable from 0.05 to 5.0. Lower values approach greedy behavior while higher values explore lower-confidence legal actions; 0.8 is the recommended starting point. Existing `greedy` Argmax remains the default, preserving old command behavior
- **Shared legal-action safeguards**: All three policies retain the existing action mask, Retry hard bans, repetition soft bans, macro-action wrapper pool, and candidate-exhaustion protection. Only the final selection among filtered legal logits changes
- **Reproducible sampled benchmarks**: Under a fixed schedule, sampling uses an independent PyTorch generator initialized from each duel seed. Benchmark results now store policy and temperature metadata. Older 3.6.4 results are interpreted as greedy; direct comparisons should share the same plan, policy, and temperature
- **Synchronized CLI and WebUI**: `duel` adds `--policy-mode greedy|training|deployment` and `--temperature`; WebUI exposes all three policies, deployment temperature, and a policy column in benchmark results
- **Fixed Field Spell display in holographic replay**: A resident Spell/Trap-zone card at sequence 5 is no longer overwritten by the fixed “Field” placeholder. Empty Field Zones retain the original label, and existing replay files need no regeneration
- **Oversized GitHub Release bundles**: The portable builder now actually compresses DLL/PYD/SO runtime libraries. When the final ZIP reaches GitHub's 2 GiB per-file limit, it automatically emits approximately 1900 MiB lossless parts, a SHA256 manifest, and a Windows merge-and-verify script. Existing ZIP files can be split directly with `--split-existing`
- **Clarified CPU validation option**: `--allow-cpu-only` is explicitly limited to skipping the CUDA release probe and no longer implies that CUDA libraries are removed from an existing portable environment. Splitting and merging change only the transfer representation, not Python, PyTorch, training behavior, or model contents
- **Training and protocols unchanged**: Network, observations, action protocol, rewards, PPO, Workers, central inference, ONNX, and checkpoints are untouched. Model Protocol remains 3 and Checkpoint Format remains 2
- **Stable-release regression**: 161 automated tests pass, with one real-Core long-game test skipped unless explicitly enabled. CPU/CUDA sampling, CLI arguments, Streamlit WebUI runtime checks, runtime-library compression, and lossless split reconstruction pass

---

## [v3.6.4] - 2026-09-07

### 🏟️ Arena Deck Selection and Reproducible Benchmarks

- **Kept normal Arena and added two-sided deck selection**: P0 can use the current globally weighted ranges, one physical pool, one virtual pool, or one exact deck. P1 can use the same four independent modes, follow P0's resolved range per game, or use P0's exact deck
- **Preserved the default behavior**: Normal Arena still defaults to P0 selecting a physical/virtual range under current global weights and P1 drawing independently from that range. Existing `duel` commands need no changes. Training continues to use its original `get_random_deck_pair()`, weight cache, and sampling order, isolated from the Arena selector
- **Added fixed-schedule benchmark mode**: Before play, a benchmark freezes both decks, deck SHA-256 values, duel seeds, and seat swaps for every game. Odd/even games alternate the physical P0/P1 seats, while winners are mapped back to logical model identities, reducing deck-sampling and first-player bias in comparisons
- **Reusable plans with drift rejection**: New plans are stored under `arena_benchmarks/plans/` and may be reused for later models. Loading resolves ordinary deck files again and checks their content hashes; a modified, removed, or replaced deck causes an explicit refusal instead of an incomparable run
- **Structured benchmark results**: Files under `arena_benchmarks/results/` contain model names and SHA-256 identities, per-game winners/reasons/aborts/decision steps, overall P0 win rate with a 95% Wilson interval, seat splits, and fallback totals. WebUI compares recent results and warns that only runs sharing a plan are directly comparable
- **Input and artifact boundaries**: Plan/result JSON rejects symlinks and non-JSON files, is capped at 16 MiB and 10,000 games, and Arena cataloging accepts ordinary `.ydk` files only. Runtime benchmark files are excluded from Git and portable release archives
- **Protocol and training unchanged**: This release only changes Arena controls, deck selection, and post-game statistics. Network, observations, actions, rewards, PPO, training sampling, and ONNX are untouched. Model Protocol remains 3 and Checkpoint Format remains 2
- **Stable-release regression**: 155 automated tests pass, with one real-Core long-game test skipped unless explicitly enabled. Both Normal and Benchmark WebUI forms pass Streamlit runtime checks

---

## [v3.6.3] - 2026-09-05

### 🛡️ V3 Audit Finalization and PPO Observability

- **Explicit diagnostic audit switch**: Training and Arena no longer collect per-process V3 reports by default. CLI now exposes `--protocol-audit`, with matching Training and Arena toggles in WebUI. RuleBot self-check remains enabled by default for dedicated protocol stress testing
- **Audit logs in Storage & Logs**: A dedicated V3 Audit Reports tab now supports JSON preview, download, batch deletion, and confirmed purge
- **Location-aware sequence validation**: Auditing no longer treats legal sequence values above 31 in Deck, Hand, Graveyard, Banished, and Extra Deck as structural errors. Monster and Spell/Trap zones retain strict fixed-slot validation
- **Arrow table display fix**: The WebUI Lua-bound semantic-slot column now uses one display type, so integer bindings and the unresolved placeholder no longer trigger mixed-type PyArrow serialization errors
- **PPO health diagnostics**: TensorBoard now records `Approx_KL`, `Clip_Fraction`, `Explained_Variance`, and pre-clipping `Gradient_Norm`. They are derived from existing tensors only at the existing once-per-20-mini-batch logging interval and never participate in loss or backpropagation
- **Throughput timing diagnostics**: Each iteration records collection seconds, PPO update seconds, and valid rollout samples per second under the TensorBoard `Performance/` group
- **Stable protocols and checkpoints**: Network inputs, rewards, actions, PPO loss, and parameter updates are unchanged. Model Protocol remains 3 and Checkpoint Format remains 2, so existing V3 checkpoints resume directly
- **Stable-release regression**: 151 automated tests pass; one long-running real-Core test remains skipped unless explicitly enabled

---

## [v3.6.2] - 2026-09-04

### 🔗 Exact Lua Runtime-Effect to Code-Semantic Binding

- **Removed the invalid low-bit assumption**: The low four bits of Core `desc` retain a Stringid index; they are no longer interpreted as `Effect.CreateEffect(c)` creation order. The framework therefore no longer sets the wrong `used_effect_mask` bit or focuses a chain on an unrelated semantic slot
- **Generated mappings from Lua object identity**: Semantic parsing follows each top-level Lua Effect object to a static `SetDescription(aux.Stringid(...))` and writes the complete runtime `desc` onto the same code-semantic slot. Printed card-description text is neither read nor parsed
- **Safe fallback for ambiguity**: Dynamic expressions, passive effects without descriptions, and a `desc` shared by multiple Effect objects are never guessed. Actions, chains, and history retain whole-card semantics, and no false used-effect bit is set
- **Exact effect semantics for action candidates**: Legal actions add a compact `act_effect_slot` input. With both an entity pointer and an exact binding, the policy head directly indexes that slot's existing Lua code vector. Flattened indexing creates only candidate-scale selected vectors, never a candidate-by-eight-slot expansion, and adds no network parameters
- **Unified runtime path**: GameState, action candidates, the 12-entry chain, eight-entry activation history, and entity used-effect bits share one mapping. Chain/history attention narrows to one slot only when binding is exact; unknown bindings preserve the former whole-card mask
- **Semantic assets and audit upgraded**: Startup validates runtime-binding uniqueness, type, and range. WebUI replaces “current V3 assumed slot” with the actual Lua-bound slot and a `binding_missing` safe-fallback status; the audit report schema advances to 2
- **Current assets migrated**: All 27,647 code-semantic slots and vector-index rows remain unchanged. The KB gains 13,831 exact runtime bindings across 8,951 cards. Ninety-four identifiers shared by multiple Lua objects were detected and intentionally omitted
- **Adjacent issue fixed**: Semantic-cache construction now honors each explicit KB `slot` instead of relying on JSON list order. Model Protocol remains V3 and checkpoint/deployment container formats remain 2, but ONNX artifacts should be re-exported for this input signature
- **Safe CUDA compile fallback**: Training now checks that CUDA Inductor's Triton backend is usable before wrapping the model. A portable environment without Triton no longer fails lazily on its first forward pass; it reports the condition and uses the result-equivalent eager path
- **Full regression**: 149 automated tests pass with one environment-dependent skip, covering exact/cross-card/ambiguous binding, invalid-asset rejection, PyTorch forward, ONNX export, and ONNX Runtime inference. One real-Core model-vs-RuleBot smoke duel also passed

---

## [v3.6.1] - 2026-09-04

### 🧪 Model Protocol V3 Stabilization

- **Strict semantic-bundle cross-validation**: Encoder startup, code-semantic generation, deployment-package import/export, and one-click preflight now cross-check every modeled KB effect slot against matrix rows and index keys. Missing, stale, duplicate, or partial assets no longer silently fall back to zero vectors
- **Automatic V3 observation audit**: Training workers, Arena, and RuleBot self-check aggregate Core message counts, chain depth, handler/trigger controller and location boundaries, runtime description slots, and semantic-slot coverage per process/iteration. Reports are atomically written under `system_logs/protocol_v3_audit/`
- **Initial effect-slot mapping audit**: Runtime collection records only in-memory facts and performs no Lua I/O. Offline inspection of `SetDescription(aux.Stringid(...))` exposed that Stringid low bits are not Lua creation slots, directly motivating the object-level binding implemented in 3.6.2
- **WebUI visualization**: Semantic Knowledge Engine now contains a V3 Observation Audit tab with on-demand full-bundle validation, source filtering, and card name/runtime description/KB slot/explicit Lua slot/count tables
- **Stable protocol**: Network inputs, parameters, actions, rewards, PPO, Worker memory, and ZMQ flows are unchanged. `MODEL_PROTOCOL_VERSION` remains 3; checkpoint-container and deployment-package formats both remain 2
- **Tightened deployment boundaries**: `.gkg` import/export now treats the knowledge base, code vectors, and vector index as one indivisible runtime semantic bundle, preventing a partial overwrite from breaking the next startup
- **Full regression**: 143 automated tests completed with one environment-dependent skip, plus one real-Core RuleBot self-check duel. All 27,647 KB effect slots, 27,647 code-vector rows, and 27,647 index keys in the current assets match exactly

---

## [v3.6.0] - 2026-09-03

### 🧠 Ordered Context and Model Protocol V3

- **Fixed ineffective chain/history ordering**: The old path added position vectors and immediately averaged the sequence, so swapping A→B into B→A left the sum unchanged. `OrderedContextPool` now binds positions, models neighboring relations with an asymmetric local convolution, and performs masked attention pooling. The 12-slot chain and eight-entry recent activation history therefore produce order-sensitive representations
- **Strict padding and empty-sequence handling**: Padding is masked before and after local mixing, so even corrupted padding values cannot leak into context. A sequence with no valid items returns a finite zero vector
- **Bound effect-slot identity**: The eight Lua effect slots now carry explicit slot embeddings, allowing bit N of an entity's `used_effect_mask` to correspond to semantic effect N instead of treating a card's effects as freely interchangeable
- **Completed active-chain context**: Each chain link now contributes its card ID, effect description, chain index, handler location, triggering location, and relative controllers in addition to card semantics; all public fields are encoded from the acting player's perspective before ordered aggregation
- **Raised Model Protocol to V3**: Chain input tensors and network structure are expanded while PPO equations remain unchanged. Existing weights are structurally incompatible, so `MODEL_PROTOCOL_VERSION` moves from 2 to 3 while the checkpoint-container format remains 2
- In the default 512×6 network, ordered pooling, effect-slot identity, and chain metadata projection add 681,984 parameters, about 1.37% of the current model. A local batch-512 microbenchmark measured the ordered branches at roughly 14.5 ms on CPU and 1.0 ms on CUDA BF16 without changing Worker, central-inference, or ONNX call flows

### 🧬 Resumable Semantic Assets

- **Fixed nondeterministic Lua semantics**: Field deduplication now preserves first occurrence instead of using unordered sets, preventing Python hash randomization from changing which values survive fixed-slot truncation; script scanning is stable as well
- **Complete GitHub semantic sync**: `--sync` now retrieves the knowledge base, Hash map, code-semantic matrix, and vector index, then automatically appends missing vectors after parsing newer local scripts. If the remote Hash map is absent, continuation records can be reconstructed from custom tags already stored in the KB; the embedding model is not loaded when no slots are missing
- **Incremental code semantics**: With a valid matrix/index pair, `--embed` computes only newly added effect slots. `--clear` removes stale vectors, while dimension, row-count, or index mismatches trigger a safe full rebuild
- **Deployment Package V2**: WebUI, CLI, and `.gkg` manifests support `hash_mapping_report.json`, `code_embeddings.npy`, and `code_embeddings_idx.json`. Matrix and index must appear together with their KB, and one-click release validation now requires the complete semantic asset set
- **Full regression**: 138 automated tests completed (one environment-dependent skip). Order swapping, effect-slot identity, chain packet/view encoding, incremental vectors, semantic-package round trips, full PyTorch forward, `torch.compile`, ONNX export, and ONNX Runtime inference all passed

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
