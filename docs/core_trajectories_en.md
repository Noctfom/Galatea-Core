# Canonical Core trajectories, AI predictions and external ingestion

This document applies to **v3.15.3**: Model Protocol **4**, Checkpoint Format **3**, schema revision **14**.
Canonical traces and prediction replay arrived in 3.13.1, fixed Core decoding in 3.13.2,
public observations in 3.13.3, and external ingestion/quality gates in 3.13.5.
Independent `TRAJECTORY_SCHEMA_VERSION=2` in `core_trajectory.py` adds
fixed message-protocol identity and automatic empty-chain responses. Raw Schema 1 traces are no
longer accepted as current traces. Holographic JSON remains format 3; formats 1/2 remain viewable.

## Enabling recording

Trajectory format 2 and holographic format 3 are maintained independently of the observation schema;
older-schema raw traces cannot use current observation checks. For public events, hint lifetimes
and known deck-position displays, see [Public observations](public_observations_en.md).

Both options default to off and affect only games selected by a positive `--thought_freq`.
They can be used independently; training Workers and PPO storage are unchanged.

```powershell
.\python_env\python.exe main.py duel --p0 .\models\galatea_iter_100.pth --thought_freq 5 --record-trajectory --record-evaluations
```

WebUI Arena offers “Record canonical Core trajectories” and “Record AI prediction curves”.
The replay “AI prediction curves” toggle displays independent win/draw probabilities, remaining-event
curves and available actual-terminal references. Its state persists across pauses and seeks.
Rule agents have no fabricated predictions; old recordings are not retroactively evaluated.

### Prediction curves and playback position

- **Horizontal axis**: zero-based replay steps, matching the slider and including model decisions, rule decisions and visible events. These are neither turns nor training-sample counts.
- **Vertical axes**: the upper chart shows each AI's prediction of its own eventual win (0–100%); the lower chart shows estimated remaining action-transition events, nonnegative and not formatted as probabilities.
- **Colors and cursor**: P0 blue, P1 orange. The dashed line marks the playback step; dots mark each side's latest valid prediction at or before it. Hover shows the source step, perspective, win/draw/loss probabilities and remaining events.
- **Per-frame evaluation**: new recordings evaluate model sides on every saved visible-event/decision snapshot, including opponent actions and terminal events. Both models have curves; RuleBot receives no invented predictions. Existing actor predictions are reused; observers run only the original state/planner weights, without action towers, sampling or new training rows.
- **Snapshot timing**: events use mirrors maintained by already-consumed messages, with dynamic attributes from the latest query correction. They do not query Core ahead or put future state into past frames. Decisions use existing query-corrected snapshots; hover identifies the source. Per-frame means recorded replay frames, not every notification, internal engine state or wall-clock instant.
- **Figures/legacy replays**: P0/P1 display predictions separately at the current step and remain visible on pause. Missing values retain only earlier predictions with source labels, never future values. Old replays retain their original decision predictions without automatic recomputation; blank spans/events are normally expected.
- **Missing-data diagnostics**: shows counts/spans and rejection reasons. New recordings count expected model-side frames without predictions; legacy recordings count only explicitly model-sourced decisions without predictions. RuleBot is excluded. Format, units, perspective and numerical failures are explained rather than shown as zero probabilities.
- **Actual outcomes**: separate references give each source decision's real outcome and remaining-event count. They do not enter predicted curves or figures. The sides' predictions need not add to one, and PPO value is not a win probability.

See [prediction validation](../decision_evaluation.py) and [prediction UI](../replay_evaluation_ui.py).
Optional same-frame `evaluations` / `evaluation_actuals` fields preserve legacy single-view fields
and holographic format3. Individual predictions remain format1; parameters, input protocols,
action selection, training data and checkpoints are unchanged.

Raw trajectories: `replays/core_trajectories/*.core.jsonl.gz`. Holographic JSON remains in
`ai_thoughts/`. Both are manageable in Storage & Logs → AI Thought Records. Trajectories contain
omniscient local raw data and both deck builds, including Side for future upper-layer adapters;
consider information boundaries before sharing. **Side never enters the single-duel Encoder.**

## One interpreter and recorded fields

`CoreTrajectoryInterpreter` reuses the existing `MessageParser`, `DuelState`, and
`AiBot._pack_response`; there is no second Core message parser. Records include:

- Source, physical seats/swapping, original builds, Core seed/reset parameters and **actual injection
  order**; Core/CDB/loaded card-cache/loaded Lua SHA-256, V4 vocabulary/semantic/protocol identities;
  model UUID, internal iteration/configuration and a checkpoint digest read once at Arena initialization.
  `core_message_protocol` identifies the bundled fixed layout; no ghost-byte toggle is recorded or applied.
- Raw Core chunks, actually consumed messages, snapshot/query timing and raw candidate pools.
  Generated macro pools and raw coordinates are retained rather than resampled; full tensors are not saved.
- Actual signed integer or ordered byte responses, actor/prompt, all matching candidates, selected index,
  actual CPU Encoder digest, and optional evaluation. Runtime metadata retains policy/macro RNG seeds.
- Empty Type 16 `automatic_response` records save the actual `-1`, prompt and actor without a model
  observation, action, transition event or behavior-cloning label. Replay reports `automatic_response_count` separately.
- Stream checksum and terminal/prefix outcome, recording/parse/Retry/mapping quality flags.

### External response-label adapters (3.15.0)

Original responses always drive submission and replay. Label comparison may use an explicit source
contract. Ordinary Arena, RuleBot, YRP and old canonical files without a declaration retain strict
`canonical_response` byte matching. Conversion of a full WindBot capture uses
`windbot_select_card_set_v1` only for seats explicitly identified as `windbot`; the new canonical
header's optional `response_label_adapters` maps physical seats to contract IDs.
[Core's session interface](../integrations/duel_session.py) owns dispatch; the set conversion itself
stays in the WindBot adapter.

Only valid Type15 physical-index sets use the same comparison key on both expert/candidate sides.
Same-code copies remain distinct; neither raw responses nor pool actions are rewritten. The existing
validator checks counts, index bounds, duplicates and cancellation first. Ordered Type21/25 and other
prompts keep byte comparison. Independent replay rechecks all matching under the recorded version,
then still compares original Core output and observation digests. Unknown contracts/wrong participant
kinds fail closed. Unique matching, the120-slot boundary, real terminal and all other gates remain;
expert answers are never inserted into candidates.

Old canonical WindBot files are not upgraded in place: re-convert the original `*.link.jsonl.gz` and
rebuild quality/split manifests. Eligibility is not an instruction to start training, and provenance
metadata is not identity authentication. See [WindBot label equivalence](../integrations/windbot/README.md#原始重放与学习标签等价性).

Ordinary Arena shuffles decks in Python as well as using Core RNG, so the seed alone is insufficient.
Replay uses actual injection order but original lists for stable `DuelState` profiles. Query markers
preserve transition-history completion boundaries and the exact pre-decision observation.

Streaming gzip avoids retaining raw histories in memory: 1 MiB per record, 64 MiB **uncompressed**
per duel, at most 100,000 records. Hitting a limit disables recording and marks truncation while
the duel continues. UUID filenames avoid same-second collisions. Readers enforce matching line/count/
decompression limits and restore only whitelisted JSON fields: no pickle, downloads, archive paths, or
private Core modifications. Checksums detect corruption; **they are not signatures or source authentication**.

## Read-only checks and deterministic replay

```powershell
# Format/boundaries/checksum only; no Core
.\python_env\python.exe main.py trajectory-check .\replays\core_trajectories\duel_UUID.core.jsonl.gz
# Additionally verify current public Core output, mappings and observations; CPU, no policy network
.\python_env\python.exe main.py trajectory-check .\replays\core_trajectories\duel_UUID.core.jsonl.gz --replay
```

A bounded temporary-file copy freezes input against post-validation path replacement. The whole file
is validated before Core execution; runtime assets must match before deck injection.
Replay follows recorded chunks/messages/pools/query markers/responses and compares every raw Core chunk,
response mapping, and available Encoder digest. Mismatching assets are rejected, not substituted with new
scripts. Existing same-process callback bindings are restored afterward; the parsing layout is fixed.

| Result | Meaning |
| --- | --- |
| `integrity_valid` | Format/checksum passed; not source trust or training eligibility |
| `replay_verified` | Recorded duel/complete prefix passes message/response/observation checks; the cosmetic exception is explicit below |
| `raw_output_exact` | Every raw Core chunk matches byte-for-byte; false always forbids behavior-cloning eligibility |
| `behavior_clone_eligible` | Additional necessary gates: real terminal, no quality flags, unique mapping and all observation checks; not authorization to start imitation training |

Recording-cap truncation cannot be replay-verified. A duel step-limit prefix can be verified, but has no
real terminal label. Rule macro responses can lack unique atomic-candidate mappings; retain raw replay
without inventing action labels. Retry-rejected responses are not treated as correct demonstrations.

Public Core resets effects through an unordered index, so multiple client-hint removals can have varying
iteration order. Only **consecutive Type 160 / subtype 7 removals for the same card/location**, with
identical message multisets, may continue diagnostic replay in recorded order while validating every
later response/observation. Report `raw_output_exact=false` and `client_hint_reordered_chunks`;
**behavior-cloning eligibility remains false**. Raw files, live Core, training and Arena are never normalized.
Cross-card ordering, hint additions, values/actions, nonconsecutive events and incomplete parsing remain
strict. This is not a claim that arbitrary old Core builds support byte-exact replay.

3.13.2 fixes the empty Type 16 coverage failure discovered in 3.13.1: forced flags are per candidate,
not in the header. Type 31's extra byte is `skip_panel`. See the [fixed message audit](core_message_audit_en.md).
Per-chunk `parsed_bytes/message_count/parse_complete` remain; other unconsumed bytes still set
`unparsed_core_bytes` and prohibit cloning. These fixes do not imply complete cognition of every message.

## Evaluation semantics

- Optional reads use the existing state planner's terminal/length decoder, not the action-conditioned
  auxiliary head or future-resource branch. No new weights, policy/value modifications, prediction-driven
  selection, tree search, or MCTS.
- Save loss/draw/win probabilities, physical perspective, UUID/iteration/checkpoint digest, policy mode/
  temperature and existing internal calibration provenance, using only that side's snapshot-time encoded
  observation. UI labels say “AI predictions”; internal `uncalibrated` metadata is not a warning title.
- Independent views/models need not produce complementary curves. `P(win)` differs from expected score
  `P(win)+0.5×P(draw)`. PPO value estimates discounted returns, not win probability.
- Recorded policy mode/temperature are provenance, **not decoder inputs**. Predictions learn the training
  continuation/opponent mixture; do not assume calibration for a deployment temperature, greedy continuation,
  or a specific human opponent.
- Length inverts the training target `log1p(events)/log1p(1500)`: an estimate, not an exact expectation.
  It counts **both sides' action-transition events**, not turns or remaining own actions. Display values
  cap at one million events with an explicit clipped flag, without affecting policy/training.
- Only real `MSG_WIN` adds separate `evaluation_actual` / dual-view `evaluation_actuals` outcomes and counts
  from the marked transition sequence (zero at the terminal event). Predictions stay immutable, without forcing
  the last model point to100% win/loss; aborted/truncated games receive no fabricated labels.
- Curves reflect the model's assessment, not actual win rates or a sole measure of playing strength.
  Standard ONNX still returns only logits/value; prediction recordings require full PTH.

## Cost and remaining boundaries

Disabled options execute no diagnostic decoder, checkpoint hashing, or raw recording. Enabled options
add state encoding/backbone inference, a small decoder, scalar GPU→CPU synchronization and prediction storage, plus candidate/observation hashing and gzip I/O when tracing is enabled, only to
logged games; checkpoint hashes are read once and asset hashes at first capture. This is not zero cost,
but PPO/shared-memory/training pools do not grow. Checks may scan gzip multiple times to validate before
execution while keeping memory bounded.

Core-side Link/YRP adapters, whole-duel/deck/player isolation and quality gates arrived in 3.13.5;
see below. These diagnostic commands do not train. See the explicit [3.15.3 BC workflow](#3153-independent-imitation-workflow);
the external Link project still needs SDK integration.
Public cognition for Types 120, 160/165, 37/38 and 162 arrived in 3.13.3. Type 35 remains deferred
by agreement and Tag 161 unsupported. The abandoned Replay Worker has been removed;
trajectory recording invokes neither a BC optimizer nor counterfactual search.

## 3.13.5: External ingestion and isolated quality gates

Phase7 batch3 implements the **Core-side** contract. The separate Link project must wire the SDK;
live online end-to-end acceptance is not claimed. Model Protocol4 / Checkpoint3 / revision14 /
static format1 / GKG4 are unchanged, as are the network, observations, rewards, PPO, standard ONNX,
and normal `GalateaEnv.reset`. After confirming no live imports, the abandoned `replay_worker.py`
and `yrp_parser.py` are removed; the new reader is `yrp_ingest.py` with the single Core interpreter.

### CLI and WebUI

```powershell
.\python_env\python.exe main.py replay-ingest .\replays\example.yrp --source yrp --inspect
.\python_env\python.exe main.py replay-ingest .\replays\example.yrp --source yrp --report .\replay_data\ingest_reports\example.json
.\python_env\python.exe main.py replay-ingest .\replays\link_captures\example.link.jsonl.gz --source link --inspect
.\python_env\python.exe main.py replay-ingest .\replays\link_captures\example.link.jsonl.gz --source link
.\python_env\python.exe main.py dataset-build --directory .\replays --output .\replay_data\datasets\review_001.json --seed 20261009
```

In3.15.3, WebUI processing is unified under **Control & Logs → Independent Imitation → tab1**.
Choose a file/directory, teacher seats and optional split settings, then one action reads the format,
reconstructs where needed, verifies labels and assigns splits. Complete Link sources reconstruct once;
canonical traces are strictly read once. Separate conversion, dataset-audit and expert-read clicks are
not required. The older CLI commands above remain developer diagnostics, not mandatory training steps.

Each task writes `preparation.json` under `replay_data/imitation/<batch>/`. Usable sources also produce
`dataset.json` and `cache/`; diagnostic-only batches explain their exclusions without creating a dataset.
Ordinary YRP without original messages/historical assets stays diagnostic: self-replay cannot prove eligibility.

**Storage & Logs → Trajectory & Imitation Files** only views, exports and cleans files. Read/refresh the
inventory, filter data batches/BC experiments/legacy source groups, select categories and adjust individual
files. New batches include captures, converted traces, caches, reports, Bot logs and optional holographic
replays; BC experiment groups include checkpoints and matching TensorBoard. Legacy copies/reports join
through direct provenance; a multi-source manifest does not join every referenced game. Deletion affects
only explicitly listed managed files, never externally referenced sources, and is disabled during managed tasks.

### YRP boundaries

`yrp_ingest.py` handles32-byte YRP1,80-byte extended-header-v1 YRP2 and explicitly framed YRP3D with
exactly one231 replay packet. No magic-byte scanning or ignored trailing garbage. Tag, single-script,
unknown extensions and ambiguous multi-replay containers fail explicitly. Limits:16MiB input,
8MiB decompressed data,64MiB LZMA dictionary,100,000 responses; actual sizes must match declarations.
Nicknames are display-only and trailing UTF16 padding is not an identity.

Reconstruction follows the public client's fixed profile: YRP1 uses the first `std::mt19937(seed)`
output as the actual Core seed; YRP2 preserves all eight seed words through the bundled public
`create_duel_v2`. Arrays are injected in stored order at position8, not the parked worker's reverse-
order guess. References: [headers](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay.h),
[StartDuel](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay_mode.cpp),
[YRP3D framing](https://github.com/purerosefallen/ygopro-yrp3d-encode).
The Core binary/source is untouched. Non-UNIFORM legacy files are inspectable, but conversion does
not switch into legacy Core modes; future migrations require separate review.

Original responses advance strictly: no skipped cancels, no guessing after Retry, and no inserting
the expert answer into the candidate pool. Macro pools reuse the existing constructor with local
fixed RNG/uniform priors and the existing120-action cap. Expert answers outside that pool remain
replayable raw responses, not supervised labels. No policy model, fabricated PPO statistics or
training optimizer is used. The conversion budget defaults to300s, configurable from1 to3600s.

Reconstruction proves only consistency **under current assets**, not historical scene accuracy.
Ordinary YRP lacks historical asset hashes and original full Core output; flags
`historical_assets_unknown/original_core_output_unavailable` keep behavior-cloning eligibility false.

### Link capture contract

The optional 3.15.0 controlled runner reuses this SDK without the Link project. Optional `external_session` contract-v1 metadata contains a session UUID, both physical participants, executors, declared source commit and actual executable digest. It is not a model input or source authentication. Conversion preserves it under `ingress`; existing captures without it retain the old contract.

Type23's responder is `payload[1]`; `payload[0]` is the selection mode. State, training dispatch, Arena, ingestion and independent replay share Core's player reader. Old traces with a mode-derived wrong actor are rejected rather than silently relabeled. Complete YRP evidence and imitation optimization remain pending; see the [controlled module](../integrations/windbot/README.md).

`LINK_CAPTURE_FORMAT_VERSION=1` lives in `trajectory_ingest.py`. The SDK records raw Core chunks and
actually submitted responses, using physical seats0/1; it does not operate the policy.

```python
from trajectory_ingest import LinkCaptureWriter
capture = LinkCaptureWriter(visibility='player_view', perspective=0,
                           player_ids={'0': 'service:user_a', '1': None})
capture.record_chunk(raw_core_chunk)  # Core bytes after removing network framing
capture.record_response(actual_response, actor=0)
capture.finish(winner=winner, reason=reason, core_terminal=real_msg_win_seen)
```

Full server/local-bridge captures use `visibility='full_core'`, plus `reset` (actual seed/seed_sequence,
injection order, LP/hand/draw/rules/initial_position), stable `decks` for both seats and original
`assets=runtime_asset_identity(env)` from the capture environment. Side metadata never enters the
single-duel Encoder. Import cannot substitute local hashes for missing historical hashes. Full
asset parity, byte-exact original output, response/actor/terminal alignment and independent replay
are prerequisites; partial views stay diagnostic without invented hidden data.

Stable anonymized `player_ids` include the source-service namespace, not nicknames or seats. YRP may
receive `--p0-player-id/--p1-player-id`; Link import cannot overwrite captured identities. This does
not modify automatic model UUIDs. Raw records can expose hidden cards, deck lists and linked identity:
redact before sharing. Hashes detect corruption, not authentic sources, signatures or expert skill.
Prefer isolated CLI processes for native replay rather than unreviewed captures in live inference.

### Dataset gate and transitive isolation

`DATASET_MANIFEST_VERSION=1` lives in `trajectory_dataset.py`; at most10,000 canonical files are
inspected using frozen bounded copies. Corruption, recording truncation, missing real terminals,
unverified origins, missing stable players, Retry, raw-output/observation divergence or unreliable
candidate mapping cannot qualify. Known-bad sources are quarantined before unnecessary native replay.

Complete-duel identity excludes recording UUID/display provenance; duplicate repackaging is removed.
Deck identity includes Main/Extra/Side and exact card counts, not filenames or input order. Shared
duels, exact decks or stable players form **transitive connected groups**, each assigned wholly to
one split. Model UUIDs/rule bots also count as shared players. Defaults select validation/test groups
at0.1 each, deterministically by seed; file counts are not guaranteed to match those fractions.
Unknown player identities remain quarantined. One connected group cannot produce a genuinely
independent holdout; warnings report insufficient groups/empty holdouts instead of splitting it.
Similar deck archetypes are not automatically clustered—only exact compositions are isolated.

Manifests contain relative paths, file/duel hashes, grouping, origin/assets/outcome and quality counts,
not tensors or PPO data. A future learner must recheck file hashes/identities; editable JSON is not
trusted training authorization. These diagnostic manifests do not start learning; independent BC is
provided by3.15.3 below. Counterfactual search, hindsight backfill and external Link client wiring
remain separate work. Ordinary PPO incurs no extra per-step work from these tools.

## 3.15.3: Independent imitation workflow

WebUI: **Control & Logs → Independent Imitation**, with Collect / Prepare Replays, Independent BC Training,
and Playing Evaluation tabs. CLI commands use `imitate`. They do not automatically join PPO,
ordinary Arena games or online-human learning. Recording a game does not start learning. WindBot as
an online PPO opponent is a separate explicit [training option](../integrations/windbot/README.md#optional-main-training-opponents)
since 3.15.4; BC-to-PPO initialization also requires explicit selection.

### 1. Teacher, validation and final-test pools

```bash
python main.py imitate collect --train-executor Blue-Eyes --games-per-executor 10 --output replay_data/imitation/example
# Optional unseen-executor evaluation:
python main.py imitate collect --train-executor Blue-Eyes --validation-executor Albaz --test-executor BlueEyesMaxDragon --games-per-executor 10 --output replay_data/imitation/executor_holdout
```

Training updates weights; validation checks demonstration fit and records the best checkpoint each epoch;
final test is reserved for explicit evaluation, never checkpoint selection. These measure learning, not
executor installation/import. Default episode splitting allows the same teachers/builds in each split,
using different complete games. Choose `--split-mode none` to train on everything. Build/executor
holdouts are optional stricter generalization checks, not a prerequisite for everyday imitation.

Only separate-executor holdouts require disjoint lists. Each executor mirrors
its paired build and both real teacher seats provide examples. `plan.json` freezes executor, declared
level, actual YDK hash, role, Core seed range and executable hash before games. Levels
`Normal/Easy/NotFinished/Test` are upstream declarations, not measured strength ratings.
Optional `--families families.json` accepts an explicit `{"executor":"family"}` mapping. Identical
builds or declared families cannot cross held-out roles; names are not guessed into families.

Collection randomizes the initial Core seed by default, incrementing it per duel and recording actual
values in the plan/captures. `--seed 73` pins Core starts; `--split-seed 73` independently controls data
assignment. Neither controls WindBot's own RNG.

Examples are prepared in the actual Core session without separate conversion/replay engines. Retry,
parse failure, disconnection or no real terminal prevents publication of that game's cache. An action
outside the model's unique120-slot mapping loses only its label and is counted; answers are never
injected into the pool. No per-game human approval is needed. Raw captures remain available for
independent diagnostic replay.

### 2. Prepare complete existing sources once

```bash
python main.py imitate prepare replays/imported_trajectories --split-mode episode --output replay_data/imitation/imported_example
```

Scans `.core.jsonl.gz`, `.link.jsonl.gz`, `.yrp` and `.yrp3d` without importing an executor catalog or
requiring WindBot installation for existing files. One task checks formats, reconstructs, extracts
labels and assigns splits. Complete Link captures produce examples during one
original-output reconstruction. Older canonical traces use one strict replay/read; epochs then use
the prepared cache without Core. The older canonical reader still requires whole-duel unique mapping;
if raw captures remain, the new raw preparation path can retain individually usable labels.
Ordinary YRP without original output/historical evidence and player-only Link captures remain
ineligible. Missing private information is not invented.

Ordinary YRP reconstruction writes diagnostic results without a redundant independent self-replay.
Diagnostic-only batches still save preparation.json, but no dataset.json or training starts.

The teacher-seat setting selects whose actions become labels, not a viewing perspective or starting role.
The default teachers are explicitly declared WindBot seats. Human/other sources require
`--expert-seat 0` and/or `--expert-seat 1`. Provenance is a declaration, not authentication or proof of
skill. Losses are not automatically excluded. `preparation.json` records excluded sources, not a queue
waiting for human approval.

| Split mode | Grouping | Does not establish |
|---|---|---|
| `none` | All games train, duplicate wrappers still deduplicated | Held-out validation or final-test conclusions |
| `episode` | Complete duel, same opening and human/unknown identities; programs/builds may be shared | Cross-build/executor generalization |
| `deck` | Above plus exact Main/Extra/Side composition | Automatic isolation of similar-build families |
| `executor` | Above plus WindBot executor; explicit validation/test names, remaining executors train | That a parent PPO model has never seen the deck |

Human/unknown associations remain transitive; program IDs remain provenance without automatically
joining the entire corpus. Duplicate wrappers of one game are deduplicated. Episode/deck modes target
80%/10%/10% by groups, not exact file counts. At least one training group remains; a single group
stays entirely in training rather than fabricating a frame-level holdout. No-validation datasets
can support engineering fitting, but cannot claim held-out checkpoint selection.

### 3. Start BC explicitly

```bash
python main.py imitate train --dataset replay_data/imitation/example/dataset.json --init-from models/your_v4_model.pth --epochs 5 --batch-size 64 --lr 0.00001 --device auto
```

Each epoch traverses training examples once, optimizing cross-entropy of the actual expert candidate
index. Forced single-choice states contribute neither policy loss nor accuracy. Inverse effective
sample counts balance teachers so long/high-volume teachers do not dominate. Existing V4 Main/Extra
inputs remain unchanged; Side is provenance/isolation metadata, not model input. Macro pools retain
canonical uniform priors and do not claim exact PPO Pass1 distribution equivalence.

Shared encoders, deck/FiLM, policy branches and the existing planning-to-policy path learn. Critic,
structured auxiliary heads, planning target decoder and planning value branch receive no direct
supervision. Their outputs can still drift when shared representations change: BC prediction curves
are not automatically revalidated win rates. There are no invented PPO advantages/behavior
probabilities or tactical rewards.

The dataset's `cache/` holds bounded compressed numeric blocks with deduplicated stable deck profiles.
Only mini-batches reach the GPU; trainable Encoder/latent outputs are never frozen into the cache.
Relative references allow moving the whole dataset. BC consumption checks vocabulary/semantic
protocol identity, not an identical native Core binary across operating systems. Collection Core/Lua/CDB
hashes remain provenance. Prepare new data after those source resources change; old caches are not
silently rewritten.

Outputs go to `models/imitation/<run_id>/`: epoch reports, saved PTH/artifact manifests and stage
results. New tasks generate new UUIDs and record parent checkpoint/dataset hashes. Epoch monitoring
reads validation, never final test. Balanced validation cross-entropy identifies the best checkpoint,
which may remain the initialization checkpoint. Action accuracy is not playing win rate.

Resume with `--resume path/to/galatea_iter_200_bc_N.pth`; `--epochs` adds more imitation passes. Dataset
digest, batch size, learning rate and seed must match; historical-best validation selection is retained.
Use `--init-from` to branch instead.

| UI choice | Weights | Optimizer / UUID / epochs | Result |
|---|---|---|---|
| New experiment | Copy selected PTH | New optimizer/UUID, epoch zero | Branch without changing the source task |
| Continue BC experiment | Restore BC PTH | Restore momentum/scaler/RNG and cumulative epochs | Append epochs; epoch5 plus2 ends at7 |

Checkpoints
retain Format3/Protocol4/revision14 and support inference. `training_stage=imitation` prevents ordinary
PPO `--resume` from restoring a BC optimizer. Continuing into PPO requires explicit weight initialization.

### Dual counters, model management and continuing PPO

`galatea_iter_200_bc_5.pth` means cumulative PPO iteration200 and BC epoch5. Continuing PPO to210
saves `galatea_iter_210_bc_5.pth`; another BC epoch saves `galatea_iter_210_bc_6.pth`. Checkpoints
also store both counters/parent provenance, independently of filenames. Local BC `iteration` remains
the current experiment's epoch, never PPO age. Separate directories/UUIDs prevent collisions even when
two experiments have identical counters.

Storage & Logs → Model Storage automatically lists BC/PPO branch directories. Tables show full names,
stages and cumulative counters; deployment packaging accepts those directories too. Main training and
ordinary Arena list PTH recursively. Original3.15.3 BC metadata supplies parent-derived display names
without silently renaming existing files or rewriting old replays.

Selecting BC on the main training page explicitly initializes a new PPO branch via `--init-from`:

```bash
python main.py train --init-from models/imitation/RUN/galatea_iter_200_bc_5.pth --additional-iterations 10 --device auto --no_compile
```

Creates `models/ppo_branches/<new UUID>/galatea_iter_200_bc_5.pth`, then trains to210. Weights are
inherited exactly; PPO optimizer/scaler and task-local update/rollout counters reset while cumulative
PPO200/BC5 remain. BC files/optimizers are untouched. The ordinary PPO algorithm, including existing
auxiliary/planning warm-up, applies; BC updates are not counted as PPO updates. Use `train --resume`
only to restore the resulting PPO branch's optimizer/progress.

### 4. Fixed-opponent game evaluation

```bash
python main.py imitate evaluate --dataset replay_data/imitation/example/dataset.json --checkpoint models/imitation/RUN/galatea_iter_200_bc_5.pth --baseline models/your_v4_model.pth --split validation --games-per-seat 5 --record-replay --output replay_data/imitation/eval_example
```

Candidate and optional baseline run the same Core-seed/both-seat schedule. Both model and Bot use
the actual held-out build frozen into the output directory, not a substituted executor default.
`--split test` explicitly runs final testing. `--executor NAME` overrides regression opponents and is
marked as a regression override, not unseen-opponent evidence.

The WebUI defaults to known-teacher regression, which requires no separate executor pool. Held-out
validation/final test use reserved data. This tab runs actual games, unlike example CE/agreement on
the training tab. Candidate/baseline share a schedule without updating weights. Random starts are
recorded by default; `--seed` or disabling random mode pins the schedule. WindBot RNG remains separate.

Default `training` policy usesT=1; `greedy` takes argmax; `deployment --temperature 0.2` samples at the
specified temperature. Core seeds do not fully control WindBot randomness, so evaluation repeats a
fixed configuration rather than guaranteeing byte-identical games. Reports include wins/losses/draws,
aborts, completed-game win rate, executor breakdowns and descriptive Wilson95% intervals. Paired seeds
are correlated; intervals are not guarantees of strictly independent trials. Infrastructure failures
are not wins or fabricated draws, and opponents are not replaced mid-game. Captures carry
`imitation_evaluation` and automatic preparation excludes them from learning.

`--record-replay` saves Holographic Replay files and defaults to recording the model side's state
predictions; use `--no-record-evaluations` to omit their extra CPU inference cost. WebUI exposes the
same choice. WindBot has no fabricated prediction curve. Candidate/baseline replays use separate
subdirectories and appear in the unified Holographic Replay selector, with full model names and
independent counters. Existing recordings are not retroactively evaluated. WebUI tab3 reads the report. TensorBoard
uses `runs/imitation_<run_id>` with BC epoch on the X-axis; see [metric definitions](tensorboard_metrics_en.md#10-independent-imitation-metrics).
Engineering regression does not establish GPU long-run behavior, stronger play across all executors
or online-human client integration.

## 3.15.2: Expert samples from complete verified traces

The reader reconstructs **V4 CPU observations + unique candidate indices of actual actions** for
demonstration-data checks. This read-only entry does not train; the independent3.15.3 workflow provides
the optimizer and experimental checkpoints.
Partial Link views, ordinary YRP without original evidence, unmapped decisions and nonterminal
diagnostic traces remain ineligible.

### CLI and WebUI

```bash
# Defaults to participants explicitly declared WindBot, not native model/RuleBot opponents
python main.py expert-read replays/imported_trajectories/duel_EXAMPLE.core.jsonl.gz

# Human/other demonstrations require explicit physical seats; both seats can be requested
python main.py expert-read replays/core_trajectories/duel_EXAMPLE.core.jsonl.gz --expert-seat 0 --preview-rows 5 --report replay_data/expert_reports/review_001.json
```

Optional `--expected-sha256` checks a manifest's `file_sha256` before native execution.
`--preview-rows` accepts0–20;0 reports counts only. Report files are created exclusively, never overwritten.
Participant/expert declarations are **not authentication or proof of playing strength**. Losses are not
automatically excluded.

`expert-read` remains a developer CLI/API for inspecting labels/input specifications, removed from
the everyday WebUI. Unified preparation already performs these checks and creates reusable caches;
there is no extra verification step to click. Old JSON reports remain viewable/downloadable/deletable
in Storage & Logs → Trajectory & Imitation Files.

### Core reader API and lifecycle

```python
from trajectory_dataset import open_verified_demonstrations

with open_verified_demonstrations(path, expert_seats=[1]) as examples:
    print(examples.report['sample_count'])
    for example in examples:
        observation = example['observation']  # All existing Encoder CPU inputs, batch=1 retained
        action_index = example['action_index']  # Actual action in the original pool, 0..119
        provenance = example['metadata']  # Seat, identity, record index and file/duel/observation hashes
```

The reader freezes a bounded source copy, checks format/provenance, performs **one** strict native replay
and observation-digest check, and stages selected samples in a temporary cache. Iteration opens only
after every whole-duel gate passes. It reuses the existing interpreter, visibility, Encoder and explicit
response-label contracts. There is no alternate parser or expert-answer injection into candidate pools.
Cancel, select/unselect and multi-select retain actual submitted responses. Automatic empty-chain
replies create no examples. Labels must be unique, within120 slots and unmasked.

Examples contain no invented PPO behavior probabilities, advantages or tactical rewards. Side remains
isolation metadata, never model input; opposing hidden builds do not become the own stable deck profile.
Limits are8MiB/sample and512MiB of anonymous temporary storage/duel. Exhaustion rejects the read rather
than releasing a qualified prefix. Numeric arrays are staged without pickle and decoded individually
on CPU. Exiting `with` closes/deletes the cache; callers retaining tensors own that memory usage.
Only explicit offline reads incur this cost. Normal PPO, Arena and WindBot sessions create no cache
or additional inference.
Core callbacks are process-global: this is not a thread-concurrent reader. Do not replay alongside
another duel in the same process; use isolated processes for parallel work. WebUI already launches
the reader in a separate CLI process.

### Program identities and isolation audit

Rebuilding a dataset adds `participants`, per-seat decision counts and `identity_audit` to the existing
format1 manifest. Older reports remain viewable. Actual splits still use transitive complete-duel,
exact Main/Extra/Side composition and **all stable participant identities**, with no policy relaxation.

The audit lists shared program identities, declared kinds and duel counts, and computes a hypothetical
group count without program-identity edges. This is diagnostic—not another split or independent
validation. Human/unknown identities remain isolated; prefixes such as `windbot:` do not determine
kind. Conflicting declarations for the same identity retain the more conservative non-program edge.

Identical builds still join groups without program edges. Generalization claims for shared models,
executors or similar builds require a defined evaluation goal and a separately reviewed split policy.
Do not invent player IDs or randomly split frames to manufacture a holdout. Future learners must
recheck referenced file digests rather than trusting a stale editable eligibility flag.

## 3.15.1: Core-owned player-view ingress

The developer SDK is [`integrations/player_view.py`](../integrations/player_view.py), with independent
`PLAYER_VIEW_FORMAT_VERSION=1`. It defines ingress and diagnostic projection for future Link reuse,
not a Link client, imitation learner or training toggle. Protocol4/Checkpoint3/revision14 are unchanged.

### Message/query contract

- `YGOProFrameDecoder.feed` handles fragmented/coalesced uint16-LE frames and preserves STOC types.
  The caller handles lobby/control packets; only type1 payloads are host GameMessages, one per call
  in this profile. Neither network frames, Start nor refreshes are original Core chunks for `record_chunk`.
- `PlayerViewAdapter.accept_game_message` separates Start4, full-zone UpdateData6, single-card UpdateCard7
  and visible events. Start preserves physical seats, LP and Main/Extra counts, without opponent builds/Side.
  Repeated Start, spectator/Tag, unsupported layouts, out-of-order IDs and malformed data fail closed.
- Explicit `query_mode='replace'` rebuilds the updated row's attributes; `'merge'` combines cached deltas
  for the same known instance. Type6 replaces the zone's slot collection; Type7 changes one slot.
  A length4 row is empty; length8/flags0 in an uncached profile means unknown attributes, not all-zero
  attributes. The host's cache settings, not guessed bytes, determine these semantics.
- The existing `parse_legacy_card_query` decodes binary fields. Sparse `ViewCard.fields` and `known_flags`
  preserve known zero versus unknown; unsupported flags are rejected. Only an explicit `requested_flags`
  contract proves the public Core's omitted `QUERY_EQUIP_CARD` means no target rather than unavailable data.
- Entire refreshes are validated/masked before atomic application. Movement, control changes, shuffling,
  set-card relocation and reload invalidate affected instance caches, without same-code state resurrection.
  Queries never establish hidden Main Deck order; opposing hand/face-down identities and dynamic values
  are masked again. Grave restrictions apply to the viewer, not the graveyard owner.
- After actually sending an own response, `record_submitted_response(ResponseSubmission(...))` uses
  session UUID/request/physical-seat checks. Refreshes and empty-chain automatic replies create no decisions.

Minimal wiring, with transport/response sending owned by the caller:

```python
from integrations.player_view import PlayerViewAdapter, YGOProFrameDecoder
from integrations.duel_session import ResponseSubmission

view = PlayerViewAdapter(
    physical_seat, query_mode='replace',
    own_main=own_main_codes, own_extra=own_extra_codes,
)
decoder = YGOProFrameDecoder()
for packet in decoder.feed(network_bytes):
    if packet[0] == 1:  # This profile carries one host GameMessage per packet.
        view.accept_game_message(packet[1:])

# Only after sending an actual response, not an unexecuted model suggestion.
request = view.gate.pending
view.record_submitted_response(ResponseSubmission(
    request.session_id, request.request_id, physical_seat, actual_response,
))
projection = view.diagnostic_projection()
print(projection.coverage)
```

### Completeness, scope and cost

`diagnostic_projection()` calls the existing `DuelState.get_snapshot` on a detached copy and returns
`PlayerViewProjection(snapshot, coverage)`, without finalizing the live transition recorder. Legacy V4
data-class defaults in the projection are **not proof that unavailable fields equal zero**. Always retain
the coverage sidecar: physical perspective, message sequence, cache mode, stale zones, missing fields,
failure/terminal state and source limitations.

This conservative profile cannot prove all opponent response boundaries, Type160 hints or dynamic
attributes. Both `v4_observation_complete` and `behavior_clone_eligible` remain false; `encode_v4`
raises `IncompletePlayerViewError` instead of silently zero-filling a model input. Matching synthetic
projections/encodings proves the known-field mapping, not complete online V4 inference. Native training,
full Core capture and existing learning gates do not pass through this partial view and lose no inputs.

WindBot reuses lightweight builders and the unchanged visibility filter only: no extra adapter, model or
per-step inference is instantiated. Explicit adapters maintain one bounded sparse cache; diagnostic reads
copy current state and are not zero-cost per-frame operations. Limits include65535-byte network packets,
8192-byte query rows,256 rows/zone and100000 messages. No engine is launched/advanced or PPO/example row
created. Actual Link networking and full-view acceptance remain separate work.

### Earlier trajectory acceptance evidence

Tests cover compression/plain data, zero/cancel bytes, framing/magic/truncation/multi-packet errors,
seed sequences, unsupported modes, partial capture refusal, asset mismatches, transitive isolation,
corrupt-file quarantine, duplicate removal and exclusive report creation. Synthetic real-Core full
Link capture passes conversion/replay/gates; same-source YRP1/YRP2 reconstruction remains ineligible
without historical evidence. All13 local YRP/YRP3D samples are inspectable;4 non-UNIFORM legacy samples
are diagnostic-only. Existing YRP1/YRP2 samples consume1,137/809 responses through genuine terminals,
with zero Retry and complete parsing;724/389 observation digests and413/420 automatic empty-chain
responses pass independent replay. Macro mappings are incomplete, so neither becomes training data.
Production long training and live online Link end-to-end acceptance are not claimed.
