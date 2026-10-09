# Canonical Core trajectories, AI predictions and external ingestion

3.13.1 completed Phase 7 batch 2; 3.13.2 stabilizes fixed Core decoding. Model Protocol **4**,
Checkpoint Format **3**, schema revision **13**, network parameters/inputs, rewards and standard
ONNX remain unchanged. Independent `TRAJECTORY_SCHEMA_VERSION=2` in `core_trajectory.py` adds
fixed message-protocol identity and automatic empty-chain responses. Raw Schema 1 traces are no
longer accepted as current traces. Holographic JSON remains format 3; formats 1/2 remain viewable.

## Enabling recording

3.13.3 adds public observations and schema revision14. Trajectory2/replay3 remain unchanged, but older-schema raw traces cannot share new observation checks. See [Public observations](public_observations_en.md). The opening unchanged-network/input statement applies only to3.13.2.

Both options default to off and affect only games selected by a positive `--thought_freq`.
They can be used independently; training Workers and PPO storage are unchanged.

```powershell
.\python_env\python.exe main.py duel --p0 .\models\galatea_iter_100.pth --thought_freq 5 --record-trajectory --record-evaluations
```

WebUI Arena offers “Record canonical Core trajectories” and “Record AI prediction curves”.
The replay expander displays independent perspective win/remaining-event curves, the current
decision's draw probability, and available actual-terminal references. Rule agents have no
fabricated predictions; old recordings are not retroactively evaluated.

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
  temperature and existing internal calibration provenance, using only that side's pre-action encoded
  observation. UI labels say “AI predictions”; internal `uncalibrated` metadata is not a warning title.
- Independent views/models need not produce complementary curves. `P(win)` differs from expected score
  `P(win)+0.5×P(draw)`. PPO value estimates discounted returns, not win probability.
- Recorded policy mode/temperature are provenance, **not decoder inputs**. Predictions learn the training
  continuation/opponent mixture; do not assume calibration for a deployment temperature, greedy continuation,
  or a specific human opponent.
- Length inverts the training target `log1p(events)/log1p(1500)`: an estimate, not an exact expectation.
  It counts **both sides' action-transition events**, not turns or remaining own actions. Display values
  cap at one million events with an explicit clipped flag, without affecting policy/training.
- Only real `MSG_WIN` adds separate `evaluation_actual` outcomes and remaining transition counts,
  including the current action. Predictions stay immutable; aborted/truncated games get no fabricated labels.
- Curves reflect the model's assessment, not actual win rates or a sole measure of playing strength.
  Standard ONNX still returns only logits/value; prediction recordings require full PTH.

## Cost and next batch

Disabled options execute no diagnostic decoder, checkpoint hashing, or raw recording. Enabled options
add a small decoder, scalar GPU→CPU synchronization, candidate/observation hashing and gzip I/O only to
logged games; checkpoint hashes are read once and asset hashes at first capture. This is not zero cost,
but PPO/shared-memory/training pools do not grow. Checks may scan gzip multiple times to validate before
execution while keeping memory bounded.

Next: Link/YRP raw-input adapters, whole-duel/deck/player isolation and quality gates. Cognition additions
for Type 120/160/165 and 35/37/38/162 await design approval; see the message audit. This batch does not
revive the parked Replay Worker, add imitation optimization, or counterfactual search.

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

WebUI: **Storage & Logs → Trajectory Ingress & Quality** has four numbered workflow sections,
with headings and input/output explanations before controls. Question-mark help covers source,
file, directory, split seed and actions.

1. **Inspect & Convert Raw Replays**: choose YRP/YRP3D or raw Link captures. Read-only inspection
   never calls Core or writes files. Managed conversion writes
   `replays/imported_trajectories/*.core.jsonl.gz` and `replay_data/ingest_reports/*.json`, without
   modifying the source or starting learning. Each source has separate file-selection state.
2. **Audit & Split Canonical Trajectories**: recursively scans only `.core.jsonl.gz`, including
   native Arena and converted diagnostic traces. Writes `replay_data/datasets/dataset_*.json`
   without moving traces. The seed controls grouped splits, not duel/model randomness; targets
   default to80%/10%/10%, but group isolation does not guarantee exact file-count proportions.
3. **View Quality Audit Results**: shows candidate train/validation/test and quarantine counts
   plus per-trace rejection reasons. Counts are files, not steps or already-trained samples.
   Expandable help explains common codes and insufficient-independent-validation warnings.
   Viewing a manifest does not re-audit; re-run after adding data or changing assets.
4. **File Management by Category**: four separate collapsible areas for raw Link captures,
   imported diagnostic traces, ingress reports and dataset manifests. Category, directory,
   suffix and count precede controls, with category-specific deletion labels. Batch deletion/
   clearing affects only matching files directly in that directory, never subdirectories or
   related artifacts. Deleting a manifest keeps its referenced traces; native traces stay in
   Thought Replays. No UI undo is available; download backups first.

Progress appears in Control & Logs and `system_logs/ReplayIngest*` / `DatasetQuality*`; an existing
managed task is not replaced. No automatic Core replay runs on page refresh and no external assets
are downloaded. This UI patch remains3.13.5 without protocol/checkpoint/training changes. Outputs
require new filenames. Conversion failures seal a diagnostic prefix and return CLI status2;
format/asset refusal returns1. Successful diagnosis does not authorize training.

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
trusted training authorization. No imitation optimizer, counterfactual search, hindsight evaluation
backfill or external Link client wiring is implemented. Ordinary training adds no per-step work or
storage when these manual tools are unused.

### Acceptance evidence

Tests cover compression/plain data, zero/cancel bytes, framing/magic/truncation/multi-packet errors,
seed sequences, unsupported modes, partial capture refusal, asset mismatches, transitive isolation,
corrupt-file quarantine, duplicate removal and exclusive report creation. Synthetic real-Core full
Link capture passes conversion/replay/gates; same-source YRP1/YRP2 reconstruction remains ineligible
without historical evidence. All13 local YRP/YRP3D samples are inspectable;4 non-UNIFORM legacy samples
are diagnostic-only. Existing YRP1/YRP2 samples consume1,137/809 responses through genuine terminals,
with zero Retry and complete parsing;724/389 observation digests and413/420 automatic empty-chain
responses pass independent replay. Macro mappings are incomplete, so neither becomes training data.
Production long training and live online Link end-to-end acceptance are not claimed.
