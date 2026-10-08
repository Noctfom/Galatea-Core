# 3.13.1: Canonical Core trajectories and state-evaluation replay

Phase 7 batch 2 preserves Model Protocol **4**, Checkpoint Format **3**, schema revision **13**,
all network parameters/inputs, rewards, and the standard ONNX contract. Independent
`TRAJECTORY_SCHEMA_VERSION=1` lives in `core_trajectory.py`. Holographic JSON advances to
format 3; formats 1/2 remain viewable.

## Enabling recording

Both options default to off and affect only games selected by a positive `--thought_freq`.
They can be used independently; training Workers and PPO storage are unchanged.

```powershell
.\python_env\python.exe main.py duel --p0 .\models\galatea_iter_100.pth --thought_freq 5 --record-trajectory --record-evaluations
```

WebUI Arena offers “Record canonical Core trajectories” and “Record state-evaluation curves”.
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
- Raw Core chunks, actually consumed messages, snapshot/query timing and raw candidate pools.
  Generated macro pools and raw coordinates are retained rather than resampled; full tensors are not saved.
- Actual signed integer or ordered byte responses, actor/prompt, all matching candidates, selected index,
  actual CPU Encoder digest, and optional evaluation. Runtime metadata retains policy/macro RNG seeds.
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
scripts. Existing same-process callback bindings and parsing dialect are restored afterward.

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

Real smoke testing found that the **existing** parser leaves some Type 16 chunks partially/unconsumed.
The complete raw chunk is retained with `unparsed_core_bytes`; raw replay can still match, but behavior-
cloning eligibility is false. This batch deliberately preserves parsing/decision behavior; audit the public
Core message dialect separately in the next batch rather than claiming complete parser coverage.
Per-chunk `parsed_bytes/message_count/parse_complete` also localize coverage gaps.

## Evaluation semantics

- Optional reads use the existing state planner's terminal/length decoder, not the action-conditioned
  auxiliary head or future-resource branch. No new weights, policy/value modifications, prediction-driven
  selection, tree search, or MCTS.
- Save loss/draw/win probabilities, physical perspective, UUID/iteration/checkpoint digest, policy mode/
  temperature, and `calibration.status=uncalibrated`, using only that side's pre-action encoded observation.
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
- Whole-duel held-out validation and calibration are required before treating these as accurate Go-AI-style
  win-rate curves. Standard ONNX still returns only logits/value; prediction recordings require full PTH.

## Cost and next batch

Disabled options execute no diagnostic decoder, checkpoint hashing, or raw recording. Enabled options
add a small decoder, scalar GPU→CPU synchronization, candidate/observation hashing and gzip I/O only to
logged games; checkpoint hashes are read once and asset hashes at first capture. This is not zero cost,
but PPO/shared-memory/training pools do not grow. Checks may scan gzip multiple times to validate before
execution while keeping memory bounded.

Next: Link/YRP raw-input adapters, public-dialect audits, whole-duel/deck/player isolation and quality
gates. This batch does not revive the parked Replay Worker, add imitation optimization, or counterfactual search.
