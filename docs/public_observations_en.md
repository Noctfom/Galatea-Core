# Public observations, prompts and chain context

> Current usage applies to v3.16.9; dated sections retain their original contracts and measurements

## 3.16.9 Current prompt and chain context

The observation grows from **108 to 116 fields, schema revision 14 to 15**. Model Protocol 4, Checkpoint Format 3, backbone configurations such as 512/8/6, and every learned parameter name and shape remain unchanged. Current facts are separate from the 16 completed transition events. No synthetic actions, rewards, supervision or PPO rows are added.

| Source | Added facts | Lifetime and uncertainty |
| --- | --- | --- |
| Type 2 / SELECTMSG(3) | Exact 32-bit selection parameter, prompt kind and recipient | Available before the choice; 15/18/20/23/24 consume it on completion; 26 retains it through repeated select/unselect; finish or leaving the flow consumes it; Retry preserves it; a new hint replaces it |
| Other known Type 2 subtypes | Received event text, selected options, declarations, card/zone hints | Recipient-specific; chain notices last until chain end, unlinked notices retain the latest 16 per viewer; raw values are not automatically effect-slot IDs |
| 70/71 | Original chain index and building/confirmed status | Nodes retain their indices until 74 |
| 72/73 | Which chain began resolution and which has resolved | 73 retains the last resolving index with an explicit resolved flag; next 72 replaces it; 74 clears it |
| 75/76 | Activation negation and effect disabling of the specified chain | Stored separately, not inferred from aggregate event bits |
| 83 | Announced objects, announcement locations, known identities and tracked current locations | Follow provable moves/swaps; shuffle or uncertain identity invalidates tracking without rebinding a new occupant of the old slot |
| 162 | Reloaded chain sources | Omitted statuses, targets and selection purposes remain unknown rather than inheriting old caches |

SELECTMSG can contain a card code for a placement prompt, not just a cost/target purpose ID. MSG83 can also be emitted by `Duel.HintSelection`, so these are **announced objects**, not guaranteed rules-level targeting relationships. They remain separate from persistent `QUERY_TARGET_CARD` relations.

Without an active chain, Core may announce each card in a group with a separate MSG83. The receiver accumulates these objects until the next decision rather than keeping only the final card. Retry retains that window; a subsequent decision or game, turn or phase boundary clears it. Repeated-state signatures include prompt and chain context so a changed selection purpose is not mistaken for the same loop.

Type 2's player byte is a wire-routing field, not a universal actor: 1/2/3/5 go to that player; 4/6/7/8/9 to the other player; 10/11 to both. The controlled WindBot bridge fixes subtype 4 routing and subtype 11 broadcast to match the [upstream client delivery logic](https://github.com/Fluorohydride/ygopro/blob/master/gframe/single_duel.cpp#L593). Unknown UI hint subtypes neither invent observations nor alone abort a valid game.

### Inputs and forward path

`public_observation.CURRENT_CONTEXT_INPUT_SPECS` defines eight `current_context_*` fields with 256 fact slots: `mask`, `kind`, `code`, `location`, `chain`, `offset`, `subtype`, `value`. Exact identities use four bytes, players use relative roles, and card codes respect viewer knowledge. Capacity overflow fails explicitly rather than silently dropping objects or evicting transition history.

`PublicContextEncoder.encode_current_context` reuses existing notification embeddings, pooling, projections and gates. Per-chain summaries enter their matching chain representations; prompts and unlinked facts enter global public context, then the shared state, policy, value and planner. The existing chain radar still has 12 slots; additional facts beyond chain 12 retain their raw index in the global group. No main 120-card tokens or learned parameters are added.

Empty context matches the previous network bit for bit. Real added facts change outputs; existing notification gates are retained, and newly used types may require learning. Continued training updates shared notification parameters. Preserving weights does not promise unchanged decisions or already-correct use of the new evidence.

Additional storage is 5,120 bytes/sample, about 160 MiB for 32,768 samples; worker and merged pools count separately. On a 5070 Ti at width 512/FP16, the new branch alone measured approximately 0.132 ms for batch 12 CUDA Graph forward and 7.65 MiB extra peak; batch 512 forward/backward measured 12.91 ms and 332.88 MiB extra peak. These are branch measurements, not whole-training overhead or long-run memory guarantees.

### Continuation, exports and replay

- PTH artifacts matching the complete known revision 14 schema can migrate to 15 with all weights retained. Normal PPO resume preserves Adam, AMP, counters and model identity; BC-to-new-PPO initialization keeps its existing separate-optimizer rules
- Ordinary loading leaves the source file untouched. The next save uses the current schema and records `observation_protocol_updates`. Vocabulary IDs, semantic encoder compatibility and unknown schema differences remain checked
- Restart trainer and workers together and collect a fresh round. Do not combine old encoded PPO rows/old log-probabilities with newly constructed observations
- Old ONNX graphs lack the added path. Use the existing Resource Sync ONNX check/update entry with matching PTH files to re-export; no semantic vectors, vocabulary or hash categories need rebuilding. Relabeling ONNX metadata cannot add the computation
- Old normalized trajectories/BC caches retain strict schema checks. Rebuilding current observations requires complete raw messages, queries and responses; replacing digests is insufficient. This release does not automatically migrate old normalized trajectories. Old holographic JSON remains viewable without invented missing facts
- New holographic states store `prompt_context`, per-chain status/objects, `resolving_chain_index` and unlinked announced objects. Core training, Arena, controlled sessions and offline reconstruction share state logic. Link model readers must adopt the eight fields and revision 15; real-server readiness gates are not automatically enabled

Rewards, temperature, GAE, KL limits, DeckFiLM strength and auxiliary targets are unchanged. Strategic decisions remain learned; the added facts reduce aliasing between genuinely different visible situations.

## 3.13.3 Historical implementation

This batch connects Types120,160/165,37/38 and162 to the shared state/observation path. It does not modify public Core, rewards, decision-request counts, MCTS or a universal active-effect tracker.

## Versions and usage

Framework3.13.3; Model Protocol4 and Checkpoint Format3 remain unchanged; schema revision14 rejects revision13 and older development models without partial loading/migration. Public-observation format1, event format2 and hint-catalog extension1 are independent. Raw Trajectory Schema2 and holographic JSON3 remain unchanged. Standard ONNX still returns logits/value, but adds public inputs. No new CLI/WebUI switch is needed; external Link adapters must update fields and schema identity.

## State and lifetime

| Message | Information | Lifetime/boundary |
| --- | --- | --- |
| 120 | Missed-timing code/location, result flag, notification order and last processing chain | Separate from the original action source; no desc/target/cause IDs and no extra penalty |
| 160 subtype0 | Clear the latest scalar hint | Independent description references remain; the clear notification uses existing event/replay fields |
| 160 subtypes1–5 | Latest typed32-bit value | Does not overwrite identity/stats/counters; ordinary moves clear the scalar, same-coordinate position changes do not |
| 160 subtypes6/7 | Instance+desc reference counts | ADD increments, REMOVE decrements; follows known movement/reindexing/swaps/material layers |
| 165 | Affected player+desc reference counts | No field-slot assumption or invented expiry; persists beyond16-step history until Core REMOVE |
| 37/38 | Reversal and legally known deck identities/positions/face-up flag | Both decks remap;38 uses logical size when consumed, including before DRAW; shuffle invalidates order |
| 162 | Fixed7/8 slots, positions/overlay counts, LP, zone counts and chains | Strict whole-packet validation before atomic rebuild; public query completes available identities, never invents unknown ones |

Hand shuffle clears mappings before Core re-emits description ADD, avoiding doubled references. Hidden set-card shuffles cannot use internal identity queries to secretly follow instances. Unrecoverable hints are cleared with an incomplete-state flag; unmatched REMOVE never produces negative counts. Player hints survive source-card departure.

### 3.14.1 hint-boundary hotfix

Type160 retains its exact nine-byte layout and accepts Core subtypes0–7. Public Core may emit CLIENT_HINT during script initialization before inserting a card into a zone: `controller=PLAYER_NONE(2), location=0`. This was reproduced with20989253. Empty zones for controller0/1 also have no visible host. Following public-client behavior, these notifications are ignored without creating P2 cards, occupying hint/history slots, marking observations incomplete or creating holographic events. Raw trajectories still retain the actual packets.

A valid-zone hint does not require its host to have reached the Python query mirror, avoiding information loss from update ordering. Subtype0 clears only the scalar;6/7 reference lifetimes remain unchanged. Unknown subtypes, invalid controllers, PLAYER_NONE with a nonempty zone and truncated/trailing bytes still fail, with location/subtype/value diagnostics. Validation is not disabled, and no guessed active-effect tracking is introduced.

Model Protocol4/Checkpoint3/revision14, all23 public input fields and semantic asset identities remain unchanged. Compatible checkpoints resume after restarting training/arena processes; updating source does not patch already-running Workers.

Hint presence does not guarantee the corresponding rule is active. Age is elapsed time since notification, not remaining duration. Effects without CLIENT_HINT, including relevant bundled Maxx “C”/Droll effects, are not magically covered. CARD_QUESTION blocks graveyard query identities/dynamic state only for the player receiving that viewing restriction; legitimate prior event memory is not erased.

## Reusing Lua semantics

No additional language model or code-vector asset is introduced. A small catalog is compiled from existing knowledge_base runtime bindings and raw_code. Unique proven initial bindings are `exact`; callback-created hints use an explicitly labeled `parent context` vector already containing their callback code. Comments/strings, dynamic expressions and ambiguous sources cannot become precise bindings. Unknown hints retain their full raw32-bit desc, encoded as four bytes; no modulo identity or desc-low-bit slot guessing is used.

Host, affected player, declared card, semantic source and parameter remain separate. Only the selected code row is projected, not all8 effect slots. Catalog records live in existing `semantic_lookup_v1.json`, participate in the semantic hash and travel through existing semantic synchronization/GKG paths. Missing extensions rebuild automatically from complete local source assets; old compiled-only deployments must be regenerated/repackaged. Existing code embeddings need not be regenerated merely for this catalog addition.

## Model and cost

Lua parsing preserves `public_hint_parent_desc_ids` while source is available; cleaned bundles may retain those IDs with matched vectors. The3.13.3 audit's legacy library retained source in only613 effects, so unproven hints remained unknown.3.13.4 implemented trusted provenance extraction/vector rebuilding, and continuation maintains hint origins. Actual bindings follow the installed assets: legacy counts are not the current asset state, and effect names cannot justify attaching current scripts to old vectors.

The legacy vector collapse was confirmed in3.13.4 as missing/empty-source generation plus key-only reuse, not coarse hashes. The trusted rebuild produced26,092 distinct vectors for27,647 slots, with27 explicitly incomplete static origins. These are dated rebuild evidence, not fixed quality requirements; inspect current assets through Code Semantic Quality or `semantic-check`. See [provenance repairs and activation](code_semantics_en.md).

The common23-field specification stores IDs/categories/parameters, not expanded vectors:128 current hints,256 positions per deck, and8 ordered notifications per event across16 events. Public coordinates retain full8-bit sequence/material indices. Hint overflow rejects incomplete observations; event-notification overflow keeps the first8 and sets MESSAGE_OVERFLOW.

A maximum128-dimensional lightweight branch pools hints/known positions. Host residuals enter the existing120-card backbone; dynamic global context joins policy/value/shared planning, without changing stable DeckProfile/style or exposing Side. Ordered notifications use the existing event encoder. Zero-initialized bounded gates preserve initial outputs for identical shared weights, but learning these new facts can subsequently change strategy. Empty pools are safe; identity IDs are not numeric magnitudes and elapsed age uses log scaling.

Local512/8/6 eager measurements compare the same network with/without the branch, not production training:

- 8,710 extra bytes/step, or272.19MiB for32,768 observations; allocation headroom/main merge pools also matter and preflight includes new fields.
- Branch665,472 parameters; including auxiliary output expansion,665,986 additional parameters (~1.26%),53,661,973 total.
- CPU batch1 forward81.99→83.81ms (+2.2%); CUDA BF16 batch6 forward15.23→16.37ms (+7.5%); batch128 forward/backward267.92→279.11ms (+4.2%), about250.69MiB additional peak allocated memory.
- Empty current-hint encoding ~0.028ms/observation. Hardware/scenes/batch/threading affect results; these are not guaranteed whole-run percentages or zero cost.

## Replay and validation

Holographic replay saves hints, known deck positions and their recorded semantic bindings. The public-state expander shows references, parameters, source/slot/quality and top offsets; new messages have event labels. Older JSON remains readable without fabricating missing facts.

Tests cover packet truncation/joining, reference lifetimes, movement/shuffle/reversal/draw order, hidden information and relative views, semantic catalogs/hash integrity, finite gradients/zero-gate equivalence, ONNX inputs/numerical agreement, public-DLL snapshots/player hints, real-duel capture/replay and a CPU centralized-inference/PPO smoke run. Final counts are in the changelog; production-scale long training remains unvalidated.

Type35 bulk Grave/Deck exchange remains deferred; Type161 Tag mode is not added. Type162 cannot restore hints/hidden order/history absent from its payload. Such observations are not claimed complete and raw trajectories retain quality gates.
