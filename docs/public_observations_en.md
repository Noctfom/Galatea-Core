# 3.13.3: Public hints, timing feedback and known deck positions

This batch connects Types120,160/165,37/38 and162 to the shared state/observation path. It does not modify public Core, rewards, decision-request counts, MCTS or a universal active-effect tracker.

## Versions and usage

Framework3.13.3; Model Protocol4 and Checkpoint Format3 remain unchanged; schema revision14 rejects revision13 and older development models without partial loading/migration. Public-observation format1, event format2 and hint-catalog extension1 are independent. Raw Trajectory Schema2 and holographic JSON3 remain unchanged. Standard ONNX still returns logits/value, but adds public inputs. No new CLI/WebUI switch is needed; external Link adapters must update fields and schema identity.

## State and lifetime

| Message | Information | Lifetime/boundary |
| --- | --- | --- |
| 120 | Missed-timing code/location, result flag, notification order and last processing chain | Separate from the original action source; no desc/target/cause IDs and no extra penalty |
| 160 subtypes1–5 | Latest typed32-bit value | Does not overwrite identity/stats/counters; ordinary moves clear the scalar, same-coordinate position changes do not |
| 160 subtypes6/7 | Instance+desc reference counts | ADD increments, REMOVE decrements; follows known movement/reindexing/swaps/material layers |
| 165 | Affected player+desc reference counts | No field-slot assumption or invented expiry; persists beyond16-step history until Core REMOVE |
| 37/38 | Reversal and legally known deck identities/positions/face-up flag | Both decks remap;38 uses logical size when consumed, including before DRAW; shuffle invalidates order |
| 162 | Fixed7/8 slots, positions/overlay counts, LP, zone counts and chains | Strict whole-packet validation before atomic rebuild; public query completes available identities, never invents unknown ones |

Hand shuffle clears mappings before Core re-emits description ADD, avoiding doubled references. Hidden set-card shuffles cannot use internal identity queries to secretly follow instances. Unrecoverable hints are cleared with an incomplete-state flag; unmatched REMOVE never produces negative counts. Player hints survive source-card departure.

Hint presence does not guarantee the corresponding rule is active. Age is elapsed time since notification, not remaining duration. Effects without CLIENT_HINT, including relevant bundled Maxx “C”/Droll effects, are not magically covered. CARD_QUESTION blocks graveyard query identities/dynamic state only for the player receiving that viewing restriction; legitimate prior event memory is not erased.

## Reusing Lua semantics

No additional language model or code-vector asset is introduced. A small catalog is compiled from existing knowledge_base runtime bindings and raw_code. Unique proven initial bindings are `exact`; callback-created hints use an explicitly labeled `parent context` vector already containing their callback code. Comments/strings, dynamic expressions and ambiguous sources cannot become precise bindings. Unknown hints retain their full raw32-bit desc, encoded as four bytes; no modulo identity or desc-low-bit slot guessing is used.

Host, affected player, declared card, semantic source and parameter remain separate. Only the selected code row is projected, not all8 effect slots. Catalog records live in existing `semantic_lookup_v1.json`, participate in the semantic hash and travel through existing semantic synchronization/GKG paths. Missing extensions rebuild automatically from complete local source assets; old compiled-only deployments must be regenerated/repackaged. Existing code embeddings need not be regenerated merely for this catalog addition.

## Model and cost

Lua parsing now persists `public_hint_parent_desc_ids` while raw code is available; cleaned remote assets can retain these small IDs without raw_code. Only613 local effects currently retain raw_code, so many old entries—including the current Magic Eye/Gravekeeper's Inscription entries—have no provable parent binding. Their full identities still enter learning. Completing these sources requires trusted full semantic extraction/generation, not slot-only refresh or silently matching current scripts to old vectors. Existing knowledge/vector assets are not rewritten by this batch.

An additional read-only audit found that the current `code_embeddings.npy` contains 27,647 rows of 384 dimensions but only 530 byte-distinct vectors; 27,007 rows are identical. Within that group, 26,942 effect records no longer retain raw_code and 65 still do. This is a priority semantic-quality risk, not proof of the original generation cause or a regression introduced here. The existing generator reads an empty string when code is absent and incremental generation reuses existing keys, which may be related. A follow-up should verify provenance, empty-code vectors and incremental invalidation before deciding on trusted regeneration. This batch does not overwrite those source vectors or claim complete local code semantics.

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
