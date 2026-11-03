# 3.13.4: Code-source provenance and content-aware semantic rebuilds

> Usage guidance applies to v3.16.7; dated repair and measurement evidence below remains historical.

## Scope

This page covers offline semantic maintenance and the 3.13.4 provenance repair. Semantic maintenance does not change Model Protocol **4**, Checkpoint Format **3**, revision **14** or static format **1**; the duel network, input shapes, rewards, PPO/GAE and legal candidates retain their existing definitions. Deployment containers separately support [GKG4/5](deployment_packages_en.md). Generation manifest format **1** lives in `code_semantic_provenance.py`; extraction versions live in `lua_semantic_source.py`.

Coarse hashes and code vectors are independent channels. The existing `_hash_code_block()` normalization still groups variables, booleans and numbers to expose shared patterns. Embeddings use actual code, preserving literals and logic. Only duplicate/stale incremental registry memberships are cleaned; classification rules are not redesigned.

## Confirmed cause

The old 27,647×384 matrix had only 530 byte-distinct vectors; 27,007 rows were identical. Of that group, 26,942 records lacked `raw_code` and 65 had empty code. Offline encoding of the empty string matched the dominant vector (cosine 1, maximum absolute difference about 2.22e-7). Missing code was silently embedded as empty text, while key-only incremental reuse perpetuated it. Coarse hashes were not the direct embedding input.

Extraction also omitted registration-only effects and SetValue callbacks and duplicated Operation code. Long inputs risked tail truncation. Matching keys/dimensions did not establish valid semantic provenance.

## Pipeline

1. Tokenize comments and strings correctly and use block stacks for nested functions, conditions, loops and repeat. Keep existing Effect-object slot ordering rather than renumbering by descriptions or hashes.
2. Per slot, collect static context, creation/registration, Clone overrides and callback closure, including SetValue and uniquely resolvable local/cross-card/utility/procedure functions. Never execute Lua or copy public native Core internals.
3. Record source SHA-256, block roles/names/spans, dependency hashes, completeness and unresolved references. Dynamic/ambiguous references retain known code and are explicitly incomplete, not empty or falsely complete.
4. Re-extract missing legacy sources and changed scripts/dependencies. Same-key changed code regenerates; removed effects do not leave stale slots. Unchanged certified assets reuse all vectors without loading the encoder.
5. Cover long code with role/function windows of at most 256 encoder tokens and 32-token overlap. Block/part positions are text prefixes. New-token-weighted mean and L2 normalization retain the existing **384-dimensional** output. Coverage does not imply program execution, full reasoning or MCTS.
6. `code_embeddings_meta.json` records actual encoder weights/config/tokenizer identity, pinned revision, chunk policy, per-slot source/encoding hashes and coverage, and vector/index file hashes. Temporary tokenizer padding/truncation state does not change encoder identity.
7. Legacy unmanifested vectors remain loadable as original model assets but cannot be trusted for incremental reuse; first local update fully rebuilds. Modern sources require a matching manifest. Source/vector mismatch fails explicitly.

Provenance is an integrity record, not a signature or proof of semantic understanding. The encoder remains the original general-purpose `all-MiniLM-L6-v2`; this does not introduce a specialized code model or new duel network.

## Build, inspect and activate

```powershell
.\python_env\python.exe main.py parse --output semantic_builds/v3.13.4/knowledge_base.json --local-update
.\python_env\python.exe main.py semantic-check --directory semantic_builds/v3.13.4
```

`semantic-check` is read-only: no Lua scan, download, embedding generation or log-file creation. WebUI: **Semantic KB Engine → Code Semantic Quality**. Checks run only on request; execution, hash browsing and card browsing share the selected asset directory. Selecting it does not activate training assets, which remain project-root assets.

During the initial 3.13.4 acceptance run, rebuilt assets were staged under `semantic_builds/v3.13.4/`
without replacing the then-current root assets; that `Galatea_Semantics_3.13.4.gkg` was about
87.32 MiB. This historical record does not mean every clone or portable bundle contains that
staging directory, nor that root assets are still the old library. Active assets are those in
the project root, checked with `semantic-check`; selecting another inspection directory does not activate it.

To activate a separate build or received GKG, use the deployment manager's extraction/staging
page to import matching complete source and compiled-runtime groups. Back up originals, stop
training/duels and restart after import. Do not replace assets in a running process.

**New vectors change `semantic_lookup_hash`, not network protocol.** From 3.16.1, compatible semantic updates can be used automatically by existing PTH weights. Content identity and weight compatibility are checked separately; see [Existing models and semantic updates](#existing-models-and-semantic-updates). Renumbered labels or a different vector encoder remain incompatible. The staging directory is gitignored; publish rebuilt assets through a GKG or complete semantic source bundle.

## Existing models and semantic updates

Semantic updates can add cards or revise existing effects and code vectors. Whether an existing model can use an updated library depends on compatible card/semantic-label IDs and vector encoding. The rules are:

| Use | Validation and behavior |
|---|---|
| PTH loading, arena/deployment, PPO resume or BC weight initialization | Card IDs must retain an append-only prefix; category/requirement IDs must not be renumbered. Encoder fingerprint, revision, dimension and chunk/aggregation policy must match. Compatible updates use current semantics and preserve learned weights |
| Deterministic trajectory replay, expert examples and prepared-data caches | Exact semantic content identity is retained. Keep original assets or prepare data again rather than substituting new observations for recorded ones |
| Exported ONNX | Static semantics are baked into the graph. Validation checks the semantic prefix for its original card range. Pure appends preserving old semantics remain valid; changes to old semantics require PTH fallback or re-export |

An appended library does **not** invalidate every old engine. The hash represents logical semantics in the old card range, not complete file equality or physical dictionary row numbers. An old graph remains usable for old cards when that prefix is unchanged. Spare vocabulary capacity does not give it new cards' static effect knowledge. This includes deck profiles, the board, actions, chains, history and public hints, rather than only cards currently on the field. Core still supplies legal candidates; missing knowledge does not mean the parser invents actions. Revised old slots, code vectors or relevant public-hint bindings can also affect old cards, so avoiding new cards is insufficient after such changes. Original matching assets can still run the original graph. Prediction quality always depends on training.

### ONNX engine updates

There is one WebUI entry: **Update Manager → ONNX Engine Update**. It uses current project-root vocabulary and semantics to maintain original training directories, without automatically building GKG packages or a separate model tree.

- **Check**: Recursively inspect PTH/ONNX pairs, protocol, semantic prefixes and external dependencies. Results distinguish current engines, pure appends without new-card coverage, revised old semantics, missing/incomplete artifacts and outdated PTH asset metadata. ONNX-only files can be checked but need PTH for re-export. Checking neither modifies models nor generates assets
- **Update**: Select PTH files across model pools or BC/PPO branches. Re-export original learned weights with current compatible semantics on CPU and compare three input cases. Replace an existing same-name ONNX or create it when missing. Refresh PTH/net_config asset versions and compatibility records when needed; retain previous provenance in semantic_asset_updates

**Model UUID, prefix, PPO/BC counters, every learned parameter, optimizer, Scaler and RNG state remain unchanged.** Vocabulary/semantic fingerprints describe asset versions and change with those assets; they are separate from model identity. A metadata refresh can change the PTH file checksum without creating a new model or retraining. PTH files already recording current assets are not repeatedly rewritten.

Maintain PTH, ONNX and the iteration manifest together in place, restoring that model's original files if replacement fails. Earlier completed models are retained. Temporary backups are removed after success and require temporary disk space. Check and Update are independent. End main training first; later Workers use the original engine path, while already-open inference sessions retain their loaded graph.

```bash
python main.py onnx-update --directory models --check
python main.py onnx-update --checkpoint models/galatea_iter_210.pth
python main.py onnx-update --directory models
```

This uses existing assets: **no vector regeneration, vocabulary renumbering or Hash reclassification**. Network, reward and PPO algorithms remain unchanged. Verification details use existing artifact manifests and system logs, without new manually maintained files. Distribution remains part of the existing deployment/packaging workflow.

Core retains strict load validation and offers no risk-acceptance bypass for old semantic graphs. PTH can rebuild from compatible new semantics; ONNX must update actual graph constants rather than hashes alone. With only an old ONNX, retain its original assets or obtain refreshed artifacts from its publisher. Link-side risk acceptance is handled separately in Link and is not changed here.

### Automatically maintained compatibility registry

The program maintains `semantic_compatibility_v1.json` automatically. **No manual JSON edits or per-update entries are needed.** Use the normal WebUI/CLI sync, continuation and packaging functions.

The file stores fixed category/requirement IDs plus registered library hashes, card-vocabulary identities and vector-encoding details. Stable IDs keep learned labels attached to their meanings. Historical hashes establish compatibility for older PTHs without embedded descriptions. The file contains no card-code vectors or network weights.

| Operation | Registry handling |
|---|---|
| Compile/rebuild static semantics | Preserve existing IDs, append new labels and register the compiled library's identity |
| WebUI/CLI sync or local continuation | Compile and update records on completion; merge a remote registry when one is supplied |
| Clear local semantics and rebuild | Retain the registry so regenerated labels keep their assigned IDs |
| Export/import GKG | Include an existing registry automatically and merge historical entries on import |
| Distribute source and semantic assets | Commit or distribute the registry with its semantic assets; receivers do not maintain it manually |

Back up the file with the semantic library rather than treating it as a temporary cache. Conflicting label IDs from independent branches need reconciliation. Older remote sources without a registry can still sync. An old PTH with neither embedded compatibility details nor a historical registry entry needs its original assets for compatibility verification.

New PTHs store compatibility details in `semantic_compatibility` and content-change provenance in `semantic_asset_updates`. Ordinary loading does not rewrite PTHs. Explicit ONNX maintenance can refresh their asset metadata while preserving identity and learning state; subsequent training saves also record current assets. Updated knowledge can change predictions even when weights are identical. Keep original assets to reproduce old predictions, and finish active training before updating the library.

## Sync and packaging

ONNX freezes the code-vector dictionary, effect slots and semantic tables as graph constants. PTH
loads learned parameters after constructing current runtime tables, which are not saved in its state.
A runtime may still open an old graph, but relaxing the hash cannot install new tables and can disagree
with current effect-slot bindings. The framework therefore keeps the PTH fallback and explains that
the frozen semantics are outdated; it does not classify this as damaged learned weights.

To use updated semantics for an old iteration, re-export its PTH with compatible current assets and
regenerate external weights and the artifact manifest. Retraining is unnecessary; editing metadata
alone is insufficient. Existing save points export the current iteration, not all historical graphs.
This release does not automatically migrate old graphs. Original matching assets can still run them.

- Sync remains download-only. Modern source bundles require KB, vectors, index and manifest; missing provenance cannot silently attach old vectors to new code. Local extraction can repair them afterward.
- Clean remote KBs may omit `raw_code` but must preserve source hashes/block metadata and matching certified vectors. They may reuse identical sources but cannot regenerate without local Lua.
- GKG automatically includes modern provenance with the resumable source group and imports it as one group. Legacy source bundles remain supported within their original boundary. Compiled-only packages still run without SentenceTransformer or Lua scanning.
- Compiled catalogs additionally verify manifest file hashes. Logical model-semantic identity still derives from runtime arrays/bindings, not generation logs or state.

## Results and cost

Local rebuild: 13,400 cards with independent effects, 27,647 slots, no missing/empty code; 26,092 byte-distinct vectors with largest identical group 657 (previously 27,007). 25,748 slots use multiple chunks, totaling 172,018 chunks. Equivalent registration code may legitimately share vectors.

27 slots retain unresolved static helper checks/references and are explicitly incomplete, with known registration/callback source still present. Expansion is bounded to 128 functions and a target 128 KiB per slot, and the file-index LRU to 256 units; capacity limits also report incompleteness. No artificial effect taxonomy is added.

Chunking increases **offline initial generation** cost. Source/provenance also enlarge packages and startup checks. Lossless compact JSON reduces the KB from about104.71MiB to73.70MiB by removing formatting whitespace, without data/semantic-identity changes. Dictionary row count, model parameters and per-step rollout storage are unchanged; the generator never runs per decision. Compiled-only deployment omits large source files. A no-change rerun reused all13,400 cards and27,647 vectors without loading the encoder; reuse with an already-loaded encoder also passes. Playing-strength/learning-efficiency gains need fresh-training comparisons.

## Historical verification and remaining work

Full-script comparison against the previous committed parser found no slot/category/requirement changes;2,156 coarse classes remain. The full suite passes324 tests and50 subtests, with four rebuilt-asset gates skipped by default and all four passing separately. Tests cover extraction, closures/Clone/literals, changed same-key code, encoder identity, complete chunk coverage, missing-source rejection, cleaned bundles, tampering, GKG and sync. Independent rebuilt-asset integration checks network shapes, PPO updates, ONNX agreement and real-Core duel/raw-trajectory replay. No production-scale long training or manual browser visual acceptance is claimed.

Core-side Phase 7 batch 3 ingestion and isolation gates arrived in 3.13.5; see
[canonical traces and external ingestion](core_trajectories_en.md). v3.14.0 enters full V4
main-framework deployment and long-training validation. Standalone imitation learning is available
from3.15.3; the deck-building upper layer remains future work. Passing inspection never starts training automatically.
