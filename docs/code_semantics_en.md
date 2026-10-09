# 3.13.4: Code-source provenance and content-aware semantic rebuilds

## Scope

This repairs offline semantic assets only. Duel-network architecture, input fields/shapes, rewards, PPO/GAE, legality, Core and Link contracts are unchanged. Model Protocol **4**, Checkpoint Format **3**, schema revision **14**, static semantic format **1**, and GKG format **4** remain unchanged. Code-generation manifest format **1** lives independently in `code_semantic_provenance.py`; extraction version lives in `lua_semantic_source.py`.

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

The rebuilt assets are staged under `semantic_builds/v3.13.4/`; original root sources/vectors are preserved. The assets-only `Galatea_Semantics_3.13.4.gkg` is about87.32MiB and also copied into `deploy_packages/` for direct WebUI selection. Extract/stage it in the deployment manager and import complete source/runtime groups to activate them. Preserve original assets, stop training/duels, and restart after import. Do not replace assets in a running process.

**New vectors change `semantic_lookup_hash`, not network protocol.** Existing identity validation rejects old checkpoints with new semantics. No migration is provided: fresh V4 training uses rebuilt assets; old models keep their original assets/separate packages. The staging directory is gitignored; source commits do not publish rebuilt assets automatically. Publish the GKG or complete semantic source bundle separately.

## Sync and packaging

- Sync remains download-only. Modern source bundles require KB, vectors, index and manifest; missing provenance cannot silently attach old vectors to new code. Local extraction can repair them afterward.
- Clean remote KBs may omit `raw_code` but must preserve source hashes/block metadata and matching certified vectors. They may reuse identical sources but cannot regenerate without local Lua.
- GKG automatically includes modern provenance with the resumable source group and imports it as one group. Legacy source bundles remain supported within their original boundary. Compiled-only packages still run without SentenceTransformer or Lua scanning.
- Compiled catalogs additionally verify manifest file hashes. Logical model-semantic identity still derives from runtime arrays/bindings, not generation logs or state.

## Results and cost

Local rebuild: 13,400 cards with independent effects, 27,647 slots, no missing/empty code; 26,092 byte-distinct vectors with largest identical group 657 (previously 27,007). 25,748 slots use multiple chunks, totaling 172,018 chunks. Equivalent registration code may legitimately share vectors.

27 slots retain unresolved static helper checks/references and are explicitly incomplete, with known registration/callback source still present. Expansion is bounded to 128 functions and a target 128 KiB per slot, and the file-index LRU to 256 units; capacity limits also report incompleteness. No artificial effect taxonomy is added.

Chunking increases **offline initial generation** cost. Source/provenance also enlarge packages and startup checks. Lossless compact JSON reduces the KB from about104.71MiB to73.70MiB by removing formatting whitespace, without data/semantic-identity changes. Dictionary row count, model parameters and per-step rollout storage are unchanged; the generator never runs per decision. Compiled-only deployment omits large source files. A no-change rerun reused all13,400 cards and27,647 vectors without loading the encoder; reuse with an already-loaded encoder also passes. Playing-strength/learning-efficiency gains need fresh-training comparisons.

## Verification and next batch

Full-script comparison against the previous committed parser found no slot/category/requirement changes;2,156 coarse classes remain. The full suite passes324 tests and50 subtests, with four rebuilt-asset gates skipped by default and all four passing separately. Tests cover extraction, closures/Clone/literals, changed same-key code, encoder identity, complete chunk coverage, missing-source rejection, cleaned bundles, tampering, GKG and sync. Independent rebuilt-asset integration checks network shapes, PPO updates, ONNX agreement and real-Core duel/raw-trajectory replay. No production-scale long training or manual browser visual acceptance is claimed.

Resume Phase7 batch3: Link/YRP ingestion, whole-duel/deck/player partitions, consistent asset/Core identities and candidate mapping, then imitation learning only from deterministically replayable data. No replay training or deck-building network is introduced here.
