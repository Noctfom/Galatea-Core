# 🗺️ Development Roadmap

> Applies to v3.17.3. Checked items have landed; unchecked items are TODOs. Full V4 deployment does not mean deck-building/BO3 or long-run acceptance is complete.

## Collection scheduling and runtime visibility

- [x] 3.17.0: Sibling training status and live collection controls; recover actual settings/progress after browser refresh, apply league/mirror probability changes at the next iteration, and measure accepted samples and worker-seconds by actual source
- [x] 3.17.1: Optional sample-quota scheduling assigns workers by estimated remaining work; switch sources between complete episodes, give low-share sources at least one game, use bounded rollout chunks, freeze the policy within each iteration and never reuse old-policy rows across PPO iterations
- [x] 3.17.2: Historical ONNX Pass1/Pass2 state reuse, checked against the original graph on real games; retain CPU thread budgets and fall back to full ONNX if reuse fails, with no re-export
- [ ] Further acceleration of ordinary single-pass historical inference and cloud concurrency validation; GPU history caching remains unimplemented and would require separate PPO release and memory-limit checks
- [x] 3.17.3: Wake WindBot on complete messages, coalesce query-frame writes and reuse worker startup assets; retain independent per-game processes/state and record source patches and executable provenance
- [ ] Validate 3.17 concurrency, memory, source quotas and long-run stability on the target Linux/Mono host; WindBot still initializes its process/card database per game, with no persistent Bot service

- [x] 3.16.9: Independent current prompts, announced objects and per-chain resolution/negation states; 116 inputs, revision 15, existing learned parameters and Checkpoint 3 retained. Known old PTH files can resume; ONNX requires re-export. Rewards, temperature and DeckFiLM tuning remain future discussion. See [contract and migration](public_observations_en.md#3169-current-prompt-and-chain-context).

See the [V4 Protocol Implementation Specification](protocol_v4_implementation_en.md) for the frozen scope, implementation order, and acceptance gates.

## Remaining Main-Framework Cognition

- [x] Bind complete runtime `desc` values to code-semantic slots through Lua Effect object identity; never infer slots from description low bits, and safely fall back for dynamic or ambiguous scripts
- [x] Complete structured auxiliary heads and a continuous goal-planning latent from verifiable immediate, multi-horizon future, and terminal posteriors; do not use hand-authored tactics or present attention/latent dimensions as explanations
- [x] Display AI terminal/length predictions in Holographic Replay, separately from actual outcomes; counterfactual evaluation awaits independent validation
- [x] Phase 7 batch 1 (3.13.0): unified GAE comparisons, PPO epoch/target-KL controls, and actual-update audits without network or reward changes
- [x] Phase 7 batch 2 (3.13.1): canonical raw Core trajectories, deterministic observation-digest replay, and optional independent state-evaluation curves; retain quality gates and separately validate counterfactual search
- [x] Parsing stability (3.13.2): fixed bundled Core Type 16/31 layout, no ghost-byte toggle, Type 21 dispatch; no network/reward changes
- [x] 3.13.3:120 visible feedback without penalties,160/165 reference lifetimes/code semantics,37/38 public position memory and162 strict rebuilding; [observations/cost](public_observations_en.md)
- [ ] Type35 bulk region exchange remains deferred; unproven old semantic assets require trusted regeneration, not forced parent-vector matching
- [x] 3.13.4 confirms missing/empty-source generation plus key-only reuse caused collapse; source provenance, content identity and complete chunking now support isolated rebuilds without network/input/coarse-classification changes. New assets have26,092 distinct vectors and27 explicitly incomplete slots; see [asset guide](code_semantics_en.md)
- [x] Phase7 batch3 (3.13.5): Core-side Link capture SDK, strict diagnostic YRP1/YRP2/YRP3D conversion, transitive whole-duel/exact-deck/stable-player isolation and quality manifests; no network/training changes or eligibility without evidence
- [x] Core defines shared external-session/capture interfaces and explicit-provenance response-label contracts; Link reuses them after Core stabilizes and is not a development prerequisite
- [ ] Historical YRP evidence and automatic similar-build family splits; accept actual Link networking separately

## Optional external opponents and demonstration learning

WindBot belongs to Core's [optional integration directory](../integrations/windbot/README.md), without a Link proxy. Controlled BO1 landed in3.15.0 and independent imitation in3.15.3; full player-only Link inference and YRP without evidence remain unsupported.

- [x] Batch 1: Core session contract v1, pinned IceYGO source, explicit CLI/WebUI controlled BO1, masked views, faithful capture and independent replay. Windows and Ubuntu/WSL2 real both-seat games and Blue-Eyes/Albaz mirrors pass; broader executor/distribution acceptance remains pending
- [ ] Batch 2: process full Link captures and YRP with original-message/asset evidence through the same canonical interpreter; compare observations and unique expert-action mapping, including multi-selection/cancellation, without inventing evidence
- [x] 3.15.2: Read-only expert CPU sample API and CLI/WebUI verification for complete eligible traces; reuse strict replay, observation digests and original candidate labels. No prefix access before whole-duel acceptance, default WindBot-only selection, no optimizer
- [x] 3.15.2: Program identity association audit reports shared identities and hypothetical group counts without program edges. Actual isolation still uses all stable identities; human/unknown handling remains conservative
- [x] 3.15.3: Explicit duel/build/executor holdout goals. Program IDs retain provenance; human/unknown identities remain transitively grouped. Explicit family configuration is supported, not guessed automatic clustering
- [x] WindBot-side Type15 physical-index-set label equivalence uses a Core-owned versioned interface for future Link reuse. All471 decisions in the original Albaz capture now map, with original responses/observations unchanged. Default/ordered matching, public ygocore, RuleBot packaging and training semantics are unchanged
- [ ] Continue reviewing true reduced-pool coverage and prompt-specific adapters; native demonstrations can already drop unmapped expert labels, while canonical traces retain completeness checks without expert-answer injection
- [x] 3.15.1: Core-owned host Start/query ingress, framing, sparse attributes and diagnostic projection. WindBot reuses packet builders/filtering; queries create neither actions nor transition events and never masquerade as original Core chunks
- [ ] Accept full V4 online inference from a single-player view separately: opponent response boundaries, Type160 hints and dynamic-attribute coverage are not fully proven by this conservative profile. Default encoding stays gated; actual Link networking is future work
- [x] 3.15.3: Independent BC optimizer, epoch budget, experimental UUID/checkpoints, one-time preparation/compressed caches, single-Core native collection and candidate/baseline fixed-build swapped-seat evaluation. Engineering closure only, not demonstrated playing-strength improvement
- [x] 3.15.3 review: Explicit BC weight initialization for a new PPO branch, cumulative PPO/BC counters, separate UUID/directory and fresh PPO optimizer/update counters. No automatic continuation; imitation and online-human learning still require separate startup
- [x] 3.15.4: Optional WindBot training opponents reuse Workers/central inference and learn only model actions, with a configurable Worker allocation rate, mirror/regular decks, suggested per-level holdouts and frozen rosters. Expert imitation and online-human games remain separate
- [x] 3.15.5: Optional customization of four absolute league probabilities; enabled WindBot replaces the base Rule15% while extra coverage remains. Prebuilt runtimes automatically adapt to Windows/Linux without an OS selector, user build workflow or upstream executor changes
- [x] Default paths do not require the module/runtime. Dependencies are checked only on activation; `--with-windbot` optionally packages source, licenses and an already installed runtime
- [x] 3.15.6: Preserve pinned central-inference staging and batch result transfers; correct self-play audit attribution. Separate WindBot CSV/dashboard and mirror/pool metrics from ordinary deck statistics without network or protocol changes
- [x] 3.15.8: Automatic idle-socket selectors remove Linux select's high-descriptor restriction while preserving deadlines, invalid-packet rejection and cleanup; the real Linux high-descriptor regression still needs target-machine execution
- [x] 3.15.9: Public GitHub download routes, local README navigation, vocabulary acknowledgement controls and optional WindBot self-check; network and protocol unchanged
- [x] 3.16.4: All Arena modes and controlled native-model duels use leaf-first closed-return retries, matching response identities across macro-pool reordering; main training unchanged
- [x] 3.16.6: Per-game configurable mirror/training-pool mixing for model decks in WindBot training; retain pure modes, actual-mode statistics and league/extra-Rule coverage behavior

Distinguish upstream sources, original licenses, pinned commits and local patches. Keep source snapshots inside the integration directory and selected public prebuilt runtime files in the repository with manifests; intermediate build caches/logs stay out of Git. Small UI/chart changes stay in their existing source files; add modules only for independently scoped components. Do not refactor existing modules solely to reorganize files.

## V4 deployment and training validation

- [x] 3.16.8: Fix the fixed13-byte Type12 description reader, successor-value bootstrapping and single-count truncation statistics; controlled WindBot native models use Pass1 macro proposals. Network, V4 fields and reward settings remain unchanged; later learning/architecture work is separate

- [x] 3.14.0: full V4 main-framework deployment, direct CUDA semantic aggregation and bilingual documentation audit; Model Protocol4/Checkpoint3/revision14 unchanged
- [x] 3.14.1: public-hint/announcement hotfixes, deployment labels and public metric-reference corrections, dual-view prediction cursors/coverage diagnostics; protocols and network unchanged
- [ ] Complete50-iteration trends,100–200-iteration medium runs and long-term fixed Arena comparisons; temporary short-run stability is not acceptance
- [x] 3.15.7: Linux setup reuses current Python/Conda, constrains existing Torch and installs missing WindBot runtime prerequisites. Shared AnyCPU files ship in source without local compilation or a new Linux ZIP
- [ ] Automatic Torch-build selection by GPU/driver on fresh machines remains pending; first installation uses requirements or an explicit wheel source without guessing compatibility
- [x] 3.16.7: GKG5 optional decks, replays, developer/trainer notes and raw presets with independent import; retain GKG4 reads/writes without attachments, never apply presets automatically. See [format and usage](deployment_packages_en.md)

## Online ingress and collection performance

Core owns controlled sessions and model-call contracts; Link owns real-server connections and its
additional uses. Controlled Core sessions reuse the WindBot-style path. Reconstructing real-server
player observations is accepted separately and does not block collection performance work.

- [x] 3.16.0 batch1: Core-owned V4 player sources, missing/invalidated data and separate eligibility; bounded native/player encoding comparison, CLI/WebUI controls and report management, with online encoding still closed
- [x] 3.16.0 batch1: Optional bounded Pass1/Pass2 timing for shared-slot filling, queueing, gathering, forward/sampling and CPU/CUDA stages; network and learning unchanged
- [x] 3.16.1 batch2: Dtype-packed shared/pinned/device buffers with a fields reference path; exact CPU/CUDA inputs, outputs and fixed-seed samples, plus real Worker/PPO regression. Target-server end-to-end gains still require comparison
- [x] 3.16.1: Compatible content updates reuse PTH weights with stable label/encoder validation separate from content identity; exact replay and ONNX remain strict, with compatibility evidence carried by sync/GKG
- [x] 3.16.1: Extra inventory/entity alignment and unknown-card transition target padding; valid-event encoding retained and online gate still closed
- [x] 3.16.2: Optional exact-batch CUDA Graph forward, sampling outside capture, updated PPO weights and capture-failure fallback; existing reports separate setup from steady execution
- [x] 3.16.3: Release graphs/GPU outputs before PPO, reuse capture streams, combine result transfers and adapt batching waits by platform; retry training ranges, draw only model-side WindBot decks, and add monotonic collection-phase diagnostics
- [x] 3.16.5: Check/replace/create ONNX in original training directories, refresh PTH asset metadata without changing identity or learning state, roll back failures and retain Core load restrictions
- [ ] Validate long-run CUDA Graph/PPO peaks and prolonged collection stalls on the target Linux server with complete Trainer/Worker logs
- [ ] Compare real-server Start/query delivery,160/165 hints and162 recovery; discuss any required change to native two-player versus online one-player history semantics before changing protocol identity or training
- [ ] Benchmark optional frozen-policy local CPU/GPU collection by resource budget, preserving iteration-boundary weight sync, stale-response rejection and trajectory identity without automatically adopting asynchronous off-policy learning

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
