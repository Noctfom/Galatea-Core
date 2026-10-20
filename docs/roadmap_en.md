# 🗺️ Development Roadmap

> Applies to v3.15.0. Checked items have landed; unchecked items are TODOs. Full V4 deployment does not mean imitation learning, deck-building/BO3 or long-run acceptance is complete.

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
- [ ] Historical YRP evidence, related-deck-family splits and an independent imitation learner after gates pass; accept actual Link networking separately

## Optional external opponents and demonstration learning

WindBot belongs to this Core repository's [optional integration directory](../integrations/windbot/README.md), without requiring a Link proxy. 3.15.0 implements batch 1's controlled BO1 prototype; Link/YRP evidence extensions and imitation training remain pending.

- [x] Batch 1: Core session contract v1, pinned IceYGO source, explicit CLI/WebUI controlled BO1, masked views, faithful capture and independent replay. Real WindBot/RuleBot both-seat and mirror gates pass; Linux and broader executor acceptance remain pending
- [ ] Batch 2: process full Link captures and YRP with original-message/asset evidence through the same canonical interpreter; compare observations and unique expert-action mapping, including multi-selection/cancellation, without inventing evidence
- [ ] Audit program-participant identity and split granularity before imitation learning: a shared model/Bot ID can merge multiple builds into one connected group. Retain truthful provenance and distinguish in-distribution validation from held-out builds/executors; never fabricate identities or split individual frames randomly to obtain validation data
- [x] WindBot-side Type15 physical-index-set label equivalence uses a Core-owned versioned interface for future Link reuse. All471 decisions in the original Albaz capture now map, with original responses/observations unchanged. Default/ordered matching, public ygocore, RuleBot packaging and training semantics are unchanged
- [ ] Separately audit true reduced-pool coverage and other prompt-specific adapters. Design per-sample eligibility independently, without expert-answer injection or automatic whole-duel gate relaxation
- [ ] Define host-generated Start/query player-view reconstruction in Core before Link reuse; keep unknown-information masks and never label derived host packets as original native Core chunks
- [ ] Batch 3: build an isolated imitation learner with separate experimental checkpoints; verify action learning and fixed Arena performance. Human demonstrations reuse capture interfaces, never fabricated PPO probabilities or advantages
- [ ] Evaluate a separate WindBot pool and small-fraction training integration only after acceptance; leave PPO, league sampling, rewards, V4 networks and public ygocore unchanged meanwhile
- [x] Default paths do not require the module/runtime. Dependencies are checked only on activation; `--with-windbot` optionally packages source, licenses and an already installed runtime

Distinguish upstream sources, original licenses, pinned commits and local patches. Keep source snapshots inside the integration directory and build outputs/logs out of Git. Small UI/chart changes stay in their existing source files; add modules only for independently scoped components. Do not refactor existing modules solely to reorganize files.

## V4 deployment and training validation

- [x] 3.14.0: full V4 main-framework deployment, direct CUDA semantic aggregation and bilingual documentation audit; Model Protocol4/Checkpoint3/revision14 unchanged
- [x] 3.14.1: public-hint/announcement hotfixes, deployment labels and public metric-reference corrections, dual-view prediction cursors/coverage diagnostics; protocols and network unchanged
- [ ] Complete50-iteration trends,100–200-iteration medium runs and long-term fixed Arena comparisons; temporary short-run stability is not acceptance
- [ ] Select compatible PyTorch by Python/GPU/driver in the Linux helper without overwriting existing cloud environments; use documented manual installation meanwhile

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
