# 🗺️ Development Roadmap

> This document records agreed directions only. Items below are not part of the current stable training protocol yet.

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
- [ ] External Link wiring/live acceptance, historical YRP evidence, related-deck-family splits, and an independent imitation learner after gates pass

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
