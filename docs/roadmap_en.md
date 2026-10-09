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
- [ ] Prioritize local code-semantic quality: 27,007 of 27,647 vectors are identical. Verify code provenance, empty-code generation and incremental invalidation before deciding on trusted regeneration; never silently replace existing model assets
- [ ] Phase 7 batch 3: Link/YRP input, deterministic replay, and whole-duel held-out quality gates before behavior cloning

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
