# 🗺️ Development Roadmap

> This document records agreed directions only. Items below are not part of the current stable training protocol yet.

See the [V4 Protocol Implementation Specification](protocol_v4_implementation_en.md) for the frozen scope, implementation order, and acceptance gates.

## Remaining Main-Framework Cognition

- [x] Bind complete runtime `desc` values to code-semantic slots through Lua Effect object identity; never infer slots from description low bits, and safely fall back for dynamic or ambiguous scripts
- [x] Complete structured auxiliary heads and a continuous goal-planning latent from verifiable immediate, multi-horizon future, and terminal posteriors; do not use hand-authored tactics or present attention/latent dimensions as explanations
- [ ] Display calibrated auxiliary predictions and counterfactual evaluations in Holographic Replay while clearly separating prediction, observed event, and explanatory inference
- [x] Phase 7 batch 1 (3.13.0): unified GAE comparisons, PPO epoch/target-KL controls, and actual-update audits without network or reward changes
- [ ] Phase 7 batch 2: one canonical trajectory and optional per-decision predictions; implement state-evaluation curves before separately validating counterfactual search
- [ ] Phase 7 batch 3: Link/YRP input, deterministic replay, and whole-duel held-out quality gates before behavior cloning

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
