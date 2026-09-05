# 🗺️ Development Roadmap

> This document records agreed directions only. Items below are not part of the current stable training protocol yet.

## Remaining Main-Framework Cognition

- [x] Bind complete runtime `desc` values to code-semantic slots through Lua Effect object identity; never infer slots from description low bits, and safely fall back for dynamic or ambiguous scripts
- [ ] Expand recent activation history from “the last eight activated card IDs” into structured events, then evaluate whether summons, attacks, moves, and selection results belong in the same stream
- [ ] Research structured auxiliary heads for tactical goals, card roles, line completion, resource deltas, and next-state prediction. Every output needs a verifiable label or self-supervised target; attention weights alone are not explanations
- [ ] Display calibrated auxiliary predictions and counterfactual evaluations in Holographic Replay while clearly separating prediction, observed event, and explanatory inference

## Separable Deck-Construction Cognition

- [ ] Extract a reusable card-semantic encoding interface while keeping the duel policy independently trainable and deployable
- [ ] Build a permutation-invariant deck relation encoder that explicitly distinguishes Main, Extra, and Side Deck sections, copy counts, and card roles
- [ ] Build BO3 siding and full deck-edit policies with Add, Remove, Swap, and Finish actions protected by legality masks
- [ ] Reuse the Galatea duel agent as executor and win-rate evaluator, learning card synergy from simulations and counterfactual win-rate changes
- [ ] Export card-relation graphs, before/after win-rate estimates, and auditable reasons for each edit
- [ ] Let the Type 142 announce pool derive candidates dynamically from the current deck, Side Deck, revealed opponent cards, and the deck module's opponent-deck posterior; retain the staple pool as a safety fallback
