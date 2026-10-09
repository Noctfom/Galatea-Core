# 3.13.2: Fixed Core messages and cognition gaps

Splitting a message, maintaining its state and exposing it to the model are separate layers.
Raw trajectory capture does not make every captured field a training input.

Scope: the bundled public Fluorohydride legacy Core ABI. No Core source/binary modifications.
Framework 3.13.2; Model Protocol 4 / Checkpoint Format 3 / network schema revision 13 unchanged.
Independent raw Trajectory Schema 2; holographic JSON remains format 3.

## Implemented: 16, 31, 21

Type 16 SELECT_CHAIN, little-endian payload:

```
player:u8, count:u8, special_count:u8, hint_timing:u32, opponent_hint_timing:u32
count entries: effect_mode:u8, forced:u8, code:u32, location:u32, desc:u32
```

Payload length is `11 + 14*count` (total `12 + 14*count`). Effect modes are ordinary 0,
operation 1, reset 2. Forced flags are per candidate; Core accepts cancellation `-1` only if
none are forced. The old parser treated a timing byte as header forced, the first candidate's
forced flag as its mode, and later modes as delimiters. Nonempty lengths coincidentally matched,
often preserving code/location/desc while corrupting flags, timings and cancel legality. Empty
prompts incorrectly expected an extra byte and went unconsumed.

`core_message_protocol.py` now supplies strict shared decoding for model and RuleBot. Truncated/
invalid fields fail explicitly. Empty chains automatically send `-1` without RNG, inference,
PPO samples, decision steps or transition events in Worker, Arena and RuleBot self-check paths.
Trajectories record/replay automatic responses separately; they are not behavior-cloning labels.

Type 31 CONFIRM_CARDS payload:

```
player:u8, skip_panel:u8, count:u8
count entries: code:u32, controller:u8, location:u8, sequence:u8
```

`skip_panel` controls display, not optional bytes or disclosure. Both values preserve the same
confirmed identities; shared decoding retains positions and the existing tracker receives codes.
No other hidden cards become visible. CLI `--standard_core`, three WebUI controls, constructor/
Worker arguments and `CORE_HAS_GHOST_BYTE` are removed. Future Core upgrades need migration
audits rather than guessed dialect switching.

Type 21 SORT_CHAIN orders supplied candidates; it does not activate a new effect. Existing
candidate/macro/response paths were disconnected from `DuelState.update()`. Dispatch is now
connected and tested with complete constructed messages. The bundled Core does not emit it in
common execution paths, so this is not claimed as naturally occurring real-game coverage.

## Awaiting approval: 120, 160, 165

### Type 120 MISSED_EFFECT

Payload: `location:u32, code:u32`. Core reports certain optional triggers missing their activation
timing when processing advances and they lack applicable delay conditions. This is not negation
of an activated effect, a player's Cancel, an illegal duel, or a universal report of unavailable effects.

Current gap: known byte length, but no dedicated DuelState event, public transition result flag
or Encoder input. Raw capture retains it without teaching the model this specific result; later
states/outcomes can still supply indirect learning signals.

Proposal: add a visible missed-timing event with source/player/location/code and a result flag,
using the existing history and replay paths, **without a manual reward penalty**. The packet lacks
desc/effect-slot identity and causal reason IDs; do not invent exact Lua-slot attribution or blame.

### Type 160 CARD_HINT

Payload: `location:u32, hint_type:u8, value:u32`, attached to a card instance at that location.

| Subtype | Meaning | Not equivalent to |
| --- | --- | --- |
| 1 TURN | Card turn-count hint | Actual counters added/removed by 101/102 |
| 2 CARD | Referenced/declared card identity | Changing this card's own identity |
| 3 RACE | Declared race hint | Current actual card race |
| 4 ATTRIBUTE | Declared attribute hint | Current actual attribute |
| 5 NUMBER | Numeric hint | Automatically ATK, cost or a counter |
| 6 DESC_ADD | Add a description hint reference | A plain Boolean active-effect flag |
| 7 DESC_REMOVE | Remove one description reference | Remove all identical hints from other sources |

Current gap: the splitter consumes 9 bytes but there is no instance-hint state table, event
encoding or model input. Queries cover some actual stats/statuses/proper-summon state, not all
declared values or descriptions. Static Lua semantics cannot tell which instance has a hint now.

Proposal: separate numeric hints from reference-counted descriptions; follow Core add/remove,
movement, reindexing and recreation lifecycles. A location is not an eternal instance ID.
Refreshes may recreate cards and resend additions, so clear prior mappings before rebuilding.
Use precise Lua bindings/raw desc identities, never `desc & 15` guesses. Unknown hints stay
unknown; do not invent effect classes or durations.

### Type 165 PLAYER_HINT

Payload: `player:u8, hint_type:u8, desc:u32`; bundled Core uses 6 add / 7 remove for player-level
description references. They need not belong to a currently occupied field slot. Some special
client descriptions also affect region-viewing permissions, not merely decorative text.

Current gap: 6 bytes are recognized but no player-hint state, event or model input exists.
Proposal: player+desc reference counts, lifecycles dictated by actual Core notifications, bounded
structured inputs to policy/value and matching replay display. Side remains outside duel observations.

**160/165 are not a universal active-effect tracker.** Core emits these only where scripts/effects
request client hints. Bundled `script/c23434538.lua` (Maxx “C”) and `script/c94145021.lua`
(Droll & Lock Bird) do not mark their relevant ongoing effects with `EFFECT_FLAG_CLIENT_HINT`.
Adding 165 alone therefore cannot guarantee recognition of those ongoing effects. Public event
inference or exact additional coverage would require a separate design; no guessed tracker is added.

All three proposals affect model/event observations, not just replay text. Encoder, rollout/shared
storage, training, ONNX and Link boundaries must be audited together before implementation.

## Deferred by agreement: 35; awaiting approval: 37/38, 162

### Type 35 SWAP_GRAVE_DECK

Payload: `player:u8`. Exchanges that player's entire Deck and Graveyard, not both players' decks
or simply returning the graveyard to the deck. Extra-deck monsters in the former graveyard return
to Extra, with appropriate sequence/status resets and shuffling. This does not authorize revealing
otherwise hidden Deck/Extra identities.

Current gap: no bulk-swap state branch. Queries may refresh graveyard entities, while unqueried
Deck/Extra lists/counts/tracker mappings remain stale. Ordinary per-card MOVE cannot fully repair
this. Stable DeckProfile must remain unchanged.

2026-10-09 user decision: defer support for this uncommon bulk-region exchange and retain the
known gap. Exchange of the Spirit is a relevant example, not the only possible user of this Core
operation. Deck/Extra remainder observations and card-memory mappings are not guaranteed correct
in such duels. This decision changes neither Core nor training behavior.

If implemented later, use atomic bulk-region updates including
extra routing, counts, instance-label cleanup, visibility and query verification, not a naive list swap.

### Type 37 REVERSE_DECK / Type 38 DECK_TOP

37 has no payload and toggles global deck-reversal state, not just the acting player's deck.
38 contains `player:u8, offset:u8, code:u32` and updates a known top-relative card. The code high
bit `0x80000000` denotes face-up position in the deck, not card identity. Neither message reveals
the entire hidden order. Current observations have no reversal/public-top state; deck composition
does not tell the next draw. Proposal: bounded public-top mappings with reversal tracking and
shuffle/draw/movement invalidation or shifting, never retaining stale next-draw identities.

### Type 162 RELOAD_FIELD

Includes rule, both LPs, occupancy/positions for 7 monster and 8 spell/trap zones, overlay counts,
Deck/Hand/Grave/Removed/Extra and face-up Extra counts, and active-chain identities/positions/
actors/desc. It is not a full card query: full identities/stats/ongoing effects need companion public
queries. Relevant to field reloading/debug setups; not every Link reconnection necessarily sends it.

Two current gaps: the length parser reads only 5 monster slots for rule < 4 although this bundled
Core always writes 7, and DuelState has no snapshot rebuild branch. Uncommon in ordinary modern
full-start duels, but required before promising mid-duel snapshot/custom-setup support. Proposal:
fixed 7/8 layout, atomic state replacement, clearing unverifiable instance histories and companion
public queries. Do not fabricate pre-snapshot full-game histories.

## Impact and verification boundaries

- Corrected cancellation/timing semantics can change decisions at previously misdecoded prompts;
  this is a bug fix, not bitwise equivalence to the faulty path.
- No PPO/GAE/reward/network/ONNX shape changes or additional effect tracker; no empty-prompt samples.
- Decoding is linear in candidate count without new background threads, inference requests or large
  persistent tables. Enabled traces store additional actually handled empty messages/responses and
  therefore have some extra I/O; no unmeasured full-training speedup is claimed.
- Verification covers malformed/boundary packets, full Type 21 dispatch, real Core model duels,
  recording-toggle equivalence, response/observation replay and WebUI widgets. Small smoke tests
  are not production-scale long-run validation.
- UI says “AI predictions”: model assessments, not actual win rates. Calculations/targets are unchanged.

## Sources

Bundled source is the adaptation baseline; public links explain mechanisms, not universal latest-Core compatibility:

- [Core playerop.cpp: chain messages/cancellation](https://github.com/Fluorohydride/ygopro-core/blob/master/playerop.cpp)
- [Core libduel.cpp: confirmation skip_panel](https://github.com/Fluorohydride/ygopro-core/blob/master/libduel.cpp)
- [Core processor.cpp: missed timing/reversal](https://github.com/Fluorohydride/ygopro-core/blob/master/processor.cpp)
- [Core field.cpp: region swap/snapshot/player hints](https://github.com/Fluorohydride/ygopro-core/blob/master/field.cpp)
- [Core card.cpp: instance hint lifecycles](https://github.com/Fluorohydride/ygopro-core/blob/master/card.cpp)
- [Public duelclient.cpp: display/reference counts](https://github.com/Fluorohydride/ygopro/blob/master/gframe/duelclient.cpp)
