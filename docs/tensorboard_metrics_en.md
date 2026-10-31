# TensorBoard Metric Reference

> Applies to Galatea-Core v3.15.7 / Model Protocol V4.
> For users and developers: chart purposes, y-axes, logging locations, and trend interpretation.
> [中文](tensorboard_metrics.md)

Default main PPO contains **80 fixed scalars, 3 planning distributions and 1 configuration text**, plus `Deck_WinRates/*`.
Explicit WindBot opponents add 7 fixed scalars, 1 configuration text and per-executor series; see section 11.
Independent imitation adds10 scalars in its own run without changing PPO charts. Prediction, update budget,
expert agreement and playing strength are different quantities.

| Group | Question it answers |
|---|---|
| League / Deck_WinRates | Which opponents and decks are easier to win against during training? |
| Rollout | How much reward does a game produce, how many decisions occur, and how does it end? |
| Auxiliary_Targets | How many supervision labels are available, and what events occurred? |
| Auxiliary_Train | Given the current state and chosen action, can its consequences be predicted? |
| Planning | Without specifying the chosen action, is the representation informative about future resources and outcomes? |
| Train / PPO_Update | How do policy/value change, how much data is actually used, and which guards trigger? |
| Performance / Configuration | Where does runtime go, and which training settings are actually used? |
| Imitation (separate run) | Does policy agree more with experts, and does agreement improve on held-out demonstrations? |

Jump to: [logging rules](#1-storage-axes-and-reading-rules) · [win rates](#2-league-training-game-win-rates) · [collection](#3-rollout-collection-outcomes-and-termination) · [labels](#4-auxiliary_targets-coverage-and-event-distribution) · [auxiliary prediction](#5-auxiliary_train-structured-predictive-learning) · [planning](#6-planning-latent-goals-and-policy-integration) · [training updates](#7-train-current-mini-batch-ppo-joint-update) · [update guards](#8-ppo_update-iteration-controls-and-actual-reuse) · [runtime and settings](#9-performance-and-configuration) · [independent imitation](#10-independent-imitation-metrics)

## 1. Storage, axes, and reading rules

### Files and interface

- Events are stored in `runs/<model_prefix>_<launch_date-time>/events.out.tfevents.*`. Each launch creates a run directory; resuming inherits iteration and train_step, so axes need not restart at zero. Matching prefixes do not establish matching model UUIDs.
- WebUI: **TensorBoard → Start TensorBoard**. Scalars shows scalar metrics, Histograms / Distributions shows planning distributions, and Text contains `Configuration/PPO`.
- Standalone launch: `python -m tensorboard.main --logdir ./runs`. Inspect resumed runs separately before comparing continuous steps.
- Sources: [SummaryWriter and configuration](../trainer.py#L606), [WebUI TensorBoard module](../app.py#L3951).

### Frequency and statistical unit

| Group | Step axis | Frequency | Scope of one point |
|---|---|---|---|
| `League_Overall/*`, `League_Self/*`, `League_Opponent/*` | Collection iteration | After each collection | Game counters from successfully loaded Worker outputs; not cumulative training win rate |
| `Rollout/*` | iteration | After each collection | Game counts or an average of Worker means; see section 3 |
| `Auxiliary_Targets/*` | iteration | After merging | Label audit over valid training rows; not fractions of games |
| `Train/*`, `Auxiliary_Train/*`, `Planning/*` | Cumulative optimizer updates, train_step | Every 20-update position | Diagnostics for the current mini-batch, not iteration averages |
| `Planning_Histograms/*` | train_step | Every 200-update position | Current batch planning vectors or current gate parameters |
| `PPO_Update/*` | iteration | After each PPO update phase | Summaries of inspected batches and actual optimizer updates |
| `Performance/*` | iteration | After the corresponding phase | Complete collection/update durations and submitted-row throughput |
| `Configuration/PPO` | train_step at launch | Once per launch | JSON text of PPO settings and actual optimizer learning rates |

train_step advances only after a successful optimizer update. Logging conditions are checked before the update: the first point can be Step 0, and FP16 overflow skipping an update can produce repeated steps. Gradient norms are measured after backward and AMP unscaling, **before clipping**. Predictions and losses come from the pre-update forward pass.

The dark Smoothed line is TensorBoard display smoothing; the light line and Value are original logged values. A smoothed `Epochs_Completed=1.44` does not mean 1.44 whole epochs were completed: raw values are integers. Automatic axis zoom can magnify tiny changes; coverage_next changing from 0.9997 to 0.9999 is only 0.02 percentage points. See the [official TensorBoard guide](https://www.tensorflow.org/tensorboard/get_started) for dashboard presentation.

### Terms and y-axis units

- **Iteration**: one collection followed by PPO updates. **Mini-batch**: training rows handled in one optimizer update; an iteration contains multiple batches. **Epoch**: one traversal of that iteration's data.
- **Training row**: one retained training decision. **Game**: one duel, which can produce many rows. Decisions by both players and Core messages also have different counting rules.
- **Fraction/accuracy**: 0.8 means 80% of the specified denominator. Coverage means labels exist, Accuracy means predictions are correct, and WinRate means games were actually won.
- **Loss**: a training error score, not reward. BCE scores multiple yes/no predictions; cross-entropy scores class predictions; Smooth L1 scores numerical differences. Lower usually means better fit only for the same task, normalization, and comparable data. There is no universal passing threshold, and the value is not the average LP, card, or turn error.
- **RMS / Std / gradient norm**: overall magnitude, variation across samples, and parameter-update signal strength, respectively. None is a percentage. Gradient clipping limits excessively large signals; charts still show pre-clipping values.
- **Seconds / rows per second / nats**: runtime, training-row throughput, and policy entropy. Network magnitudes and normalized errors are not measured in original game units.

For example, `coverage_terminal=0.8` means 80% of training rows have terminal labels; `Terminal_Accuracy=0.8` means 80% of eligible rows' outcomes were classified correctly; `RuleBot=0.8` means 80% of that iteration's counted RuleBot games were won. Identify the quantity before interpreting its trend.

## 2. League: training game win rates

Logging: [trainer.py collection statistics](../trainer.py#L1207). Counters: [worker.py game settlement](../worker.py#L1260). Related design: [league training](architecture_en.md#league-training-mechanism), [Arena benchmarks and policy modes](features_en.md#launch--monitor-hub).

All values below are **wins / games counted in the group**, in [0,1]. Results are counted for train_p_id. First/Second identifies the training player's first/second seat, not the current decision actor. Draws and nonterminal truncations do not count as wins and lower the rate when included in the denominator. Aborted games roll back and are not submitted.

| Chart | Group and y-axis meaning | Interpretation and limits |
|---|---|---|
| `League_Overall/WinRate_Total` | Mixed Self/Hist/Rule win rate | Improvement in the mixed environment; changing opponent proportions also changes this value |
| `League_Overall/WinRate_First` | First-seat mixed-opponent win rate | Compare seats only after considering decks, opponents, and game counts |
| `League_Overall/WinRate_Second` | Second-seat mixed-opponent win rate | Exceeding first-seat performance does not automatically imply identity mislabelling |
| `League_Self/WinRate_Total` | Win rate against this iteration's current-weight copy | Approximately 0.5 under symmetric seats/decks and no truncation; not a strength curve that must rise |
| `League_Self/WinRate_First` | First-seat Self win rate | Reveals seat differences; finite samples can deviate from 0.5 |
| `League_Self/WinRate_Second` | Second-seat Self win rate | Truncation, inference paths, and deck asymmetry can also cause differences |
| `League_Opponent/Historical_AI` | Win rate against this model's historical pool | The pool changes as checkpoints are added; not a fixed opponent benchmark |
| `League_Opponent/RuleBot` | Win rate against the rule-based bot | Measures adaptation to this rule policy, not human/general-model strength |

**A zero denominator also produces a logged zero.** Before historical checkpoints exist, the zero line does not indicate continuous losses. A later zero can likewise mean no valid submitted games in that group. Game denominators are not logged alongside these rates, so the curves cannot distinguish the cases or support reliable pooled game-weighted win rates and confidence intervals.

`Deck_WinRates/{deck}_Total/First/Second` also uses the current collection; a missing subgroup produces no point. Thousands of TensorBoard cards are thousands of metric series, not thousands of independent evaluation games.

## 3. Rollout: collection outcomes and termination

Logging: [trainer.py](../trainer.py#L1239). Generation: [Worker rewards and GAE](../worker.py#L1284). Related design: [training loop](architecture_en.md#training-loop), [PPO controls](quickstart_en.md).

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Rollout/Average_Reward` | Equal-weight average of nonzero Worker mean episode rewards | Rising under matched conditions is encouraging; not a win rate or game-weighted mean; excluding zero means introduces bias |
| `Rollout/Average_Length` | Equal-weight average of nonzero Worker mean ep_steps | Falling can mean faster wins/fewer empty actions, different opponents, or different truncation frequencies |
| `Rollout/Timeout_Rate` | Worker timeouts counter / total counted games | Currently includes duplicate counting; not a literal fraction of timed-out games |
| `Rollout/Deadlock_Rate` | Worker deadlocks counter / total counted games | Aborts generally roll back before this counter; zero does not establish absence of errors/deadlocks |
| `Rollout/Draw_Rate` | Worker draws counter / total counted games | Tracks draw-like settlements, not all truncations or aborted games |

Episode reward is **the training trajectory's summed step_reward + final_reward**, not a discounted return. Current rewards are not strictly win/loss-only:

- Wins at up to 50 turns give +1; later wins give +0.05. Losses give −1; draws/nonterminal truncations have zero terminal reward.
- Model decisions after turn 40 incur −0.0001; after 300 model decisions within one turn, an additional −0.0005 applies.
- Repeated identical candidates in identical states without progress can incur −0.005 after the threshold: 5 for Cancel/Unselect, 10 for other actions. Retry can incur −0.001. This does not penalize every multi-material selection indiscriminately.
- Auxiliary/planning losses **are not rewards** and do not turn prediction accuracy into additional wins or reward.

ep_steps is not Core-message count, turn count, or submitted training-row count. Self/Hist accumulates both model actors' decisions; RuleBot-side rule actions are not counted the same way. Worker-budget/engine-wait truncations can also assign ep_steps=1500 directly. Length therefore combines actual interaction differences and termination bookkeeping.

### Confirmed statistical limitations

1. worker.py increments timeout when ep_steps reaches 1500, then increments it again in the no-winner/cap-reached branch. One nonterminal truncation can count twice. Do not blindly halve the entire curve: real terminal cases and mixed branches require identification. This is a counter problem, not duplicate reward punishment.
2. Worker means are not weighted by game counts, and exactly-zero mean rewards are excluded. Workers with few and many games have equal weight.
3. Excluding aborted games protects training data; it is not error monitoring. Deadlock_Rate does not cover every rollback path. Inspect Worker error logs separately.

These are limitations of current bookkeeping, not evidence that all corresponding games failed. Read them alongside Worker logs and game settlements; they do not independently establish strength or capacity acceptance.

## 4. Auxiliary_Targets: coverage and event distribution

Logging: [trainer.py label audit](../trainer.py#L1243). Computation: [summarize_auxiliary_targets](../auxiliary_targets.py#L357). Events: [event_history.py](../event_history.py#L326). Related functionality: [events and public hints](public_observations_en.md), [network and auxiliary targets](architecture_en.md#phase-4-deck-protocol-boundary).

The first four denominators are merged training-row counts. The last five denominators are **rows with valid NEXT_DECISION labels**. All values are proportions in [0,1], not prediction accuracy or reward.

| Chart | Y-axis meaning | Interpretation and limits |
|---|---|---|
| `Auxiliary_Targets/coverage_next` | Rows with a reliable next-decision/Retry/true-terminal boundary | Near 1 is desirable; small unclosed tails can be missing |
| `Auxiliary_Targets/coverage_chain` | Rows with a reliable chain-end future summary | Only eligible chain contexts have this target; 0.4 does not imply 60% parsing loss |
| `Auxiliary_Targets/coverage_turn` | Rows with a reliable turn-end future summary | Higher coverage supplies more labels; truncated tails can reduce it slightly |
| `Auxiliary_Targets/coverage_terminal` | Rows with true Core-terminal future/outcome labels | Not the fraction of completed games; a game contributes many rows |
| `Auxiliary_Targets/retry_rate` | Valid immediate rows with the RETRY bit | Sustained growth warrants response/parser checks; near zero does not cover every failure |
| `Auxiliary_Targets/cancel_rate` | Valid immediate rows with CANCELLED | Includes legitimate cancellation; not all Decline/selection completion; rising does not prove a loop |
| `Auxiliary_Targets/state_changed_rate` | Valid immediate rows with STATE_CHANGED | LP/zone counts/chain depth changes or transition messages can set it; not tactical success |
| `Auxiliary_Targets/negated_rate` | Rows whose transition includes CHAIN_NEGATED | Can involve either player and a chain context; not successful AI interruption rate |
| `Auxiliary_Targets/disabled_rate` | Rows whose transition includes CHAIN_DISABLED | Distinct from activation negation; sparse rates do not directly measure strength |

Bits can coexist and need not sum to 1. coverage_terminal=0.9663 means about 96.63% of **rows** support true-terminal supervision, not 96.63% wins or only 3.37% truncated **games**.

## 5. Auxiliary_Train: structured predictive learning

Logging: [trainer.py joint loss and diagnostics](../trainer.py#L1712). Computation: [auxiliary_heads.py](../auxiliary_heads.py#L233). Related functionality: [auxiliary heads and gradient boundaries](architecture_en.md#phase-4-deck-protocol-boundary).

These charts measure **consequence prediction**: after a specified action, will cancellation occur, which messages appear, how do resources change, and how and when does the game end? They do not score agreement with a hand-written optimal strategy or convert prediction accuracy into game reward.

Each point measures a training mini-batch, not an independent validation set. Heads consume the current representation and the **executed candidate** representation to predict consequences. Labels are constructed after collection and never used as inference inputs. An empty valid mask returns zero masked means; zero loss can therefore mean no usable targets, not perfect prediction.

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Auxiliary_Train/Total_Loss` | Overall consequence-prediction error; Weighted mean of seven tasks: 1/1/0.5/1/1/0.5/0.25, total weight 5.25 | Excludes the outer 0.1 coefficient; compare matched versions and label distributions |
| `Auxiliary_Train/Backbone_Scale` | Strength of auxiliary backpropagation into shared representations; Gradient-bridge multiplier: zero for 100 updates, then ramps to 0.2 over 900 | A schedule, not prediction quality or a guarantee of a 20% total-gradient share |
| `Auxiliary_Train/Gradient_Norm` | Update signal strength on prediction decoders; Pre-clipping L2 norm of pure prediction/target-decoder parameters; cap 1.0 | Includes the planner target decoder; not auxiliary influence on the shared backbone |
| `Auxiliary_Train/Result_Loss` | Error predicting resolution, cancellation, Retry, negation, and other immediate results; Masked BCE-with-logits over 11 immediate result bits | Many negative sparse bits; low total loss alone is insufficient |
| `Auxiliary_Train/Message_Loss` | Error predicting which public message categories follow the action; Masked BCE over occurrence bits for 11 public Core-message groups | Does not reconstruct complete messages, card names, or every effect result |
| `Auxiliary_Train/Boundary_Loss` | Error predicting continued decisions, Retry, or game end; Masked three-class CE: next decision / Retry / terminal | Ordinary next decisions dominate; low loss can partly reflect majority-class frequency |
| `Auxiliary_Train/Immediate_Delta_Loss` | Error predicting LP, card-count, and chain-depth changes before the next boundary; Masked Smooth L1 over 20 normalized fields: two LP, 16 zone deltas, final/max chain depth | Many selections have zero resource change; not error in original units |
| `Auxiliary_Train/Future_Delta_Loss` | Error predicting resource changes at chain-end, turn-end, and terminal; Masked Smooth L1 over 18 LP/zone fields at chain-end, turn-end, terminal | Indicates future-resource learning; label coverage and changing board scales affect it |
| `Auxiliary_Train/Terminal_Outcome_Loss` | Error predicting loss/draw/win after the specified action; Masked CE for true-terminal loss/draw/win | Predicts actual future outcomes, not value regression targets |
| `Auxiliary_Train/Terminal_Remaining_Loss` | Error predicting remaining transition events until true terminal; Masked Smooth L1 for log1p(remaining transition events)/log1p(1500) | Not absolute turn error; 1500 normalizes labels rather than clipping them |
| `Auxiliary_Train/Result_Bit_Accuracy` | Per-bit accuracy over valid immediate rows×11 bits, logit≥0 means true | 99% is not 99% fully correct transitions or rare-negation recall |
| `Auxiliary_Train/Message_Bit_Accuracy` | Per-bit accuracy over valid rows×11 message groups | Class imbalance makes high values easy; independent per-class checks remain needed |
| `Auxiliary_Train/Boundary_Accuracy` | Argmax accuracy for three boundary classes on valid rows | Compare with an always-next-decision baseline |
| `Auxiliary_Train/Terminal_Accuracy` | True-terminal labelled rows' loss/draw/win classification accuracy | Not model win rate; repeated epochs and many states from one game are not independent tests |
| `Auxiliary_Train/Future_Valid_Fraction` | Mean validity across three future horizons in the current batch | Related to mean future coverage; batch and iteration statistics need not match |

The 11 result bits are Resolved, Retry, Cancelled, Declined, SelectionFinished, ChainNegated, ChainDisabled, Terminal, StateChanged, MessageOverflow, and MissedTiming. The 11 message groups cover turn/phase, movement/position, field control, summon, chain, target/equip, draw, LP, counters, battle, and public notifications. See [event definitions](../data_types.py#L107) and [message groups/zone order](../event_history.py#L24).

Continuous targets are actor-relative: LP delta/8000 and zone-count delta/5, both clamped to [−4,4]; chain depth/16 clamped to [0,4]. Zones include both players' deck, hand, monster, spell/trap, graveyard, banished, extra, and overlay. Remaining length counts both players' inter-decision transition events, including the current event, not only training-player actions.

## 6. Planning: latent goals and policy integration

Logging: [trainer.py](../trainer.py#L1890). Computation: [goal_planner.py](../goal_planner.py#L198). Integration: [galatea_net.py](../galatea_net.py#L1127). Related functionality: [planning paths and training boundaries](architecture_en.md#phase-4-deck-protocol-boundary).

These charts measure a **future-oriented summary of the current position**. Unlike the action-conditioned auxiliary heads, the planner does not read which action was actually selected. State and deck context produce a vector informative about future resources, outcomes, and remaining length. Neighbouring-pair consistency measures continuity; cross-sample variation measures whether different positions remain distinguishable.

The planner regenerates a 128-dimensional plan_latent from current state, stable deck style, and global context. Its target decoder runs for training/optional diagnostics; bounded independent residuals connect the representation to policy and value. Unlike a pure auxiliary probe, this representation participates in decisions. **It is neither MCTS nor an already-labelled combo/terminal-action script.**

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Planning/Total_Loss` | Combined future-prediction and continuity error; Future/TerminalOutcome/Remaining/Consistency weighted 1/0.5/0.25/0.1, divided by 1.85 | Excludes outer 0.05; changing tasks/data can prevent monotonic decline |
| `Planning/Gradient_Norm` | Joint update signal strength on the planning path; Pre-clipping L2 norm of non-decoder planner, projections, and gates; cap 1.0 | Includes PPO and planning supervision; not directly comparable parameter importance |
| `Planning/Future_Loss` | Error predicting chain-end, turn-end, and terminal resources from the current position; Masked Smooth L1 for three 18-field future LP/zone summaries | Same targets as auxiliary future prediction, but no executed-candidate input |
| `Planning/Terminal_Outcome_Loss` | Error predicting eventual loss/draw/win from the position; True-terminal three-class CE from the planning representation | Training predictive performance, not independent validation or strength |
| `Planning/Terminal_Remaining_Loss` | Error predicting remaining transition events from the position; Smooth L1 over the same log-normalized remaining-event target | Low error does not imply exact absolute-turn prediction |
| `Planning/Consistency_Loss` | Difference between plan directions at eligible neighbouring decisions; mean(1−cosine(plan_a,plan_b)) over eligible neighbouring pairs | Lower means more directional continuity; forcing zero can erase needed plan changes |
| `Planning/Latent_RMS` | Overall planning-vector magnitude; sqrt(mean(z²)) across the current 128-dimensional batch | LayerNorm makes approximately 1 normal; does not imply constant plans |
| `Planning/Latent_Mean_Dimension_Std` | Average plan variation across batch samples; Batch std for each dimension, then mean over 128 dimensions | Near zero warns of cross-state collapse; growth alone does not establish interpretable goals |
| `Planning/Latent_Active_Dimension_Fraction` | Fraction of planning dimensions that vary across this batch; Fraction of dimensions with batch std>0.01 | 1 means all dimensions vary, not full matrix rank or independent meanings |
| `Planning/Pair_Cosine` | Plan-direction similarity at eligible neighbouring decisions; Mean cosine similarity of eligible neighbouring pairs | Approximately 1−Consistency_Loss when pairs exist; inspect cross-batch std too |
| `Planning/Pair_Valid_Fraction` | Coverage of eligible pair starts used for continuity supervision; Valid adjacent pair starts / (batch size−1) | Not fraction of participating rows; maximum at batch512 is about 256/511≈0.501 |
| `Planning/Terminal_Accuracy` | Planning outcome classification accuracy over true-terminal labelled rows | Neither eventual game win rate nor held-out validation accuracy |
| `Planning/Future_Valid_Fraction` | Mean validity across the batch's three future horizons | Same definition as the auxiliary metric; coverage, not prediction quality |
| `Planning/Policy_Gate_RMS` | Gate magnitude on the planning-to-action-probability path; sqrt(mean(tanh(policy gate)²)), in [0,1] | Slow opening indicates participation; 0.045 does not mean 4.5% strength contribution |
| `Planning/Value_Gate_RMS` | Gate magnitude on the planning-to-return-estimation path; sqrt(mean(tanh(value gate)²)), in [0,1] | Learns independently; near-one saturation merits review, not celebration |

The actual residual is `tanh(gate) × 0.5 × tanh(projection(plan))`. Gate RMS omits the projection and 0.5 multiplier; it is not a logit, value, or gradient contribution percentage. Shared inputs use the Backbone_Scale bridge; the planner/gates and prediction decoders occupy separate parameter groups.

Consistency pairs contain adjacent training decisions by the same actor in the same turn/phase. Non-overlapping two-row units survive shuffling. At batch 512, Pair_Valid_Fraction≈0.44 corresponds to approximately 88% of rows in real pairs, **not 44% row coverage**. Different players, phases, and games need not have consistent plans.

### Planning_Histograms: not scalar y-axes

| Distribution | Logged tensor | How to read it |
|---|---|---|
| `Planning_Histograms/Plan_Latent` | All current batch plan_latent elements | Inspect width, tails, or collapse to constants; bell shape does not establish tactics |
| `Planning_Histograms/Policy_Gate` | Current model's channelwise tanh(policy gate) | Gradual spreading from zero means opening; a central spike can mean inactive channels |
| `Planning_Histograms/Value_Gate` | Current model's channelwise tanh(value gate) | Compare with policy gates; distributions need not match |

Histogram value axes contain tensor values, bucket heights count elements, and stacked/time axes use train_step. Distributions uses percentile bands. Step values on the time/stack axis are optimizer update counts, **not latent values or sample confidence**. These distributions are not predicted win-rate curves.

## 7. Train: current mini-batch PPO joint update

Logging/computation: [trainer.py](../trainer.py#L1712). Policy shift: [ppo_control.py](../ppo_control.py#L20). Related design: [PPO framework](architecture_en.md#ppo-training-framework), [KL early stopping](architecture_en.md#interpreting-kl-early-stopping).

**Policy** is the probability distribution over legal actions; **Value** is expected training return, not win probability. GAE advantage estimates whether a sampled action was better or worse than the value baseline. Policy updates increase positive-advantage action probabilities and decrease negative-advantage probabilities. Value fits GAE return targets combining later rewards and estimated values, not just terminal ±1 outcomes. See [PPO controls](quickstart_en.md#ppo-update-controls-3130).

Let `d=log π_new(a|s)−log π_old(a|s)`, `r=exp(d)`, and A be the iteration-normalized GAE advantage. Metrics below use sampled actions in the current batch, not the entire state space.

| Chart | Y-axis and formula | Interpretation and limits |
|---|---|---|
| `Train/Total_Loss` | Combined objective optimized by this update; Policy + 0.5×Value − entropy_coef×Entropy + 0.1×AuxiliaryTotal + 0.05×PlanningTotal | Can be negative; not an unweighted sum or a supervised loss that must always fall |
| `Train/Policy_Loss` | Objective adjusting action probabilities according to advantage; −mean(min(rA,clip(r,1−ε,1+ε)A)) | Positive/negative values and fluctuations can be normal; advantages change each iteration |
| `Train/Value_Loss` | Difference between estimated return and this batch's GAE return target; Smooth L1 between value and GAE lambda-return | Not error against true win probability or pure Monte Carlo returns |
| `Train/Entropy` | How spread out choices are: higher is more random, lower is more concentrated; Mean entropy of legal-candidate masked policy, in natural-log nats | Maximum is about log(candidate count), zero for one option; option structure affects averages |
| `Train/Approx_KL` | How much the policy changed relative to the collection policy; Sampled-action estimate mean(exp(d)−1−d) | Policy shift, not intelligence; not exact full-distribution KL on a fixed evaluation set |
| `Train/Clip_Fraction` | Fraction of sampled-action probability ratios outside PPO's permitted interval; mean(abs(r−1)>ε) | ε defaults to 0.2; out-of-range ratios, not exact fraction of gradients clipped |
| `Train/Explained_Variance` | How well value predictions explain variation in batch return targets; 1−Var(return−value)/Var(return); zero for tiny target variance | Can be negative, maximum 1; low residual variance does not guarantee unbiasedness/generalization |
| `Train/Gradient_Norm` | Joint update signal strength on the shared backbone; Pre-clipping backbone-group L2 norm; cap 0.5 | Includes auxiliary/planning gradients on shared parameters, excludes separate groups; >0.5 does not mean clipping failed |

Entropy is not confidence in a correct answer. Three equally likely candidates have entropy ln(3)≈1.10; a single legal candidate has zero entropy even without a learned strategy. Falling entropy more clearly indicates increasingly distinct preferences only when candidate counts and types are comparable.

KL measures the magnitude of change, not its quality. Low KL can mean stability or almost no updates. Clip_Fraction=0.2 means about 20% of sampled-action ratios are outside the interval, not that 20% of training data was removed.

High explained variance is not “95% terminal prediction accuracy”: GAE targets include value bootstrapping and this measures training samples. Outcome-classification accuracy, return-regression explained variance, and game win rate are different quantities.

## 8. PPO_Update: iteration controls and actual reuse

Definitions: [PPOUpdateAudit](../ppo_control.py#L47). Ordering: [trainer.py](../trainer.py#L1763). Related functionality: [PPO framework](architecture_en.md#ppo-training-framework), [KL explanation](architecture_en.md#interpreting-kl-early-stopping).

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `PPO_Update/Requested_Epochs` | Configured maximum traversals, default 4 | An upper limit, not a guarantee of four learning passes |
| `PPO_Update/Epochs_Started` | Entered epochs, including the interrupted pass | Exceeding Completed usually means mid-iteration KL stopping |
| `PPO_Update/Epochs_Completed` | Entirely traversed mini-batch loops | Integer; a loop can finish despite skipped nonfinite batches |
| `PPO_Update/Optimizer_Updates` | Successfully completed mini-batch optimizer steps | Persistent zeros/collapse merit checks; not forward-pass count |
| `PPO_Update/Effective_Epochs` | Actual sample uses in successful updates / accepted rows | Can be fractional; real reuse, not a smoothed KL statistic |
| `PPO_Update/Samples_Updated` | Sample uses in completed optimizer updates, counting repeated epochs | Not unique environment interactions |
| `PPO_Update/Skipped_Tail_Samples_Per_Epoch` | Accepted row count % mini_batch_size | Random incomplete tails are unused; up to511 at batch512, not all discarded rows |
| `PPO_Update/KL_Early_Stop` | Whether KL stopping triggered, 0/1 | Persistent1 means update-limited, not failed training or proven convergence |
| `PPO_Update/Target_KL` | Configured target, default0.02 | Zero disables the finite threshold, not nonfinite protection; frequent stops alone do not establish an incorrect threshold |
| `PPO_Update/KL_Stop_Threshold` | 1.5×Target_KL, default0.03 | A trigger, not a strict bound on every logged value |
| `PPO_Update/Approx_KL_Mean` | Mean of inspected mini-batch mean KL values | Not final full-batch re-evaluation; includes checks along the update path |
| `PPO_Update/Approx_KL_P50` | Median of those mini-batch means | Not a percentile of individual actions' KL |
| `PPO_Update/Approx_KL_P95` | 95th percentile of mini-batch means | Higher means some batches shift more; not per-action tails |
| `PPO_Update/Approx_KL_Max` | Maximum inspected mean KL, including the stopping batch | Crossing0.03 can be the trigger; not automatically a failed guard |
| `PPO_Update/Clip_Fraction_P50` | Median mini-batch clip fraction | Need not match a Train single point |
| `PPO_Update/Clip_Fraction_P95` | 95th percentile of mini-batch clip fraction | Does not mean95% of samples were clipped |
| `PPO_Update/Nonfinite_Batches` | Detected nonfinite KL/clip/loss or FP16 scaler skips | Persistent nonzero values require investigation; zero does not cover every error/memory condition |

### Why can KL_Max exceed the threshold?

Each batch is forwarded, its KL/clip registered, then checked. KL above 0.03 stops **before backward on that batch**, ending remaining joint policy/value/auxiliary/planning updates. Already-completed updates stay in place and the next iteration collects normally. The stopping batch enters PPO_Update Max/percentiles but not the ordinary Train backward-diagnostic points.

This does not project the policy back into a strict KL region. Completed updates may measure 0.05 on a subsequent batch; stopping prevents further updates rather than guaranteeing every state's KL≤0.03. PPO clipping is not a strict full-distribution constraint either. See [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) for the rationale behind additional KL stopping.

For example, an iteration with Effective_Epochs≈1.05, Completed=1, Started=2 completed one traversal and stopped early in the second, **not no learning**. Persistent first-batch stops/near-zero updates warrant checking old log-probs, sampling/update modes, learning rate, and data shifts before simply widening the threshold.

## 9. Performance and Configuration

Sources: [collect_rollouts](../trainer.py#L921), [run_training_loop](../trainer.py#L1954), [configuration text](../trainer.py#L606). Related design: [central inference](architecture_en.md#central-batched-inference-service), [CUDA semantic allocation boundaries](architecture_en.md#direct-cuda-semantic-aggregation-2026-10-10).

| Chart/text | Y-axis or contents | Interpretation and limits |
|---|---|---|
| `Performance/Collection_Seconds` | Complete collect_rollouts seconds, including launch/loading, waiting, cleanup, merge/statistics | Near-timeout behavior points to a collection budget, not GPU-update duration |
| `Performance/PPO_Update_Seconds` | Complete update_policy seconds including transfers, reconstruction, forward/backward, audit | KL reducing update count also reduces time; not direct operator speedup |
| `Performance/Rollout_Steps_Per_Second` | Accepted merged training rows / Collection_Seconds | Not all Core actions/sec or per-Worker throughput; multiply by duration to estimate accepted rows |
| `Configuration/PPO` | gamma, lambda, requested/actual lr, entropy, clip, epochs, target KL, multiplier, mini_batch at launch | Text, not a scalar; actual optimizer lr matters when resuming |

Configuration currently **omits** Worker count, collection timeout, layer configuration, Core/asset fingerprints, and per-group game counts. Events cannot establish CUDA peak/RSS stability; consult startup/memory logs separately.

## 10. Independent imitation metrics

Source: [independent trainer](../integrations/imitation/workflow.py), writing
`runs/imitation_<run_id>/events.out.tfevents.*`. Ten additional scalar tags are separate from PPO.
Each records once per completed BC epoch; X is an imitation pass, not a PPO iteration or Core event.
BC resume preserves run/epoch and purges later stale event displays. Training statistics span changing
mini-batch weights during an epoch; validation is read-only with final epoch weights. Single-choice
forced states contribute to neither.

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Imitation_Train/cross_entropy`, `Imitation_Validation/cross_entropy` | Mean natural-log `-log π(expert action)` over effective examples | Lower means assigning more probability to demonstrations, not higher playing strength. Falling train/rising validation may indicate overfitting |
| `Imitation_Train/balanced_cross_entropy`, `Imitation_Validation/balanced_cross_entropy` | Mean CE with inverse teacher-effective-count weights | Teachers contribute equally overall; balanced validation CE identifies best checkpoints without test data |
| `Imitation_Train/top1_accuracy`, `Imitation_Validation/top1_accuracy` | Fraction of effective examples whose argmax equals the expert label,0–1 | Agreement, not proof the label is uniquely good or optimal |
| `Imitation_Train/effective_samples`, `Imitation_Validation/effective_samples` | Effective multi-choice examples measured this epoch | Usually constant for fixed data; not cumulative events or forced/unmapped/invalid examples |
| `Imitation_Train/learning_rate` | BC optimizer's configured learning rate | Currently fixed and separate from main PPO |
| `Imitation_Train/optimizer_updates` | Cumulative BC mini-batch optimizer steps, continuing on resume | Training budget, not games or playing strength |

Per-teacher/prompt counts, CE and Top1 are in each experiment's `epoch_*.json` under `breakdown`, not
thousands of scalar cards. `result_*.json` records parent, dataset digest, saved and best-validation
checkpoints. Actual game outcomes, aborts and completed-game win rates are in `evaluation.json` and
cannot be replaced by action accuracy. See [explicit workflow and cache usage](core_trajectories_en.md#3153-independent-imitation-workflow).

## 11. WindBot main-training opponent metrics

Explicit WindBot training writes to the ordinary PPO `runs/<prefix>_<time>/events.out.tfevents.*`.
X is cumulative PPO iteration, not BC epoch, with one point per rollout merge. Definitions are implemented
in [Trainer](../trainer.py) from [Worker](../worker.py) commit/rollback records.

| Chart/text | Vertical axis or recording | Interpretation and limits |
|---|---|---|
| `League_Opponent/WindBot` | Model wins / trustworthy completed WindBot games this iteration, 0–1 | Real draws count in the denominator; failures/truncations do not. No completed games means no zero point. Changing executor/build mixes differ from a fixed benchmark |
| `WindBot_Training/Games_Completed` | Games with a genuine Core terminal and successful processing | A count, not win rate; zero when no WindBot games complete |
| `WindBot_Training/Games_Aborted` | Uploaded diagnostics for startup, connection, parsing or target-alignment failures | Failed games supply no training actions. Workers killed before upload are absent; also consult process logs |
| `WindBot_Training/Games_Truncated` | Budget/long-game truncations without a genuine terminal or infrastructure failure | Original truncation/GAE rules apply without invented outcomes |
| `WindBot_Training/Accepted_Model_Steps` | Model action rows from WindBot games actually committed to PPO | Excludes Bot replies, Pass1 queries and rolled-back rows. Sample share need not match Worker allocation probability |
| `WindBot_Training/Bot_Responses` | Successfully received Bot replies in uploaded diagnostics | Not training steps; a selection chain may contain multiple replies |
| `WindBot_Training/Bot_Response_Seconds` | Summed wall time for those successful replies, including refreshes/waiting | Neither pure Bot compute nor an average; compare alongside response/game counts |
| `WindBot_Mirror/WinRate` | Per-mode model wins / genuine completed games this iteration, 0–1 | Mirror and training-pool distributions remain separate; draws count, aborts/truncations do not; no point without completed games |
| `WindBot_Mirror/DrawRate` | Genuine draws / completed games for the mode, 0–1 | Budget truncations are not draws |
| `WindBot_Mirror/Games` | Completed games for the mode this iteration | Denominator count; zero when none complete |
| `WindBot_Mirror/WinRate_First` | Model wins / completed games in the corresponding mode and seat, 0–1 | No point without that seat's completed games; the model is not always P0/first |
| `WindBot_Mirror/WinRate_Second` | Model wins / completed games in the corresponding mode and seat, 0–1 | No point without that seat's completed games; the model is not always P0/first |
| `WindBot_Mirror/Games_First` | Completed-game denominator for the corresponding seat rate | Zero does not mean a0% win rate |
| `WindBot_Mirror/Games_Second` | Completed-game denominator for the corresponding seat rate | Zero does not mean a0% win rate |
| `WindBot_Pool/WinRate` | Per-mode model wins / genuine completed games this iteration, 0–1 | Mirror and training-pool distributions remain separate; draws count, aborts/truncations do not; no point without completed games |
| `WindBot_Pool/DrawRate` | Genuine draws / completed games for the mode, 0–1 | Budget truncations are not draws |
| `WindBot_Pool/Games` | Completed games for the mode this iteration | Denominator count; zero when none complete |
| `WindBot_Pool/WinRate_First` | Model wins / completed games in the corresponding mode and seat, 0–1 | No point without that seat's completed games; the model is not always P0/first |
| `WindBot_Pool/WinRate_Second` | Model wins / completed games in the corresponding mode and seat, 0–1 | No point without that seat's completed games; the model is not always P0/first |
| `WindBot_Pool/Games_First` | Completed-game denominator for the corresponding seat rate | Zero does not mean a0% win rate |
| `WindBot_Pool/Games_Second` | Completed-game denominator for the corresponding seat rate | Zero does not mean a0% win rate |
| `WindBot_Executors/{Mirror,Pool}/<executor>/WinRate` | Model wins / trustworthy completed games for that mode and executor, 0–1 | Written only with completed games; executor labels are not measured ratings |
| `WindBot_Executors/{Mirror,Pool}/<executor>/Games` | Completed-game denominator for the associated win-rate point | Check sample size before interpreting fluctuations |
| `Configuration/WindBot` | Active/reserved rosters, fixed builds, upstream levels, file/runtime hashes, allocation and model deck mode | Text, not a scalar. Records this launch's settings; resuming does not automatically enable opponents |

From3.15.5, `league_weights` in the existing `Configuration/PPO` Text panel records the four absolute
RuleBot/WindBot/history/self probabilities, not realized sample shares. No scalar tag is added.
See [WindBot training opponents](../integrations/windbot/README.md#optional-main-training-opponents)
for configuration, per-level holdouts and CLI examples.

From3.15.6, `Deck_WinRates/*` and ordinary pool dashboards exclude WindBot. Separate
`web_data/windbot_history_<run_id>.csv` files power the WindBot tab, with mode/executor/model-pool
filters, genuine W/D/L, abort/truncation counts, seat denominators and per-game build hashes.
These records are written once per collection merge; they are neither model inputs nor PPO losses.
The storage repository's match-data controls manage both CSV families.

Old `WindBot_Executors/<executor>/*` tags combine modes and are no longer extended. Existing events
are not rewritten. `League_Opponent/WindBot` remains the combined-mode rate; `League_Overall/*`
continues to describe the actual league mixture, including WindBot. Neither is a fixed benchmark.
Identifiable legacy `windbot:` mirror rows are shown separately; older pool rows without opponent
provenance cannot be reliably split. Old WindBot contributions to scalar deck rates cannot be removed
by reverse inference.

Existing denominator differences also matter: ordinary deck CSV and `Deck_WinRates` record decisive
games only, while WindBot rates include genuine draws. The old RuleBot/Historical_AI league tags write
zero when that opponent has no games in an iteration; this does not mean all games were lost. This
batch does not rewrite those historical tags. Worker central round-trip timing now includes both
self-play sides, queue/batch waits, transfers and the forward pass. It is not pure ZMQ transport time;
historical CPU forward passes and WindBot refresh/reply waits are separate.

Ordinary `Deck_WinRates` still groups by deck name rather than pool/build fingerprint, so distinct
same-name builds across pools may share a series. Ordinary WebUI pool rankings group by pool;
the WindBot tab retains separate provenance fields. These views have different grouping rules.
