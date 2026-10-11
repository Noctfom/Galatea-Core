# TensorBoard Metric Reference and Initial V4 Training Analysis

> Applies to Galatea-Core v3.14.0 / Model Protocol V4. This reference documents current logging behavior; it changes no networks, rewards, training settings, or protocol versions.
> [中文](tensorboard_metrics.md)

This document explains what each chart measures, how to distinguish ordinary variation from problems, and what the V4 cloud run started on 2026-10-10 currently demonstrates. The implementation has **80 fixed scalars, 3 planning distributions, and 1 configuration text**. Variable per-deck `Deck_WinRates/*` series are not individually catalogued. Fixed metrics hidden by screenshot pagination are included.

The preliminary assessment is: **clear progress within the training distribution, currently stable numerical updates, and no obvious constant planning representation collapse. This does not establish generalization, exclude specialization against RuleBot, or demonstrate a 20-fold improvement over V3.**

Jump to: [logging rules](#1-storage-axes-and-reading-rules) · [target coverage](#4-auxiliary_targets-coverage-and-event-distribution) · [auxiliary learning](#5-auxiliary_train-structured-predictive-learning) · [planning](#6-planning-latent-goals-and-policy-integration) · [PPO guards](#8-ppo_update-iteration-controls-and-actual-reuse) · [cloud analysis](#10-cloud-run-raw-events-rather-than-smoothed-screenshots) · [possible causes](#11-why-might-v4-learn-faster-than-v3)

## 1. Storage, axes, and reading rules

### Files and interface

- Events are stored in `runs/<model_prefix>_<launch_date-time>/events.out.tfevents.*`. Each launch creates a run directory; resuming inherits iteration and train_step, so axes need not restart at zero. Matching prefixes do not establish matching model UUIDs.
- WebUI: **TensorBoard → Start TensorBoard**. Scalars shows scalar metrics, Histograms / Distributions shows planning distributions, and Text contains `Configuration/PPO`.
- Standalone launch: `python -m tensorboard.main --logdir ./runs`. Inspect resumed runs separately before comparing continuous steps.
- Sources: [SummaryWriter and configuration](../trainer.py#L604), [WebUI TensorBoard module](../app.py#L3949).

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

## 2. League: training game win rates

Logging: [trainer.py collection statistics](../trainer.py#L1194). Counters: [worker.py game settlement](../worker.py#L1260). Related design: [league training](architecture_en.md#league-training-mechanism), [Arena benchmarks and policy modes](features_en.md#launch--monitor-hub).

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

Logging: [trainer.py](../trainer.py#L1211). Generation: [Worker rewards and GAE](../worker.py#L1284). Related design: [training loop](architecture_en.md#training-loop), [PPO controls](quickstart_en.md).

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

This document records these gaps; **it does not change statistics or training behavior**. Until aggregation is corrected, these curves remain trend references rather than standalone strength/capacity acceptance tests.

## 4. Auxiliary_Targets: coverage and event distribution

Logging: [trainer.py label audit](../trainer.py#L1240). Computation: [summarize_auxiliary_targets](../auxiliary_targets.py#L357). Events: [event_history.py](../event_history.py#L326). Related design: [auxiliary targets](protocol_v4_implementation_en.md#44-auxiliary-heads-and-planning), [public hints and parsing boundaries](core_message_audit_en.md).

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

Logging: [trainer.py joint loss and diagnostics](../trainer.py#L1712). Computation: [auxiliary_heads.py](../auxiliary_heads.py#L233). Related design: [auxiliary specification](protocol_v4_implementation_en.md#44-auxiliary-heads-and-planning), [network/gradient boundaries](architecture_en.md#phase-4-deck-protocol-boundary).

Each point measures a training mini-batch, not an independent validation set. Heads consume the current representation and the **executed candidate** representation to predict consequences. Labels are constructed after collection and never used as inference inputs. An empty valid mask returns zero masked means; zero loss can therefore mean no usable targets, not perfect prediction.

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Auxiliary_Train/Total_Loss` | Weighted mean of seven tasks: 1/1/0.5/1/1/0.5/0.25, total weight 5.25 | Excludes the outer 0.1 coefficient; compare matched versions and label distributions |
| `Auxiliary_Train/Backbone_Scale` | Gradient-bridge multiplier: zero for 100 updates, then ramps to 0.2 over 900 | A schedule, not prediction quality or a guarantee of a 20% total-gradient share |
| `Auxiliary_Train/Gradient_Norm` | Pre-clipping L2 norm of pure prediction/target-decoder parameters; cap 1.0 | Includes the planner target decoder; not auxiliary influence on the shared backbone |
| `Auxiliary_Train/Result_Loss` | Masked BCE-with-logits over 11 immediate result bits | Many negative sparse bits; low total loss alone is insufficient |
| `Auxiliary_Train/Message_Loss` | Masked BCE over occurrence bits for 11 public Core-message groups | Does not reconstruct complete messages, card names, or every effect result |
| `Auxiliary_Train/Boundary_Loss` | Masked three-class CE: next decision / Retry / terminal | Ordinary next decisions dominate; low loss can partly reflect majority-class frequency |
| `Auxiliary_Train/Immediate_Delta_Loss` | Masked Smooth L1 over 20 normalized fields: two LP, 16 zone deltas, final/max chain depth | Many selections have zero resource change; not error in original units |
| `Auxiliary_Train/Future_Delta_Loss` | Masked Smooth L1 over 18 LP/zone fields at chain-end, turn-end, terminal | Indicates future-resource learning; label coverage and changing board scales affect it |
| `Auxiliary_Train/Terminal_Outcome_Loss` | Masked CE for true-terminal loss/draw/win | Predicts actual future outcomes, not value regression targets |
| `Auxiliary_Train/Terminal_Remaining_Loss` | Masked Smooth L1 for log1p(remaining transition events)/log1p(1500) | Not absolute turn error; 1500 normalizes labels rather than clipping them |
| `Auxiliary_Train/Result_Bit_Accuracy` | Per-bit accuracy over valid immediate rows×11 bits, logit≥0 means true | 99% is not 99% fully correct transitions or rare-negation recall |
| `Auxiliary_Train/Message_Bit_Accuracy` | Per-bit accuracy over valid rows×11 message groups | Class imbalance makes high values easy; independent per-class checks remain needed |
| `Auxiliary_Train/Boundary_Accuracy` | Argmax accuracy for three boundary classes on valid rows | Compare with an always-next-decision baseline |
| `Auxiliary_Train/Terminal_Accuracy` | True-terminal labelled rows' loss/draw/win classification accuracy | Not model win rate; repeated epochs and many states from one game are not independent tests |
| `Auxiliary_Train/Future_Valid_Fraction` | Mean validity across three future horizons in the current batch | Related to mean future coverage; batch and iteration statistics need not match |

The 11 result bits are Resolved, Retry, Cancelled, Declined, SelectionFinished, ChainNegated, ChainDisabled, Terminal, StateChanged, MessageOverflow, and MissedTiming. The 11 message groups cover turn/phase, movement/position, field control, summon, chain, target/equip, draw, LP, counters, battle, and public notifications. See [event definitions](../data_types.py#L107) and [message groups/zone order](../event_history.py#L24).

Continuous targets are actor-relative: LP delta/8000 and zone-count delta/5, both clamped to [−4,4]; chain depth/16 clamped to [0,4]. Zones include both players' deck, hand, monster, spell/trap, graveyard, banished, extra, and overlay. Remaining length counts both players' inter-decision transition events, including the current event, not only training-player actions.

## 6. Planning: latent goals and policy integration

Logging: [trainer.py](../trainer.py#L1870). Computation: [goal_planner.py](../goal_planner.py#L198). Integration: [galatea_net.py](../galatea_net.py#L1127). Related design: [auxiliary heads and planning](protocol_v4_implementation_en.md#44-auxiliary-heads-and-planning).

The planner regenerates a 128-dimensional plan_latent from current state, stable deck style, and global context. Its target decoder runs for training/optional diagnostics; bounded independent residuals connect the representation to policy and value. Unlike a pure auxiliary probe, this representation participates in decisions. **It is neither MCTS nor an already-labelled combo/terminal-action script.**

| Chart | Y-axis and aggregation | Interpretation and limits |
|---|---|---|
| `Planning/Total_Loss` | Future/TerminalOutcome/Remaining/Consistency weighted 1/0.5/0.25/0.1, divided by 1.85 | Excludes outer 0.05; changing tasks/data can prevent monotonic decline |
| `Planning/Gradient_Norm` | Pre-clipping L2 norm of non-decoder planner, projections, and gates; cap 1.0 | Includes PPO and planning supervision; not directly comparable parameter importance |
| `Planning/Future_Loss` | Masked Smooth L1 for three 18-field future LP/zone summaries | Same targets as auxiliary future prediction, but no executed-candidate input |
| `Planning/Terminal_Outcome_Loss` | True-terminal three-class CE from the planning representation | Training predictive performance, not independent validation or strength |
| `Planning/Terminal_Remaining_Loss` | Smooth L1 over the same log-normalized remaining-event target | Low error does not imply exact absolute-turn prediction |
| `Planning/Consistency_Loss` | mean(1−cosine(plan_a,plan_b)) over eligible neighbouring pairs | Lower means more directional continuity; forcing zero can erase needed plan changes |
| `Planning/Latent_RMS` | sqrt(mean(z²)) across the current 128-dimensional batch | LayerNorm makes approximately 1 normal; does not imply constant plans |
| `Planning/Latent_Mean_Dimension_Std` | Batch std for each dimension, then mean over 128 dimensions | Near zero warns of cross-state collapse; growth alone does not establish interpretable goals |
| `Planning/Latent_Active_Dimension_Fraction` | Fraction of dimensions with batch std>0.01 | 1 means all dimensions vary, not full matrix rank or independent meanings |
| `Planning/Pair_Cosine` | Mean cosine similarity of eligible neighbouring pairs | Approximately 1−Consistency_Loss when pairs exist; inspect cross-batch std too |
| `Planning/Pair_Valid_Fraction` | Valid adjacent pair starts / (batch size−1) | Not fraction of participating rows; maximum at batch512 is about 256/511≈0.501 |
| `Planning/Terminal_Accuracy` | Planning outcome classification accuracy over true-terminal labelled rows | Neither eventual game win rate nor held-out validation accuracy |
| `Planning/Future_Valid_Fraction` | Mean validity across the batch's three future horizons | Same definition as the auxiliary metric; coverage, not prediction quality |
| `Planning/Policy_Gate_RMS` | sqrt(mean(tanh(policy gate)²)), in [0,1] | Slow opening indicates participation; 0.045 does not mean 4.5% strength contribution |
| `Planning/Value_Gate_RMS` | sqrt(mean(tanh(value gate)²)), in [0,1] | Learns independently; near-one saturation merits review, not celebration |

The actual residual is `tanh(gate) × 0.5 × tanh(projection(plan))`. Gate RMS omits the projection and 0.5 multiplier; it is not a logit, value, or gradient contribution percentage. Shared inputs use the Backbone_Scale bridge; the planner/gates and prediction decoders occupy separate parameter groups.

Consistency pairs contain adjacent training decisions by the same actor in the same turn/phase. Non-overlapping two-row units survive shuffling. At batch 512, Pair_Valid_Fraction≈0.44 corresponds to approximately 88% of rows in real pairs, **not 44% row coverage**. Different players, phases, and games need not have consistent plans.

### Planning_Histograms: not scalar y-axes

| Distribution | Logged tensor | How to read it |
|---|---|---|
| `Planning_Histograms/Plan_Latent` | All current batch plan_latent elements | Inspect width, tails, or collapse to constants; bell shape does not establish tactics |
| `Planning_Histograms/Policy_Gate` | Current model's channelwise tanh(policy gate) | Gradual spreading from zero means opening; a central spike can mean inactive channels |
| `Planning_Histograms/Value_Gate` | Current model's channelwise tanh(value gate) | Compare with policy gates; distributions need not match |

Histogram value axes contain tensor values, bucket heights count elements, and stacked/time axes use train_step. Distributions uses percentile bands. The screenshots' right-side 500–7500 values are update steps, **not latent values or sample confidence**. These distributions are not predicted win-rate curves.

## 7. Train: current mini-batch PPO joint update

Logging/computation: [trainer.py](../trainer.py#L1712). Policy shift: [ppo_control.py](../ppo_control.py#L20). Related design: [PPO framework](architecture_en.md#ppo-training-framework), [KL early stopping](architecture_en.md#interpreting-kl-early-stopping).

Let `d=log π_new(a|s)−log π_old(a|s)`, `r=exp(d)`, and A be the iteration-normalized GAE advantage. Metrics below use sampled actions in the current batch, not the entire state space.

| Chart | Y-axis and formula | Interpretation and limits |
|---|---|---|
| `Train/Total_Loss` | Policy + 0.5×Value − entropy_coef×Entropy + 0.1×AuxiliaryTotal + 0.05×PlanningTotal | Can be negative; not an unweighted sum or a supervised loss that must always fall |
| `Train/Policy_Loss` | −mean(min(rA,clip(r,1−ε,1+ε)A)) | Positive/negative values and fluctuations can be normal; advantages change each iteration |
| `Train/Value_Loss` | Smooth L1 between value and GAE lambda-return | Not error against true win probability or pure Monte Carlo returns |
| `Train/Entropy` | Mean entropy of legal-candidate masked policy, in natural-log nats | Maximum is about log(candidate count), zero for one option; option structure affects averages |
| `Train/Approx_KL` | Sampled-action estimate mean(exp(d)−1−d) | Policy shift, not intelligence; not exact full-distribution KL on a fixed evaluation set |
| `Train/Clip_Fraction` | mean(abs(r−1)>ε) | ε defaults to 0.2; out-of-range ratios, not exact fraction of gradients clipped |
| `Train/Explained_Variance` | 1−Var(return−value)/Var(return); zero for tiny target variance | Can be negative, maximum 1; low residual variance does not guarantee unbiasedness/generalization |
| `Train/Gradient_Norm` | Pre-clipping backbone-group L2 norm; cap 0.5 | Includes auxiliary/planning gradients on shared parameters, excludes separate groups; >0.5 does not mean clipping failed |

High explained variance is not “95% terminal prediction accuracy”: GAE targets include value bootstrapping and this measures training samples. Outcome-classification accuracy, return-regression explained variance, and game win rate are different quantities.

## 8. PPO_Update: iteration controls and actual reuse

Definitions: [PPOUpdateAudit](../ppo_control.py#L47). Ordering: [trainer.py](../trainer.py#L1763). Related design: [training/data constraints](protocol_v4_implementation_en.md#5-training-and-data), [KL explanation](architecture_en.md#interpreting-kl-early-stopping).

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
| `PPO_Update/Target_KL` | Configured target, default0.02 | Zero disables the finite threshold, not nonfinite protection; do not disable just to avoid stops |
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

At iteration 76, Effective_Epochs=1.0473, Completed=1, Started=2 means one full traversal followed by early stopping in the second, **not no learning**. Persistent first-batch stops/near-zero updates warrant checking old log-probs, sampling/update modes, learning rate, and data shifts before simply widening the threshold.

## 9. Performance and Configuration

Sources: [collect_rollouts](../trainer.py#L921), [run_training_loop](../trainer.py#L1954), [configuration text](../trainer.py#L604). Related design: [central inference](architecture_en.md#central-batched-inference-service), [CUDA semantic allocation boundaries](architecture_en.md#direct-cuda-semantic-aggregation-2026-10-10).

| Chart/text | Y-axis or contents | Interpretation and limits |
|---|---|---|
| `Performance/Collection_Seconds` | Complete collect_rollouts seconds, including launch/loading, waiting, cleanup, merge/statistics | Near-timeout behavior points to a collection budget, not GPU-update duration |
| `Performance/PPO_Update_Seconds` | Complete update_policy seconds including transfers, reconstruction, forward/backward, audit | KL reducing update count also reduces time; not direct operator speedup |
| `Performance/Rollout_Steps_Per_Second` | Accepted merged training rows / Collection_Seconds | Not all Core actions/sec or per-Worker throughput; multiply by duration to estimate accepted rows |
| `Configuration/PPO` | gamma, lambda, requested/actual lr, entropy, clip, epochs, target KL, multiplier, mini_batch at launch | Text, not a scalar; actual optimizer lr matters when resuming |

Configuration currently **omits** Worker count, collection timeout, layer configuration, Core/asset fingerprints, and per-group game counts. Events cannot establish CUDA peak/RSS stability; consult startup/memory logs separately.

## 10. Cloud run: raw events rather than smoothed screenshots

### Scope and comparisons

File: `events.out.tfevents.1791642844.autodl-container-280f4f89e7-89ac241d.48076.0`.
User-confirmed configuration: **512/8/6 network, 32768 target rows/iteration, 512 mini-batch, 12 Workers, timeout 600 seconds**. Configuration text confirms gamma=0.998, lambda=0.98, actual lr=0.0001, entropy=0.03, clip=0.2, at most 4 epochs, and target KL=0.02.

Screenshots mostly reach iteration 76; the file contains **79 collections / 78 completed PPO phases**. Collection 79 is not evidence of a completed 79th learning phase. The file has 80 fixed scalars, 3191 additional per-deck scalar series, 3 distributions, and configuration text. Windows below stay within 1–76 to explain the screenshots; two later completed iterations are insufficient to declare a trend reversal.

Values are **arithmetic means of raw points**, not TensorBoard smoothing or pooled game-weighted rates. Train/Auxiliary/Planning points are partitioned by iteration PPO-end timestamps and averaged within each window. Repeated updates are not independent games.

| Metric | Iterations 1–10 | 11–30 | 31–50 | 51–76 |
|---|---:|---:|---:|---:|
| Mean per-iteration RuleBot win rate | 46.81% | 68.83% | 81.13% | 85.49% |
| Mean historical-opponent win rate | 0(no pool yet) | 47.86% | 50.78% | 62.72% |
| Mean Self win rate | 46.86% | 46.31% | 47.39% | 48.23% |
| Mean mixed-league win rate | 46.99% | 50.73% | 54.43% | 56.48% |
| Average_Reward | −0.0962 | 0.0290 | 0.1331 | 0.1742 |
| Average_Length | 526.6 | 430.8 | 332.6 | 305.8 |
| Logged Timeout_Rate(duplicate counts) | 15.45% | 12.85% | 9.46% | 8.88% |
| Mean effective epochs | 2.832 | 1.776 | 1.567 | 1.573 |
| KL-stop iterations/window size | 7/10 | 20/20 | 20/20 | 26/26 |
| Collection seconds | 605.5 | 606.2 | 605.6 | 605.3 |
| PPO seconds | 181.0 | 111.9 | 100.2 | 97.8 |
| Accepted collection rows/second | 48.57 | 47.42 | 48.36 | 46.89 |
| Train/Entropy,nats | 1.291 | 1.206 | 1.078 | 1.041 |
| Train/Explained_Variance | 0.872 | 0.898 | 0.909 | 0.898 |
| Train/Value_Loss | 0.0070 | 0.0106 | 0.0136 | 0.0170 |
| Auxiliary future-resource loss | 0.0951 | 0.0777 | 0.0678 | 0.0647 |
| Planning future-resource loss | 0.0797 | 0.0746 | 0.0669 | 0.0647 |
| Mean planning dimension batch std | 0.589 | 0.648 | 0.665 | 0.662 |
| Policy gate RMS | 0.0071 | 0.0218 | 0.0337 | 0.0419 |
| Value gate RMS | 0.0032 | 0.0207 | 0.0400 | 0.0551 |

### Health assessment

**Evidence of useful learning:** RuleBot, historical, and mixed rates improve while Self stays near 0.5 rather than all groups suspiciously approaching 1. Reward becomes positive and length falls. Future-resource losses improve; entropy decreases gradually instead of collapsing. Planning dimensions still vary, active fraction remains 1, and gates open progressively without obvious constant collapse/saturation. All 78 PPO Nonfinite_Batches values are zero; inspected fixed scalars contain no NaN/Inf.

**Reasons not to label everything healthy unconditionally:**

1. KL stops every iteration from 4 through 78: **75/78 phases**. Later effective epochs average about 1.6, so learning continues, but the 4-epoch ceiling is rarely used. KL_Max reaches **0.0803** globally; its 51–76 mean is about 0.0381. Skipping the trigger batch does not guarantee all completed policy changes are small. Retain the guard and compare fixed benchmarks rather than disabling it to complete four passes.
2. Mean Value_Loss rises from 0.0070 to 0.0170 while explained variance remains about 0.90. Changing return/opponent distributions can increase absolute error; this does not look like numerical collapse, but value learning is not uniformly improving.
3. Auxiliary terminal accuracy averages about 90.7%/89.6% in 11–30/51–76; planning terminal accuracy about 91.8%/90.5%. Planning Total_Loss is not monotonically falling either. These tasks still need training and independent validation; a screenshot endpoint near 95% does not establish completion.
4. Iteration 76 raw reward is **0.0744**, below displayed smoothing near 0.1837. Iterations 77/78 then report about 0.330/0.193, and collection 79 about 0.073. Genuine variation exists; neither one low point nor a favourable smoothed endpoint settles the trend.
5. Later cancellation rises and state-change rates are nonmonotonic. Some cancellations are sensible, others may indicate unresolved loops. Inspect complete selection flows in replay rather than classifying by rate alone. Retry/deadlock coverage limits remain relevant.

### Collection-efficiency boundary

Collection mostly reaches the **600-second budget**; roughly 606 seconds also includes launch/cleanup/merge overhead. Workers reserve time before the deadline and can truncate active games; Trainer also has wait/termination guards. This is not 600 seconds of PPO, and timeout_rate does not independently recover budget-truncation frequencies.

Duration×throughput reconstructs about **2.2523 million accepted rows over the 78 completed phases, averaging 28875 rows/iteration: approximately 88.1% of 32768**. Collection 79 separately adds about 26970 rows. These exclude epoch reuse but are not statistically independent game samples; estimates are rounded from float32 events. Completed phases performed **7831 mini-batch optimizer updates**, about 4.0095 million sample uses including reuse across epochs.

Much of falling PPO duration matches fewer effective passes, not necessarily faster operators. Later collection throughput declines modestly from about 48.6 to 46.9 rows/sec rather than accelerating continuously. Collection dominates wall time; identify environment/historical-opponent/central-inference costs before redesigning the trainer. Changing timeout is a collection-distribution/budget experiment, not a guaranteed strength improvement.

## 11. Why might V4 learn faster than V3?

These are **implementation-consistent mechanisms and hypotheses**, not a causal decomposition established by one run. The V3 near-1000-iteration reference comes from the user's observations; matched V3 events, opponents, decks, assets, and budgets were not supplied here, so a rigorous speedup factor cannot be calculated.

### 11.1 Information becomes distinguishable rather than guessed

Exact card vocabulary removes old modulo-ID aliasing. Main/Extra partitions, categorical phase/zone/position, decision-versus-turn identity, operation families, and summon/selection/public relationships distinguish situations previously merged or underspecified. Equal network dimensions do not mean equal usable information.

When inputs cannot distinguish situations with different correct actions, additional training must compromise across them. Better inputs can make activation versus setting, attacking versus ending, or extra-deck endpoints easier to condition on. This does not supply optimal rules; it reduces representation ambiguity and the burden of inferring absent fields from outcomes. See [target data protocol](protocol_v4_implementation_en.md#3-target-data-protocol).

### 11.2 Semantic-asset repair may matter greatly, separately from architecture

The old 27647-slot vector library had only 530 distinct vectors, with 27007 identical entries, mostly encoding empty source. Rebuilding produced 26092 distinct vectors and retained registrations, callbacks, and long-source tails. See [semantic provenance repair](code_semantics_en.md).

**If that V3 run used the audited old assets**, many effects' code-semantic channel effectively supplied almost identical information. Rebuilt assets let the existing module exploit more realistic differences, potentially a larger benefit than merely adding branches. This remains a plausible cause, not proof that a general sentence encoder understands all Lua or that the planner deserves all gains. Keep old models with their original assets; swapping semantic libraries invalidates identity/input comparisons.

### 11.3 Deck style and FiLM condition decisions without rule rewards

Stable Main/Extra profiles produce learned Deck Encoder style latents. Layered FiLM separates deck-style modulation from phase/player/global context. Similar local actions can therefore be evaluated differently for different deck endpoints and situations instead of sharing one unconditional preference.

This is learned composition-to-policy conditioning, not manually fixing decks to attack/defence categories. It may also specialize to training preferences; unseen-family testing is needed. See [Deck Encoder](protocol_v4_implementation_en.md#41-reusable-deck-encoder) and [layered FiLM](protocol_v4_implementation_en.md#42-layered-film).

### 11.4 Auxiliary tasks provide nearer, more specific learning signals

Previously, terminal rewards mainly relied on value/advantage propagation to assign credit across actions. Now actual actions also supervise public events, immediate resource changes, chain/turn-end summaries, and genuine outcomes. External rewards remain predominantly outcome-based, but representation learning no longer depends only on sparse wins/losses.

Ordinary auxiliary heads neither inject predicted probabilities into decisions nor execute rules. Small loss weights and controlled shared gradients encourage consequential features. Improving future losses support learning, not a measured win-rate contribution. Auxiliary shared-representation learning has research support; this implementation differs from [UNREAL](https://arxiv.org/abs/1611.05397), whose acceleration figures cannot be transferred to this project.

### 11.5 Planning feeds future-oriented representations back into decisions

The planner sees current state/deck/context, not real future answers. Multi-horizon targets supervise its summary, then gradually opening bounded residuals influence policy/value. This creates explicit pressure to organize expected resources/outcomes beyond isolated candidate scores.

Consistency discourages abrupt representation resets within a phase without prescribing actions. It may make combo goal/process relations easier to share, but bell-shaped latents cannot establish specific learned sequences. See [planning specification](protocol_v4_implementation_en.md#44-auxiliary-heads-and-planning).

### 11.6 GAE changed too: the differences are not only cognition plus KL

The file confirms lambda=0.98, compared with the old default 0.95. With gamma=0.998, GAE weights more distant TD residuals by `(gamma×lambda)^distance`:

| Setting | gamma×lambda | Geometric weight sum 1/(1−gamma×lambda) | Residual weight 100 steps away |
|---|---:|---:|---:|
| lambda=0.95 | 0.94810 | 19.27 | 0.00485 |
| lambda=0.98 | 0.97804 | 45.54 | 0.10856 |

More distant residuals decay less rapidly, plausibly helping credit assignment for multi-stage plays. **These are GAE weights, not 19/46-step model memory limits or guarantees of fully propagating terminal rewards 100 steps.** Bootstrapping and repeated training still matter. Higher lambda also raises estimation variance; it is not universally better. Recompute comparisons if the old run used a nondefault setting. See the [GAE paper](https://arxiv.org/abs/1506.02438) and [3.13.0 change record](changelog_en.md).

### 11.7 KL guards and data feedback may preserve progress

KL adds no observation information or good-action labels. It limits repeated updates on stale-policy samples and can help retain improvements. This does not establish that overshooting caused all V3 inefficiency.

If genuine games become shorter, a fixed interaction budget can supply more completed outcomes/future labels, reinforcing useful learning. Falling length and high terminal coverage fit this hypothesis, but Worker averaging/truncation biases require per-game checks. CUDA semantic aggregation preserves learning inputs/targets and addresses temporary allocations; it is not added cognition.

## 12. Checking overfitting and next steps

Continuing the current configuration is reasonable while retaining KL protection; do not aim to force four epochs simply because stops are frequent. Add independent evidence using existing Arena benchmarks. **This documentation task did not launch training, alter settings, or run paid cloud evaluations.**

1. **Freeze test conditions:** Core, CDB, scripts, banlist, deck schedule, and temperature; swap seats. Evaluate training-distribution sampling first, then Link deployment temperature separately. Sampling at 1.0 and 0.2/greedy are different policies.
2. **Hold out deck families:** Group near-identical builds/theme families together rather than splitting copied filenames. Report familiar decks, unseen builds, and unseen families separately.
3. **Use fixed diverse opponents:** Separate RuleBot, frozen V4 snapshots, independent models, and human games. Exploiting RuleBot can be genuine opponent adaptation without establishing general strength.
4. **Retain game denominators:** Record wins/losses/draws/truncations/aborts, seats, and opponents. Estimate intervals by games/schedule groups; hundreds of states from one game are not hundreds of independent trials. Missing-group zeros should not become evaluation losses.
5. **Compare real budgets:** Accepted interaction rows, completed games, optimizer updates, and wall time. V4's 78 phases provide about 2.25 million rows; iteration counts alone do not establish a speedup against 1000 V3 rounds. Use independent V3 software/original assets for V3 comparisons, never V3 checkpoints inside V4 or substituted semantic tables.
6. **Use controlled ablations if attribution is needed:** Keep V4 protocol/assets/schedules fixed and train from scratch across multiple seeds while varying auxiliary/planning gradients, deck modulation, lambda, and KL. Disabling branches or changing assets in an already-trained model is not a rigorous contribution measurement.
7. **Validate predictors independently:** Hold out entire games/deck groups; separate opening/midgame/near-terminal states. Add per-class positive recall and majority baselines. 99% training bit accuracy does not establish rare timing/negation/target understanding.

Acceptance asks whether independent conditions improve, not whether every training loss declines together. Current status: **valuable early/mid-run V4 progress worth continued verification, not completed generalization or final-strength acceptance.**
