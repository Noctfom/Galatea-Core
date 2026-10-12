# Initial V4 cloud training analysis

> Development analysis of the single v3.14.0 / Model Protocol V4 experiment launched on 2026-10-10. Not a general user manual or release acceptance report.
> [中文](training_analysis_v4_initial.md)

This record examines training win rates, predictive errors, policy updates, and runtime. Results support in-distribution progress and numerical stability, not demonstrated generalization or a controlled speedup over V3. General chart definitions are in the [TensorBoard Metric Reference](tensorboard_metrics_en.md).

## 1. Run data and results

### Scope and comparisons

File: `events.out.tfevents.1791642844.autodl-container-280f4f89e7-89ac241d.48076.0`.
Run configuration: **512/8/6 network, 32768 target rows/iteration, 512 mini-batch, 12 Workers, timeout 600 seconds**. Configuration text confirms gamma=0.998, lambda=0.98, actual lr=0.0001, entropy=0.03, clip=0.2, at most 4 epochs, and target KL=0.02.

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

## 2. Possible causes of faster learning

These are **implementation-consistent mechanisms and hypotheses**, not a causal decomposition established by one run. The V3 near-1000-iteration reference comes from the user's observations; matched V3 events, opponents, decks, assets, and budgets were not supplied here, so a rigorous speedup factor cannot be calculated.

### 2.1 Information becomes distinguishable rather than guessed

Exact card vocabulary removes old modulo-ID aliasing. Main/Extra partitions, categorical phase/zone/position, decision-versus-turn identity, operation families, and summon/selection/public relationships distinguish situations previously merged or underspecified. Equal network dimensions do not mean equal usable information.

When inputs cannot distinguish situations with different correct actions, additional training must compromise across them. Better inputs can make activation versus setting, attacking versus ending, or extra-deck endpoints easier to condition on. This does not supply optimal rules; it reduces representation ambiguity and the burden of inferring absent fields from outcomes. See [target data protocol](protocol_v4_implementation_en.md#3-target-data-protocol).

### 2.2 Semantic-asset repair may matter greatly, separately from architecture

The old 27647-slot vector library had only 530 distinct vectors, with 27007 identical entries, mostly encoding empty source. Rebuilding produced 26092 distinct vectors and retained registrations, callbacks, and long-source tails. See [semantic provenance repair](code_semantics_en.md).

**If that V3 run used the audited old assets**, many effects' code-semantic channel effectively supplied almost identical information. Rebuilt assets let the existing module exploit more realistic differences, potentially a larger benefit than merely adding branches. This remains a plausible cause, not proof that a general sentence encoder understands all Lua or that the planner deserves all gains. Keep old models with their original assets; swapping semantic libraries invalidates identity/input comparisons.

### 2.3 Deck style and FiLM condition decisions without rule rewards

Stable Main/Extra profiles produce learned Deck Encoder style latents. Layered FiLM separates deck-style modulation from phase/player/global context. Similar local actions can therefore be evaluated differently for different deck endpoints and situations instead of sharing one unconditional preference.

This is learned composition-to-policy conditioning, not manually fixing decks to attack/defence categories. It may also specialize to training preferences; unseen-family testing is needed. See [Deck Encoder](protocol_v4_implementation_en.md#41-reusable-deck-encoder) and [layered FiLM](protocol_v4_implementation_en.md#42-layered-film).

### 2.4 Auxiliary tasks provide nearer, more specific learning signals

Previously, terminal rewards mainly relied on value/advantage propagation to assign credit across actions. Now actual actions also supervise public events, immediate resource changes, chain/turn-end summaries, and genuine outcomes. External rewards remain predominantly outcome-based, but representation learning no longer depends only on sparse wins/losses.

Ordinary auxiliary heads neither inject predicted probabilities into decisions nor execute rules. Small loss weights and controlled shared gradients encourage consequential features. Improving future losses support learning, not a measured win-rate contribution. Auxiliary shared-representation learning has research support; this implementation differs from [UNREAL](https://arxiv.org/abs/1611.05397), whose acceleration figures cannot be transferred to this project.

### 2.5 Planning feeds future-oriented representations back into decisions

The planner sees current state/deck/context, not real future answers. Multi-horizon targets supervise its summary, then gradually opening bounded residuals influence policy/value. This creates explicit pressure to organize expected resources/outcomes beyond isolated candidate scores.

Consistency discourages abrupt representation resets within a phase without prescribing actions. It may make combo goal/process relations easier to share, but bell-shaped latents cannot establish specific learned sequences. See [planning specification](protocol_v4_implementation_en.md#44-auxiliary-heads-and-planning).

### 2.6 GAE changed too: the differences are not only cognition plus KL

The file confirms lambda=0.98, compared with the old default 0.95. With gamma=0.998, GAE weights more distant TD residuals by `(gamma×lambda)^distance`:

| Setting | gamma×lambda | Geometric weight sum 1/(1−gamma×lambda) | Residual weight 100 steps away |
|---|---:|---:|---:|
| lambda=0.95 | 0.94810 | 19.27 | 0.00485 |
| lambda=0.98 | 0.97804 | 45.54 | 0.10856 |

More distant residuals decay less rapidly, plausibly helping credit assignment for multi-stage plays. **These are GAE weights, not 19/46-step model memory limits or guarantees of fully propagating terminal rewards 100 steps.** Bootstrapping and repeated training still matter. Higher lambda also raises estimation variance; it is not universally better. Recompute comparisons if the old run used a nondefault setting. See the [GAE paper](https://arxiv.org/abs/1506.02438) and [3.13.0 change record](changelog_en.md).

### 2.7 KL guards and data feedback may preserve progress

KL adds no observation information or good-action labels. It limits repeated updates on stale-policy samples and can help retain improvements. This does not establish that overshooting caused all V3 inefficiency.

If genuine games become shorter, a fixed interaction budget can supply more completed outcomes/future labels, reinforcing useful learning. Falling length and high terminal coverage fit this hypothesis, but Worker averaging/truncation biases require per-game checks. CUDA semantic aggregation preserves learning inputs/targets and addresses temporary allocations; it is not added cognition.

