# TensorBoard 指标手册

> 适用于 Galatea-Core v3.14.1 / Model Protocol V4。
> 面向使用者和开发者：说明每项图表的功能、纵轴、记录位置和趋势含义。
> [English](tensorboard_metrics_en.md)

训练面板包含 **80 个固定标量、3 个规划分布和 1 个配置文本**，另有随卡组变化的 `Deck_WinRates/*`。这些图表分别回答不同问题，不能把“预测更准”“更新更多”和“对局更强”当成同一个指标。

| 分组 | 回答的问题 |
|---|---|
| League / Deck_WinRates | 训练内对哪些对手、用哪些卡组更容易赢？ |
| Rollout | 一局获得多少奖励、进行多少操作、如何结束？ |
| Auxiliary_Targets | 实际采集的数据能提供多少监督标签、发生过哪些事件？ |
| Auxiliary_Train | 给定当前状态和本次动作，能否预测其后果？ |
| Planning | 不预先指定动作时，能否形成对未来资源和结果有用的表示？ |
| Train / PPO_Update | 策略和值如何更新，实际用了多少样本，是否触及保护边界？ |
| Performance / Configuration | 时间花在哪里，实际按什么训练参数运行？ |

快速跳转：[记录规则](#1-记录位置横轴与读图规则) · [对局胜率](#2-league训练内对局胜率) · [采集结果](#3-rollout采集结果与终止情况) · [标签覆盖](#4-auxiliary_targets标签覆盖与事件分布) · [辅助预测](#5-auxiliary_train结构化预测学习) · [目标规划](#6-planning目标隐变量与策略接入) · [训练更新](#7-train当前小批的-ppo-联合更新) · [更新保护](#8-ppo_update整轮更新控制与实际复用) · [耗时与配置](#9-performance-与-configuration)

## 1. 记录位置、横轴与读图规则

### 文件与界面

- 文件位于 `runs/<模型前缀>_<启动日期-时间>/events.out.tfevents.*`。每次启动创建新目录；恢复训练继承 iteration 和 train_step，不保证横轴从零开始。前缀相同不代表模型 UUID 相同。
- WebUI：**训练流形图 → 启动 TensorBoard**。标量在 Scalars，规划分布在 Histograms / Distributions，启动参数在 Text 的 `Configuration/PPO`。
- 单独启动可用 `python -m tensorboard.main --logdir ./runs`。多次恢复训练的文件应先按 run 分开核对，再比较连续步数。
- 入口源码：[SummaryWriter 与配置文本](../trainer.py#L606)、[WebUI TensorBoard 模块](../app.py#L3951)。

### 频率与统计单位

| 分组 | 横轴 Step | 写入频率 | 每个点的统计范围 |
|---|---|---|---|
| `League_Overall/*`、`League_Self/*`、`League_Opponent/*` | 采集轮次 iteration | 每轮采集后 | 汇总成功读入的 Worker 对局计数，不是整个训练的累计胜率 |
| `Rollout/*` | iteration | 每轮采集后 | 对局计数，或 Worker 均值的再平均，详见第 3 节 |
| `Auxiliary_Targets/*` | iteration | 每轮合并后 | 本轮有效训练样本行上的标签审计，不是对局比例 |
| `Train/*`、`Auxiliary_Train/*`、`Planning/*` | 累计优化器更新数 train_step | 每 20 次更新位置 | 当前一个 mini-batch 的诊断，不是整轮平均 |
| `Planning_Histograms/*` | train_step | 每 200 次更新位置 | 当前批规划向量，或当前模型的门参数 |
| `PPO_Update/*` | iteration | 每轮 PPO 后 | 所有被检查小批的摘要，以及实际完成的优化器更新 |
| `Performance/*` | iteration | 每轮对应阶段后 | 完整采集/更新耗时及提交样本吞吐 |
| `Configuration/PPO` | 启动时 train_step | 每次启动一次 | JSON 文本：本次 PPO 配置与实际优化器学习率 |

`train_step` 仅在实际优化器更新成功后递增。诊断条件在更新前判断：第一个点可以是 Step 0；若 FP16 溢出导致更新跳过，同一步可能重复写点。梯度范数在反向、AMP unscale 后测量，**记录的是裁剪前范数**；损失/预测对应本次更新前的前向。

深色 Smoothed 是 TensorBoard 的显示平滑，浅色和 Value 是原始写入值。不要把平滑后的 `Epochs_Completed=1.44` 理解为完成了 1.44 个整 epoch；原始值是整数。也不要把纵轴自动缩放后的剧烈抖动当成大幅变化，例如 coverage_next 从 0.9997 到 0.9999 只差 0.02 个百分点。界面显示方式见 [TensorBoard 官方说明](https://www.tensorflow.org/tensorboard/get_started)。

### 术语与纵轴单位

- **轮次 iteration**：一次采集和随后的 PPO 更新。**小批 mini-batch**：一次优化器更新处理的样本行；一轮包含多个小批。**epoch**：遍历本轮数据一次。
- **样本行**：一个被保留的训练决策。**对局**：一次完整决斗。一局可产生很多行，双方决策和 Core 消息也不是同一计数单位。
- **比例/准确率**：0.8 表示相应分母的 80%。coverage 是“有标签”，Accuracy 是“预测正确”，WinRate 是“实际赢局”，三者不能互换。
- **Loss（损失）**：训练用的误差分数，不是奖励。BCE 衡量多个是/否预测；交叉熵衡量类别预测；Smooth L1 衡量数值预测与目标的差距。比较相同任务、归一化方式和数据分布时，通常越低表示拟合越好。它没有通用的“合格线”，也不等于平均错了多少 LP、多少张卡或多少回合。
- **RMS / Std / 梯度范数**：分别表示数值整体幅度、不同样本间的变化程度、参数受到的更新信号强度；都不是百分比。梯度裁剪限制过大的更新信号，图表仍记录裁剪前的值。
- **秒 / 行每秒 / nats**：分别用于耗时、训练样本吞吐和策略熵。除明确标注的计数与时间外，网络幅度和归一化误差不使用游戏原单位。

例如 `coverage_terminal=0.8` 是 80% 的训练行有终局标签，`Terminal_Accuracy=0.8` 是有效行中 80% 的终局分类预测正确，`RuleBot=0.8` 才是该轮对 RuleBot 的胜局占比。读取趋势前先确认是哪一种。

## 2. League：训练内对局胜率

记录入口：[trainer.py 的采集统计](../trainer.py#L1207)；计数来源：[worker.py 对局结算](../worker.py#L1260)。对应方案：[联盟训练机制](architecture.md#联盟训练机制)、[竞技场固定基准与推理模式](features.md#启动与监控中枢)。

以下纵轴均为 **胜局数 / 计入该组的对局数**，范围 0～1。训练方只统计本局 train_p_id 的结果；First/Second 是训练方的先/后手座位，不是当前决策者身份。平局与非终局截断不计为胜，计入对局时会压低该组胜率；异常局回滚后不提交。

| 图表 | 分组与纵轴含义 | 走向与限制 |
|---|---|---|
| `League_Overall/WinRate_Total` | Self/Hist/Rule 混合总胜率 | 上升是混合训练环境改善；对手比例变化也会影响结果 |
| `League_Overall/WinRate_First` | 混合对手中的先手胜率 | 与后手比较时需同时核对卡组、对手和样本量 |
| `League_Overall/WinRate_Second` | 混合对手中的后手胜率 | 高于先手不自动说明身份错标，也可能是环境或卡组偏向 |
| `League_Self/WinRate_Total` | 对本轮当前权重副本的胜率 | 在双方/座位/卡组对称且无截断时应接近 0.5；不是必须持续上涨的棋力曲线 |
| `League_Self/WinRate_First` | Self 对局中的先手胜率 | 用于观察座位差异；有限采样允许偏离 0.5 |
| `League_Self/WinRate_Second` | Self 对局中的后手胜率 | 同上；截断、双方推理路径和卡组不对称也能造成差异 |
| `League_Opponent/Historical_AI` | 对本模型历史池的胜率 | 池会随检查点更新，基准不是固定不动的对手 |
| `League_Opponent/RuleBot` | 对规则机器人胜率 | 衡量对此规则策略的适应，不等于对真人/其他模型的胜率 |

**分母为零时当前代码也写 0。** 因此首次历史模型出现前的零线不是连续全败；中间某轮的零值也可能没有有效提交的该类对局。文件没有同步记录这些胜率的 games 分母，不能仅凭曲线区分这两种情况，也无法计算可靠的整段对局加权胜率或置信区间。

`Deck_WinRates/{卡组}_Total/First/Second` 也是本轮统计，某张卡组无对应对局时不写该分项。数千张 cards 指的是数千条 TensorBoard 指标序列，不等于数千局独立测试。

## 3. Rollout：采集结果与终止情况

记录：[trainer.py](../trainer.py#L1239)。生成：[Worker 奖励与 GAE](../worker.py#L1284)。对应方案：[训练流程](architecture.md#训练流程)、[PPO 更新控制](quickstart.md#ppo-更新控制3130)。

| 图表 | 纵轴与记录方式 | 走向与限制 |
|---|---|---|
| `Rollout/Average_Reward` | 各 Worker 的单局总奖励均值，再对非零 Worker 均值做等权平均 | 同分布上升通常有利；不是胜率，也不是所有对局的加权平均；零均值被排除会造成偏差 |
| `Rollout/Average_Length` | 各 Worker 单局 ep_steps 均值再等权平均，排除零均值 | 降低可能是更快取胜/更少空操作，也可能改变了对手结构或截断比例 |
| `Rollout/Timeout_Rate` | Worker 的 timeouts 计数 / 总计入对局数 | 当前存在重复计数，不能直接解释成真实超时局百分比 |
| `Rollout/Deadlock_Rate` | Worker 的 deadlocks 计数 / 总计入对局数 | 当前异常局通常在此计数前回滚；零线不能证明没有异常或死锁 |
| `Rollout/Draw_Rate` | Worker 的 draws 计数 / 总计入对局数 | 观察真实平局类结算；不包含所有非终局截断或异常回滚 |

单局奖励是训练方轨迹的 **step_reward 总和 + final_reward**，不是折扣回报。当前也并非严格“只有胜负奖励”：

- 正常胜利在不超过 50 回合时为 +1，超过 50 回合胜利为 +0.05；失败为 −1，平局/非终局截断的终局部分为 0。
- 超过 40 回合的模型决策步 −0.0001；单回合超过 300 个模型决策步额外 −0.0005。
- 相同场面、相同候选的无进展重复达到阈值会扣 −0.005，Cancel/Unselect 阈值为 5，其他为 10；Retry 可扣 −0.001。这不是按“进行了多次材料选择”统一惩罚。
- 辅助/规划 Loss **不是奖励**，不会把预测准确率转换成额外胜局或额外 Reward。

ep_steps 不等于 Core 报文数、回合数或提交训练行数：Self/Hist 时会累积双方模型决策，RuleBot 侧规则操作不按同样方式累积；Worker 时间预算/引擎等待截断还可能把它直接设为 1500。因此平均长度既含真实交互差异，也含终止口径影响。

### 已确认的统计限制

1. `worker.py` 在 ep_steps 达到 1500 时加一次 timeout；无真实胜者且达到该上限的分支又加一次。同一非终局截断可能计两次。不能盲目把曲线除以二：正常终局与混合分支仍需辨别。这是计数问题，不是重复扣分。
2. Trainer 对 Worker 均值不按其对局数加权，且排除恰为零的奖励均值。少局 Worker 与多局 Worker 权重相同。
3. 异常局不提交是数据保护，不是异常监控。Deadlock_Rate 没覆盖所有回滚路径，需同时看 Worker 错误日志。

这些是当前记录口径的限制，不意味着所有对应对局都异常。修正前应结合 Worker 日志和逐局结算读图，不能仅凭这些统计证明容量或棋力达标。

## 4. Auxiliary_Targets：标签覆盖与事件分布

记录：[trainer.py 标签审计](../trainer.py#L1243)；计算：[summarize_auxiliary_targets](../auxiliary_targets.py#L357)；原事件：[event_history.py](../event_history.py#L326)。相关功能：[事件与公开提示](public_observations.md)、[网络与辅助目标](architecture.md#phase-4-卡组协议边界)。

前四项分母是本轮合并的训练样本行数；后五项分母是其中 **NEXT_DECISION 标签有效的样本行数**。纵轴都是 0～1 的比例，不是预测准确率，不直接参与奖励。

| 图表 | 纵轴含义 | 走向与限制 |
|---|---|---|
| `Auxiliary_Targets/coverage_next` | 能找到本次动作到下一决策/Retry/真实终局边界的行比例 | 接近 1 有利；少量未闭合尾部允许缺失 |
| `Auxiliary_Targets/coverage_chain` | 有可靠连锁末未来摘要的行比例 | 只有符合连锁上下文的样本有此目标；约 0.4 不代表丢了 60% 解析 |
| `Auxiliary_Targets/coverage_turn` | 有可靠回合末未来摘要的行比例 | 上升表示标签更完整；截断尾部可使其略低于 1 |
| `Auxiliary_Targets/coverage_terminal` | 有真实 Core 终局未来/胜负标签的行比例 | 不能解释为正常结束对局比例；一局会贡献很多行 |
| `Auxiliary_Targets/retry_rate` | 有 RETRY 结果位的即时有效行比例 | 持续升高需排查响应/解析；接近零是有利证据，不等于无所有错误 |
| `Auxiliary_Targets/cancel_rate` | 有 CANCELLED 位的即时有效行比例 | 合法取消也计入；不包含所有 Decline/完成选择；上升不自动代表无意义循环 |
| `Auxiliary_Targets/state_changed_rate` | 有 STATE_CHANGED 位的即时有效行比例 | LP、区域数量、连锁深度变化或出现转移消息可置位；不是“战术进展成功率” |
| `Auxiliary_Targets/negated_rate` | 转移中有 CHAIN_NEGATED 位的行比例 | 可能涉及双方及对应连锁，不是 AI 成功交康率；极低值通常是稀疏标签 |
| `Auxiliary_Targets/disabled_rate` | 转移中有 CHAIN_DISABLED 位的行比例 | 与发动无效不是同一标签；同样不能直接判断对局能力 |

事件位可以同时为真，不是一组总和必须为 1 的类别。coverage_terminal=0.9663 表示约 96.63% **行**可用于真实终局监督，不表示 AI 胜率 96.63%，也不表示只截断了 3.37% **局**。

## 5. Auxiliary_Train：结构化预测学习

记录：[trainer.py 联合 Loss 与诊断](../trainer.py#L1712)；计算：[auxiliary_heads.py](../auxiliary_heads.py#L233)。相关功能：[辅助头与梯度隔离](architecture.md#phase-4-卡组协议边界)。

这里衡量的是**后果预测**：选定一个动作后，会不会取消、出现何类消息、改变多少资源，以及最后是否获胜、多久结束。这不是评判该动作“符不符合某个人工最优规则”，也不会把预测准确率加进对局奖励。

每点是当前训练小批，不是独立验证集。辅助头读取当前状态表示和**已执行候选**表示，预测它的后验；标签只在采集结束后构建，不作为推理输入。无有效标签时对应掩码均值为零，不能把零损失当作完美预测。

| 图表 | 纵轴与公式口径 | 走向与限制 |
|---|---|---|
| `Auxiliary_Train/Total_Loss` | 辅助头整体后果预测误差；七个分任务的加权平均，权重依次为 1/1/0.5/1/1/0.5/0.25，总权重 5.25 | 不乘外层 0.1；同版本、相似标签分布下下降才可比 |
| `Auxiliary_Train/Backbone_Scale` | 辅助任务对共享表示的反向影响强度；辅助表示梯度桥乘数：前 100 次为 0，随后 900 次线性升至 0.2 | 是既定日程，不是模型学得好坏；不保证辅助梯度只占总梯度 20% |
| `Auxiliary_Train/Gradient_Norm` | 预测解码器受到的更新信号强度；纯预测/目标解码参数组裁剪前 L2 梯度范数，裁剪上限 1.0 | 包含规划目标解码器；不是共享主干受到辅助影响的比例 |
| `Auxiliary_Train/Result_Loss` | 预测动作是否完成、取消、重试或引发无效等即时结果的误差；11 个即时结果位的 masked BCE-with-logits | 降低代表这些位更容易预测；稀疏负类很多，不能只看总值 |
| `Auxiliary_Train/Message_Loss` | 预测本动作之后会出现哪些公开消息类别的误差；11 个公开 Core 消息组出现位的 masked BCE | 不是预测逐条完整报文、卡名或所有效果结果 |
| `Auxiliary_Train/Boundary_Loss` | 预测动作后继续决策、被要求重试或结束决斗的误差；下一决策 / Retry / 终局三分类 masked 交叉熵 | 普通下一决策占多数，低值可能部分来自多数类 |
| `Auxiliary_Train/Immediate_Delta_Loss` | 预测本动作到下一边界期间 LP、卡片数量与连锁深度变化的误差；即时 20 维摘要的 masked Smooth L1：双方 LP、16 区域变化、结束/最大连锁深度 | 许多选择不改变 LP/资源，零目标很多；不是原单位的误差 |
| `Auxiliary_Train/Future_Delta_Loss` | 预测后续连锁末、回合末、终局资源变化的误差；连锁末、回合末、终局的 18 维 LP/区域摘要 masked Smooth L1 | 可观察未来资源理解；覆盖率、场面尺度变化会影响损失 |
| `Auxiliary_Train/Terminal_Outcome_Loss` | 预测从这次动作之后最终会败、平还是胜的误差；真实终局败/平/胜三分类 masked 交叉熵 | 预测未来真实结果，不是当前 value 的回归误差 |
| `Auxiliary_Train/Terminal_Remaining_Loss` | 预测距真实终局还有多少转移事件的误差；对 log1p(剩余转移事件数)/log1p(1500) 的 masked Smooth L1 | 不是“还差多少回合”的绝对误差；1500 是归一化尺度，不是标签截断上限 |
| `Auxiliary_Train/Result_Bit_Accuracy` | 即时有效行×11 结果位上的逐位正确率，logit≥0 判真 | 99% 不等于 99% 的完整事件正确，也不等于稀有无效事件识别率 |
| `Auxiliary_Train/Message_Bit_Accuracy` | 即时有效行×11 消息组上的逐位正确率 | 正负类不平衡时高值容易；需后续逐类验证 |
| `Auxiliary_Train/Boundary_Accuracy` | 即时有效行上三分类 argmax 正确率 | 高值应与“永远猜下一决策”的基线比较 |
| `Auxiliary_Train/Terminal_Accuracy` | 有真实终局标签的行上败/平/胜分类正确率 | 不是模型胜率；长局、多次 epoch、临近终局行并非独立测试样本 |
| `Auxiliary_Train/Future_Valid_Fraction` | 当前批三种未来 horizon 的有效位均值 | 与三个未来 coverage 的均值相关；本批抽样与全轮统计不必相同 |

11 结果位：Resolved、Retry、Cancelled、Declined、SelectionFinished、ChainNegated、ChainDisabled、Terminal、StateChanged、MessageOverflow、MissedTiming。11 消息组：回合/阶段、移动/表示、场面控制、召唤、连锁、目标/装备、抽卡、LP、指示物、战斗、公开通知。详情见 [事件定义](../data_types.py#L107)、[消息组与区域顺序](../event_history.py#L24)。

连续目标统一采用行动者相对视角：LP 差÷8000，区域数量差÷5，分别裁至 [−4,4]；连锁深度÷16，裁至 [0,4]。区域包括双方卡组、手牌、怪兽、魔陷、墓地、除外、额外、叠放。Remaining 是双方决策间转移事件数，包含当前事件，不是训练方独自的动作数。

## 6. Planning：目标隐变量与策略接入

记录：[trainer.py 规划诊断](../trainer.py#L1890)；计算：[goal_planner.py](../goal_planner.py#L198)；接入：[galatea_net.py](../galatea_net.py#L1127)。相关功能：[规划路径与训练边界](architecture.md#phase-4-卡组协议边界)。

这里衡量的是**当前局面的未来摘要**。与辅助头先给定已执行动作不同，规划器不预先读取这次实际选了哪个动作；它从场面和卡组条件提取对未来资源、终局和剩余长度有预测力的向量。相邻决策的一致性反映这个摘要是否延续，跨样本差异反映不同场面是否被区分。

规划器从当前状态、稳定卡组风格、全局时局重新生成 128 维 plan_latent。其目标解码器只在训练/可选诊断执行；规划表示通过两条独立有界残差进入策略与价值，因此规划和普通辅助探针不完全相同。**它不是 MCTS，也不是已经拥有命名 combo/终端的脚本。**

| 图表 | 纵轴与记录方式 | 走向与限制 |
|---|---|---|
| `Planning/Total_Loss` | 规划表示的未来预测与连续性综合误差；Future/TerminalOutcome/Remaining/Consistency 按 1/0.5/0.25/0.1 加权，再÷1.85 | 不乘外层 0.05；分项和数据分布改变时不要求单调下降 |
| `Planning/Gradient_Norm` | 规划路径受到的联合更新信号强度；规划器非解码部分、两条投影与门的裁剪前 L2 范数，裁剪上限 1.0 | 包含 PPO 与规划监督的联合作用；不能和主干范数直接比较“重要性” |
| `Planning/Future_Loss` | 从当前局面预测连锁末、回合末和终局资源变化的误差；从 plan_latent 预测三种未来 18 维 LP/区域摘要的 masked Smooth L1 | 与辅助未来头同标签，但此头不输入本次已执行候选 |
| `Planning/Terminal_Outcome_Loss` | 从当前局面预测最终败、平、胜的误差；从规划表示预测真实终局三分类交叉熵 | 下降可说明状态摘要更有预测力；不是独立验证或棋力 |
| `Planning/Terminal_Remaining_Loss` | 从当前局面预测距终局还有多少转移事件的误差；同上节 log 归一化剩余事件目标的 Smooth L1 | 小误差不等于准确预测绝对回合数 |
| `Planning/Consistency_Loss` | 相邻同阶段决策的计划方向差异；合规相邻配对的 mean(1−cosine(plan_a,plan_b)) | 越低方向越连续，但强求零可能抹平必要的计划变化 |
| `Planning/Latent_RMS` | 计划向量数值的整体幅度；当前批 128 维向量整体 sqrt(mean(z²)) | 输出有 LayerNorm，约 1 正常；不能据此证明不同状态共享同一计划 |
| `Planning/Latent_Mean_Dimension_Std` | 计划在不同样本之间的平均变化程度；每个维度在当前批不同样本间的 std，再对 128 维平均 | 接近零需警惕跨状态坍塌；增加也不保证获得可解释语义 |
| `Planning/Latent_Active_Dimension_Fraction` | 在本小批中确实随样本变化的计划维度比例；批间 std>0.01 的维度数÷128 | 1 表示每维在变化，不证明矩阵满秩或每维有独立用途 |
| `Planning/Pair_Cosine` | 相邻合规决策的计划方向相似程度；合规相邻配对的平均余弦相似度 | 有配对时约等于 1−Consistency_Loss；高值必须结合跨样本 std 看 |
| `Planning/Pair_Valid_Fraction` | 本小批实际用于连续性监督的配对起点覆盖率；当前批合规相邻配对起点数÷(批大小−1) | 不是参与配对的行比例；512 批量的最大值约为 256/511≈0.501 |
| `Planning/Terminal_Accuracy` | 有真实终局标签的行上规划三分类正确率 | 不是当局最终胜率，不是独立验证集正确率 |
| `Planning/Future_Valid_Fraction` | 当前批三类未来目标有效位均值 | 与辅助未来有效率相同口径，不是预测质量 |
| `Planning/Policy_Gate_RMS` | 规划残差通向动作概率路径的门幅度；sqrt(mean(tanh(策略门参数)²))，范围 0～1 | 从零缓慢打开说明规划路径开始参与；0.045 不等于贡献 4.5% 棋力 |
| `Planning/Value_Gate_RMS` | 规划残差通向回报估计路径的门幅度；sqrt(mean(tanh(价值门参数)²))，范围 0～1 | 与策略门独立学习；快速接近 1 需复核，而非越大越好 |

最终残差为 `tanh(gate) × 0.5 × tanh(projection(plan))`，门 RMS 没有包含后面的投影和 0.5。因此不能把门幅度当作 logits、value 或梯度占比。共享输入的梯度桥沿用 Backbone_Scale，但门/规划器与预测解码器有独立参数组。

一致性只配同一行动者、同一回合、同一阶段的相邻训练决策，并保留不重叠双样本单元进行 shuffle。Pair_Valid_Fraction≈0.44，在 512 批量下约对应 88% 的行参加真实配对，**不是只覆盖了 44% 的行**；它不要求不同阶段、不同玩家或不同对局的计划一致。

### Planning_Histograms：不是普通标量纵轴

| 分布 | 被记录的值 | 怎么读 |
|---|---|---|
| `Planning_Histograms/Plan_Latent` | 当前批所有 plan_latent 元素 | 观察宽度、长尾和是否退成常数；钟形分布不代表已学出具体战术 |
| `Planning_Histograms/Policy_Gate` | 当前模型逐通道 tanh(策略门) | 从零附近逐步展开是正常开放；尖峰仍可能表示部分通道不开门 |
| `Planning_Histograms/Value_Gate` | 当前模型逐通道 tanh(价值门) | 与策略门对照；不需要两者分布一致 |

Histograms 的值轴是张量数值，桶高是该区间的元素计数，叠层/时间轴是 train_step；Distributions 会使用分位数带显示。时间/叠层轴上的 Step 是更新步数，**不是 latent 数值或样本置信度**。这些分布不是预测胜率曲线。

## 7. Train：当前小批的 PPO 联合更新

记录与计算：[trainer.py](../trainer.py#L1712)；偏移计算：[ppo_control.py](../ppo_control.py#L20)。对应方案：[PPO 框架](architecture.md#ppo-训练框架)、[KL 提前停止](architecture.md#kl-提前停止如何解读)。

**Policy** 是“各合法动作的选择概率”，**Value** 是“当前状态的预计训练回报”，并不是胜率。GAE advantage（优势）估计一个已采样动作相对价值基线好或坏多少；策略更新提高正优势动作的概率、降低负优势动作的概率。价值头拟合的是 GAE 构建的回报目标，目标包含后续奖励和价值估计，不只是真实终局的 ±1。参数解释见 [PPO 更新控制](quickstart.md#ppo-更新控制3130)。

令 `d=log π_new(a|s)−log π_old(a|s)`、`r=exp(d)`、A 为本轮归一化的 GAE advantage。以下计算都针对采样到的动作和当前小批，不是完整状态空间。

| 图表 | 纵轴与公式 | 走向与限制 |
|---|---|---|
| `Train/Total_Loss` | 本次联合更新所优化的综合目标；Policy + 0.5×Value − entropy_coef×Entropy + 0.1×AuxiliaryTotal + 0.05×PlanningTotal | 可为负；不是所有分项的等权和，也不要求像监督学习一样持续下降 |
| `Train/Policy_Loss` | 根据优势调整动作概率的优化目标；−mean(min(rA,clip(r,1−ε,1+ε)A)) | 正负/波动都可能正常；优势每轮改变，不能单独测棋力 |
| `Train/Value_Loss` | 预计回报与本批 GAE 回报目标的数值差距；value 与 GAE lambda-return 的 Smooth L1 | 不是与真实胜率或完整 Monte Carlo 回报的误差；回报变难时可上升 |
| `Train/Entropy` | 当前选择有多分散：越高越随机，越低越集中；有合法候选掩码的动作分布熵均值，单位 nats，自然对数 | 最高约 log(候选数)，单候选为零；候选结构不同也会改变平均熵 |
| `Train/Approx_KL` | 当前策略相对采集时策略改变了多少；mean(exp(d)−1−d) 的 sampled-action 近似量 | 新旧策略偏移，越大不代表越聪明；不是相邻轮固定测试集的精确全分布 KL |
| `Train/Clip_Fraction` | 采样动作概率比值越出 PPO 允许区间的比例；mean(abs(r−1)>ε) | 默认 ε=0.2；统计比率越界，而非梯度被实际截掉的精确比例 |
| `Train/Explained_Variance` | 价值预测解释本批回报目标变化的程度；1−Var(return−value)/Var(return)；目标方差极小时写零 | 可小于零、上限为 1；接近 1 只说明本批目标残差方差小，不保证预测无偏或泛化 |
| `Train/Gradient_Norm` | 共享主干受到的联合更新信号强度；PPO 主干参数组的裁剪前 L2 范数，裁剪上限 0.5 | 共享参数的辅助/规划梯度也在其中；不含独立规划与解码参数组；超过 0.5 不代表裁剪失效 |

熵不是“对正确答案的置信度”：三个等概率候选的熵为 ln(3)≈1.10，只剩一个合法候选时为 0，即使模型没有学会策略也会如此。只有在候选数量和类型相近时，逐渐下降才更能说明选择倾向变得明确。

KL 衡量策略改变的幅度，不衡量改变后的优劣；低 KL 可能表示稳定，也可能表示几乎没更新。Clip_Fraction=0.2 表示约 20% 的采样动作比值越界，不表示删掉了 20% 的训练数据。

高 Explained_Variance 也不能直接说明“AI 预测终局准确 95%”：GAE return 自带价值 bootstrap，且这里在训练样本上测量。终局分类准确率、回报回归解释率、对局胜率是三个不同概念。

## 8. PPO_Update：整轮更新控制与实际复用

统计定义：[PPOUpdateAudit](../ppo_control.py#L47)；触发顺序：[trainer.py](../trainer.py#L1763)。相关功能：[PPO 训练框架](architecture.md#ppo-训练框架)、[KL 功能解释](architecture.md#kl-提前停止如何解读)。

| 图表 | 纵轴与记录方式 | 走向与限制 |
|---|---|---|
| `PPO_Update/Requested_Epochs` | 本轮配置的最大遍历数，默认 4 | 是上限，不是实际学习四次的保证 |
| `PPO_Update/Epochs_Started` | 已进入的 epoch 数，包括中途停止的那次 | Started>Completed 通常只是本轮中途 KL 停止 |
| `PPO_Update/Epochs_Completed` | 完整走完小批循环的 epoch 数 | 整数；跳过非有限批也可能走完循环，需结合更新数 |
| `PPO_Update/Optimizer_Updates` | 实际完成的优化器小批更新次数 | 零或长期骤减需排查；不是本轮前向次数 |
| `PPO_Update/Effective_Epochs` | 实际更新样本次数总和 / 本轮有效样本行数 | 可为小数；衡量真实复用，不是 KL 平滑指标 |
| `PPO_Update/Samples_Updated` | 完成优化器更新的样本次数总和，重复 epoch 重复计数 | 不等于独立环境交互数量 |
| `PPO_Update/Skipped_Tail_Samples_Per_Epoch` | 本轮样本数 % mini_batch_size | 当前不更新随机尾部；512 批量最多 511 行，不等于本轮全部丢弃量 |
| `PPO_Update/KL_Early_Stop` | 本轮是否触发 KL 停止，0/1 | 长期 1 说明更新受边界限制，不是每轮训练失败或已经收敛 |
| `PPO_Update/Target_KL` | 配置目标值，默认 0.02 | target=0 可关闭有限值阈值，非有限保护仍保留；频繁停止本身不代表阈值错误 |
| `PPO_Update/KL_Stop_Threshold` | 1.5×Target_KL，默认 0.03 | 是触发条件，不是所有记录值的严格上界 |
| `PPO_Update/Approx_KL_Mean` | 所有登记小批 KL 均值的再均值 | 不是更新后重新评估整批的最终 KL；包含过程中的检查 |
| `PPO_Update/Approx_KL_P50` | 上述小批 KL 均值序列的中位数 | 不是 50% 动作各自的 KL |
| `PPO_Update/Approx_KL_P95` | 上述序列的第 95 百分位 | 不是逐动作尾部；较高表示部分小批偏移更强 |
| `PPO_Update/Approx_KL_Max` | 上述序列最大值，包括触发停止的小批 | 超过 0.03 正是可能触发停止的记录；不是保护自动失效 |
| `PPO_Update/Clip_Fraction_P50` | 各小批 clip fraction 的中位数 | 与 Train 的单点不必相同 |
| `PPO_Update/Clip_Fraction_P95` | 各小批 clip fraction 的第 95 百分位 | 不代表 95% 样本被截断 |
| `PPO_Update/Nonfinite_Batches` | 审计发现非有限 KL/clip/loss、或 FP16 scaler 跳步的计数 | 长期非零需查明；零只说明这些检查未命中，不是覆盖所有异常/内存问题 |

### 为什么 KL_Max 超过阈值，而训练仍然正常？

每个小批先前向，登记 KL/clip，再判断停止。若 KL 超过 0.03，**该小批不反向更新**，并结束本轮剩余策略/价值/辅助/规划联合更新；此前已经完成的更新保留，下一轮正常采集。因此停止小批会进入 PPO_Update 的 Max/分位数，却不会出现在普通 Train 的反向诊断点里。

这并不把策略投影回严格 KL 边界：已执行更新在后续某批上可能测出 0.05，停止只能避免继续扩大偏移，不能保证模型所有状态的 KL≤0.03。PPO clipping 本身也不是严格的全分布约束，KL early stop 是额外保护；相关原理见 [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

例如某轮 Effective_Epochs≈1.05、Completed=1、Started=2，表示完整走了一次并在第二次早期停止，**不是没有训练**。若长期第一批就停止、更新数逼近零，才需优先排查 old log-prob、采样/更新推理模式、学习率与数据分布；不能直接认定阈值太小。

## 9. Performance 与 Configuration

记录：[collect_rollouts](../trainer.py#L921)、[run_training_loop](../trainer.py#L1954)、[配置文本](../trainer.py#L606)。对应方案：[CPU/CUDA 中央推理](architecture.md#中央批量推理服务)、[CUDA 语义聚合边界](architecture.md#cuda-语义直接聚合2026-10-10)。

| 图表/文本 | 纵轴或内容 | 走向与限制 |
|---|---|---|
| `Performance/Collection_Seconds` | 本轮完整 collect_rollouts 秒数，含启动/资源载入、等待、回收、合并及统计 | 接近 timeout 时优先理解成采集时间预算，不是单局变慢或 GPU 更新耗时 |
| `Performance/PPO_Update_Seconds` | 本轮完整 update_policy 秒数，含 CPU/GPU 搬运、重建、前后向和审计 | KL 缩短更新数也会降低耗时；不能单凭下降声称算子加速 |
| `Performance/Rollout_Steps_Per_Second` | 实际合并的训练样本行数÷Collection_Seconds | 不是所有 Core 动作/秒或每 Worker 吞吐；乘回采集秒数可近似还原实际提交行数 |
| `Configuration/PPO` | 启动时记录 gamma、lambda、请求/实际学习率、entropy、clip、epochs、target KL、multiplier、mini_batch | Text，不是标量图；恢复时看实际优化器学习率，不能只看命令行请求值 |

配置文本目前**不记录** Worker 数、采集 timeout、网络层数、Core/资产指纹或逐组对局数。事件文件也不能证明 CUDA 峰值/RSS 无增长，需另看启动与内存日志。
