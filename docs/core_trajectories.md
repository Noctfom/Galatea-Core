# 规范 Core 轨迹、AI 预测回放与外部数据入口

3.13.1 完成阶段 7 批次 2，3.13.2 补齐固定 Core 解析稳定性修正。Model Protocol **4**、
Checkpoint Format **3**、结构修订 **13** 以及全部网络参数/输入不变。
`TRAJECTORY_SCHEMA_VERSION=2` 独立维护于 `core_trajectory.py`，新增固定报文身份及空连锁
自动应答；旧格式 1 的原始轨迹不再接受为当前协议轨迹。全息 JSON 仍为格式 3，旧格式 1/2 仍可查看。

## 开启方式

3.13.3新增公开观测并升至结构修订14；轨迹格式2/全息格式3不变，但旧结构原始轨迹不能混用新观测校验。新增显示见 [公开观测](public_observations.md)。本页开头的“网络/输入不变”仅指3.13.2批次。

默认关闭，只为 `--thought_freq` 选中的录像局记录，不改变训练 Worker 或 PPO 存储。
两个开关可以分别使用：

```powershell
.\python_env\python.exe main.py duel --p0 .\models\galatea_iter_100.pth --thought_freq 5 --record-trajectory --record-evaluations
```

WebUI：竞技场设置的“记录规范 Core 轨迹”和“记录 AI 预测曲线”。频率为 0 时拒绝启用。
全息回放展开“AI 预测曲线”，查看双方独立的终局胜利概率、剩余事件曲线、
当前决策的平局概率以及可用的真实终局对照。规则方没有模型预测，旧录像不会反推预测。

原始轨迹位于 `replays/core_trajectories/*.core.jsonl.gz`；全息 JSON 仍位于 `ai_thoughts/`。
两类文件都可在“存储与日志仓库 → AI 读心记录”管理删除。轨迹含本地全知原始数据和双方
构筑，包括为未来上层预留的 Side；分享前注意信息边界。**Side 不进入单局 Encoder。**

## 记录内容与唯一解释器

`CoreTrajectoryInterpreter` 直接调用现有 `MessageParser`、`DuelState` 和 `AiBot._pack_response`。
记录与验证不维护第二套消息解析规则。每局依次保存：

- 头部：来源、物理玩家座位、交换座位标记、原始构筑、Core seed/启动参数、**实际注入顺序**、
  Core/CDB/实际内存卡片缓存/实际加载 Lua 的 SHA-256，以及 V4 词表、语义、观测协议身份。
  模型身份含 UUID/内置轮次/网络配置；开启记录时检查点文件摘要仅在 Arena 初始化时读取一次。
  `core_message_protocol` 固定为随包 Core 的报文布局身份，不再保存或切换幽灵字节开关。
- 原始 Core 数据块、实际消费消息、状态查询时点及查询前的原始候选池。
  宏动作保留真实生成结果和原始坐标，不要求重新抽样选池，不保存完整观测张量。
- 实际响应（保留整数正负语义或字节顺序）、物理行动玩家、对应提示、全部候选匹配项、
  选中编号、实际 CPU Encoder 输出摘要；可选状态预测。独立策略/宏池随机种子另存运行元数据。
- 空 Type 16 的 `automatic_response`：只保存真实 `-1` 应答和提示/玩家，不生成观测摘要、
  模型动作、转移事件或行为克隆标签；重放单独返回 `automatic_response_count`。
- 尾部：真实终局或截断原因、流摘要、容量/解析/Retry/映射质量标记。

普通竞技还会在 Python 层打乱卡组，所以只保留 seed 不够；复现必须使用已记录的实际注入
顺序，但 `DuelState` 的稳定构筑仍使用原始列表。保存查询时点是为了复现事件完成边界，
避免重放的历史摘要与原决策时不一致。

逐行 gzip 写入，不把原始历史累积在内存。单条上限 1 MiB、每局上限 64 MiB **未压缩数据**、
最多 100,000 条；达到上限仅停用记录并标记截断，对局继续。使用 UUID 文件名避免同秒覆盖。
读取同样限制行长、记录数和累计解压尺寸；只还原白名单 JSON 字段，不使用 pickle、不下载
资产、不解压路径、不接受私有 Core 改动。摘要用于检测损坏，**不是来源签名或可信认证**。

## 只读检查与确定性重放

```powershell
# 只检查格式、字段边界和流摘要，不启动 Core
.\python_env\python.exe main.py trajectory-check .\replays\core_trajectories\duel_UUID.core.jsonl.gz
# 使用当前公开 Core/本地资产额外验证（CPU，无需载入策略网络）
.\python_env\python.exe main.py trajectory-check .\replays\core_trajectories\duel_UUID.core.jsonl.gz --replay
```

重放先制作有界临时文件副本，防止校验后源路径被替换；再完整校验格式与运行资产身份，
之后按记录顺序注入卡组、消费消息、恢复候选及查询
时点、提交原始响应。逐块比较 Core 输出，逐决策比较响应映射和 Encoder 摘要。资产不一致
直接拒绝，不以最新脚本强行解释旧轨迹。结束时恢复同进程既有 Core 回调绑定；解析布局固定。

三个结果必须分开理解：

| 字段 | 含义 |
| --- | --- |
| `integrity_valid` | 格式与摘要通过；不代表来源可信、可用于训练 |
| `replay_verified` | 当前资产下，整局或完整前缀的消息/响应/编码观测验证；纯提示撤销重排例外见下一项 |
| `raw_output_exact` | 每个 Core 原始块逐字节一致；false 时绝不具备行为克隆资格 |
| `behavior_clone_eligible` | 额外满足真实终局、零质量标记、全响应唯一映射且全部观测校验等必要条件；仍不是自动开始模仿训练的授权 |

文件记录被容量上限截断则不能验证；对局自身达到步数上限，可验证已完整记录的前缀，但
不提供真实终局标签。规则宏响应可能无法唯一映射到原子候选，仍可重放，只是不成为监督
动作标签。出现 Retry 也不自动把被拒绝的响应当成正确动作。

公开 Core 的效果重置索引采用无序容器，多个客户端提示撤销可能存在遍历顺序差异。
仅当同一卡片、同位置的**连续 Type 160 / subtype 7 提示撤销**消息集合完全相同而排列
不同，离线校验允许按原始记录顺序继续比较全部后续响应/观测，返回
`raw_output_exact=false` 和 `client_hint_reordered_chunks`。这只是诊断性状态验证，
**行为克隆资格仍为 false**，原始文件不重写、真实 Core/训练/竞技场完全不归一化。
跨卡、添加提示、数值/动作改变、非连续事件或不完整解析均不放宽。此例外不能用来
宣称任意旧 Core 都支持严格逐字节重放；未来数据接入仍需要专项审计。

3.13.1 压测发现的 Type 16 空候选未消费问题已在 3.13.2 修正：实际消息头没有全局 forced，
每项分别携带 effect mode 和 forced；Type 31 的额外字节是 `skip_panel`。详见
[固定报文审计](core_message_audit.md)。每块仍保存 `parsed_bytes/message_count/parse_complete`；
其他未消费字节仍标记 `unparsed_core_bytes` 并拒绝行为克隆，不能把本批修复当成全部消息认知已覆盖。

## 状态预测的解释边界

- 仅可选调用既有 `goal_planner.target_decoder` 的状态终局/长度分支；不再跑动作条件辅助头
  或未来资源分支，不新增权重，不改 logits/value，不用预测调整动作，不增加搜索/MCTS。
- 保存 `loss/draw/win` 三分类概率，视角为实际物理 P0/P1，附模型 UUID/轮次/文件摘要、
  策略模式/温度及既有内部 `calibration` 来源元数据。界面统一称“AI 预测”，不把内部
  `uncalibrated` 状态作为警告标题；推断只使用动作提交前的本方编码观测。
- P0/P1 掌握的信息可能不同、模型也可能不同，曲线不要求互补。`P(win)` 是胜利概率，
  `P(win)+0.5×P(draw)` 是另一种期望比分；PPO value 是折扣回报，不是胜率。
- 记录中的策略模式/温度只是来源元数据，**不是解码器输入**。预测学习的是训练续局与
  对手混合分布；不能假定它已对部署温度、贪心策略或某个真人对手的实际胜率校准。
- 长度按监督时的 `log1p(events)/log1p(1500)` 逆变换，为估计值而非精确期望。
  **事件数包含双方动作转移，不是本方剩余动作数或游戏回合数**。展示极端输出以 100 万
  事件为上限并标记 clipped，不改变策略或训练。
- 仅真实 `MSG_WIN` 后添加独立 `evaluation_actual`：本视角胜负和从该决策开始（含当前
  动作）的实际剩余转移数量。预测内容不覆写，异常/截断局不伪造终局答案。
- 曲线反映模型对当前局面的理解，不代表实际胜率，也不应作为评判强弱的唯一依据。
  常规 ONNX 仍仅输出 logits/value；本批预测录像需要完整 PTH。

## 成本与后续批次

关闭开关不跑诊断解码器、不计算模型文件摘要、不写原始轨迹。开启后仅录像局承担小型
解码器、标量 GPU→CPU 同步、候选/观测摘要及压缩写盘的开销；首次 Arena 初始化额外读取
模型摘要，首次记录计算资产摘要。它不是零成本功能，但不增加 PPO/共享内存/训练池占用。
只读检查可能多次扫描压缩文件，以换取校验先于 Core 执行和有界内存。

下一批是 Link/YRP 原始数据适配、整局/卡组/玩家隔离划分及数据质量门禁。
Type 120/160/165 与 35/37/38/162 的认知补全须先确认设计，见报文审计文档。
当前不接管旧的搁置 Replay Worker、不实现行为克隆优化器，也不添加反事实搜索。

## 3.13.5：外部数据入口与隔离门禁

阶段7批次3建立 **Core侧** 数据入口；外部 Link 项目仍需调用采集 SDK，不宣称已完成联网端到端验收。
Model Protocol4 / Checkpoint Format3 / 结构修订14 / 静态语义格式1 / GKG4不变。网络、
编码输入、奖励、PPO、正常 `GalateaEnv.reset` 与部署输出均不变。已确认无活跃导入后删除
废弃的 `replay_worker.py` 与 `yrp_parser.py`；新入口只使用 `yrp_ingest.py` 和唯一 Core 解释器。

### CLI 与 WebUI

```powershell
# 只读格式检查，不调用 Core；.yrp3d 同样使用 --source yrp
.\python_env\python.exe main.py replay-ingest .\replays\example.yrp --source yrp --inspect
# 在当前资产下重建诊断轨迹，再独立重放；这不是开始监督训练
.\python_env\python.exe main.py replay-ingest .\replays\example.yrp --source yrp --report .\replay_data\ingest_reports\example.json
# Link SDK 产生的原始捕获
.\python_env\python.exe main.py replay-ingest .\replays\link_captures\example.link.jsonl.gz --source link --inspect
.\python_env\python.exe main.py replay-ingest .\replays\link_captures\example.link.jsonl.gz --source link
# 严格审查规范轨迹，按连通组划分；输出必须使用新文件名
.\python_env\python.exe main.py dataset-build --directory .\replays --output .\replay_data\datasets\review_001.json --seed 20261009
```

WebUI：**存储与日志仓库 → 轨迹与数据门禁**。页面按操作阶段分为四区，交互前显示
标题、输入/输出与用途；来源、文件、路径、种子和操作按钮的小问号提供解释。

1. **原始录像检查与诊断转换**：选 YRP/YRP3D 或 Link 原始捕获。只读检查不调用 Core、
   不写文件；后台转换生成 `replays/imported_trajectories/*.core.jsonl.gz` 与
   `replay_data/ingest_reports/*.json`，不修改来源，也不启动学习。两种来源独立保存选择状态。
2. **规范轨迹审查与隔离划分**：仅递归扫描所选目录内的 `.core.jsonl.gz`，可覆盖原生
   竞技场轨迹和导入诊断轨迹。生成 `replay_data/datasets/dataset_*.json`，不移动轨迹。
   种子仅控制关联组划分，不影响对局重建或模型；默认目标80%/10%/10%，组隔离不保证精确数量。
3. **查看质量审查结果**：显示训练/验证/测试候选及隔离数量和逐轨迹拒绝原因；数量是
   文件数，不是步数或已训练样本数。折叠说明解释常见拒绝代码和独立验证不足的警告。
   读取已有清单不会重新审查；新增数据或变更资产后需重新生成清单。
4. **分类文件管理**：四个独立折叠分类区分别管理 Link 原始捕获、导入诊断轨迹、
   入口诊断报告、数据质量清单。每区先显示类别、目录、后缀和文件数；删除按钮标明类别。
   批量删除/清空只作用于对应目录当前匹配后缀的文件，不递归、不同步删除其他产物；
   清单删除不会删除其引用的轨迹，原生轨迹仍在“读心记录”页管理。不可在界面撤销，请先下载备份。

转换与审查进度在“启动与监控中枢”及 `system_logs/ReplayIngest*`、`DatasetQuality*` 查看；
同一会话已有托管任务时不抢占进程登记。不上传或下载外部资产，不在每次页面刷新时
自动执行 Core 重放。上述 UI 调整仍属于3.13.5，不变更协议、检查点或训练路径。

### YRP 支持边界

标准32字节 YRP1、80字节扩展头版本1的 YRP2，以及长度严格分帧且恰有一个231 replay
包的 YRP3D 可读取。Tag、残局脚本、未知扩展头和含歧义多个 replay 包的容器拒绝猜测。
不再用全文 `rfind` 魔数，也不忽略尾部残渣。文件上限16MiB、解压输出8MiB、LZMA字典
64MiB、响应100,000项；声明与实际大小必须相符。昵称只作显示，不当作稳定玩家身份。

固定重建路径按公开客户端执行：YRP1 将头部 seed 经 `std::mt19937(seed)()` 转成真实
Core seed；YRP2 使用完整8项序列和已随包导出的 `create_duel_v2`；保存数组按原顺序
注入、初始表示8。不沿用旧模块的倒序猜测或直接头部 seed。依据为公开的
[ReplayHeader](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay.h)、
[StartDuel](https://github.com/Fluorohydride/ygopro/blob/master/gframe/replay_mode.cpp) 与
[YRP3D分帧](https://github.com/purerosefallen/ygopro-yrp3d-encode)。不修改本地 Core。
无UNIFORM标志的旧录像可只读检查，但不切换旧 Core 模式来重建；未来版本迁移单独审查。

仅按原始响应逐条推进：**取消不跳过、Retry不盲猜下一响应、答案不注入候选池**。
宏候选使用现有构建器、固定局部 RNG 和均匀先验；超过120的选择池按既有规则缩减，
专家答案可能不在池中，此时保留实际响应重放，但不产生可靠动作标签。模型不用于选池，
没有伪造 log_prob、value、return 或 advantage。预算默认300秒，1～3600秒可调。

成功重建只证明当前资产下响应流自洽，**不证明原录像使用相同历史资产或原始场面**。
普通 YRP 缺历史 Core/脚本/CDB摘要及全量原始 Core 报文，因此仍标记
`historical_assets_unknown/original_core_output_unavailable`，不自动具备行为克隆资格。
转换失败封存诊断前缀，CLI返回2；解析/资产拒绝返回1，报告不会覆盖已有同名文件。

### Link 采集合同

`LINK_CAPTURE_FORMAT_VERSION=1` 独立维护于 `trajectory_ingest.py`。
SDK只保存原始 Core 块及**实际已提交**响应，物理玩家始终用0/1，不读取或改变策略。

```python
from trajectory_ingest import LinkCaptureWriter

# 普通在线单方捕获，不补造对方卡组、服务器种子或未知资产。
capture = LinkCaptureWriter(visibility='player_view', perspective=0,
                           player_ids={'0': 'service:user_a', '1': None})
capture.record_chunk(raw_core_chunk)       # 已移除网络帧头的原始 Core 字节
capture.record_response(actual_response, actor=0)
capture.finish(winner=winner, reason=reason, core_terminal=real_msg_win_seen)
```

完整服务端/本地桥接可使用 `visibility='full_core'`，额外传入：`reset`（实际 Core seed
或完整 seed_sequence、双座位实际注入列表、LP/起手/抽牌/规则位、initial_position）、
`decks`（双方稳定 Main/Extra，Side可留在元数据但不进入单局 Encoder）、`assets`
（在真正采集环境调用 `runtime_asset_identity(env)` 的原始结果）。不能在导入时以当前
本地摘要替代历史摘要。只有资产完全一致、原始块逐字节相同、响应/玩家/终局顺序一致，
再经独立规范重放，才可能通过动作标签门禁。单方捕获仅诊断，不补齐隐藏数据。

`player_ids` 是来源服务命名空间内稳定匿名ID，不是昵称、座位、模型UUID的手工改名。
YRP可用 `--p0-player-id/--p1-player-id` 追加来源标识；Link导入不覆写原捕获身份。
完整原始资料可能包含隐藏卡、构筑与关联身份，分享前自行脱敏。摘要是损坏检测，**不是
数字签名、来源可信认证或专家水平证明**。原生执行建议使用独立CLI进程，不把未审查
捕获送入在线推理进程。

### 隔离划分与质量门禁

`DATASET_MANIFEST_VERSION=1` 独立维护于 `trajectory_dataset.py`。扫描最多10,000局，
每局固定有界输入副本、只读审查。损坏、截断、非真实终局、来源缺证据、未知稳定玩家、
Retry、原始报文偏差、观测摘要不符、非唯一或不可见候选映射均不能成为可训练条目。
无终局和已知坏来源先隔离，避免无意义原生重放；必要条目才执行严格重放。

划分不是“把某局前80%状态放训练、后20%放验证”。完整局身份按开局/资产/原始消息与
响应生成，去掉录制UUID/展示来源差异；重复封装去重。构筑按 Main/Extra/Side、精确
卡号及投入数量哈希，不按文件名或卡片顺序。共享任一完整局、精确构筑或稳定玩家的
轨迹形成**传递连通组**，整组只进入一个集合；默认验证/测试各0.1，按组确定性抽样，
不是保证文件数量恰好10%。相同规则Bot/模型UUID也算共享玩家。

当前只隔离**精确相同构筑**，不宣称相似卡组/同系列构筑已经自动聚类隔离；后续可在
清单上增加人工或上层构筑家族标签。没有稳定玩家身份则先隔离，不假装做到了玩家隔离。
若所有局因共享玩家/卡组连成一组，独立验证集可能为空，清单明确警告，不强行拆开。

清单仅记录相对文件路径、文件/整局摘要、分组、来源、资产、结果和质量统计。
后续学习器必须再次核对文件摘要与身份，不把可修改JSON清单当成可信训练授权。
本批不实现行为克隆优化器、预测曲线回填或外部Link联网接线；YRP后验预测可在下一
批基于已验证的玩家观测独立接入。正常训练不开启这些工具时没有新增逐步计算或存储。

### 验收证据

已验证压缩/未压缩、取消/零尾字节、YRP3D伪魔数/截断/多包、YRP2完整种子、未知模式、
部分Link拒绝原生执行、来源摘要、传递隔离、损坏文件隔离、重复局去重及报告不覆写。
真实Core合成完整Link捕获可转换并通过质量门禁；YRP1/YRP2同源响应可重建和独立重放，
但仍遵守缺历史证据的拒绝规则。目录现有13份YRP/YRP3D均可检查，其中4份旧非UNIFORM
只诊断；现有YRP1的1,137条响应和YRP2的809条响应均到真实终局，零Retry、完整解析，
分别724/389个模型决策观测摘要及413/420个自动空连锁应答通过独立重放。
历史录像的宏池存在未唯一映射，训练资格保持false。未做真实在线Link端到端或生产长训。
