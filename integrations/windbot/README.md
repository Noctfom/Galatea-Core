# WindBot 可选集成

此目录提供 Galatea-Core 3.15.4 的可选 WindBot 集成：受控 BO1 会话自3.15.0提供，可选主训练对手自3.15.4提供。一个公开 Core 驱动对局，WindBot通过本地协议桥响应。默认训练不依赖本模块；真人客户端和BO3尚未接入。

3.15.3的独立模仿组件位于 `../imitation/`，提供教师/留出池、一次准备缓存、BC优化器和固定
换边评测。必须显式通过WebUI“独立模仿学习”或 `imitate` CLI启动，不由普通WindBot对局
自动触发，也不自动接续主训练或真人在线学习。BC权重可由用户显式选择 `train --init-from`
建立新的PPO分支；不恢复BC优化器，PPO/BC累计计数分别保留。[完整操作说明](../../docs/core_trajectories.md#3153独立模仿学习工作流)。

3.15.2可从完整合格规范轨迹读取真实专家样本，默认只选WindBot，不自动模仿陪练模型；
仅离线CPU读取与报告，不改对局执行或启动学习。见 [专家样本CLI/API与身份审计](../../docs/core_trajectories.md#3152完整轨迹的专家样本读取)。

## 3.15.4主训练对手

WebUI在 **启动与监控中枢 → 发起训练 → WindBot主训练对手** 勾选启用，读取兼容执行器后
选择训练/留出名单、Worker分配比例及模型侧卡组选法。控件附带说明；关闭时不导入可选模块，
也不检查或启动WindBot。已有检查点仍能照常推理、继续PPO或显式作为BC后的权重来源。

```bash
# 默认20%概率给每个Worker分配WindBot，其余保留原联盟抽样
python main.py train --resume models/galatea_iter_200.pth --additional-iterations 10 --windbot --windbot-rate 0.2 --no_compile

# 手动指定训练与留出执行器，模型侧可改用常规训练卡组池
python main.py train --resume models/galatea_iter_200.pth --additional-iterations 10 --windbot --windbot-executor Blue-Eyes --windbot-holdout Albaz --windbot-deck-mode training-pool --no_compile
```

- **卡组与席位**：默认每局镜像WindBot固定主/额外构筑；`training-pool`仅将模型侧改为常规
  训练卡组抽样，WindBot构筑保持不变。模型物理席位每局随机。Side只用于外部握手与来源记录，
  不进入单局模型观测。每个WindBot Worker按局从训练名单等概率选执行器。
- **默认名单**：建议使用当前CDB/词表兼容的`Normal`和`Easy`执行器，每个有多个执行器的
  层级保留一个完整执行器；相同主/额外构筑的别名跟随留出，不以Side差异伪装独立构筑。
  `NotFinished`与`Test`默认不选，仍可手动选择。上游层级是脚本声明，不代表实测强度。
  留出只约束WindBot训练对手，不能证明模型在其他卡组池或BC中从未见过该构筑。
- **比例与预算**：比例是Worker分配概率，不是对局数或模型样本占比；小规模单轮可能未抽中。
  混合模式保留RuleBot补局，优先交给普通AI Worker；比例为100%时只对WindBot，不插入Rule局。
  不另建模型副本、GPU服务或主进程经验池。机器人启动、查询刷新和CPU决策会消耗采集时间，
  实际吞吐取决于执行器与局长；不能保证与self/hist相同。
- **单一学习来源**：Worker仍使用原中央推理、模型候选、log-prob、FiLM掩码、辅助标签与GAE。
  WindBot响应直接提交同一Core，作为环境变化供模型观察；不生成教师动作标签、不调用BC，
  不加载真人对局或重放器。普通受控对局和独立模仿工作流也不会因此自动加入PPO。
- **异常与回收**：缺依赖或不兼容名单在启动前报错；断线、无效响应、解析失败等沿用整局回滚，
  不替换对手、不伪造胜负。每局回收Bot与连接；Trainer强制终止Worker后按登记的PID与创建时刻
  清理所属Bot，防止误杀复用PID。无可用动作时仅上传小型诊断，跳过PPO而不分配大型合并池。
- **来源与恢复**：检查点和TensorBoard `Configuration/WindBot`保存本次名单、构筑、层级、
  文件摘要、程序摘要及比例。恢复PPO不会自动启用WindBot，需再次传`--windbot`或在UI勾选；
  会按当前本地资产重新检查。网络、模型协议4、检查点格式3及修订14不变。

Bot日志仍在`system_logs/windbot/`，训练不默认写整局捕获或全息录像。查看模型对局与保留执行器
表现时，使用原受控对局或独立评测入口显式录制。主训练新增指标见
[TensorBoard对手统计](../../docs/tensorboard_metrics.md#11-windbot主训练对手指标)。

## 接口归属与运行边界

Core 定义项目组唯一的对局会话、观测、动作与采集规范，并负责解析、玩家可见信息、V4 编码和数据门禁。WindBot 的启动、连接、执行器配置与进程退出留在此模块；Link 在 Core 接口稳定后复用，不作为本模块开发或运行的前置条件。

启用 WindBot 必须是显式操作。默认启动不得导入此模块、探测外部运行环境、下载依赖或启动 Bot；模块不存在时，原有训练、竞技场、回放和打包仍须正常运行。缺少依赖只在选择 WindBot 时给出安装提示，不静默换成其他对手。

Core 会话合同 v1 位于 `../duel_session.py`，包含会话 UUID、单调请求编号、物理响应者及实际响应提交。旧式 YGOPro 不携带请求编号，因此桥接器严格串行响应；超时/断线即封存异常局，不重连复用旧响应，也不替换对手。监听仅限 loopback，每个会话使用独立房间口令；客户端提交的构筑须与受控构筑一致。

原始完整 Core 块和实际响应由既有 Link 采集 SDK 保存。网络握手、派生玩家视角和 query 刷新不混入原始块；转换仍需原资产一致和独立重放。客户端不能查询对方未公开手牌/盖卡/里侧额外卡组身份或未公开的主卡组顺序；私有确认及错过时点通知按接收者分流。当前对外桥不发送 Type 160 卡片提示，未知消息拒绝广播；这是本批外部客户端适配边界，不删除完整捕获或原生 V4 中的信息。

## 上游来源与许可证

- 上游项目：[IceYGO/windbot](https://github.com/IceYGO/windbot)。本次原样快照固定为 `master` 的 `dec0535f8669dca01457a898890a968139711b19`，来源记录见 `UPSTREAM.json`，不在运行时跟随分支变化。未改上游决策代码。
- 许可证：[MIT](https://github.com/IceYGO/windbot/blob/master/LICENSE)，版权声明为 `Copyright (c) 2015-2017 IceYGO`。导入源码或分发编译包时保留原许可证、版权声明和所带第三方依赖的通知。
- 上游是独立 C# YGOPro 客户端，不是 Galatea 原创代码或通用动作 RPC；用法见 [上游说明](https://github.com/IceYGO/windbot#usage)。

## 源码与 Git 管理

Galatea 原创的适配代码在 `bridge.py`、`runtime.py` 和 `session.py`；上游完整源码快照在 `vendor/`，不拆散到项目根目录。`UPSTREAM.json` 记录仓库、分支、提交、导入日期和原始归档摘要；上游许可证保留在 `vendor/LICENSE`，依赖原有通知随快照保留。Git 提交应区分原样源码导入、Galatea 适配和后续上游更新。

不提交嵌套 `.git`。源码、许可证、来源记录及可复核的必要补丁进入 Galatea 仓库；本地可执行运行目录 `runtime/`、构建目录 `build/` 和上游 `bin/obj` 产物由根 `.gitignore` 排除。若需要改动上游，单独标记补丁及理由，不把改动描述为未经修改的上游代码。

一键包默认排除 `vendor/`、`runtime/` 和 `build/`，保留原创适配代码与说明；`build_portable_package.py --with-windbot` 显式携带源码、许可证和已安装的 `runtime/`，始终排除构建工作区。此选项不下载或编译组件，不改变 GKG 模型包。

## 构建与使用

Windows 需 MSBuild 和 .NET Framework 4.8 开发引用包；Linux 需可构建该旧式项目的 Mono/MSBuild，以及 SQLite 原生运行库。上游不是现代 .NET SDK 项目，不使用 `pip` 安装，不修改主项目的 Python 依赖。

在项目根目录执行以下标准构建命令，将产物放入本模块的忽略目录；PowerShell 可将 `msbuild` 替换为实际 MSBuild.exe 路径。

```text
msbuild integrations/windbot/vendor/WindBot.csproj /t:Build /p:Configuration=Release /p:Platform=AnyCPU /p:OutputPath=../runtime/
python main.py windbot --inspect
python main.py windbot --executor Blue-Eyes --opponent-executor Blue-Eyes --verify
python main.py windbot --executor Blue-Eyes --windbot-seat 0 --verify
python main.py windbot --executor Blue-Eyes --model models/your_v4_model.pth --verify
python main.py windbot --executor Blue-Eyes --native-deck-source physical:your_pool --games 10 --record-replay
python main.py windbot --executor Blue-Eyes --model models/your_v4_model.pth --record-replay --record-evaluations
python main.py windbot --executor Blue-Eyes --model models/your_v4_model.pth --policy-mode deployment --temperature 0.2
```

`--runtime` 可指定其他已构建的 WindBot.exe；Linux 可用 `--mono` 指定 Mono。必须保留 exe 同目录的配置、SQLite 运行文件与 Dialogs；单拷贝 exe 不构成完整运行包。未自定义原生侧 YDK 时，使用对应执行器的相同构筑，不从普通训练池随机换牌。

WebUI 入口在竞技场的“WindBot 可选受控对局”。点击读取/刷新后，执行器与当前兼容 V4 PTH 通过下拉列表选择；切换另一侧类型立即显示对应控件，不再等待表单提交。缺少构筑的执行器不进入列表，旧协议/资产不匹配模型列入诊断说明，不伪装为可用模型。模型最终仍由原生严格加载器验证。

WebUI和CLI默认随机Core开局，各局使用起始种子加局数偏移；实际值记入日志与捕获。
取消“随机Core开局”或传入 `--seed 73` 可固定序列。该种子不控制WindBot自身随机性。
同样的随机/固定选项也用于独立模仿采集和评测，数据划分种子与Core开局独立。

选择 V4 模型后，可独立设置“原生模型决策策略”：默认 `training` 按 `Categorical(logits)` 采样，温度固定1.0；`greedy` 取最高 Logit，不使用温度；`deployment` 使用0.05～5.0的可调温度，初始值0.8。对应 CLI 为 `--policy-mode` 和仅部署采样使用的 `--temperature`。这些设置不沿用普通竞技场控件，也不控制 WindBot。本入口宏池仍使用均匀先验，因此训练式采样不表示完全复现训练 Pass1。结果 JSON 的 `native_policy` 和全息录像元数据保存实际生效策略/温度；非模型对局的 `native_policy` 为 null。

原生侧默认使用该 WindBot 的相同构筑，也可以关闭此选项，复用普通竞技场的权重区间、物理池、虚拟池或单卡组选择器。CLI 对应 `--native-deck-source` 与 `--deck_dir`；每局用本局种子在所选范围抽取，兼容性规则与普通竞技场相同，不能与旧式 `--native-deck` 同时使用。只改变原生 RuleBot/V4 模型的卡组，WindBot 始终使用其执行器绑定的构筑；双 WindBot 模式没有原生选池。

原始捕获始终写入 `replays/link_captures/`，Bot 日志写入 `system_logs/windbot/`；`--verify` 通过既有入口生成 `replays/imported_trajectories/` 中的规范轨迹。`--record-replay` 独立追加 `ai_thoughts/` 中的全息录像，保存双方决策、场面事件和开局主/额外构筑，可在现有全息回放页播放；`--replay-root` 可指定其他输出目录。外部动作记录实际响应，支持多选卡片展示，不生成 WindBot 置信度。`--record-evaluations` 需要全息录像和原生模型，逐已记录状态追加该模型侧预测，会增加 CPU 推理时间；WindBot/RuleBot 不生成预测曲线。三类文件继续使用存储与日志仓库管理。Bot 日志限 8 MiB，异常局只有诊断前缀/截断录像，无伪造终局标签。

### 初始化次数与校验开销

当前受控原型每局调用一次独立 runner，重新载入原生模型并建立 Core，不像普通竞技场在多局间复用同一模型/环境。开启终局校验且顺利完成时，每局通常有三次脚本预加载：实际对局、离线转换、独立重放各一次。五局因此是十五次预加载，不是十五局；校验不载入策略网络或重新运行 WindBot。各阶段按顺序建立/关闭 Core，而非同时启动三个对局。

日志以 `[WindBot/对局]` 和 `[WindBot/校验]` 区分阶段。报告 `seconds` 保持原有受控会话计时（不包含前置模型加载和终局校验），额外的 `verification_seconds` 记录终局转换及重放耗时，不是实际比赛时长。只需观察对局时可取消 WebUI 的“终局后独立转换与重放”或不传 `--verify`，每局只初始化一次 Core；原始捕获及已启用的全息录像仍保存，可稍后通过 `replay-ingest --source link` 校验。未完成校验的捕获不能直接作为已验收学习数据。

## 验证范围与后续边界

本批验证 Windows 实际编译、原生 RuleBot 对 WindBot 双座位、WindBot 镜像、终局原始输出/观测摘要独立重放，以及提前停止的进程回收。Linux 实际联网、全部执行器、真人客户端和 BO3 尚待验收。

烙印 `Albaz` 镜像原先有 2 条 Type15 未映射。复审确认两处都是 8 选 2，28 个组合完整入池，不存在 120 项缩减：实际响应 `02 07 00` 对应池内 `02 00 07`，实际 `02 07 05` 对应池内 `02 05 07`。专家与池选择同一组物理卡片，却使用相反的索引排列；此前“宏池覆盖缺口”的归因应以此复审为准。通过下述 WindBot 适配器重新转换后，471 条决策均唯一映射，整局通过现有行为克隆资格门禁；原始输出逐字节一致，实际响应与所有观测摘要保持不变。这不等于已经训练模仿学习器。

### 原始重放与学习标签等价性

两处均位于结束阶段，紧邻选择提示 501（弃牌）；8 张手牌选择 2 张，双方实际都选择了融合派兵与烙印开幕。公开 Core `processor.cpp` 的手牌上限分支把响应索引写入 `card_set` 后送墓；`libgroup.cpp` 的普通选卡也写入 Group 的集合。因此这里不是融合素材解析失败，而是合法集合响应的顺序等价未纳入标签比较。

原始响应是重放证据，始终原样保存/提交；不为了生成标签修改它。`bridge.py::selection_label_bytes` 仅为 WindBot Type15 提供比较键：复用现有选择校验器检查数量、索引范围、重复索引及取消权限，再对**已使用的物理索引集合**排序。候选与专家两侧使用同一比较键，不依赖 RuleBot 的权重排序；未使用的缓冲尾部不参与标签身份。不同个体即使卡密相同也不合并。Core 与 WindBot 均把 cancelable 的非零值视为可取消，包含合法值 `2`，不局限于字面 `1`。

公共入口由 `../duel_session.py` 定义：`response_label_bytes(adapter, response, prompt_type, payload)`。首个合同 ID 为 `windbot_select_card_set_v1`，只允许显式 WindBot 来源座位使用；不是运行时插件加载器，不从文件内容加载任意模块。Link 后续可复用合同入口与门禁，并为自己的来源显式定义适配，不应冒充 WindBot 来源。

全量捕获转换时，转换器依据已验证的参与者元数据，将每座位合同写入新规范轨迹头部的 `response_label_adapters`，并保留来源信息。独立重放按同一合同重新计算全部匹配项，同时仍向 Core 提交原始响应、逐块核对原始输出、逐决策校验观测摘要。无唯一匹配、超出120槽或缺少真实终局等条件仍拒绝学习；不向候选池注入专家答案。未知合同及不适用的参与者类型拒绝处理。

`canonical_response`、默认匹配、RuleBot 包装、训练、网络和公开 ygocore 均不改语义；Type21/25 排序与 Type20/23/26 等其他响应保持原有顺序敏感比较。旧规范轨迹未声明合同时继续使用旧字节规则，既有文件不自动改写。需要修复旧 WindBot 标签时，在 WebUI“轨迹与数据门禁”重新转换原始 `*.link.jsonl.gz`，或执行 `python main.py replay-ingest PATH --source link`；新建规范文件后重新生成质量/划分清单，不沿用旧审查结果。

响应顺序问题与实际池覆盖是不同问题。后者仍可能来自 5000 组合枚举上限、等价副本合并或超过 120 项的抽样；届时应分别检查生成前候选、缩减后候选和专家响应。即使某条专家响应合法，也不能静默塞入池里制造表面上的训练/部署同分布。整局资格还是逐样本资格、确定性候选还是分解动作，是后续数据批次的设计选择。

### Link 玩家视角中的派生查询接口

本地原生 `get_message` 块与在线客户端收到的宿主协议不是同一层。宿主还生成 Start（GameMsg4）、UpdateData（6）/UpdateCard（7）等初始化和 query 刷新，包含 LP/区域数量或 query flags 指定的属性。这些不是一次新动作，不应作为原始 Core 事件直接送入现有解析器，更不能声明成 `full_core` 来源。

当前受控桥已把这些查询发给 WindBot，由它自己的客户端解码；Galatea 原生侧仍通过自己的 Core 查询取得状态。因此当前本地闭环可运行，但这不意味着未来 Link 只保存过滤后的 TCP 包就能独立重建同一 V4 观测。

3.15.1已在 `../player_view.py` 定义有界分帧、Start/整区/单卡刷新、稀疏查询和诊断投影接口。
WindBot封包与可见性过滤复用该Core模块，字节行为不变，也不额外创建一份状态或推理。
新SDK明确物理座位、缓存/无缓存语义、离场/洗牌失效、未知字段和隐私掩码；
只接受本方构筑，不暴露对方手牌、盖卡、主卡组隐藏顺序或Side。
当前保守profile仍缺对手响应边界/Type160提示/完整动态属性的证明，默认V4编码及学习资格均不开放。
仅玩家视角的数据保持 `player_view`，不承诺从缺资产/种子/原始消息的录像恢复完整重放。
接口、调用示例及完整性报告见 [Core玩家视角SDK](../../docs/core_trajectories.md#3151core-玩家视角接收接口)。

本机标准构建遇到缺少 4.8 引用包；本地验证产物用已安装框架程序集进行临时编译，未修改上游项目目标，也未安装系统 SDK。正式构建应使用完整开发引用包；本地 smoke 不替代跨平台发行验证。

受控局的 P0 固定先手，交换座位不会更改这一规则。种子只控制 Core，不控制上游 Bot 的时间初始化随机数，因此该入口不等于固定竞技基准。原生模型目前使用 CPU，并沿用竞技场策略采样；宏候选采用 Core 规范重建的均匀先验，不宣称与训练 Pass1 选池完全同分布。可选全息录像不提供 WindBot 置信度，也不把显示用响应解码当作学习标签。

普通 `--verify` 和旧质量清单仍按严格整局门禁运行。独立 `imitate collect` 在原生实际对局中
直接准备样本；`imitate prepare` 对旧完整Link只重建一次并同步产样，不重复调用Core。
原生/重建源的个别未映射动作只排除标签，状态失效仍拒绝整局。独立数据集可明确选择整局、
构筑或执行器留出；同构筑新对局不冒充跨构筑泛化。完整YRP旁证和真人在线接线仍待开发。

## English summary

Galatea-Core 3.15.4 owns this optional integration: controlled BO1 arrived in 3.15.0 and optional training opponents in 3.15.4. One public Core drives a loopback WindBot client. Source remains pinned to `dec0535f8669dca01457a898890a968139711b19` without upstream decision patches. Human clients and BO3 remain future work.

### Optional main-training opponents

Enable the WindBot section on the main training tab, or pass `train --windbot`. Disabled mode neither
imports the optional module nor probes its runtime. The default Worker allocation probability is 0.2;
remaining Workers use the original league. It is not a guaranteed game/sample share. Each external
Worker samples its executor uniformly per game, with random model seats and mirror builds by default.
Mixed mode retains RuleBot coverage, preferably on regular AI Workers; a 100% setting uses only WindBot.
`--windbot-deck-mode training-pool` uses regular sampling only on the model side.

Compatible Normal/Easy executors are suggested, reserving one per multi-executor level plus aliases
with identical Main/Extra builds. Side differences do not create an independent BO1 build. Unfinished/
Test scripts remain manual choices. Upstream levels are labels, not measured strength. Holdouts exclude
WindBot training opponents only; the same cards/builds may appear in other pools or BC. Explicit
`--windbot-executor` and `--windbot-holdout` lists override suggestions.

Only model actions reach ordinary PPO/GAE/auxiliary targets through the existing Worker and central
inference. Bot responses are environmental transitions, never BC labels. Side is transport/provenance
only. No additional GPU policy, replay pass or experience pool is created. Startup, refreshes and Bot
CPU inference consume collection time; throughput is executor-dependent.

Failures roll back games without replacement opponents or fabricated outcomes. Each game closes its
Bot/socket; forced Worker cleanup matches the registered child PID and creation time. Diagnostic-only
results do not allocate a large merge pool or invoke PPO. Checkpoints and `Configuration/WindBot`
retain the active roster, builds, file/runtime hashes and allocation settings. Resume requires another
explicit opt-in and checks current local assets. Network/Protocol4/Checkpoint3/revision14 stay unchanged.
Training writes Bot logs, not full captures/replays by default. Use controlled duels/evaluation to record
held-out games. See [metric definitions](../../docs/tensorboard_metrics_en.md#11-windbot-main-training-opponent-metrics).

The independent `../imitation/` workflow adds teacher/holdout pools, one-time preparation, a BC optimizer
and swapped-seat fixed evaluation. Only explicit WebUI/`imitate` commands start it; ordinary games do
not train automatically or auto-join PPO/online-human learning. Explicit `train --init-from` can
initialize a separate PPO branch from BC weights, with a fresh optimizer and independent cumulative
PPO/BC counters. See the [workflow guide](../../docs/core_trajectories_en.md#3153-independent-imitation-workflow).
Ordinary `--verify` retains strict whole-duel replay. Independent native collection prepares samples
in the actual duel; complete older Link sources reconstruct and produce examples in one Core pass.
Individual unmapped labels are skipped, while invalid states reject the game. Explicit duel/build/
executor splits replace universal program-ID grouping only in the separate imitation dataset.

3.15.2 reads real expert examples from complete eligible canonical traces, defaulting to WindBot rather
than native opponents. This is offline CPU reading/reporting, not altered duel execution or learning.
See [expert CLI/API and identity audits](../../docs/core_trajectories_en.md#3152-expert-samples-from-complete-verified-traces).

Core owns the shared session, observation, action and capture contracts. WindBot lifecycle management belongs here; Link reuses stable Core interfaces later and is not a prerequisite. Normal startup must not import, probe, download or launch this component. Its absence must not affect existing training, Arena, replay or packaging. Selecting WindBot without its dependencies must report the missing setup instead of silently substituting another opponent. Future Link player-view reconstruction needs Core's host-generated Start/query adapter; these packets are not original native Core chunks.

The upstream is [IceYGO/windbot](https://github.com/IceYGO/windbot). Its [MIT license](https://github.com/IceYGO/windbot/blob/master/LICENSE), copyright and dependency notices are retained in the source snapshot. `UPSTREAM.json` identifies the exact snapshot and original archive; Git excludes nested repositories and build/runtime outputs. Portable packages exclude vendor/runtime by default; `--with-windbot` includes source, licenses and an already installed runtime, without downloading or building. GKG is unchanged.

Build the legacy project using MSBuild/Mono and .NET Framework 4.8 references, with `OutputPath=../runtime/`, then use `python main.py windbot --inspect` or the commands above. Keep native SQLite/configuration/Dialogs beside the exe; `--runtime` and `--mono` override paths. The Arena panel reads executor/compatible V4 checkpoint lists on request and reacts immediately to side-type changes. Native opponents can use the same Bot build or the regular weighted/physical/virtual/single-deck Arena selector (`--native-deck-source`, `--deck_dir`); WindBot always keeps its own executor deck. Raw captures are always saved; `--verify` converts/replays without learning. Independent `--record-replay` saves both sides and board events for Holographic Replay. Optional `--record-evaluations` requires a native model and adds only its state predictions, at extra CPU cost. External confidence is not fabricated; aborts do not receive terminal labels.

With a V4 opponent, the WebUI exposes an independent native-policy selector. The existing default `training` samples `Categorical(logits)` at fixed T=1.0; `greedy` uses argmax without a temperature; `deployment` exposes T=0.05–5.0, initially0.8. CLI equivalents are `--policy-mode` and `--temperature` for deployment. These settings affect neither WindBot nor normal Arena controls; uniform macro priors mean training-style sampling is not full Pass1 equivalence. Result JSON `native_policy` and Holographic metadata record the effective mode/temperature; non-model sessions report null.

The prototype rebuilds its native model/Core per controlled game rather than reusing a normal Arena instance. With successful terminal verification, each game loads scripts three times: actual play, offline conversion and independent replay. Five games therefore produce fifteen preload pairs, not fifteen games. Verification neither loads a policy network nor reruns WindBot; Core instances are created/closed sequentially. `[WindBot/对局]` and `[WindBot/校验]` identify these stages. Existing `seconds` measures the controlled session excluding initial model loading and terminal verification; `verification_seconds` separately measures conversion/replay. To watch games without immediate checks, disable terminal verification or omit `--verify`: one Core initialization per game, with raw captures/selected Holographic Replay still saved. Verify later using `replay-ingest --source link`; unverified captures are not accepted learning data.

Contract v1 serializes one pending physical-player request. Legacy YGOPro cannot echo request tokens, so timeout/disconnect ends the session instead of reconnecting. Loopback peers use separate room tokens and verify deck submissions. Hidden card identities/Main Deck order are masked; Type160 is not sent externally, and unknown messages fail closed. Complete original chunks remain separate from all derived client updates.

Windows real-Bot/Core smoke and replay are validated; Linux, all executors, human clients and BO3 remain pending. Local builds used installed framework assemblies because the development reference pack was missing; standard release builds require proper references. P0 always starts; Core seeds do not control Bot randomness. Native models use CPU with uniform canonical macro pools, not full training Pass1 equivalence. Complete YRP evidence and isolated imitation optimization remain pending.

The two Albaz Type15 misses were **response ordering, not pool truncation**: each end-phase discard prompt (hint501) has all 28 combinations of eight cards taken two at a time. Both players discard Fusion Deployment and Branded Opening. Expert bytes `02 07 00`/`02 07 05` differ from pool bytes `02 00 07`/`02 05 07`, although they refer to the same physical subsets. The public Core's hand-limit branch inserts responses into a `card_set`, and ordinary selection returns a Group set. Re-conversion now uniquely maps all 471 decisions and passes the existing behavior-cloning eligibility gates. Original response bytes, exact raw replay and every observation digest are unchanged; this is not imitation optimization.

Core owns the closed `response_label_bytes(adapter, response, prompt_type, payload)` interface in `../duel_session.py`. WindBot's `bridge.py::selection_label_bytes` implements `windbot_select_card_set_v1`: validate Type15 constraints first, then compare the used physical-index set on both expert and candidate sides. Same-code copies stay distinct; cancellation follows the public Core's nonzero-byte convention, including `2`. Unused buffer tails do not identify labels. Conversion selects this contract only for explicitly identified WindBot seats and stores per-seat `response_label_adapters` in the new canonical header. Replay independently checks matching under that contract while submitting the original response bytes. Unknown contracts/wrong participant kinds fail closed; no arbitrary module is loaded from capture content. Future Link can reuse the interface with its own explicit provenance/adapter rather than pretending to be WindBot.

Ordinary `canonical_response`, default matching, RuleBot packaging, training, network and public ygocore semantics are unchanged. Other prompt types, including ordered Type21/25 and Type20/23/26, retain their original byte-order comparison. Existing canonical files without a contract retain strict matching. Re-convert the original WindBot capture through WebUI or `python main.py replay-ingest PATH --source link` and rebuild quality/split manifests; no old file is overwritten automatically. Ambiguity, missing candidates, slots beyond120, missing real terminal/evidence and all other whole-duel gates still reject eligibility. True coverage gaps from enumeration/equivalence reduction/120-slot sampling remain separate future cases; expert answers are never inserted into policy candidates.

3.15.1 provides Core-owned bounded framing, Start/zone/card refreshes, sparse attributes and diagnostic projection in `../player_view.py`. WindBot reuses its builders/filter without changed bytes, another state adapter or extra inference. Physical seats, cache modes, instance invalidation and unknown/private fields are explicit; only the own build is accepted. The conservative profile still cannot prove all opponent response boundaries, Type160 hints or dynamic fields, so default V4 encoding/learning eligibility remain closed. Host refreshes never become actions or `full_core` evidence. This does not validate a standalone online Link client or exact raw replay from filtered captures. See [Core's player-view SDK](../../docs/core_trajectories_en.md#3151-core-owned-player-view-ingress).
