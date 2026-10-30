# 🔧 架构设计

> 本文档深入介绍 Galatea-Core 的技术架构与核心算法，适合想要深入了解或参与开发的用户。

> 文档适用于 **Galatea-Core v3.15.4**：Model Protocol 4 / Checkpoint Format 3 / 结构修订 14 不变。独立模仿工作流与可选WindBot训练对手分开启动，前者监督示范动作，后者仅以PPO学习模型自己的动作；不自动学习真人对局。单方玩家视角接收仍仅提供诊断投影，不冒充完整原生观测。

> 💡 **框架的独特处理逻辑**（语义化模块、142宣言池、多选题组块包装、记牌器、卡组权重、伪装池）请参考 [特殊处理逻辑文档](special_handling.md)。

---

## 📋 目录

- [系统架构概览](#系统架构概览)
- [神经网络架构](#神经网络架构)
- [特征编码系统](#特征编码系统)
- [语义知识库](#语义知识库)
- [PPO 训练框架](#ppo-训练框架)
- [OCGCore 环境封装](#ocgcore-环境封装)
- [性能优化技术](#性能优化技术)

---

## 系统架构概览

Galatea-Core 采用模块化设计，由以下核心子系统构成：

```text
WebUI / CLI
    ├─ Trainer → CPU Workers → 公开 Core（DLL / SO）
    │                │ 解析 / 玩家视角快照 / V4 Encoder
    │                └─ 共享观测槽 + ZMQ → CPU/CUDA 中央推理
    │                   当前策略与 self 共用中央模型；hist 为本地 CPU 对手
    ├─ Arena → 同一解析器 / Encoder / 响应包装器 → 模型决策
    └─ 录像工具 → 原始 Core/Link/YRP → 诊断重放 / 质量门禁

V4 观测 + 精确词表 + 编译语义表
    ├─ 卡片主干 + Deck Encoder → 全局/卡组分层 FiLM → Transformer
    └─ 连锁 / 最近发动 / 16步事件 / 公开提示 → 共享状态
                         ↓
                   目标规划 latent
                         ↓
                  策略头 + 价值头
                         └─ PPO：辅助/规划后验监督 + KL 更新保护
```

### 核心文件说明

| 文件 | 层级 | 功能 |
|------|------|------|
| `galatea_net.py` | 认知层 | Transformer 神经网络定义 |
| `feature_encoder.py` | 感知层 | 游戏状态特征编码 |
| `semantic_kb.py` | 感知层 | 语义知识库查询 |
| `semantic_lookup.py` | 资产层 | 精确 token 静态语义表的编译、加载与身份校验 |
| `protocol_schema.py` | 协议层 | V4 字段、Model Protocol 与结构修订的统一定义 |
| `galatea_env.py` | 环境层 | OCGCore 环境封装 |
| `gamestate.py` | 环境层 | 游戏状态解析（核心！） |
| `deck_protocol.py` | 协议层 | 完整构筑、单局卡组画像与 BO3 上层边界 |
| `deck_trajectory.py` | 轨迹层 | Worker 卡组画像目录、索引重映射与静态字段无损重建 |
| `deck_encoder.py` | 认知层 | Main/Extra 无序集合编码、卡组风格与上层复用接口 |
| `goal_planner.py` | 认知层 | 决策级规划隐变量、策略/价值门控与目标解码器 |
| `auxiliary_heads.py` / `auxiliary_targets.py` | 学习层 | 训练专用预测头与真实结果后验标签 |
| `ppo_control.py` | 学习层 | GAE/PPO 默认控制、KL 保护与实际更新审计 |
| `core_trajectory.py` | 轨迹层 | 唯一 Core 解释器、规范记录与重放验证 |
| `trajectory_ingest.py` / `trajectory_dataset.py` | 数据层 | 外部捕获入口、质量门禁与整组隔离划分 |
| `trainer.py` | 应用层 | PPO 训练器 |
| `worker.py` | 应用层 | 多进程数据采集 |
| `model_versus.py` | 应用层 | 普通竞技、逻辑/物理座位映射，以及贪心/训练同分布/部署温度策略 |
| `arena_benchmark.py` | 应用层 | 固定赛程、卡组/模型哈希及基准结果统计；不参与训练或网络前向 |

---

## 神经网络架构

### GalateaNet 结构

GalateaNet 是一个基于 Transformer Encoder 的策略-价值网络，v3.2.0 引入了 FiLM 全局调制与 SwiGLU 门控前馈：

```text
输入：120个卡片槽 / 最多120个动作；17维全局量 + 阶段/玩家角色类别
     稳定 Main/Extra 画像 + 动态剩余卡组；Side 不进入单局观测
语义：精确 token → 静态 Lua 效果槽/384维代码向量 → 可训练投影/Slot Attention
主干：卡片数值/类别/公开关系 → Transformer（分层 FiLM + SwiGLU）
上下文：连锁12项、最近发动8项、决策转移16项、当前公开提示与已知卡顶
卡组：8个学习型 latent → deck_style → FiLM / 共享状态 / 规划器
输出：动作匹配策略头 + 价值头；128维 plan_latent 经独立有界门控接入
训练专用：动作条件辅助头 + 规划目标解码器（标准 ONNX 不输出诊断预测）
```

这是当前数据流示意，不是可直接运行的建网代码。字段清单及版本以
`protocol_schema.py` 为准，具体张量路径见 `galatea_net.py`。

### 分层 FiLM：全局时局与卡组风格

FiLM（逐特征线性调制）在 LayerNorm 之后对特征作 `x * (1 + γ) + β`。
V4 使用两个独立低秩生成器：全局分支读取17维时局量和16维阶段嵌入，包含先后手、
自身/对方回合、回合进度、LP 等；卡组分支读取学习得到的 `deck_style`。
每层注意力和 FFN 分别取得 γ/β，不再全层共用一组参数。

两分支以 `0.5 * tanh(global_raw + tanh(deck_gate) * deck_raw)` 合并，总幅度限制在±0.5。
卡组门零初始化，训练中按整局10%概率关闭并在 PPO 中复用同一掩码；直接进入策略/价值
共享状态的卡组通道仍保留。它让模型**学习**时局与构筑条件下的特征选择，不手工指定
“某卡组一定进攻/防守”，也不会为一局生成并替换整套网络权重。

### SwiGLU 门控前馈网络 (v3.2.0 新增)

Transformer 前馈子层将传统 MLP (Linear→ReLU→Linear) 替换为 SwiGLU 门控线性单元：

```python
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, multiple_of=64):
        # 自动补齐至 64 的倍数，对齐 Tensor Core 硬件
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        # SiLU(Gate) × Up → Down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**核心优势**：

- **门控机制**：SiLU 激活的 Gate 分支对 Up 分支进行逐元素筛选，让网络自行决定哪些特征通过
- **无偏置设计**：该 SwiGLU 模块的三个 Linear 层不使用 bias，减少相应参数；不表示整网所有 Linear 都无偏置
- **Tensor Core 64 对齐**：`hidden_features` 自动补齐至 64 倍数，充分利用 GPU 硬件算力

### 双塔匹配机制

策略头采用双塔（Dual-Tower）架构进行动作评估：

```
意图塔 (Intent Tower)              选项塔 (Option Tower)
      ↓                                   ↓
┌─────────────┐                   ┌─────────────┐
│ 全局局面    │                   │ 目标卡片    │
│ 特征向量    │                   │ 动作类型    │
│ (v_input)   │                   │ 响应/约束   │
└──────┬──────┘                   └──────┬──────┘
       │                                 │
       └────────────┬────────────────────┘
                    ↓
              ┌───────────┐
              │  拼接融合  │
              │ (Concat)  │
              └─────┬─────┘
                    ↓
              ┌───────────┐
              │ Policy MLP│
              └─────┬─────┘
                    ↓
              动作概率分布
```

### 好奇心驱动探索

GalateaCore 通过 **熵正则化** 和 **历史模型对战联盟** 鼓励探索、增加对手多样性，
降低过早收敛和策略遗忘的风险，但不保证避免局部最优或自动获得更强棋力。

## 特征编码系统

### 编码维度

| 特征类型 | 维度 | 说明 |
|----------|------|------|
| 全局连续特征 | 17 | 回合数、相对先后手/回合身份、双方回合数、LP 与区域计数 |
| 全局离散特征 | 1 + 3 | 阶段类别；决策者/回合者/先手的相对角色 |
| 卡片特征 | 66 + 2 类别 + 紧凑公共状态 | 数值属性、效果位、类型/连接掩码；区域、表示、关系与状态类别 |
| 语义特征 | 128×8 | 每张卡最多 8 个效果槽 |

### 卡片特征详解

```python
feat_numeric = [
    owner,              # 控制者 (1.0/-1.0)
    sequence / 10.0,    # 序列
    current_atk / 4000, # 当前攻击力
    current_def / 4000, # 当前防御力
    base_atk / 4000,    # 基础攻击力
    base_def / 4000,    # 基础防御力
    pos_x, pos_y,       # 场上坐标
    level / 12.0,       # 等级
    lscale / 13.0,      # 左灵摆刻度
    rscale / 13.0,      # 右灵摆刻度
    is_public,          # 是否公开
    overlay_count / 5,  # 超量素材数
    counter_count / 10, # 指示物数
    is_equipped,        # 是否装备
    used_effect_mask[0:8], # 本回合已发动的效果槽位
    rank / 13.0,        # 阶级
    link_rating / 8.0,  # 连接值
]
# + 独立 card_zone / card_position 离散类别 Embedding
# + 32 维类型掩码 (怪兽/魔法/陷阱/效果/融合/同调/超量/灵摆/连接...)
# + 9 维连接箭头
```

3.8.2 起，标准公开 `query_card/query_field_count` 还提供动态卡名、原因位、原持有者、`DISABLED/PROC_COMPLETE/FORBIDDEN`、装备/取对象/原因卡关系、最多 16 张叠放素材、8 类带类型指示物和 32 位禁用区域。原因、指示物类型和区域掩码在轨迹中按小端字节无损存储，网络前向时再展开，避免让 PPO 轨迹池保存冗余逐位矩阵。查询在玩家视角遮罩之后才进入网络；对手里侧卡不会借此泄露身份或状态。

CDB 虽把等级、阶级和连接值复用在同一原始整数内，Core 查询会分别给出 `level/rank/link_rating`。超量与连接怪兽的 `level` 通常为 0；模型把 0 当作“不适用”而非缺失值，不会再用静态阶级或连接值回填等级。

Core 内部还维护精确召唤方式、来源和玩家，但当前公开查询 API 不导出这些字段。Galatea 不维护私有内核分支，也不从卡类或移动顺序猜测；目前只使用公开的 `STATUS_PROC_COMPLETE` 表示“已正规出场”，其余列为上游 API 待办。

### 动作编码

```python
act_dict = {
    'act_card_idx': [...],   # 可见目标实体索引 [120, 5]
    'act_type': [...],       # 动作类型
    'act_desc': [...],       # 效果描述 Hash
    'act_mask': [...],       # 有效动作掩码
    'act_race': [...],       # 宣言种族
    'act_attr': [...],       # 宣言属性
    'act_code': [...],       # 候选/宣言卡片代码
    'act_place': [...],      # 放置位置 [120, 5]
    'act_operation': [...],  # Yes/No/Select/Unselect/Finish/Cancel 等真实操作
    'act_summon_method': [...], # 召唤分类；未知特殊召唤使用独立类别
    'act_response': [...],   # 语义响应值
    'act_signature': [...],  # 完整动作语义的 4 字节稳定签名
    'act_context': [...],    # min/max/结果数量/完成与取消条件 [120, 6]
    'act_target_code': [...],# 隐藏区或宏动作目标卡密 [120, 5]
    'act_target_value': [...],# 祭品值/双星级/指示物分配 [120, 5, 2]
    'act_controller': [...], # 行动方视角控制者
    'act_location': [...],   # 引擎区域
    'act_sequence': [...],   # 区域内序号
    'act_position': [...],   # 目标表示形式类别
}
```

### 顺序上下文聚合（Model Protocol V3）

连锁和最近发动历史不能使用“位置向量相加后求均值”。因为
`Σ(语义ᵢ + 位置ᵢ) = Σ语义ᵢ + Σ位置ᵢ`，交换两个事件不会改变结果。
`OrderedContextPool` 改为以下流程：

```text
语义 token + 固定槽位向量
           ↓
深度可分离的一维局部卷积（区分前项/当前项/后项）
           ↓
通道混合 + 残差归一化
           ↓
带有效项掩码的注意力汇聚
           ↓
顺序敏感的单个上下文向量
```

连锁第 1～12 项保持 Core 入栈顺序；每项还编码卡号、效果描述、链序、处理卡位置、触发位置和双方相对控制者。历史第 0 槽仍表示最近一次发动。双向局部混合只处理已经发生的事件，不会读取未来信息。全空上下文返回零向量，填充槽无法参与卷积输入或最终权重。

每张卡的 8 个效果槽另有独立槽位嵌入。这样 `used_effect_mask` 的第 N 位才能与 Lua Parser 的第 N 个效果语义建立可学习对应，而不会在 Slot Attention 中退化为无序效果集合。

动作协议 V2 不把 `GameAction.index` 当作学习语义。`index` 与 `decision_bytes` 只负责把最终选择翻译回 Core；策略头看到的是操作、卡密、位置、约束和结果集合。Type 26 保持 Core 原生的逐次 Select/Unselect 过程，每一步都生成新快照并进入轨迹；Type 15/20/21/22/23/25 等静态组合消息则先由合法性枚举器生成完整响应，再由策略头选择。Type 21 的同时触发效果排序与 Type 25 共用排序回包，Type 22 会在 Pass1 暴露候选卡、指示物类型、目标总量和各卡可用数量。

V4 结构修订 4 将 `MSG_SELECT_IDLECMD` 的六个固定列表分别编码为通常召唤、特殊召唤、改变表示、盖放怪兽、盖放魔陷和发动，不再全部误标为发动。`act_operation` 与 `act_summon_method` 均为 `[120] uint8` 当前提示快照字段，并进入动作签名、策略网络、ONNX 与回放。标准 Core 的列表序号只能证明通常/特殊召唤这一层级，无法证明融合、仪式、同调、超量、灵摆或连接；因此特殊召唤统一使用 `SPECIAL_UNKNOWN`，只有未来获得 Core 消息或经验证运行上下文的直接证据时才写入精确方式。来源区域和怪兽卡类可以单独展示，但不得被当作召唤方式标签。通常召唤列表同样不推断是否需要解放；`TRIBUTE` 类别仅为直接证据预留。

V4 结构修订 5 按 `playerop.cpp` 逐条固定选择状态机：Type 15 校验张数范围，Type 20 同时校验最大张数与 `release_param` 总和，Type 23 使用必选占位、双值参数及 Core 的 Exact/SumGreater 边界且绝不伪造取消，Type 26 每次只返回一个 Select/Unselect 索引或 `int32(-1)` 的 Finish/Cancel。Type 20/23 现在也生成 Pass1 单素材候选，使模型偏好参与宏组合缩减；Type 26 的终止动作固定置于动作池头部，候选超过 120 时仍保留合法出口。`set_responseb` 缓冲严格匹配 Core 固定读取的 512 字节，消除了旧 64 字节截断与越界读取。

`MODEL_PROTOCOL_VERSION` 独立于框架版本和检查点容器版本维护。它同时写入 PTH 顶层、`net_config`、模型状态、ONNX 元数据与制品清单；版本不同表示输入张量或动作头权重不兼容，加载器会直接拒绝。

### 精确卡片身份（Model Protocol V4）

3.7.0 开始的 V4 不再用 `code % 19990` 把卡密折叠到同一 Embedding 行。根目录
`card_vocab.json` 为真实卡密分配唯一、连续、只追加的 token：`0` 为填充，`1` 为未知，
`2/3` 为对手隐藏手牌/魔陷占位，真实卡片从 `10` 开始。当前 `cards.cdb` 共登记
14,981 张卡，固定容量仍为 20,000，因此 Embedding 参数量没有增加。

场面卡、叠放顶卡、初始卡组、连锁卡、候选/宣言卡、宏动作目标和 Lua 语义引用
共用该词表。结构 `protocol_schema_hash` 不随合法追加变化；PTH、`net_config`、ONNX、
制品清单和 `.gkg` 另行保存训练时的词表 SHA-256 与卡数。加载时要求模型映射是当前
模型谱系词表的精确前缀，既允许旧 V4 模型在新增卡后继续使用，也拒绝重排或机器间分叉。
普通资源更新默认同步 Galatea 权威词表，也可指定自建词表源；自制卡可通过 WebUI 或
`python main.py vocab` 从本地 CDB 只追加，但同一模型谱系的所有机器必须分发同一份文件。
若 CDB 已出现权威词表尚未编号的新卡，仅含这些卡的卡组退出抽样，其余运行不受影响。
3.9.0 起，Lua 静态语义也拥有独立的只追加前缀身份。`semantic_lookup.py` 把每张卡的
8 个 Lua 效果槽编译到相同 token 行，PTH、ONNX、制品清单和部署包记录逻辑语义哈希。
新增卡及其效果不会破坏旧前缀；修改旧卡结构字段或代码向量会被明确拒绝。3.9.1 起，
Encoder 只输出卡片 token 和连锁/历史效果槽 ID，网络在设备侧还原同一份静态语义；
观测内容不变，但共享内存、Worker 轨迹和 PPO 合并池不再重复保存语义矩阵。
3.9.2 将编译结果物化为 `semantic_lookup_v1.npz`（模型数值表）和
`semantic_lookup_v1.json`（效果槽运行目录及完整性信息）。训练/PTH 启动直接加载前者；
spawn Worker 只加载后者；ONNX 在导出时把同一数值表固化进主图/`.onnx.data`，运行时不再
读取源知识库。两份编译资产是可重建缓存，不提交 Git，但一键包和 `.gkg` 会自动携带。
V4 的 `MODEL_PROTOCOL_VERSION=4` 由 `protocol_schema.py` 唯一维护，
`CHECKPOINT_FORMAT_VERSION=3` 由 `checkpoint_utils.py` 维护；V3 检查点不进行迁移。

### Phase 4 卡组协议边界

3.10.0 新增独立 `DeckSpec`、`DeckProfile`、`MatchContext` 和 `DuelSummary` 协议。
`DeckSpec` 保存主卡组、额外卡组和 Side Deck 的完整构筑，内容身份按“分区 + 卡密 +
投入数量”计算，不受文件名或 `.ydk` 排列影响；`DeckProfile` 只由主卡组和额外卡组派生，
其序列化结构中不存在 Side 字段，是当前 BO1 策略允许接收的唯一稳定卡组边界。

`.ydk` 的 `!side` 现会被资源层读取并单独校验。仅 Side 含尚未收录的新卡时，WebUI
发出警告但不阻断当前 BO1 训练/竞技抽样；主卡组或额外卡组异常仍会退出抽样。
`DuelState` 同时保留一局内不变的初始主/额外画像和随 Core 移动变化的剩余卡组，避免
后续编码器把“已抽走的卡”误认为构筑本身不存在。`MatchContext`/`DuelSummary` 仅为未来
BO3 换备与组卡层提供外部接口，当前返回空策略输入，不进入单局观测、网络或 PPO 轨迹。

在 3.10.1 批次中，稳定画像先接入训练轨迹，当时尚不改变模型输入。每个 Worker 以 `profile_id`
登记实际训练座位的一局画像，逐步样本只保存 `int32 deck_profile_index`；动态
`deck_idx/deck_mask` 继续逐步保存。`deck_race/deck_attr/deck_setcodes` 属于精确
card token 的静态属性，改为每个 Worker 只登记一次，Trainer 合并时检查冲突并建立
本轮查找表，在 PPO mini-batch 前按原 token 顺序重建。重建值与旧 Encoder 展开结果
逐元素一致，因此 GalateaNet、中央推理、PTH/ONNX 和协议结构修订均保持不变。

内部画像目录只含张量，可由 `weights_only=True` 加载；局部索引会在合并时重映射成
Trainer 全局索引。3.10.1 时画像索引仅用于身份校验和预留入口；3.10.2 起才重建下述
Deck Encoder 输入，索引数值本身不作为策略特征。Side 仍从 `DeckProfile` 的类型层排除。

3.10.2 为稳定画像新增 128 条目的模型输入。每个唯一条目携带精确卡片 token、
主卡组/额外卡组分区、初始投入数量、当前剩余数量及原有种族/属性/字段标签；同一卡密
即使同时位于不同分区也分别计数。稳定字段继续由每局画像目录在 PPO mini-batch 前
重建，只有 `deck_profile_remaining_count` 随决策步保存。当前 BO1 Encoder 只生成
Main/Extra，`DeckSection.SIDE` 仅保留给未来位于 `DuelState` 外部的组卡/BO3 调用方。

独立 `DeckEncoder` 不使用位置编码，以 8 个学习型 latent 对无序卡组条目执行交叉注意力，
再输出置换不变的 `deck_style`、固定形状 latent 和随输入同步换序的逐卡表示。卡片基础表示
复用精确 token 与静态属性，并对该卡全部有效 Lua 代码向量作掩码均值；它不依赖卡面文本，
也不在热路径读取脚本或 CDB。当前策略仅把 `deck_style` 加入统一意图表示，因此同时影响
策略头和价值头，并跳过未消费的逐卡回写；独立接口被上层显式调用时仍返回完整三类输出。
分层 FiLM 和规划 latent 已在后续3.11/3.12批次接入，上层组卡消费者仍待实现。原动态
`deck_idx/deck_mask` 暂时保留，确保对局中被效果加入卡组、但不属于初始构筑的卡仍可见。

这里的 8 个学习型 latent 不是人工定义的“启动点/终端”等标签，而是参与反向传播的查询
向量。它们分别读取全部无序卡组条目，能在 PPO 策略损失和价值损失驱动下自行形成不同的
关系摘要；含义不被框架强制命名，也不会跨局保存。`deck_style` 是 latent 的置换不变汇总，
`deck_latents` 保留固定数量的多关系表示，`per_card_features` 则与输入卡片一一对应并随排列
同步换序。三者可通过独立接口完整取得，未来组卡层无需反向解析当前动作头即可直接复用。

3.11.0 将原来“所有层、注意力和 FFN 共用一组 γ/β”的全局 FiLM 拆为两个低秩分支：
全局分支读取当前时局与阶段，卡组分支读取 `deck_style`；每个 Transformer 层的注意力与
FFN 各自取得独立 γ/β。两者按
`0.5 * tanh(global_raw + tanh(deck_gate) * deck_raw)` 合并，因此总调制始终位于 ±0.5。
卡组门逐层、逐子层、逐通道零初始化，训练开始时对主干影响严格为零；但卡组分支本身使用
非零初始化，所以主损失第一步即可给门产生梯度，不会形成双零死路。直接进入策略/价值意图
的 `deck_style` 通道继续保留，即使门尚未开启，Deck Encoder 仍能正常学习。

为降低模型对固定构筑身份的过拟合，训练 Worker 以独立随机源按整局 10% 概率关闭卡组
FiLM，并把 `deck_film_mask` 同步写入中央推理与每一步 PPO 轨迹；更新时复用相同掩码，避免
普通训练态 Dropout 导致 old/new log-prob 条件不一致。该掩码不关闭直接策略/价值通道；
竞技场、Link 和部署默认始终开启。Deck Encoder 本体及 `deck_style` 直接通道不使用普通
Dropout：直接通道承担稳定构筑识别，稀有关键卡关系不应被逐步随机删除；FiLM 掩码只约束
更深层的可选调制。在线卡组多样性会降低对单一列表的记忆，但不能替代留出卡组族评估；
若后续确认直接通道过拟合，应采用整局可复现的结构化消融或同卡组变体一致性训练，而不是
让 PPO 更新时重新随机采样逐元素 Dropout。

3.11.1 建立独立事件协议 `TRANSITION_EVENT_FORMAT_VERSION=1`，从一次动作成功提交给 Core
开始，到下一决策、Retry 或终局结束，保留最近 16 个已完成状态转移。每项包含执行者、
回合/阶段、动作语义、来源/目标/效果槽/召唤方式、公开 Core 消息标签，以及 LP、双方八类
区域数量和连锁深度变化。取消、拒绝、选择完成、Retry、连锁无效与禁用均有确定性结果位。
事件本体、意图、来源和目标分别携带双玩家可见掩码；对手完全不可见的内部多选步骤会从其
视角序列中删除，而公开结果仍保留。

3.11.2 在视角脱敏之后，把上述事件编码为 19 个固定形状紧凑张量。来源卡按精确 Lua 效果槽
恢复静态语义，目标卡使用身份与全代码语义摘要；事件内先聚合最多 5 个目标，事件间再通过固定
时间位置、深度局部卷积和注意力池化保留顺序。空历史严格为零。最终事件向量以零初始化逐通道
门加入策略/价值共享 `v_input`：初始网络行为不变，但门在第一次反向传播即可学习开启。
相同张量合同贯穿中央推理、Worker/PPO 轨迹与 ONNX，且不改变奖励、合法动作或 GAE。

3.12.0 新增独立训练后验协议 `AUXILIARY_TARGET_FORMAT_VERSION=1`。网络继续只读取最近
16 项可见事件；训练 Worker 显式要求事件记录器在单局内保存完整的已完成事件引用，以动作返回的严格递增
序号对齐训练步，并在单局提交时生成下一决策、连锁结束、回合结束和真实 Core 终局四个
尺度的标签。标签包含有效掩码、累计 LP/双方八区域变化、事件数量，以及即时结果位、公开
消息组、边界、召唤方式、连锁深度、玩家视角胜负、终局原因和剩余事件数。异常局仍整体
回滚；超时截断或尚未完成的末动作使用缺失掩码，不以零值伪造答案。完整事件引用和构建
时前缀和均在局末释放，只有固定 187 字节/步的紧凑目标随 rollout 进入 Trainer；竞技场、
Link 和普通推理不启用完整事件留存。

这些目标位于观测之外，不进入中央推理、标准 ONNX、PPO 损失或模型协议哈希；Trainer
目前只把四尺度覆盖率及 Retry/取消/状态变化/连锁无效率写到 TensorBoard 的
`Auxiliary_Targets/*`。因此 3.12.0 不改变策略、价值、奖励、GAE、网络参数或部署行为，
其唯一目的，是在后续辅助梯度接入前验证时间对齐、可见性和标签分布。

3.12.1 将后验协议升至 `AUXILIARY_TARGET_FORMAT_VERSION=2`，新增每步执行者座位，并接入
训练专用 `StructuredAuxiliaryHeads`。辅助头以共享状态表示和本次已经执行的候选表示为输入，
通过独立 256 维 MLP 预测即时结果位/公开消息/边界、即时资源与连锁变化、连锁末/回合末/
终局资源变化、终局胜负和剩余事件量。分类目标分别使用 BCE 或交叉熵，连续目标使用
Smooth L1；每一时间尺度先按有效掩码归一化，因此缺失未来不参与 Loss。

辅助总损失以 `0.1` 系数加入 PPO，但没有任何预测值回到策略 logits 或 value。前 100 次
mini-batch 更新把共享表示完全 `detach`，只将辅助头作为探针训练；随后 900 次把返回共享
主干的梯度乘数从 0 线性升至 0.2；这是梯度缩放系数，不是辅助梯度占 PPO 梯度范数的
20% 保证。主干/辅助参数分别按 0.5/1.0 裁剪，但共享参数上的 PPO 与辅助梯度仍会相加，
可能协同也可能冲突。进度由检查点内已有 `train_step` 恢复，不会在续训时重置。

普通三输出前向不执行辅助 MLP；中央推理、竞技场、Link 与历史 ONNX 对手均走该路径。
标准 ONNX 包装器仍只返回 `action_logits` 和 `values`，导出器会裁掉未使用的辅助权重。
PTH 训练检查点则保存辅助参数以支持精确续训。训练曲线位于 `Auxiliary_Train/*`，原始标签
分布继续位于 `Auxiliary_Targets/*`。结构修订 12 用于阻止缺少辅助参数的开发检查点被误恢复。

3.12.2 将后验协议升至 `AUXILIARY_TARGET_FORMAT_VERSION=3`，为每步追加一个规划一致性配对
角色。目标构建器只把同一行动者、同一回合且同一阶段的相邻决策组成不重叠配对；PPO 再以
双样本单元打乱偶数 mini-batch，不拆开真实配对或重复样本；不足完整批次的随机尾部不更新。
这个一致性项只
约束两个 128 维规划向量的方向，以 `0.1` 权重进入已经乘 `0.05` 的规划损失，不把“同一
阶段”误当作“必须执行同一动作”，也不产生人工战术标签。

`GoalPlanner` 在每次前向从共享 `state_repr`、稳定 `deck_style` 和全局/阶段 `film_context`
分别投影后融合为 `plan_latent`。训练目标解码器使用已有缺失安全后验预测连锁末、回合末、
终局的资源摘要、真实终局胜负与剩余长度；前 100 次更新仍通过共享梯度桥保持主干隔离，
随后与 3.12.1 相同地在 900 次内开放至 0.2，但规划器自身始终可学习。规划参数与纯训练
解码参数独立于原 PPO 主干裁剪。

策略和值分别使用独立线性投影、`tanh` 零初始化逐通道门和 ±0.5 残差上界。零门保证初始
输出逐位不变，PPO 梯度先训练门，再逐步开放规划投影对决策的影响。这里没有跨决策保存的
离散 option、动作脚本或树搜索；状态变化后会重新规划。标准 ONNX 因此必须保留规划器和
两条有界接入路径，但仍只输出 logits/value，并会裁掉目标解码器和结构化辅助预测头。
TensorBoard 的 `Planning/*` 与 `Planning_Histograms/*` 只用于验证损失、门开启、配对覆盖
和 latent 是否坍塌；单个 latent 维度没有预定义的“进攻/防守/某卡组路线”语义。

### V4 玩家与离散全局状态

Core 的 `MSG_NEW_TURN` 独立维护 `turn_player`、首次出现的 `starting_player` 及双方回合数；
每个交互消息只更新 `decision_player`。编码器从当前行动方视角生成未知/己方/对方三类
角色，因此“对手回合由己方连锁响应”不会再被误认为己方回合。阶段使用 11 类有界
Embedding，并同时进入全局 token 和 FiLM；卡片、连锁与动作的区域/表示也使用离散
Embedding，不再把位掩码除以常数当作连续距离。

---

## 语义知识库

3.9.1 将 `[card_token, effect_slot]` 静态表注册为模型侧非参数缓冲。场面和卡组直接按
`card_idx/deck_idx` 查表；连锁与最近历史额外携带 1～8 的效果槽 ID，0 表示绑定未知并
回退整卡语义。该路径与 3.9.0 展开张量逐元素等价；三件套不完整、槽位/向量索引错位
或旧卡语义身份不一致时仍在建网前停止。3.9.2 后，已通过完整校验的编译运行资产可独立
替代源三件套建网；若选择携带源资产，则仍必须三件齐全且与编译表一致。

### 构建流程

```
Lua 脚本 (c12345678.lua)
         ↓
    ┌─────────────┐
    │ Lua Parser  │  ← 词法/块栈与静态引用解析
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ 效果分类    │  ← CATEGORY_XXX
    │ 发动条件    │  ← RACE/ATTR/SETCODE
    │ 特殊 Hash   │  ← 代码块聚类
    └──────┬──────┘
           ↓
    knowledge_base.json + hash_mapping_report.json
code_embeddings.npy + code_embeddings_idx.json
           ↓ 一次性编译
semantic_lookup_v1.npz + semantic_lookup_v1.json
```

### 语义特征结构

每张卡片最多 8 个效果槽，每个槽包含：

| 字段 | 维度 | 说明 |
|------|------|------|
| category | 8 | 效果类型 ID |
| requirements | 128 | 发动条件多热向量 |
| setcode | 4 | 关联字段 |
| numbers | 4 | 魔法数字参数 |
| ref_codes | 4 | 关联卡片 ID |
| race | 4 | 关联种族 |
| attr | 4 | 关联属性 |

效果槽除表中内容外还包含0～7的显式槽位身份。同步以完整语义源组为单位：知识库、Hash接续映射、代码矩阵/索引和新版 `code_embeddings_meta.json` 生成清单。3.13.4起接续按源码、依赖、编码器及分块策略校验；未变化槽复用，同键内容改变重新生成，无清单旧向量首次更新重建。不能把“键已存在”当成内容未变化。详见 [语义资产指南](code_semantics.md)。

3.6.1 在编码器初始化前交叉校验知识库中所有 0～7 效果槽、代码向量行和索引键。3.6.2 进一步在语义生成阶段按 `Effect.CreateEffect(c)` 对象追踪其 `SetDescription(aux.Stringid(...))`，把完整运行时 `desc` 绑定到同一个 Lua 代码语义槽。GameState、候选动作、连锁、历史和 `used_effect_mask` 共用这份映射；不能静态证明的动态写法回退到整卡语义。卡面描述文本与 Stringid 数值均不参与槽位定义。

### Hash 聚类算法

对于无法用标准 CATEGORY 分类的效果，使用代码块 Hash 进行聚类：

```python
def _hash_code_block(self, code_block):
    # 1. 词法规范化
    clean_code = code_block
    clean_code = re.sub(r'\b1\s*-\s*tp\b', '<OPPO>', clean_code)
    clean_code = re.sub(r'\b(tp|ep|rp)\b', '<PLAYER>', clean_code)
    clean_code = re.sub(r'\b\d+\b', '<NUM>', clean_code)
    # ...
    
    # 2. 计算 MD5 Hash
    hash_val = hashlib.md5(clean_code.encode()).hexdigest()[:8]
    
    return f"CUSTOM_HASH_{hash_val.upper()}"
```

---

## PPO 训练框架

各项训练/辅助/规划指标的准确统计口径与源码入口见 [TensorBoard 指标手册](tensorboard_metrics.md)，其中也区分训练拟合、实际更新和独立棋力评估。

3.13.0 起，`ppo_control.py` 统一默认 GAE lambda=0.98、最多 4 个 epoch、目标 KL=0.02。
CLI/WebUI 可显式对照 lambda=0.95/0.99，或以 target KL=0 关闭有限偏移阈值。每个小批
在反向前用已选动作的 old/new log-prob 估计 `mean(expm1(log_ratio)-log_ratio)`，超过
1.5×目标后停止该 rollout 剩余的策略、价值、辅助与规划更新；非有限偏移始终停止。
这不会额外调用网络，也不是严格信赖域或回滚机制。固定 mini-batch 及原尾部舍弃方式
保持不变，避免动态形状重新编译。

`PPO_Update/*` 按采集轮次记录实际更新数、完整/开始 epoch、有效样本复用次数、尾部数、
KL/clip fraction 的小批均值分位数和停止标志；它们不是单个动作的全分布分位数。
`Configuration/PPO` 记录当前配置、请求学习率与优化器真实学习率，续训仍按当前启动
参数选择 GAE/KL 等控制，不把文件内审计值自动当作命令行覆盖。`train_step` 只统计实际
优化器更新，FP16 GradScaler 跳步不会推进辅助梯度渐入日程。模型协议与网络均未修改。

### KL 提前停止如何解读

KL 衡量本轮采样时的旧策略与当前策略的分布偏移，不是胜率、Loss百分比或“学了多少知识”。
PPO clipping 不能保证整个动作分布始终贴近旧策略，因此还有提前停止保护，见
[OpenAI PPO 说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。

代码使用当前小批**已采样动作**的 log-prob，计算
`mean(expm1(new_log_prob-old_log_prob)-(new_log_prob-old_log_prob))`。
这是非负近似估计，不是整轮精确 KL，也没有额外调用网络。
默认目标 `target_kl=0.02`，实际阈值为 `1.5×0.02=0.03`。
检查发生在小批前向后、反向前；超过阈值或偏移非有限就退出本轮所有剩余 epoch。

已完成的参数更新保留，**不回滚模型、不结束训练、不把整轮样本判成异常局**。
策略、价值、辅助与规划共享网络，所以剩余联合更新一并停止；下一轮正常用新策略采集。
这会减少旧经验的复用次数，换取较小偏移和更新鲜的采样，并非零影响，但不会关闭正常学习。
它不保证绝不会策略退化，也不意味着已经收敛。

日志 `KL=0.03670 > 0.03000，已更新165小批` 表示保护正常触发。
若样本31,648、切片512、上限4个 epoch，每次遍历有61个完整小批、最多244次更新；
165次是两次完整遍历加第三次43个小批，有效复用率为 `165×512/31648≈2.67`。
尾部416条沿用固定批次舍弃；每次重新打乱，并非固定丢弃同一批局。
触发检查的小批不再反向，所显示 KL 也不是更新该小批后的再次评估。

在第3轮或更早触发不代表异常：保护针对每轮偏移，不需要等训练很久才生效。
在 **训练流形图 → TensorBoard → Scalars** 结合固定竞技场基准观察：

| 指标 | 解读 |
| --- | --- |
| `PPO_Update/Effective_Epochs`、`Optimizer_Updates` | 实际样本复用次数和更新数 |
| `PPO_Update/KL_Early_Stop`、`Approx_KL_P95`、`Approx_KL_Max` | 停止频率和小批平均偏移，不是棋力 |
| `PPO_Update/Clip_Fraction_P95` | 比率越过裁剪区间的样本比例 |
| `PPO_Update/Nonfinite_Batches` | 应保持0，数值异常与有限 KL 停止分开排查 |

若持续在第一或少数小批就停、有效 epoch 很低，或同时出现非有限值/基准退化，再检查
old log-prob、采样/更新条件和学习率；不能仅凭一次日志放宽阈值。
`Configuration/PPO` 文本页保留当前参数。保留显存与 CPU 合并池跨轮复用本身不证明泄漏，
应比较预热后多个轮次同阶段的量；Worker回收后的低值也不代表 PPO 的反向峰值。

### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                      PPO 训练循环                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 多进程采集 (Rollout Collection)                         │
│     ┌─────────┐  ┌─────────┐  ┌─────────┐                  │
│     │Worker 1 │  │Worker 2 │  │Worker N │                  │
│     │自对局60%│  │历史25% │  │RuleBot15%│                  │
│     └────┬────┘  └────┬────┘  └────┬────┘                  │
│          │            │            │                        │
│          └────────────┼────────────┘                        │
│                       ↓                                     │
│  2. 经验汇总 (Memory Aggregation)                           │
│     ┌─────────────────────────────────────┐                │
│     │ obs, actions, rewards, log_probs    │                │
│     │ + GAE 优势估计                       │                │
│     └──────────────────┬──────────────────┘                │
│                        ↓                                    │
│  3. 策略优化 (Policy Update)                                │
│     for epoch in range(4):                                  │
│         for mini_batch in shuffle(memory):                  │
│             ┌─────────────────────────────┐                │
│             │ PPO Clip Loss               │                │
│             │ Value Loss                  │                │
│             │ Entropy Regularization      │                │
│             └─────────────────────────────┘                │
│                        ↓                                    │
│  4. 模型保存 (每 10 轮)                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 联盟训练机制

为防止策略退化，采用混合对手训练：

| 对手类型 | 比例 | 目的 |
|----------|------|------|
| 自对局 | 60% | 追求当前最优策略 |
| 历史模型 | 25% | 防止遗忘旧策略 |
| RuleBot | 15% | 基准锚点，确保基本能力 |

对手类型先按 Worker 权重分配。若本轮没有抽到 RuleBot，系统会轮换选择一个 AI
Worker，将它的首局临时改为 RuleBot 对局，提交后恢复原本的 self/hist 配置。这样既保留
加权随机性，也避免 Worker 数量较少时连续多轮缺少规则基准局。没有历史检查点时，hist
权重会自然回落到 self。

上表是默认联盟。3.15.4显式开启WindBot时，先按配置概率为Worker分配外部对手，其余
Worker继续按上表的条件比例抽样；混合模式优先由普通AI Worker承担Rule补局，100% WindBot
则不插入Rule局。外部对手逐局复用Worker已有Core，只有模型自己的候选动作、log-prob与
价值进入原PPO/GAE；WindBot响应只是环境转移。接口与构筑留出说明见
[WindBot训练对手](../integrations/windbot/README.md#3154主训练对手)。

### 中央批量推理服务

```
Worker 1 ──┐
Worker 2 ──┼──> ZMQ 请求 ──> CPU/CUDA 中央推理 ──> 共享结果槽 ──┬──> Worker 1
Worker 3 ──┘                                              ├──> Worker 2
                                                          └──> Worker 3
```

**优势**：

- Worker 固定使用 CPU；当前策略与 self 对手不创建重复的本地网络或 CUDA 上下文
- 多个 Worker 固定共享一个中央推理服务
- `device=auto/cpu/cuda` 只决定中央推理和 PPO 更新设备
- `auto` 优先使用可用 CUDA，否则回落 CPU；显式 `cuda` 在不可用时会提前拒绝启动
- ZMQ 只传递 Worker/请求编号，观测与结果通过共享内存槽交换，并校验 64 位完成号
- Windows 会在拉起 Worker 前校验系统提交内存余量，资源不足时提前停止而不是让原生通讯库崩溃

---

## OCGCore 环境封装

### 消息解析

`gamestate.py` 是整个项目最核心的文件，负责解析 OCGCore 发送的二进制消息：

```python
class MessageParser:
    def parse_message(self, msg_type, data):
        if msg_type == MSG_SELECT_IDLECMD:
            return self._parse_idle_cmd(data)
        elif msg_type == MSG_SELECT_CHAIN:
            return self._parse_chain(data)
        elif msg_type == MSG_SELECT_CARD:
            return self._parse_select_card(data)
        # ... 100+ 种消息类型
```

### 状态同步

状态快照会通过随包 Core 的公开旧式查询接口校准场、手牌、墓地和除外区，并以消息流维护回合、LP、连锁和禁用区域。采用新式 API 的上游 Core 版本使用结构化 `OCG_DuelQuery`；未来升级必须整体适配公开 ABI，而不是给本地二进制追加私有字段。

当前 `galatea_env.py` 绑定的是 `create_duel/process/get_message/set_response*/query_card` 旧式导出，不能直接用新版 `OCG_DuelProcess/OCG_DuelGetMessage/OCG_DuelSetResponse/OCG_DuelQuery` 二进制覆盖。新版还增加 64 位种族、TLV 查询字段及 Type 121/122/123/180/190 等消息。升级必须新增公开双后端适配器并完成 Windows/Linux 回归；在该迁移完成前，应继续使用随包公开旧式 Core。

```python
class DuelState:
    def __init__(self):
        self.turn_count = 0
        self.phase = 0
        self.current_player = 0
        self.lp = [8000, 8000]
        self.entities = []           # 场上所有卡片实体
        self.chain_stack = []        # 连锁堆栈
        self.history_stack = []      # 历史动作记录
        self.known_hand_codes = [[], []]  # 记牌器
```

---

## 性能优化技术

### 1. CUDA 混合精度训练

```python
# 自动检测硬件能力
if torch.cuda.is_bf16_supported():
    self.amp_dtype = torch.bfloat16  # Ampere+ 架构
else:
    self.amp_dtype = torch.float16   # 老显卡

# 使用 AMP 上下文
with torch.amp.autocast('cuda', dtype=self.amp_dtype):
    logits, values, v_input = self.agent.net(batch)
```

CPU 模式使用 FP32，不启用 CUDA autocast、GradScaler 或锁页内存。

#### CUDA 语义直接聚合（2026-10-10）

`GalateaNet.process_semantics()` 的类别、关联字段、卡片引用、种族和属性仍逐效果槽求和；
CUDA 使用同一个 Embedding 权重的 `embedding_bag(mode='sum')`，避免先生成全部引用向量。
它不是均值、去重、裁剪或静态缓存：重复引用计数不变，八个效果槽、代码向量、查询与掩码不变，
也不冻结任何可训练权重。非零 padding 行仍参与前向，其贡献通过 detached 行补回，padding
梯度仍为零。CPU Worker、CPU 训练、ONNX 导出及非默认 Embedding 配置沿用原算子。

混合精度没有改变原 FP32 Embedding 求和。512/75/8/4/512 形状的前向峰值增量实测从
2.9297 GiB 降到 0.6123 GiB，消除了单个 2.34375 GiB 的引用展开；这不是整网峰值降幅。
数学目标和梯度路径保持一致，但不同 GPU/算子归约顺序可能存在浮点末位差异，不保证所有
训练轨迹逐位重现。Model Protocol 4 / Checkpoint Format 3 / schema revision 14、参数名称/
形状、语义资产身份和 PPO/GAE/辅助/规划目标不变，已有兼容 V4 检查点可以继续恢复，无需重建资产。

本机 RTX 5070 Ti 16 GiB、Torch 2.9.1+cu130、512/8/6、BF16、未编译，在包含真实语义卡片的
合成观测/标签上通过现有完整 PPO 前向、反向和 Adam 更新；每种批量重复三次。建立优化器状态后：

| mini-batch | 原实现活动显存峰值 | 直接聚合活动显存峰值 | 更新结束后的保留显存：原 → 新 |
|---|---:|---:|---:|
| 128 | 7.2790 GiB | 7.3387 GiB | 8.3594 → 7.4570 GiB |
| 256 | 13.8014 GiB | 13.9213 GiB | 15.1895 → 14.2051 GiB |

直接聚合需要反向索引/分组元数据，本测试整网活动峰值约高 0.9%，并未明显降低；收益主要是
去掉语义阶段的大块临时分配及减少缓存保留压力。每次释放梯度后活动显存稳定在约 0.67 GiB，
没有连续更新累积；保留显存不是活动张量或泄漏。不能据此保证云端 32 GiB 的 mini-batch 512
能完成整个 PPO。此修补不自动改批量、不新增梯度累积或语义梯度检查点。

开发回归覆盖重复引用、非零 padding、越界引用原有截取、空槽、混合精度、有限梯度、真实
CUDA PPO、ONNX/Core 与输入/参数/协议身份。512/8/6 BF16 夹具的所有输出及 307 个有梯度
参数与原算子完全对齐；这不是长期学习强度保证。`tests/benchmark_semantic_aggregation.py`
可复测完整 PPO，测试目录仍遵循现有忽略约定。编译检查仅覆盖 Dynamo eager 完整图，未宣称
验收 Linux Inductor。用户已报告相同32 GiB云端配置修补后短期暂时稳定；仍需多场景峰值、
50轮趋势及生产长训/棋力验收，不能以短期反馈代替。

### 2. 静态内存池

首轮合并轨迹时按固定上限分配 CPU 内存池，后续轮次从索引 0 覆盖有效样本前缀，避免 Windows
反复申请与释放数 GiB 连续内存造成提交量阶梯式增长。PPO 始终只读取本轮 `total_steps` 范围，
不会读到上一轮残留尾部；内存告急时预检可释放已完成训练的旧池并复检，最终由 `close()` 统一释放：

```python
class PPOTrainer:
    def merge_rollouts(self, first_block):
        # 首轮延迟分配，后续轮次复用并覆盖有效区
        self.merged_memory = {
            'action': torch.empty(self.max_buffer_steps, ...),
            'log_prob': torch.empty(self.max_buffer_steps, ...),
            # ...
        }
```

训练器会在 Worker 启动前、Worker 回收后、轨迹合并后和 PPO 更新后输出提交量、RSS、合并池及
CUDA 内存快照；Worker 初始化对手后也会记录一次进程内存，便于区分 hist/ONNX 单轮峰值和主进程跨轮常驻量。

### 3. 共享内存槽与中央权重

当前策略与 self 对手不在 Worker 内加载 PyTorch 权重；它们通过 ZMQ 请求中央模型，并以
`share_memory_()` 创建的固定槽位交换观测、动作结果、完整 logits 和请求完成号：

```python
# Worker：只发送请求身份，观测已写入对应共享槽
socket.send(encode_inference_request(worker_id, request_id))

# 高频观测与结果通过预分配共享内存槽传递
shared_tensor.share_memory_()
```

self 对手使用中央模型；hist 对手只从同一 `model_id` 的正式制品池选择，优先挂载同轮次完整
ONNX，缺失或失效时才延迟加载 `.pth` 回退网络。

### 4. TF32 加速

在 Ampere+ 架构 GPU 上启用 TF32：

```python
if torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 5. ZMQ 请求路由与共享内存通信

使用 ZeroMQ ROUTER 路由请求，并将大体积 Tensor 保留在共享内存中，降低序列化和复制开销：

```python
context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # ROUTER 模式支持路由式微批处理
socket.setsockopt(zmq.ROUTER_HANDOVER, 1)  # 自动负载均衡
```

**核心优势**：
- **微批处理**：多个 Worker 的推理请求被聚合为一个批次，一次性送入所选训练设备，减少调度开销
- **按设备优化**：CUDA 使用 pinned memory 和 `non_blocking=True`；CPU 使用普通内存与同步复制
- **超时隔离**：请求携带轮次与局部序号；超时或响应号不匹配时重建 Worker Socket，不读取陈旧结果

### 6. ONNX 推理加速 (v3.2.0 新增)

启用 `--use_onnx` 后，在每 10 轮检查点保存点同步导出 ONNX。Worker 只对历史对手使用
ONNX Runtime；当前策略和 self 对手继续通过中央推理服务执行：

```python
# 训练主进程同步导出 ONNX
class ONNXWrapper(torch.nn.Module):
    def __init__(self, net, keys):
        super().__init__()
        self.net = net
        self.keys = keys

    def forward(self, *args):
        # 将展平输入重组为字典，传给原网络
        batch_dict = {k: v for k, v in zip(self.keys, args)}
        logits, values, _ = self.net(batch_dict)
        return logits, values

# Worker 校验主图、外置权重、UUID 和轮次后加载历史 ONNX
```

**核心优势**：
- **完整制品组**：`.onnx`、其引用的 `.onnx.data` 与 `.artifacts.json` 一并保存和鉴权
- **输入类型适配**：根据 ONNX Runtime 会话声明转换 FP16/FP32 等输入类型
- **延迟安全回退**：ONNX 不完整、身份不匹配或运行失败时，才加载历史 PTH 执行 CPU 推理；正常 ONNX 路径不会同时常驻两套历史网络

---

## 下一步

- 📝 查看 [更新日志](changelog.md) 了解版本变化
- 📚 阅读 [功能详解](features.md) 了解使用方法
- 🚀 返回 [快速上手](quickstart.md) 开始训练
