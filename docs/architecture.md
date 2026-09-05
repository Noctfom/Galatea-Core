# 🔧 架构设计

> 本文档深入介绍 Galatea-Core 的技术架构与核心算法，适合想要深入了解或参与开发的用户。

> 文档适用于 **Galatea-Core v3.6.3**。

> 💡 **框架的独特处理逻辑**（语义化模块、142宣言池、多选题组块包装、记牌器、卡组权重、伪装池）请参考 [特殊处理逻辑文档](special_handling.md)。

---

## 📋 目录

- [系统架构概览](#系统架构概览)
- [神经网络架构](#神经网络架构)
- [特征编码系统](#特征编码系统)
- [语义知识库](#语义知识库)
- [PPO 训练框架](#ppo-训练框架)
- [OCGCore 环境封装](#ocgcore-环境封装)

---

## 系统架构概览

Galatea-Core 采用模块化设计，由以下核心子系统构成：

```
┌─────────────────────────────────────────────────────────────────┐
│                         应用层 (Application)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   WebUI     │    │  Trainer    │    │   Arena     │         │
│  │  (app.py)   │    │(trainer.py) │    │(model_vs.py)│         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
├─────────┴──────────────────┴──────────────────┴─────────────────┤
│                         认知层 (Cognition)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GalateaNet                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │ Embedding │  │Transformer│  │  Policy   │            │   │
│  │  │   Layer   │→ │  Encoder  │→ │   Head    │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘            │   │
│  │                                 ┌───────────┐            │   │
│  │                                 │  Value    │            │   │
│  │                                 │   Head    │            │   │
│  │                                 └───────────┘            │   │

│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                         感知层 (Perception)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Feature    │    │  Semantic   │    │    Card     │         │
│  │  Encoder    │    │     KB      │    │   Reader    │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
├─────────┴──────────────────┴──────────────────┴─────────────────┤
│                         环境层 (Environment)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GalateaEnv                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │  Message  │  │   Duel    │  │   Action  │            │   │
│  │  │  Parser   │  │   State   │  │  Handler  │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    OCGCore (DLL)                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 核心文件说明

| 文件 | 层级 | 功能 |
|------|------|------|
| `galatea_net.py` | 认知层 | Transformer 神经网络定义 |
| `feature_encoder.py` | 感知层 | 游戏状态特征编码 |
| `semantic_kb.py` | 感知层 | 语义知识库查询 |
| `galatea_env.py` | 环境层 | OCGCore 环境封装 |
| `gamestate.py` | 环境层 | 游戏状态解析（核心！） |
| `trainer.py` | 应用层 | PPO 训练器 |
| `worker.py` | 应用层 | 多进程数据采集 |

---

## 神经网络架构

### GalateaNet 结构

GalateaNet 是一个基于 Transformer Encoder 的策略-价值网络，v3.2.0 引入了 FiLM 全局调制与 SwiGLU 门控前馈：

```python
class GalateaNet(nn.Module):
    def __init__(self, config):
        # 1. 基础物理感知层
        self.card_embed = nn.Embedding(vocab_size, d_model)      # 卡片 ID 嵌入
        self.feat_proj = nn.Linear(66, d_model)                   # 数值特征投影
        self.race_embed = nn.Embedding(30, d_model)               # 种族嵌入
        self.attr_embed = nn.Embedding(10, d_model)               # 属性嵌入
        self.setcode_embed = nn.Embedding(4096, d_model)          # 字段嵌入
        
        # 2. 语义解析皮层
        self.sem_cat_embed = nn.Embedding(4000, d_sem)            # 效果类型嵌入
        self.sem_req_proj = nn.Linear(128, d_sem)                 # 发动条件投影
        self.effect_slot_embed = nn.Embedding(8, d_model)         # 效果槽身份
        self.sem_fusion_proj = nn.Sequential(...)                 # 语义融合

        # 3. FiLM 全局调制器
        self.film_gen = FiLMGenerator(condition_dim=15, d_model=d_model)

        # 4. Transformer Encoder (每层含 FiLM 调制 + SwiGLU 门控)
        self.transformer = GalateaTransformerStack(d_model, n_heads, n_layers)

        # 5. 顺序上下文编码
        self.chain_context_pool = OrderedContextPool(d_model, 12) # 连锁先后关系
        self.history_context_pool = OrderedContextPool(d_model, 8)# 最近发动先后关系
        self.place_weights = buffer([1.0, 0.8, 0.6, 0.4, 0.2])   # 排序操作加权
        
        # 6. 输出头 (SwiGLU 门控)
        self.policy_head = SwiGLU(d_model*2) → Linear(1)         # 策略头
        self.value_head = SwiGLU(d_model) → Linear(1)            # 价值头
```

### FiLM 全局状态调制 (v3.2.0 新增)

FiLM (Feature-wise Linear Modulation) 根据当前游戏阶段/回合/LP 等全局信号，动态调整 Transformer 每层的推理倾向：

```python
class FiLMGenerator(nn.Module):
    def __init__(self, condition_dim, d_model):
        self.proj = nn.Linear(condition_dim, 2 * d_model)  # 输出 γ 和 β
        nn.init.zeros_(self.proj.weight)  # 零初始化确保训练初期不干扰

    def forward(self, condition):
        out = self.proj(condition)
        gamma, beta = out.chunk(2, dim=-1)  # 各取一半
        return gamma.unsqueeze(1), beta.unsqueeze(1)

# 在 Transformer Block 中的应用：
x = x * (1.0 + gamma) + beta  # 先做 LayerNorm 再进行 FiLM 调制
```

**设计思想**：不同游戏阶段的决策逻辑截然不同（起手展开 vs 中盘拉扯 vs 斩杀计算），FiLM 让同一个网络能根据全局状态自动切换"思考模式"，无需增加额外的分支网络。

### SwiGLU 门控前馈网络 (v3.2.0 新增)

传统 MLP (Linear→ReLU→Linear) 被全面替换为 SwiGLU 门控线性单元：

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
- **无偏置设计**：所有 Linear 层去除 bias，减少参数量并提升训练稳定性
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

GalateaCore 主要通过 **熵正则化** 和 **历史模型对战联盟** 来驱动探索行为，确保 AI 既不会过早收敛到局部最优，也不会遗忘已经学到的策略。

## 特征编码系统

### 编码维度

| 特征类型 | 维度 | 说明 |
|----------|------|------|
| 全局特征 | 15 | 回合数、阶段、LP、各区域卡片数 |
| 卡片特征 | 66 | 数值属性 + 已发动效果位 + 类型掩码 + 连接箭头 |
| 语义特征 | 128×8 | 每张卡最多 8 个效果槽 |

### 卡片特征详解

```python
feat_numeric = [
    owner,              # 控制者 (1.0/-1.0)
    location / 100.0,   # 位置
    sequence / 10.0,    # 序列
    current_atk / 4000, # 当前攻击力
    current_def / 4000, # 当前防御力
    base_atk / 4000,    # 基础攻击力
    base_def / 4000,    # 基础防御力
    pos_x, pos_y,       # 场上坐标
    level / 12.0,       # 等级/阶级
    lscale / 13.0,      # 左灵摆刻度
    rscale / 13.0,      # 右灵摆刻度
    position / 10.0,    # 表示形式
    is_public,          # 是否公开
    overlay_count / 5,  # 超量素材数
    counter_count / 10, # 指示物数
    is_equipped,        # 是否装备
    used_effect_mask[0:8], # 本回合已发动的效果槽位
]
# + 32 维类型掩码 (怪兽/魔法/陷阱/效果/融合/同调/超量/灵摆/连接...)
# + 9 维连接箭头
```

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
    'act_response': [...],   # 语义响应值
    'act_signature': [...],  # 完整动作语义的 4 字节稳定签名
    'act_context': [...],    # min/max/结果数量/完成与取消条件 [120, 6]
    'act_target_code': [...],# 隐藏区或宏动作目标卡密 [120, 5]
    'act_target_value': [...],# 祭品值/双星级/指示物分配 [120, 5, 2]
    'act_controller': [...], # 行动方视角控制者
    'act_location': [...],   # 引擎区域
    'act_sequence': [...],   # 区域内序号
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

动作协议 V2 不把 `GameAction.index` 当作学习语义。`index` 与 `decision_bytes` 只负责把最终选择翻译回 Core；策略头看到的是操作、卡密、位置、约束和结果集合。Type 26 保持 Core 原生的逐次 Select/Unselect 过程，每一步都生成新快照并进入轨迹；Type 15/20/22/23/25 等静态组合消息则先由合法性枚举器生成完整响应，再由策略头选择。

`MODEL_PROTOCOL_VERSION` 独立于框架版本和检查点容器版本维护。它同时写入 PTH 顶层、`net_config`、模型状态、ONNX 元数据与制品清单；版本不同表示输入张量或动作头权重不兼容，加载器会直接拒绝。

---

## 语义知识库

### 构建流程

```
Lua 脚本 (c12345678.lua)
         ↓
    ┌─────────────┐
    │ Lua Parser  │  ← 正则表达式提取
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

效果槽除表中内容外还包含 0～7 的显式槽位身份。GitHub 同步会以同一目录中的四个语义资产为一组：结构化知识库负责可解释字段，Hash 映射用于继续聚类，代码语义矩阵及索引用于恢复已有 Lua 向量。向量生成器会在资产一致时只追加新效果槽；这属于真正的增量接续，而不是重新编码全部旧脚本。

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
