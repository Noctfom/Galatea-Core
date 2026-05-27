# 🔧 架构设计

> 本文档深入介绍 Galatea-Core 的技术架构与核心算法，适合想要深入了解或参与开发的用户。

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
        self.feat_proj = nn.Linear(58, d_model)                   # 数值特征投影
        self.race_embed = nn.Embedding(30, d_model)               # 种族嵌入
        self.attr_embed = nn.Embedding(10, d_model)               # 属性嵌入
        self.setcode_embed = nn.Embedding(4096, d_model)          # 字段嵌入
        
        # 2. 语义解析皮层
        self.sem_cat_embed = nn.Embedding(4000, d_sem)            # 效果类型嵌入
        self.sem_req_proj = nn.Linear(128, d_sem)                 # 发动条件投影
        self.sem_fusion_proj = nn.Sequential(...)                 # 语义融合

        # 3. FiLM 全局调制器
        self.film_gen = FiLMGenerator(condition_dim=15, d_model=d_model)

        # 4. Transformer Encoder (每层含 FiLM 调制 + SwiGLU 门控)
        self.transformer = GalateaTransformerStack(d_model, n_heads, n_layers)

        # 5. 位置编码
        self.chain_pos_embed = nn.Parameter(...)                  # 连锁堆栈位置编码(12槽)
        self.history_pos_embed = nn.Parameter(...)                # 历史动作位置编码(8槽)
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
│ (v_input)   │                   │ 效果描述    │
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
| 卡片特征 | 58 | 数值属性 + 类型掩码 + 连接箭头 |
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
]
# + 32 维类型掩码 (怪兽/魔法/陷阱/效果/融合/同调/超量/灵摆/连接...)
# + 9 维连接箭头
```

### 动作编码

```python
act_dict = {
    'act_card_idx': [...],   # 目标卡片索引 [80, 5]
    'act_type': [...],       # 动作类型
    'act_desc': [...],       # 效果描述 Hash
    'act_mask': [...],       # 有效动作掩码
    'act_race': [...],       # 宣言种族
    'act_attr': [...],       # 宣言属性
    'act_code': [...],       # 宣言卡片
    'act_place': [...],      # 放置位置
}
```

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
    knowledge_base.json
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

### 异步推断服务器

```
Worker 1 ──┐
Worker 2 ──┼──> 请求队列 ──> GPU 推断服务器 ──> 响应队列 ──┬──> Worker 1
Worker 3 ──┘                                              ├──> Worker 2
                                                          └──> Worker 3
```

**优势**：
- Worker 进程不需要加载模型到显存
- 多个 Worker 共享同一个 GPU 推理服务
- 显存占用降低 70%+

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

### 1. 混合精度训练

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

### 2. 静态内存池

预分配固定大小的内存，避免训练过程中的内存碎片：

```python
class PPOTrainer:
    def __init__(self):
        # 预分配内存池
        self.max_buffer_steps = self.update_timesteps + (self.num_workers * 1000)
        self.merged_memory = {
            'action': torch.empty(self.max_buffer_steps, ...),
            'log_prob': torch.empty(self.max_buffer_steps, ...),
            # ...
        }
```

### 3. 权重共享

Windows Spawn 模式下，使用 `share_memory_()` 让所有子进程共享同一份模型权重：

```python
# 主进程
weights = model.state_dict()
for v in weights.values():
    v.share_memory_()

# 子进程直接使用共享内存中的权重
```

### 4. TF32 加速

在 Ampere+ 架构 GPU 上启用 TF32：

```python
if torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### 5. ZMQ 零拷贝进程通信 (v3.2.0 新增)

使用 ZeroMQ ROUTER 微批处理架构替换原有管道通信，大幅降低采集延迟：

```python
context = zmq.Context()
socket = context.socket(zmq.ROUTER)  # ROUTER 模式支持路由式微批处理
socket.setsockopt(zmq.ROUTER_HANDOVER, 1)  # 自动负载均衡
```

**核心优势**：
- **微批处理**：多个 Worker 的推理请求被聚合为一个批次，一次性送入 GPU，减少调度开销
- **锁页内存 DMA**：数据预置于 pinned memory (`tensor.pin_memory()`)，通过 `non_blocking=True` 实现 CPU↔GPU 异步搬运，不阻塞主线程
- **采集时间减半**：相比旧管道方案，整体数据采集耗时减少约 50%

### 6. ONNX 推理加速 (v3.2.0 新增)

训练时自动将 PyTorch 模型导出为 ONNX 计算图，Worker 端通过 ONNX Runtime 进行极速推理：

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

# Worker 端加载 ONNX 模型进行推理，无需导入 PyTorch 模型
```

**核心优势**：
- **零显存推理**：Worker 使用 ONNX Runtime 进行 CPU 推理，完全不占用 GPU 显存
- **图优化**：ONNX 导出时进行计算图级融合优化，消除冗余算子
- **提速显著**：在中低端 GPU 上，数据采集阶段整体吞吐量提升 30%+

---

## 下一步

- 📝 查看 [更新日志](changelog.md) 了解版本变化
- 📚 阅读 [功能详解](features.md) 了解使用方法
- 🚀 返回 [快速上手](quickstart.md) 开始训练
