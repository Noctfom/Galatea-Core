# 🧬 框架特殊处理逻辑

> 本文档详细介绍 Galatea-Core 中为解决框架固有局限性而特殊构建的模块，这些模块是框架的核心竞争力之一。

> 文档适用于 **Galatea-Core v3.5.0**。

---

## 📋 目录

- [语义化模块（Semantic KB）](#语义化模块semantic-kb)
- [142 宣言池包装逻辑](#142-宣言池包装逻辑)
- [多选题组块包装逻辑](#多选题组块包装逻辑)
- [简易记牌器逻辑](#简易记牌器逻辑)
- [卡组权重调整（Global Weights）](#卡组权重调整global-weights)
- [虚拟伪装池模块（Virtual Mix Pools）](#虚拟伪装池模块virtual-mix-pools)

---

## 语义化模块（Semantic KB）

### 为什么需要它

游戏王有超过 10,000 张卡片，每张卡都有独特的效果。如果 AI 只能通过卡片 ID 来识别卡片，它将无法理解"两张不同的卡片具有相似效果"这一概念。

例如"灰流丽"和"效果遮蒙者"都能无效怪兽效果，但它们的卡片密码完全不同。传统方法下，AI 需要分别学会应对这两张卡片，无法举一反三。

### 工作原理

语义化模块通过解析每张卡片的 Lua 脚本，提取出标准化的语义特征向量：

```
Lua 脚本 → Lua Parser → 效果分类 (CATEGORY_XXX)
                        → 发动条件 (RACE/ATTR/SETCODE)
                        → 特殊 Hash (代码块聚类)
                                 ↓
                        knowledge_base.json
```

每张卡片最多提取 **8 个效果槽**，每个效果槽包含：

| 字段 | 含义 | 示例 |
|------|------|------|
| `category` | 效果类型 | 破坏、无效、检索、特殊召唤... |
| `requirements` | 发动条件 | 需要场上有特定种族、特定阶段... |
| `setcode` | 关联字段 | 影依、英雄、烙印... |
| `numbers` | 魔法数字 | ATK 变化量、等级变化量... |
| `ref_codes` | 关联卡片 | 检索目标的卡片密码... |
| `race` | 关联种族 | 龙族、魔法师族... |
| `attr` | 关联属性 | 光、暗、水... |

### Hash 聚类算法

对于无法用标准 CATEGORY 分类的效果，使用代码块 Hash 进行聚类：

```python
def _hash_code_block(self, code_block):
    # 1. 词法规范化：将具体数值替换为占位符
    clean_code = re.sub(r'\b1\s*-\s*tp\b', '<OPPO>', code_block)
    clean_code = re.sub(r'\b(tp|ep|rp)\b', '<PLAYER>', code_block)
    clean_code = re.sub(r'\b\d+\b', '<NUM>', code_block)
    
    # 2. 计算 MD5 Hash
    hash_val = hashlib.md5(clean_code.encode()).hexdigest()[:8]
    
    return f"CUSTOM_HASH_{hash_val.upper()}"
```

这意味着：如果两张卡的某个效果代码结构相似（仅仅是参数不同），它们会被归为同一个 Hash 标签，AI 可以直接复用经验。

### 在 WebUI 中操作

在 **🧠 语义知识库引擎** 模块中：
1. 勾选 **🌐 从 Github 拉取基础卡库同步**（首次使用必须）
2. 点击 **开始提取卡片语义**
3. 等待解析完成

---

## 142 宣言池包装逻辑

### 问题背景

OCGCore 在某些效果（如"禁止令"、"抹杀之指名者"）下会发送 `MSG_ANNOUNCE_CARD` (type=142)，要求玩家从数万张卡片中宣言一张。如果直接把数万个合法选项丢给神经网络，动作空间的维度会瞬间爆炸，导致显存溢出和训练无法收敛。

### 解决方案

Galatea-Core 构建了一套 **三层过滤机制**，将数万候选压缩到 AI 可理解的几十个选项：

#### 第一层：RPN 逆波兰表达式过滤

OCGCore 发送的 142 消息中包含一串逆波兰表达式（RPN）操作码，描述了合法卡片的条件（如"必须是龙族"、"必须是等级 4 以下"等）。

框架实现了完整的 RPN 虚拟机：

```python
# 支持的 RPN 操作码
OP_ISCODE     = 0x40000100  # 是否匹配特定卡密
OP_ISSETCARD  = 0x40000101  # 是否匹配字段
OP_ISTYPE     = 0x40000102  # 是否匹配卡片类型
OP_ISRACE     = 0x40000103  # 是否匹配种族
OP_ISATTRIBUTE= 0x40000104  # 是否匹配属性
OP_ISLEVEL    = 0x40000105  # 是否匹配等级
OP_ISLINK     = 0x40000107  # 是否匹配连接值
OP_AND        = 0x40000004  # 逻辑与
OP_OR         = 0x40000005  # 逻辑或
OP_NOT        = 0x40000007  # 逻辑非
```

#### 第二层：常识候选池收集

框架从以下来源收集可能合法的卡片候选：

1. **自身卡组/额外卡组**：你自己的卡组里的卡最可能被宣言
2. **已知手牌**：通过记牌器追踪到的对手手牌
3. **公开情报**：场上、墓地、除外区中公开表示的卡片
4. **泛用卡池**（meta_staples.json）：环境中最常见的泛用卡（灰流丽、增殖的G等）

#### 第三层：优先级排序

收集到的候选卡片按重要性排序：

```python
def get_priority_score(c):
    score = 0
    if c in my_cards: score += 100     # 自己卡组最高优先级
    if c in known_hand_codes: score += 50  # 已知对手手牌
    if c in public_zones: score += 50      # 公开区域
    if c in meta_staples: score += 10      # 环境泛用卡
    return score
```

### 防崩溃回退机制

如果 RPN 解析有瑕疵导致候选池被全灭，框架会自动回退到全候选池，交给 RuleBot 强行穷举，保证程序不会崩溃。

### 在 WebUI 中操作

在 **🗃️ 资产与卡组管理 → 🃏 泛用卡池配置 (142 兜底)** 中：
- 可以添加/移除环境常用的泛用卡
- 模板默认包含：灰流丽、增殖的G、墓穴的指名者、泡影等

---

## 多选与顺序选择逻辑

### 问题背景

OCGCore 有两类容易混淆的选择协议：Type 15/20/22/23 等消息要求客户端**一次返回完整组合**；Type 26 (`MSG_SELECT_UNSELECT_CARD`) 才会在每次 Select/Unselect 后由引擎重新计算候选并发出下一条消息。二者不能使用同一种静态包装方式。

### 静态组合：宏动作（Macro Action）

对一次提交完整响应的消息，合法性枚举器先构造可直接发给 Core 的候选套餐，策略网络再从套餐中选择：

```
Core 请求：从 {A, B, C} 中选择 2 张
合法性枚举器：{A,B} / {A,C} / {B,C}
策略网络：选择一个完整套餐
响应封包：count + 原始候选索引列表
```

宏动作不仅保存 `decision_bytes`，还向模型公开目标卡密、顺序、位置和规则数值：

```python
action.macro_targets = [...]        # 可见实体索引
action.macro_target_codes = [...]   # 隐藏区/卡组选项卡密
action.macro_target_values = [...]  # 祭品值、双星级、指示物分配
action.macro_places = [...]         # 格子组合
action.decision_bytes = b'...'      # Core 原始完整响应
```

Type 20 会按 Core 的 `release_param` 总和判定祭品值，不再按选中卡片张数近似；Type 140/141 的多种族/多属性宣言也会组合成恰好包含 `count` 个 bit 的整数响应。

### Type 26：引擎原生顺序决策

Type 26 不做静态终局组合。任意 Lua `special_check` 只有引擎知道，单个数据包不足以可靠枚举所有终局集合；强行用 RuleBot 遍历会改变训练主体，并可能遗漏合法路径。

框架因此保留原生流程：

```
快照 N：当前已选 {A}，可 Select B/C，可 Unselect A，可 Finish
模型动作：Select B
Core 校验并执行 Lua 约束
快照 N+1：当前已选 {A,B}，候选和 finishable 状态重新生成
模型动作：Finish
```

每个动作都编码 Select/Unselect/Finish/Cancel 语义、动作后的完整已选集合、候选卡密与位置、min/max、finishable/cancelable，并单独写入 PPO 轨迹。网络虽然没有循环隐状态，但观测已包含当前选择状态，终局奖励会通过 GAE 回传到同一连贯过程中的各步，所以完成这种动作不依赖 MCTS。

#### 适用场景

| 消息类型 | 场景 | 处理方式 |
|----------|------|----------|
| `MSG_SELECT_CARD/TRIBUTE` (15/20) | 常规多选效果 | 包装合法选项/祭品组合 |
| `MSG_SELECT_PLACE/DISFIELD` (18/24) | 位置选择/封锁 | 包装合法位置组合 |
| `MSG_SELECT_COUNTER` (22) | 指示物选择 | 包装完整数量分配 |
| `MSG_SELECT_SUM` (23) | 同调/仪式等凑值 | 包装通过 Core 同等求和规则的组合 |
| `MSG_SORT_CARD` (25) | 排序 | 包装合法次序 |
| `MSG_ANNOUNCE_RACE/ATTRIB` (140/141) | 多项宣言 | 包装完整 bitmask |
| `MSG_SELECT_UNSELECT_CARD` (26) | 动态选择/撤销 | 模型逐消息顺序决策 |

#### 额外优化

静态组合枚举最多保留 5000 项，最终动作池最多 120 项。单卡 Pass1 分数用于加权随机缩减，每个合法组合保留最低探索权重；非场上同名同参数副本按“选择数量”生成规范代表，避免重复副本挤占组合池。该缩减仍保留随机性，不会固定为 RuleBot 的最高分答案。

---

## 简易记牌器逻辑

### 为什么需要它

在游戏王中，很多效果会让对手的手牌短暂暴露（如检索效果）。人类玩家会记住这些信息，但 AI 默认只能看到当前时刻的公开信息，无法利用历史暴露的手牌情报。

### 工作原理

`DuelState` 中维护了一个 `known_hand_codes` 字典，追踪对手手牌中已知的卡片：

```python
class DuelState:
    def __init__(self):
        self.known_hand_codes = {0: [], 1: []}  # [己方已知, 对方已知]
```

#### 进池（记录）：当卡片公开可见地进入对手手牌时

```python
# 1. 公开检索进手（如"增援"）
if new_l == Zone.HAND and is_public_move and pure_code != 0:
    self.known_hand_codes[new_c].append(pure_code)

# 2. 被查看后进手
if pure_code in self.recently_confirmed:
    self.known_hand_codes[new_c].append(pure_code)
```

#### 出池（遗忘）：当卡片从隐藏区域打出或丢弃时

```python
# 从手牌离开
if old_l == Zone.HAND and pure_code in self.known_hand_codes[old_c]:
    self.known_hand_codes[old_c].remove(pure_code)

# 盖伏的卡片被翻开暴露真实身份
if is_from_hidden and pure_code in self.known_hand_codes[old_c]:
    self.known_hand_codes[old_c].remove(pure_code)
```

### 记牌器的用途

1. **142 宣言池增强**：记牌器中的卡片会优先加入 142 候选池
2. **动作优先级评分**：AI 在决策时可以参考已知对手手牌
3. **态势感知**：作为全局特征的一部分提供给神经网络

---

## 卡组权重调整（Global Weights）

### 问题背景

在训练过程中，不同卡组/环境池的重要性不同。如果你希望 AI 更多地在"竞技卡组"环境中训练，而较少在"娱乐卡组"中训练，就需要调整各环境池的采样权重。

### 工作原理

`decks/` 目录下维护的 `global_weights.json` 文件控制各环境池的采样概率：

```json
{
    "tier1_meta": 3.0,     // 竞技卡组权重 3 倍
    "tier2_rogue": 2.0,    // 二线卡组权重 2 倍
    "fun_decks": 0.5,      // 娱乐卡组权重 0.5 倍
    "ygopd_MetaDecks_Latest": 1.5  // 在线抓取池权重 1.5 倍
}
```

Worker 在采样对局时会根据权重进行 `random.choices(env_choices, weights=weights)`。

### 在 WebUI 中操作

在 **🗃️ 资产与卡组管理 → ⚖️ 动态环境权重调度** 中：
- 为每个环境池设置权重滑块（0.0 ~ 10.0）
- 支持批量设值（一键覆盖同类别的所有权重）
- 权重会实时写入 `global_weights.json`，Worker 下局即生效

### 动态调整策略建议

| 训练阶段 | 权重策略 |
|----------|----------|
| 初期 (~500轮) | 分散权重，让 AI 接触多种卡组 |
| 中期 (~2000轮) | 加大主流卡组权重，强化核心能力 |
| 后期 (2000轮+) | 动态平衡，根据胜率自动调整 |

---

## 虚拟伪装池模块（Virtual Mix Pools）

### 问题背景

在训练过程中，有时候你希望 AI 能够接触"竞技卡组 vs 娱乐卡组"这样的跨环境对战（Cross-Pool Match），但又不希望物理移动卡组文件。传统的子文件夹结构只能实现"同池内战"。

### 工作原理

虚拟伪装池允许你创建一个"配方"，混合多个物理环境池的卡组：

```json
// decks/virtual_pools.json
{
    "Meta_VS_Fun": {
        "tier1_meta": 0.7,   // 70% 概率从竞技池抽卡组
        "fun_decks": 0.3     // 30% 概率从娱乐池抽卡组
    },
    "Online_VS_Local": {
        "ygopd_MetaDecks_Latest": 0.6,
        "tier1_meta": 0.4
    }
}
```

当 Worker 抽中虚拟池时：
1. 先在配方内按权重随机选择两个物理池
2. 再从每个物理池中各自随机选择一个卡组
3. 两个不同池的卡组进行跨环境对战

### 优势

- **无需物理移动文件**：减少磁盘操作
- **灵活的实验设计**：可以创建任意比例的混合配方
- **扩大对手多样性**：防止 AI 只学会"内战"

### 在 WebUI 中操作

在 **🗃️ 资产与卡组管理 → 🧠 虚拟环境构建器** 中：
- 创建新的虚拟拼装池
- 为每个物理池设置混合权重
- 创建后，虚拟池会出现在全局权重面板中

---

## 下一步

- 🔧 阅读 [架构设计](architecture.md) 深入理解框架原理
- 📚 阅读 [功能详解](features.md) 了解 WebUI 使用方法
- 🚀 返回 [快速上手](quickstart.md) 开始训练
