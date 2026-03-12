# Galatea-Core

<div align="center">

**基于深度强化学习的游戏王AI训练框架**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

</div>

## 📖 项目简介

Galatea-Core 是一个使用 **Transformer 架构** 和 **PPO (Proximal Policy Optimization)** 强化学习算法训练游戏王决斗AI的完整框架。项目通过与 OCGCore 引擎交互,实现了从零开始的自我博弈训练,能够学习复杂的卡组策略和决策逻辑。


## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选,用于 GPU 加速)
- OCGCore DLL (游戏王核心引擎)
- 游戏王卡片数据库 (cards.cdb)

### 安装依赖

```bash
pip install tensorboard numpy
```

⚠️ 版权与免责声明：本项目不包含任何受版权保护的 YGOPro 核心引擎文件或卡图数据。在运行本项目前，请自行准备相关环境。
项目运行之前，请确认已经在项目根目录放置了以下两个文件：
ygocore.dll  --ygo核心引擎，用于游戏环境搭建
cards.cdb    --卡片数据库

### 项目结构

```
Galatea-Core/
├── main.py                 # 主程序入口
├── trainer.py              # PPO 训练器
├── galatea_net.py          # Transformer 神经网络
├── galatea_env.py          # OCGCore 环境封装
├── ai_bot.py               # AI 决策模块
├── feature_encoder.py      # 特征编码器
├── gamestate.py            # 游戏状态管理
├── worker.py               # 多进程采集 Worker
├── model_versus.py         # 模型竞技场
├── rule_bot.py             # 规则脚本 Bot
├── deck_utils.py           # 卡组工具
├── ocgcore.dll             # 游戏王核心引擎
├── cards.cdb               # 卡片数据库
├── decks/                  # 卡组文件夹 (.ydk)
├── models/                 # 模型保存目录
└── runs/                   # TensorBoard 日志
```

### 准备卡组

将 `.ydk` 格式的卡组文件放入 `decks/` 目录:

```
decks/
├── 青眼白龙.ydk
├── 真红眼黑龙.ydk
└── 电子龙.ydk
```

## 🎓 使用指南

### 快速开始
```bash
streamlit run webui.py
```

在浏览器中打开生成的本地链接，你可以直接在图形界面中配置网络参数、一键开启强化学习训练，并在右侧直接观看 TensorBoard 的 Loss 曲线！


### 1. 训练模式

从零开始训练一个新模型:

```bash
python main.py train --dir ./models --batch_size 4096 --mini_batch 512 --workers 4 --steps 1000
```

**高级训练配置** (推荐用于高性能 GPU):

```bash
python main.py train \
  --dir ./models \
  --batch_size 32768 \
  --mini_batch 1024 \
  --workers 12 \
  --steps 5000 \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --async_infer \
  --no_compile
```

**恢复训练**:

```bash
python main.py train \
  --resume ./models/galatea_iter_100.pth \
  --batch_size 32768 \
  --mini_batch 1024 \
  --workers 12 \
  --steps 5000 \
  --async_infer
```

### 2. 竞技场模式

测试模型性能,与 RuleBot 对战:

```bash
python main.py duel --p0 ./models/galatea_iter_100.pth --num 100
```

模型 vs 模型对战:

```bash
python main.py duel \
  --p0 ./models/galatea_iter_100.pth \
  --p1 ./models/galatea_iter_200.pth \
  --num 100 \
  --device cuda
```

### 3. 自我博弈测试

快速验证环境配置:

```bash
python main.py play --num 10
```

## ⚙️ 核心参数说明

### 模型架构参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `d_model` | 256 | 特征维度 | 越高越聪明,但计算量成倍增加。必须能被 `n_heads` 整除 |
| `n_heads` | 4 | 注意力头数 | 通常设为 4 或 8。头越多,处理复杂关系能力越强 |
| `n_layers` | 2 | Transformer 层数 | 2层适合快速实验,4-6层适合竞技卡组 |
| `vocab_size` | 20000 | 卡片词表大小 | 只要比实际卡片ID总数大即可 |

### 训练参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `batch_size` | 4096 | 采集总步数 | 越大越稳定,但需要更多内存 |
| `mini_batch` | 512 | GPU 训练批量 | 越大更新越快,但可能不稳定 |
| `workers` | 4 | 采集进程数 | 根据 CPU 核心数调整,通常 4-12 |
| `worker_device` | cuda | Worker 推理设备 | `cuda` 或 `cpu` |
| `async_infer` | False | 异步推断服务器 | 启用后大幅节省显存并提速 |
| `no_compile` | False | 禁用 torch.compile | Windows 或老旧环境建议启用 |

## 🏗️ 技术架构

### 神经网络设计

Galatea 由四个高度解耦的子系统构成：

Environment (环境层)：负责与 ocgcore.dll 通信，处理残缺包拦截与状态同步。

Encoder (感知层)：将 100 张卡牌的异构数据转换为定长的 Tensor 矩阵。

Policy Network (认知层)：自研配置的 Transformer 模型（支持自定义 d_model, n_heads, n_layers）。

Arena & Trainer (调度层)：支持多进程 CPU 经验采集 + 独立 GPU 异步推理的 PPO 训练循环，并附带用于评估的 model_versus.py 竞技场。


### PPO 训练流程

```
1. 多进程采集 (Rollout Collection)
   ├── Worker 1: 自我博弈收集经验
   ├── Worker 2: 自我博弈收集经验
   └── Worker N: 自我博弈收集经验
       ↓
2. 经验汇总 (Memory Aggregation)
   ├── 观察 (Observations)
   ├── 动作 (Actions)
   ├── 奖励 (Rewards)
   └── 优势估计 (GAE Advantages)
       ↓
3. 策略优化 (Policy Update)
   ├── Mini-Batch 采样
   ├── PPO Clip 损失计算
   ├── 价值函数损失
   ├── 熵正则化
   └── 梯度裁剪 + 反向传播
       ↓
4. 模型保存 (Checkpoint)
   └── 每 10 轮保存一次
```

### 特征编码

#### 卡片特征 (53维)
- **基础数值** (12维): Owner, Location, Sequence, ATK, DEF, Level, Scale, Position, Public
- **类型展开** (32维): 怪兽/魔法/陷阱/效果/融合/同调/超量/灵摆/连接...
- **连接箭头** (9维): 左下/下/右下/左/右/左上/上/右上/中

#### 全局特征 (15维)
- 回合数、阶段、当前玩家
- 双方 LP
- 双方各区域卡片数量 (手牌/卡组/墓地/除外/额外)

#### 动作特征 (Action Head)
- **目标卡片索引**: 指向 entities 列表中的具体卡片
- **动作类型**: 召唤(0)/攻击(1)/发动(5)/进入战阶(6)...
- **效果描述**: Hash 映射到 1024 以内的 ID

## 🔧 高级功能

### 异步推断服务器

启用 `--async_infer` 后,训练器会启动一个独立的推断线程:

**优势**:
- Worker 进程不再需要加载模型到显存
- 多个 Worker 共享同一个 GPU 推理服务
- 显存占用降低 70%+
- 采集速度提升 2-3 倍

**原理**:
```
Worker 1 ──┐
Worker 2 ──┼──> 请求队列 ──> GPU 推断服务器 ──> 响应队列 ──┬──> Worker 1
Worker 3 ──┘                                              ├──> Worker 2
                                                          └──> Worker 3
```

### 混合精度训练

自动检测硬件能力:
- **Ampere+ 架构** (RTX 30/40/50系): 启用 TF32 加速
- **支持 BF16**: 使用 BFloat16 混合精度 (推荐)
- **仅支持 FP16**: 回退到 Float16 混合精度

### 权重共享机制

Windows Spawn 模式下,使用 `share_memory_()` 让所有子进程共享同一份模型权重:
- 避免权重被复制 N 份
- 显著降低 RAM 占用 (节省数 GB)

## 🐛 常见问题


### Q: 显存不足 (CUDA Out of Memory)

**解决方案**:
```bash
# 方案 1: 启用异步推断
python main.py train --async_infer --worker_device cpu

# 方案 2: 降低模型规模
python main.py train --d_model 128 --n_layers 2 --mini_batch 256

# 方案 3: 减少 Worker
python main.py train --workers 2
```

### Q: Windows 下编译失败

**解决**: 禁用 torch.compile
```bash
python main.py train --no_compile
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request!

### 开发路线图

- [x] 完善底层协议解析与残缺包处理

- [x] 同调/连接/超量召唤的 DFS 算法辅助兜底

- [x] Hash 持久化特征词表

- [ ] 阶段二进化：自回归动作选择，让 AI 学会微观级别的素材挑选与排连锁。

## 📄 许可证

本项目采用 GPL3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [OCGCore](https://github.com/Fluorohydride/ygopro-core) - YGOPRO核心引擎,万物之源
- [MDPro3](https://code.moenext.com/sherry_chaos/MDPro3) - MdPro3,目前优先适配的端


## 📧 联系方式

如有问题或建议,欢迎通过 Issue 联系我。
或是添加作者qq2721298904，加入作者个人q群635990103(目前只是聊天群，项目完善之后会单独分一个新的交流群),感谢你的意见与支持！

---

<div align="center">

**⭐ 如果这个项目对你有帮助,请给个 Star! ⭐**

</div>
