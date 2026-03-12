from dataclasses import dataclass, field
from typing import List

# ==========================================
#  Galatea AI 数据协议定义 (Schema V2.0)
# ==========================================

@dataclass
class GlobalFeature:
    """全局环境特征：描述整局游戏的宏观状态"""
    turn_count: int       # 当前回合数
    phase_id: int         # 当前阶段ID
    to_play: int          # 当前行动玩家 (0或1)
    
    # 核心资源
    my_lp: int
    op_lp: int
    
    # 区域资源统计 (用于宏观判断卡差)
    my_hand_len: int      # 我方手牌数
    op_hand_len: int      # 对方手牌数
    my_deck_len: int      # 我方卡组剩余
    op_deck_len: int      # 对方卡组剩余
    my_grave_len: int     # 我方墓地数
    op_grave_len: int     # 对方墓地数
    my_removed_len: int   # 我方除外数
    op_removed_len: int   # 对方除外数
    my_extra_len: int     # 我方额外卡组数
    op_extra_len: int     # 对方额外卡组数

@dataclass
class CardEntity:
    """
    全息卡片实体：描述一张卡的所有细节
    融合了【静态数据】(来自 cards.cdb) 和 【动态数据】(来自游戏引擎)
    """
    # --- 1. 动态状态 (来自 Game Engine) ---
    code: int             # 卡片密码 (若是对方盖卡/手牌，在编码阶段会被 Mask 掉)
    owner: int            # 持有者 (0/1)
    location: int         # 区域 (MZONE, SZONE, HAND...)
    sequence: int         # 序号 (0-6)
    position: int         # 表示形式 (表攻/表守/里守...)
    current_atk: int      # 当前攻击力
    current_def: int      # 当前防御力
    
    # --- 2. 静态属性 (来自 Card DB) ---
    # 这些属性帮助 AI 理解这张卡是干嘛的
    type_mask: int        # 类型 (怪兽/魔法/陷阱...)
    race: int             # 种族
    attribute: int        # 属性
    level: int            # 等级/阶级/连接值
    base_atk: int         # 原攻击力
    base_def: int         # 原防御力
    lscale: int = 0       # 灵摆左刻度
    rscale: int = 0       # 灵摆右刻度
    link_marker: int = 0  # 连接箭头 (Bitmask)
    
    # --- 3. 辅助标记 ---
    is_public: bool = False      # 是否公开可见 (表侧卡=True)

@dataclass
class GameAction:
    """
    [新增] 定义一个原子操作
    AI 的任务就是从 valid_actions 列表中选一个 Action 执行
    """
    action_type: int      # 0=Summon, 1=SpSummon, 5=Activate, 16=Chain, ...
    index: int            # 在 YGOPro 原始列表中的索引
    
    # 指针信息 (Pointer Network 用)
    # 如果这个动作是针对某张卡的(比如攻击/连锁)，记录这张卡在 entities 列表里的下标
    target_entity_idx: int = -1 
    
    # 描述信息 (供人类调试用，比如 "发动 增殖的G")
    desc_str: str = ""

    desc_id: int = 0      # 效果ID，用于区分同一张卡的不同效果

@dataclass
class GameSnapshot:
    """单一决策帧的完整快照"""
    global_data: GlobalFeature
    entities: List[CardEntity]
    
    # [新增] 当前所有合法的动作列表
    # 如果为空，说明当前不需要/不能操作 (或者在处理效果中)
    valid_actions: List[GameAction] = field(default_factory=list)