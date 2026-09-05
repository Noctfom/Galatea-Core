from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


# 动作协议固定使用 5 个显式目标槽；超出部分由完整语义签名继续区分
ACTION_TARGET_SLOTS = 5
ACTION_OPERATION_COUNT = 32
ACTION_RESPONSE_BUCKETS = 512
ACTION_SIGNATURE_BYTES = 4
ACTION_CONTEXT_DIM = 6
# 连锁条目保留处理卡位置、触发位置、链序和效果槽等 9 项结构化信息
CHAIN_CONTEXT_DIM = 9


class ActionOperation(IntEnum):
    """标记同一引擎消息内部的真实操作语义"""

    DEFAULT = 0
    YES = 1
    NO = 2
    OPTION = 3
    SELECT = 4
    UNSELECT = 5
    FINISH = 6
    CANCEL = 7
    POSITION_ATTACK = 8
    POSITION_ATTACK_DOWN = 9
    POSITION_DEFENSE = 10
    POSITION_SET = 11
    SHUFFLE = 12
    DIRECT_ATTACK = 13
    ATTACK = 14
    ACTIVATE = 15
    CHAIN = 16
    PHASE = 17
    PLACE = 18
    ANNOUNCE = 19
    MACRO_SELECT = 20
    MACRO_SORT = 21
    REMOVE_COUNTER = 22

# ==========================================
#  Galatea AI 数据协议定义 (Schema V2.0)
# ==========================================

@dataclass
class GlobalFeature:
    """全局环境特征：描述整局游戏的宏观状态"""
    turn_count: int       # 当前回合数
    phase_id: int         # 当前阶段ID
    to_play: int          # 当前行动玩家 (0或1)
    
    # 核心资源（固定座位顺序：历史命名 my=P0，op=P1；编码器再转换为行动方视角）
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
    setcodes: tuple = (0, 0, 0, 0) # 字段集合 (一张卡可能拥有多个字段)
    
    # --- 3. 辅助标记 ---
    is_public: bool = False      # 是否公开可见 (表侧卡=True)

    counter_count: int = 0       # 指示物数量
    overlay_count: int = 0       # 叠放的超量素材数量
    is_equipped: bool = False    # 是否有装备卡/取对象羁绊
    used_effect_mask: int = 0    # 已经发动过的效果 (Bitmask，区分同一张卡的不同效果)

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
    effect_slot: int = -1 # Lua 代码语义槽（零基）；无法精确绑定时保持 -1

    # 动作协议 V2：把过去只存在于 index/文字/原始响应里的语义显式交给模型。
    code: int = 0
    operation_id: int = int(ActionOperation.DEFAULT)
    response_value: Optional[int] = None
    target_location_raw: int = -1
    selection_min: int = 0
    selection_max: int = 0
    selection_count: int = 0
    finishable: bool = False
    cancelable: bool = False
    context_value: int = 0
    prompt_flags: int = 0
    prompt_value: int = 0
    prompt_value2: int = 0

    # [合法化] 宏动作专属属性 (默认为 None，兼容单卡逻辑)
    macro_targets: list = None
    macro_places: list = None
    macro_target_codes: list = None
    macro_target_values: list = None
    macro_target_locations: list = None
    decision_bytes: bytes = b''
    decision_value: Optional[int] = None

@dataclass
class GameSnapshot:
    """单一决策帧的完整快照"""
    global_data: GlobalFeature
    entities: List[CardEntity]
    
    # [新增] 当前所有合法的动作列表
    # 如果为空，说明当前不需要/不能操作 (或者在处理效果中)
    valid_actions: List[GameAction] = field(default_factory=list)

    # [新增] 上帝视角：AI 当前剩余的主卡组和额外卡组卡密列表
    p0_deck_codes: List[int] = field(default_factory=list)
    p0_extra_codes: List[int] = field(default_factory=list)
    p1_deck_codes: List[int] = field(default_factory=list)
    p1_extra_codes: List[int] = field(default_factory=list)

    chain_stack: List[dict] = field(default_factory=list)
    history_stack: List[dict] = field(default_factory=list)
