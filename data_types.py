from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import List, Optional


# 动作协议固定使用 5 个显式目标槽；超出部分由完整语义签名继续区分
ACTION_TARGET_SLOTS = 5
ACTION_OPERATION_COUNT = 32
SUMMON_METHOD_COUNT = 11
CARD_RELATION_SLOTS = 16
CARD_RELATION_TYPE_COUNT = 5
CARD_ADDITIONAL_OVERLAY_SLOTS = 15
CARD_COUNTER_SLOTS = 8
CARD_STATUS_BITS = 3
FIELD_ZONE_BITS = 32
CARD_REASON_BYTES = 4
CARD_COUNTER_TYPE_BYTES = 2
FIELD_ZONE_BYTES = 4
ACTION_RESPONSE_BUCKETS = 512
ACTION_SIGNATURE_BYTES = 4
ACTION_CONTEXT_DIM = 6
# 连锁连续特征只保留角色、序号、链序与效果槽；区域和表示改走离散输入
CHAIN_CONTEXT_DIM = 6
# V4 全局连续特征与玩家角色槽位采用固定维度，供编码器、网络和共享内存共同引用
GLOBAL_FEATURE_DIM = 17
PLAYER_CONTEXT_SLOTS = 3
PLAYER_ROLE_COUNT = 3
CARD_NUMERIC_FEATURE_DIM = 66
PHASE_CATEGORY_COUNT = 11
ZONE_CATEGORY_COUNT = 9
POSITION_CATEGORY_COUNT = 64
# 训练局中按整局关闭卡组 FiLM 的概率；掩码会写入轨迹以保持 PPO 同分布
DECK_FILM_DROPOUT = 0.1

# 状态转移历史固定保留最近 16 次已完成决策，3.11.2 编码后进入网络
TRANSITION_EVENT_HISTORY_SIZE = 16
TRANSITION_ZONE_COUNT = 8
TRANSITION_EVENT_TARGET_SLOTS = ACTION_TARGET_SLOTS
TRANSITION_EVENT_RESULT_BYTES = 2
TRANSITION_EVENT_MESSAGE_BYTES = 2
TRANSITION_EVENT_LOCATION_DIM = 4
TRANSITION_EVENT_CHAIN_DIM = 3
TRANSITION_EVENT_COUNT_DIM = 2
TRANSITION_EVENT_LP_DIM = 2
TRANSITION_BOUNDARY_COUNT = 4


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
    NORMAL_SUMMON = 23
    SPECIAL_SUMMON = 24
    CHANGE_POSITION = 25
    MONSTER_SET = 26
    SPELL_TRAP_SET = 27
    TRIBUTE_SUMMON = 28


class SummonMethod(IntEnum):
    """只记录 Core 或可验证上下文直接证明的召唤方式"""

    NONE = 0
    NORMAL = 1
    TRIBUTE = 2
    SPECIAL_UNKNOWN = 3
    RITUAL = 4
    FUSION = 5
    SYNCHRO = 6
    XYZ = 7
    PENDULUM = 8
    LINK = 9
    OTHER_SPECIAL = 10


class CardRelationType(IntEnum):
    """表示公开卡片个体之间由 Core 直接给出的关系类型"""

    NONE = 0
    EQUIP_TARGET = 1
    EFFECT_TARGET = 2
    REASON_CARD = 3
    EQUIP_SOURCE = 4


class TransitionBoundary(IntEnum):
    """标记一次决策间状态转移由什么边界结束"""

    NEXT_DECISION = 1
    RETRY = 2
    TERMINAL = 3


class TransitionResultFlag(IntFlag):
    """记录可由动作或公开 Core 消息确定的转移结果"""

    NONE = 0
    RESOLVED = 1 << 0
    RETRY = 1 << 1
    CANCELLED = 1 << 2
    DECLINED = 1 << 3
    SELECTION_FINISHED = 1 << 4
    CHAIN_NEGATED = 1 << 5
    CHAIN_DISABLED = 1 << 6
    TERMINAL = 1 << 7
    STATE_CHANGED = 1 << 8
    MESSAGE_OVERFLOW = 1 << 9

# ==========================================
#  Galatea AI 数据协议定义 (Schema V2.0)
# ==========================================

@dataclass
class GlobalFeature:
    """全局环境特征：描述整局游戏的宏观状态"""
    turn_count: int       # 当前回合数
    phase_id: int         # 当前阶段ID
    to_play: int          # 兼容字段：最近交互玩家；V4 决策使用下方显式身份
    
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

    # V4 显式拆分决策权、回合归属、先手身份与双方已开始的回合数
    decision_player: int = -1
    turn_player: int = -1
    starting_player: int = -1
    p0_turn_count: int = 0
    p1_turn_count: int = 0
    disabled_field_mask: int = 0

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
    level: int            # 等级；超量/连接怪兽通常为 0
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

    # --- 4. V4 公开个体状态与关系（仅使用公开 Core 查询） ---
    current_code: int = 0        # 动态卡名；0 表示与原卡密一致或不可见
    original_owner: int = -1
    rank: int = 0
    link_rating: int = 0
    reason: int = 0
    status_mask: int = 0
    properly_summoned: bool = False
    is_disabled: bool = False
    is_forbidden: bool = False
    is_equip_source: bool = False
    reason_card_location: int = 0
    equip_target_location: int = 0
    target_locations: list = field(default_factory=list)
    overlay_codes: list = field(default_factory=list)
    counter_items: list = field(default_factory=list)
    relation_indices: list = field(default_factory=list)
    relation_types: list = field(default_factory=list)

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
    summon_method_id: int = int(SummonMethod.NONE)
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


@dataclass(frozen=True)
class StateTransitionEvent:
    """一次已提交动作到下一决策边界之间的确定性公开状态转移"""

    sequence_id: int
    actor: int
    turn_count: int
    phase_id: int
    prompt_type: int
    operation_id: int
    summon_method_id: int
    source_code: int
    source_location_raw: int
    effect_slot: int
    target_locations: tuple
    target_codes: tuple
    target_count: int
    event_visibility_mask: int
    intent_visibility_mask: int
    source_visibility_mask: int
    target_visibility_mask: int
    boundary: int
    result_flags: int
    message_types: tuple
    message_count: int
    lp_delta_p0: int
    lp_delta_p1: int
    zone_deltas: tuple
    chain_depth_before: int
    chain_depth_after: int
    max_chain_depth: int

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

    # 一局内保持不变的初始卡组画像；与上方动态剩余卡组严格分离
    p0_initial_deck_codes: List[int] = field(default_factory=list)
    p0_initial_extra_codes: List[int] = field(default_factory=list)
    p1_initial_deck_codes: List[int] = field(default_factory=list)
    p1_initial_extra_codes: List[int] = field(default_factory=list)

    chain_stack: List[dict] = field(default_factory=list)
    history_stack: List[dict] = field(default_factory=list)
    # 3.11.2 事件协议历史，按视角脱敏后进入模型
    transition_history: List[StateTransitionEvent] = field(default_factory=list)
