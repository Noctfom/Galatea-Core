'''
游戏常量定义模块 (V3.0 最终完备版)
包含 Zone, Position, Phases, CardType, LocationInfo
'''

class Zone:
    UNKNOWN = 0x00
    DECK    = 0x01
    HAND    = 0x02
    MZONE   = 0x04
    SZONE   = 0x08
    GRAVE   = 0x10
    REMOVED = 0x20
    EXTRA   = 0x40
    OVERLAY = 0x80 # 超量素材 (关键！防止 AI 处理素材时位置识别错误)
    ONFIELD = MZONE | SZONE

    @staticmethod
    def get_str(loc):
        if loc == 0: return "未知"
        parts = []
        if loc & Zone.DECK: parts.append("卡组")
        if loc & Zone.HAND: parts.append("手卡")
        if loc & Zone.MZONE: parts.append("前场")
        if loc & Zone.SZONE: parts.append("后场")
        if loc & Zone.GRAVE: parts.append("墓地")
        if loc & Zone.REMOVED: parts.append("除外")
        if loc & Zone.EXTRA: parts.append("额外")
        if loc & Zone.OVERLAY: parts.append("素材")
        return "|".join(parts) if parts else f"Loc({loc})"

class Position:
    FACEUP_ATTACK    = 0x1
    FACEDOWN_ATTACK  = 0x2 # 极其罕见（如黑暗安眠曲）
    FACEUP_DEFENSE   = 0x4
    FACEDOWN_DEFENSE = 0x8
    
    # 辅助掩码 (方便位运算判断)
    FACEUP           = 0x5 # 0x1 | 0x4
    FACEDOWN         = 0xA # 0x2 | 0x8
    ATTACK           = 0x3 # 0x1 | 0x2
    DEFENSE          = 0xC # 0x4 | 0x8

    @staticmethod
    def get_str(pos):
        if pos == Position.FACEUP_ATTACK: return "表攻"
        if pos == Position.FACEDOWN_ATTACK: return "里攻"
        if pos == Position.FACEUP_DEFENSE: return "表守"
        if pos == Position.FACEDOWN_DEFENSE: return "里守"
        # 组合情况
        parts = []
        if pos & Position.FACEUP: parts.append("表侧")
        if pos & Position.FACEDOWN: parts.append("里侧")
        if pos & Position.ATTACK: parts.append("攻击")
        if pos & Position.DEFENSE: parts.append("守备")
        return "".join(parts) if parts else f"Pos({pos})"

class Phases:
    DRAW          = 0x01
    STANDBY       = 0x02
    MAIN1         = 0x04
    BATTLE_START  = 0x08
    BATTLE_STEP   = 0x10
    DAMAGE        = 0x20
    DAMAGE_CAL    = 0x40
    BATTLE        = 0x80
    MAIN2         = 0x100
    END           = 0x200
    
    @staticmethod
    def get_str(p):
        if p == Phases.DRAW: return "抽卡阶段"
        if p == Phases.STANDBY: return "准备阶段"
        if p == Phases.MAIN1: return "主要阶段1"
        if p >= Phases.BATTLE_START and p <= Phases.BATTLE: return "战斗阶段"
        if p == Phases.MAIN2: return "主要阶段2"
        if p == Phases.END: return "结束阶段"
        return f"阶段({p})"

class CardType:
    """卡片类型定义 (YGOPro Constants)"""
    MONSTER     = 0x1
    SPELL       = 0x2
    TRAP        = 0x4
    
    NORMAL      = 0x10
    EFFECT      = 0x20
    FUSION      = 0x40
    RITUAL      = 0x80
    TRAPMONSTER = 0x100
    SPIRIT      = 0x200
    UNION       = 0x400
    DUAL        = 0x800
    TUNER       = 0x1000
    SYNCHRO     = 0x2000
    TOKEN       = 0x4000
    
    QUICKPLAY   = 0x10000
    CONTINUOUS  = 0x20000
    EQUIP       = 0x40000
    FIELD       = 0x80000
    COUNTER     = 0x100000
    
    FLIP        = 0x200000
    TOON        = 0x400000
    XYZ         = 0x800000
    PENDULUM    = 0x1000000
    LINK        = 0x4000000

class LocationInfo:
    @staticmethod
    def decode(val):
        """
        解码 OCG 4字节位置信息
        Format: [Position(8)] [Sequence(8)] [Location(8)] [Controller(8)]
        """
        controller = val & 0xFF
        location = (val >> 8) & 0xFF
        sequence = (val >> 16) & 0xFF
        position = (val >> 24) & 0xFF
        return controller, location, sequence, position

    @staticmethod
    def encode(controller, location, sequence, position=0):
        """
        [新增] 编码 OCG 4字节位置信息 (decode 的逆运算)
        用于将解析出的分散信息打包回整数，以便存入 target_entity_idx
        """
        return (
            (controller & 0xFF) | 
            ((location & 0xFF) << 8) | 
            ((sequence & 0xFF) << 16) | 
            ((position & 0xFF) << 24)
        )