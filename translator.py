'''
OCGTranslator 模块
用于将 OCGCore 消息类型和动作类别转换为可读文本
'''
class OCGTranslator:
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'MAGENTA': '\033[95m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m'
    }

    MSG_MAP = {
        1:  ("⚠️ [重试] 操作被核心拒绝 (RETRY)", 'YELLOW'),
        2:  ("💡 [提示] 核心提示 (HINT)", 'RESET'),
        4:  ("⏳ [等待] 等待对方 (WAITING)", 'RESET'),
        5:  ("🏆 [结束] 决斗胜负已分 (WIN)", 'GREEN'),
        10: ("⚔️ [战令] 请选择战斗指令 (BATTLE)", 'RED'),
        11: ("🟢 [指令] 请选择操作 (IDLE)", 'GREEN'),
        12: ("❓ [询问] 是否发动效果? (EFFECT_YN)", 'MAGENTA'),
        13: ("❓ [询问] 是/否? (YES/NO)", 'MAGENTA'),
        14: ("🔢 [选择] 请选择选项 (OPTION)", 'MAGENTA'),
        15: ("🎴 [选择] 请选择卡片 (CARD)", 'MAGENTA'),
        16: ("🔗 [连锁] 请选择连锁 (CHAIN)", 'MAGENTA'),
        18: ("📍 [选择] 请选择位置 (PLACE)", 'BLUE'),
        19: ("🔄 [选择] 请选择表示形式 (POSITION)", 'BLUE'),
        20: ("💀 [选择] 请选择祭品 (TRIBUTE)", 'BLUE'),
        22: ("🔢 [选择] 请选择计数器 (COUNTER)", 'BLUE'),
        23: ("🔢 [选择] 请选择数值 (SUM)", 'BLUE'),
        
        40: ("⏱️ [流程] 新的回合 (NEW_TURN)", 'CYAN'),
        50: ("📦 [移动] 卡片移动 (MOVE)", 'RESET'),
        54: ("🃏 [放置] 卡片盖放 (SET)", 'RESET'),
        53: ("🔄 [形式] 改变表示形式 (POS_CHANGE)", 'RESET'),
        60: ("🌟 [召唤] 通常召唤宣言 (SUMMONING)", 'CYAN'),
        61: ("✨ [召唤] 通常召唤成功 (SUMMONED)", 'CYAN'),
        62: ("🌟 [特召] 特殊召唤宣言 (SPSUMMONING)", 'CYAN'),
        70: ("⛓️ [连锁] 效果发动 (CHAINING)", 'MAGENTA'),
        73: ("✅ [连锁] 效果处理完毕 (CHAIN_SOLVED)", 'RESET'),
        90: ("🃏 [抽卡] 玩家抽卡 (DRAW)", 'CYAN'),
        # ... 修正以下 ID ...
        91: ("💥 [伤害] 受到伤害 (DAMAGE)", 'RED'),
        94: ("❤️ [数值] LP 变化 (LP)", 'RED'),
        110: ("🔥 [攻击] 攻击宣言！(ATTACK)", 'RED'),
        113: ("⚔️ [伤判] 进入伤害计算 (DAMAGE_STEP)", 'BOLD'),
    }
    
    ACTION_MAP_IDLE = {
        0: "通常召唤", 1: "特殊召唤", 2: "改变形式",
        3: "盖放怪兽", 4: "盖放魔陷", 5: "发动效果",
        6: "进入战斗阶段", 7: "结束回合"
    }

    @staticmethod
    def _color(text, color_key):
        return f"{OCGTranslator.COLORS.get(color_key, '')}{text}{OCGTranslator.COLORS['RESET']}"

    @staticmethod
    def translate_msg(msg_type):
        if msg_type in OCGTranslator.MSG_MAP:
            text, color = OCGTranslator.MSG_MAP[msg_type]
            return OCGTranslator._color(text, color)
        return f"未知消息 ({msg_type})"

    @staticmethod
    def translate_action(cat):
        return OCGTranslator._color(
            OCGTranslator.ACTION_MAP_IDLE.get(cat, f"动作-{cat}"), 
            'BLUE'
        )