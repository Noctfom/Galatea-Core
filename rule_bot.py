'''
规则型 Bot 的核心决策逻辑
保证在所有交互请求中，100% 返回合法格式的数据，防止超时
'''


import struct
import io
import random
import itertools
from game_constants import LocationInfo, Zone
from card_reader import card_db
from core_message_protocol import CoreMessageProtocolError, parse_chain_prompt
from selection_protocol import (
    CANCEL_RESPONSE,
    SELECTION_RESPONSE_MSGS,
    parse_selection_prompt,
    project_selection_payload,
    validate_parsed_selection_response,
    validate_selection_response,
)

# --- 消息类型常量 ---
MSG_SELECT_BATTLECMD = 10
MSG_SELECT_IDLECMD = 11
MSG_SELECT_EFFECTYN = 12
MSG_SELECT_YESNO = 13
MSG_SELECT_OPTION = 14
MSG_SELECT_CARD = 15
MSG_SELECT_CHAIN = 16
MSG_SELECT_PLACE = 18
MSG_SELECT_POSITION = 19
MSG_SELECT_TRIBUTE = 20
MSG_SORT_CHAIN = 21
MSG_SELECT_COUNTER = 22
MSG_SELECT_SUM = 23
MSG_SELECT_DISFIELD = 24
MSG_SORT_CARD = 25
MSG_SELECT_UNSELECT_CARD = 26
MSG_ANNOUNCE_RACE = 140
MSG_ANNOUNCE_ATTRIB = 141
MSG_ANNOUNCE_CARD = 142
MSG_ANNOUNCE_NUMBER = 143
MATERIAL_SELECTION_MSGS = frozenset((MSG_SELECT_CARD, MSG_SELECT_TRIBUTE, MSG_SELECT_SUM))

# --- 常量定义 ---
POS_FACEUP_ATTACK = 0x1
POS_FACEDOWN_ATTACK = 0x2
POS_FACEUP_DEFENSE = 0x4
POS_FACEDOWN_DEFENSE = 0x8

# --- 辅助解析函数 ---

_shared_valid_actions = []

def sync_valid_actions(actions):
    """接收来自 gamestate 完美解析的合法动作池"""
    global _shared_valid_actions
    _shared_valid_actions = actions

# --- 辅助算法: Subset Sum ---
def solve_subset_sum(target_val, candidates, min_c, max_c):
    """
    寻找组合，使数值之和等于 target_val
    """
    # 尝试从 min_c 到 max_c 的所有数量组合
    for r in range(min_c, max_c + 1):
        for combination in itertools.combinations(candidates, r):
            current_sum = sum(c['val'] for c in combination)
            if current_sum == target_val:
                return [c['index'] for c in combination]
    return None

def parse_idle_cmd(msg_data):
    """解析 IDLE 消息，提取所有合法动作"""
    stream = io.BytesIO(msg_data)
    stream.read(1) # player
    legal_actions = []
    
    # 0:Summon, 1:SpSummon, 2:Repos, 3:MSet, 4:SSet, 5:Activate
    for cat in range(6):
        b = stream.read(1)
        if not b: break
        count = struct.unpack('<B', b)[0]
        
        for i in range(count):
            stream.read(7) # code(4) + con(1) + loc(1) + seq(1)
            if cat == 5: stream.read(4) # desc
            legal_actions.append({'cat': cat, 'idx': i})          

    b = stream.read(1)
    if b and struct.unpack('<B', b)[0]: legal_actions.append({'cat': 6, 'idx': 0}) # BP
    b = stream.read(1)
    if b and struct.unpack('<B', b)[0]: legal_actions.append({'cat': 7, 'idx': 0}) # EP
    return legal_actions

def parse_battle_cmd(msg_data):
    """解析 BATTLE 消息"""
    stream = io.BytesIO(msg_data)
    stream.read(1) # player
    legal_actions = []
    
    # Activatable (发动效果)
    b = stream.read(1) 
    if b:
        count = struct.unpack('<B', b)[0]
        for i in range(count): 
            stream.read(11) # Skip details
            legal_actions.append({'cat': 0, 'idx': i})
            
    # Attackable
    b = stream.read(1) 
    if b:
        count = struct.unpack('<B', b)[0]
        for i in range(count):
            code = struct.unpack('<I', stream.read(4))[0]
            stream.read(4) 
            legal_actions.append({'cat': 1, 'idx': i, 'code': code})
            
    # M2 / EP
    b = stream.read(1); 
    if b and struct.unpack('<B', b)[0]: legal_actions.append({'cat': 2, 'idx': 0}) 
    b = stream.read(1); 
    if b and struct.unpack('<B', b)[0]: legal_actions.append({'cat': 3, 'idx': 0}) 
    return legal_actions

# --- 核心决策逻辑 ---

def get_rule_decision(player_id, msg_type, msg, gamestate, ignore_actions=None):
    """
    处理所有交互请求，保证 100% 返回合法格式的数据，防止超时判负。
    """
    if ignore_actions is None: ignore_actions = []
    payload = msg[1:] # 去掉 msg_type 头
    stream = io.BytesIO(payload)
    
    decision = None # 最终决策结果，用于返回和记录

    try:
        # ==================== 1. 基础战斗/闲置逻辑 ====================
        # =================================================================
        # [终极修复] 7. 闲置命令 (Idle)：字符串 Key 过滤法
        # =================================================================
        if msg_type == MSG_SELECT_IDLECMD:
            actions = parse_idle_cmd(payload)
            valid_actions = []
            
            # 1. 构建黑名单 Key 集合
            ignored_keys = set()
            for val in ignore_actions:
                if isinstance(val, int):
                    i_cat = val & 0xFFFF
                    i_idx = (val >> 16) & 0xFFFF
                    ignored_keys.add(f"{i_cat}:{i_idx}")
                elif isinstance(val, str):
                    ignored_keys.add(val)

            # 2. 筛选合法操作
            for a in actions:
                cat = a['cat']
                idx = a['idx']
                if f"{cat}:{idx}" not in ignored_keys:
                    resp = (idx << 16) | cat
                    valid_actions.append(resp)
            
            if valid_actions:
                decision = random.choice(valid_actions)
            else:
                # [关键改动] 绝境处理：所有操作都被拉黑了
                # 检查 "To EP" (Cat=7) 和 "To BP" (Cat=6) 是否在黑名单里
                
                cmd_ep = (0 << 16) | 7
                cmd_bp = (0 << 16) | 6
                
                # 如果 EP 没被拉黑，且 actions 里包含 EP (虽然被上面的 key 过滤了，但我们这里是强制尝试)
                # 实际上 parse_idle_cmd 会解析出是否有 EP/BP 选项。
                # 简单粗暴点：
                
                if f"7:0" not in ignored_keys:
                    decision = cmd_ep # 尝试结束回合
                elif f"6:0" not in ignored_keys:
                    decision = cmd_bp # 尝试进战阶
                else:
                    # EP 和 BP 都不行？那只能随便发一个“投降”或“空操作”来触发外部熔断了
                    # 这里发一个不存在的 Cat 10，让 Core 报错而不是卡逻辑
                    decision = (0 << 16) | 10

        elif msg_type == MSG_SELECT_BATTLECMD:
            actions = parse_battle_cmd(payload)
            valid_actions = []
            
            # 黑名单过滤
            for a in actions:
                if a['cat'] == 1: test_val = (a['idx'] << 16) | 1
                else: test_val = (0 << 16) | a['cat']
                if test_val not in ignore_actions:
                    valid_actions.append(a)

            if not valid_actions: 
                # 🌟 核心修复：不要硬编码，动态检查引擎到底允许进哪个阶段
                can_m2 = any(a['cat'] == 2 for a in actions)
                can_ep = any(a['cat'] == 3 for a in actions)
                
                # 优先尝试 M2，再尝试 EP，且避开被拉黑的选项
                if can_m2 and ((0 << 16) | 2) not in ignore_actions:
                    decision = (0 << 16) | 2
                elif can_ep and ((0 << 16) | 3) not in ignore_actions:
                    decision = (0 << 16) | 3
                else:
                    decision = (0 << 16) | 10 # 彻底卡死时的无奈之举，触发外界熔断
            else:
                choice = random.choice(valid_actions)
                if choice['cat'] in [0, 1]: 
                    decision = (choice['idx'] << 16) | choice['cat']
                else:
                    decision = (0 << 16) | choice['cat']

        # ==================== 2. 简单二选一/多选一 ====================
        # [修复] Type 13: 必须检查 ignore_actions
        elif msg_type in [MSG_SELECT_YESNO, MSG_SELECT_EFFECTYN]:
            candidates = [0, 1]
            valid = [x for x in candidates if x not in ignore_actions]
            if valid:
                decision = random.choice(valid)
            else:
                decision = random.choice([0, 1])

        # =================================================================
        # [修复] 4. 选项选择：排除错误选项
        # =================================================================
        # [修复] Type 14: 必须检查 ignore_actions 且不能返回死锁的 0
        elif msg_type == MSG_SELECT_OPTION:
            # [修复] 防空包崩溃
            if len(payload) < 2: return bytes([0])
            
            stream.read(1) 
            try:
                count = struct.unpack('<B', stream.read(1))[0]
            except:
                return bytes([0])
            possible_choices = list(range(count))
            
            valid_choices = [i for i in possible_choices if i not in ignore_actions]
            
            if valid_choices:
                decision = random.choice(valid_choices)
            else:
                decision = -1 # 尝试取消

        elif msg_type == MSG_SELECT_CHAIN:
            prompt = parse_chain_prompt(payload)
            candidates = list(range(len(prompt['candidates'])))
            if not prompt['forced']:
                candidates.append(-1)
            valid_choices = [c for c in candidates if c not in ignore_actions]
            if not prompt['candidates']:
                return -1
            # 黑名单耗尽也只退回 Core 合法集合，不伪造非法发动/取消
            return random.choice(valid_choices or candidates)

        # ==================== 3. 宣言与竞猜 (新增与补全) ====================
        # 这类消息如果不处理，遇到《抹杀之指名者》等卡会直接卡死
        
        # =================================================================
        # 10. 种族/属性宣言 (位掩码处理)
        # =================================================================
        elif msg_type in [MSG_ANNOUNCE_RACE, MSG_ANNOUNCE_ATTRIB]:
            stream.read(1) # Player
            count = struct.unpack('<B', stream.read(1))[0] # 需要选几个
            available = struct.unpack('<I', stream.read(4))[0] # 可选掩码
            
            options = []
            for i in range(32): # 遍历 32 位
                bit = 1 << i
                if available & bit:
                    options.append(bit)
            
            if len(options) <= count:
                selected = options
            else:
                selected = random.sample(options, count)
            
            result_mask = 0
            for bit in selected:
                result_mask |= bit
            
            # 修复：直接返回整数，不再使用 struct.pack 打包为 bytes
            decision = result_mask

        # =================================================================
        #  卡片/数字宣言 
        # =================================================================
        elif msg_type == MSG_ANNOUNCE_CARD: # 142: 必须返回真实卡密
            stream.read(1)
            count = struct.unpack('<B', stream.read(1))[0]
            for _ in range(count): stream.read(4)
            
            valid_codes = [act.desc_id for act in _shared_valid_actions if act.action_type == 142]
            
            # 修复：保持整数对比，不再将 ignore_actions 和 code 视为 bytes
            safe_codes = [code for code in valid_codes if code not in ignore_actions]
            
            if safe_codes:
                decision = random.choice(safe_codes) # 完美过关，返回整数
            else:
                print("🚨 [RuleBot 警报] Type 142 (宣言卡片) 没有合法选项可选！")

        elif msg_type == MSG_ANNOUNCE_NUMBER: # 143: 必须返回索引
            stream.read(1)
            count = struct.unpack('<B', stream.read(1))[0]
            for _ in range(count): stream.read(4) 
            
            decision = 0
            if count > 0:
                # 修复：直接使用整数索引过滤
                valid_indices = [i for i in range(count) if i not in ignore_actions]
                if valid_indices:
                    decision = random.choice(valid_indices)

        # ==================== 4. 复杂对象选择 (位置/卡片) ====================
        
        # =================================================================
        # [修复] 5. 表示形式选择：修正解析偏移 + 移除非法选项
        # =================================================================
        # [修复] Type 19: 必须使用 valid_ops 而不是 options
        elif msg_type == 19: # MSG_SELECT_POSITION
            try:
                stream.read(1); stream.read(4)
                mask_byte = stream.read(1)
                if not mask_byte: decision = bytes([1])
                else:
                    mask = struct.unpack('<B', mask_byte)[0]
                    options = []
                    if mask & 0x1: options.append(1)
                    if mask & 0x2: options.append(2)
                    if mask & 0x4: options.append(4)
                    if mask & 0x8: options.append(8)
                    
                    # 过滤黑名单
                    valid_ops = [o for o in options if o not in ignore_actions and bytes([o]) not in ignore_actions]
                    
                    if valid_ops:
                        # [关键] 从 valid_ops 选
                        decision = bytes([random.choice(valid_ops)])
                    elif options:
                        # 绝境：随机盲选
                        decision = bytes([random.choice(options)])
                    else:
                        decision = bytes([1])
            except Exception as e:
                print(f"⚠️ [RuleBot] 处理 MSG_SELECT_POSITION 时发生异常: {e}")
                decision = bytes([1])

        # 普通多选、解放与凑星都复用同一套 Core 合法组合生成器
        elif msg_type in [MSG_SELECT_CARD, MSG_SELECT_TRIBUTE, MSG_SELECT_SUM]:
            options = get_macro_options(
                msg_type,
                payload,
                gamestate,
                limit=5000,
            )
            ignored_set = {
                bytes(value) for value in ignore_actions
                if isinstance(value, (bytes, bytearray))
            }
            available = [
                option for option in options
                if bytes(option.get('bytes', b'')) not in ignored_set
            ]
            non_cancel = [
                option for option in available
                if bytes(option.get('bytes', b'')) != CANCEL_RESPONSE
            ]
            if non_cancel:
                # 规则机器人优先完成选择；同等合法响应中保留随机性
                if msg_type in (MSG_SELECT_CARD, MSG_SELECT_TRIBUTE):
                    largest = max(len(option.get('locs', ())) for option in non_cancel)
                    non_cancel = [
                        option for option in non_cancel
                        if len(option.get('locs', ())) == largest
                    ]
                decision = bytes(random.choice(non_cancel)['bytes'])
            elif available:
                decision = bytes(random.choice(available)['bytes'])
            elif options:
                # Retry 黑名单耗尽时仍提交一个已验证合法响应，避免伪造格式
                decision = bytes(options[0]['bytes'])
            else:
                raise RuntimeError(
                    f"Type {msg_type} 没有生成任何满足 Core 约束的完整组合"
                )

        # =================================================================
        # [新增] 9. 复杂选卡 (Select Unselect) - Type 26
        # =================================================================
        elif msg_type == MSG_SELECT_UNSELECT_CARD:
            # [防爆1] 基础长度检查：P(1)+Fin(1)+Can(1)+Min(1)+Max(1) = 5字节
            if len(payload) < 5: 
                decision = struct.pack('<i', 0)
            else:
                try:
                    # [防爆2] 原有逻辑包裹在 try 中
                    # 源码结构: P(1)+Finish(1)+Can(1)+Min(1)+Max(1) + SizeA(1) + ...
                    stream.read(1) # Player
                    finishable = struct.unpack('<B', stream.read(1))[0]
                    cancelable = struct.unpack('<B', stream.read(1))[0]
                    min_c = struct.unpack('<B', stream.read(1))[0]
                    max_c = struct.unpack('<B', stream.read(1))[0]
                    
                    size_a = struct.unpack('<B', stream.read(1))[0]
                    stream.read(size_a * 8) # 跳过 List A
                    
                    size_b = struct.unpack('<B', stream.read(1))[0]
                    stream.read(size_b * 8) # 跳过 List B
                    
                    # 策略：能结束就结束，否则从A里选一张
                    if finishable:
                        decision = struct.pack('<i', -1)
                    elif size_a > 0:
                        # 选中 A 列表的第一张
                        decision = bytes([1, 0])
                    elif cancelable:
                        decision = struct.pack('<i', -1)
                    else:
                        decision = struct.pack('<i', 0)
                except Exception as e:
                    print(f"⚠️ [RuleBot] 处理 MSG_SELECT_UNSELECT_CARD 时发生异常: {e}")
                    # 解析中途失败（如数据包截断），默认选0
                    decision = struct.pack('<i', 0)

        # ==================== 6. 排序与位置 (MSG_SORT_CARD) ====================
        elif msg_type in (MSG_SORT_CHAIN, MSG_SORT_CARD):
            stream.read(1) # Player
            count = struct.unpack('<B', stream.read(1))[0]
            # 后面是 count * 7 字节的卡片信息，跳过
            
            # 逻辑：返回一个全新的索引顺序。
            # 比如有3张卡，我们返回 [2, 0, 1] 表示原第3张放第1，原第1张放第2...
            indices = list(range(count))
            random.shuffle(indices)
            decision = bytes(indices)

        # =================================================================
        # [终极修正] 6. 全局位置选择 (Place/Disfield)
        # =================================================================
        # [RuleBot 核弹级修复] 6. 全局位置选择 (Place/Disfield)
        elif msg_type in [MSG_SELECT_PLACE, MSG_SELECT_DISFIELD]:
            stream.seek(0) 
            req_player = struct.unpack('<B', stream.read(1))[0]
            count = struct.unpack('<B', stream.read(1))[0]
            # 防止核心发疯传来 count=0 导致 b'' 和 loc=0 越界崩溃
            count = max(1, count)
            mask = struct.unpack('<I', stream.read(4))[0]
            
            # 1. 黑名单强力清洗：只保留长度为 3 的 bytes
            safe_ignore_set = set()
            for x in ignore_actions:
                if isinstance(x, (bytes, bytearray)) and len(x) == 3:
                    safe_ignore_set.add(bytes(x))
            
            # 2. 合法位置生成 (确保绝对是 3 字节)
            valid_locs = []
            for i in range(32):
                if not (mask & (1 << i)):
                    p = 0
                    l = 0x04
                    s = 0
                    if i & 16: p = 1
                    if i & 8:  l = 0x08
                    s = i & 0x7 
                    
                    if l == 0x04 and s > 6: continue
                    if l == 0x08 and s > 7: continue 
                    
                    # === 修复开始 ===
                    # 原始逻辑: target_p = req_player if p == 0 else (1 - req_player)
                    # 问题: 如果 req_player > 1，(1-req) 会变成负数，导致 bytes() 崩溃
                    
                    # 修复逻辑: 无论算出什么，强制取模或限制在 0-1
                    raw_p = req_player if p == 0 else (1 - req_player)
                    target_p = 1 if raw_p == 1 else 0 # 任何非1的值都变成0，防止负数
                    loc_bytes = bytes([target_p, l, s]) # 绝对是 3 字节
                    valid_locs.append(loc_bytes)

            # 3. 决策生成
            candidates = []
            if count == 1:
                for loc in valid_locs:
                    # 构造完整决策包 (单选时就是 loc 本身)
                    # 检查是否在黑名单
                    if loc not in safe_ignore_set:
                        candidates.append(loc)
                
                # 绝境回退：如果全被拉黑，尝试随机选一个合法的（撞大运）
                if not candidates and valid_locs:
                    candidates = valid_locs
                    print(f"🚨 [RuleBot 警报] Type {msg_type} (站位/封锁) 合法位置全被拉黑！")
                    print(f"   -> 引擎提供可用位置: {valid_locs}")
                    print(f"   -> 当前黑名单: {safe_ignore_set}")
            else:
                candidates = valid_locs # 多选暂不过滤

            resp_buf = bytearray()
            
            if candidates:
                random.shuffle(candidates)
                # 确保取出的数量不超过 count
                # 如果 candidates 不够，就取全部，后面补 0
                selected = candidates[:count]
                
                for loc in selected:
                    resp_buf.extend(loc)
            
            # 4. 最终守门员：强制补齐与校验
            expected_len = max(1, count) * 3
            
            # 如果长度不够，补 0
            while len(resp_buf) < expected_len:
                resp_buf.extend([0, 0, 0])
            
            # 如果长度超了（理论不应发生），截断
            if len(resp_buf) > expected_len:
                resp_buf = resp_buf[:expected_len]
                
            decision = bytes(resp_buf)
            
            # [双重保险] 如果 decision 居然还是 2 (比如被某些诡异逻辑覆盖了)
            # 这里做最后的类型检查
            if not isinstance(decision, bytes) or len(decision) != expected_len:
                decision = bytes([0] * expected_len)

        # =================================================================
        # 严格修正：MSG_SELECT_COUNTER (22)
        # =================================================================
        elif msg_type == MSG_SELECT_COUNTER:
            # 结构解析 (基于 C++ field::select_counter)
            stream.read(1) # Player
            stream.read(2) # Type
            qty = struct.unpack('<H', stream.read(2))[0] # 需要移除的总数
            size = struct.unpack('<B', stream.read(1))[0] # 列表长度
            
            cards = []
            for i in range(size):
                # 9 Bytes: Code(4)+C(1)+L(1)+S(1)+Avail(2)
                stream.read(7)
                avail = struct.unpack('<H', stream.read(2))[0]
                cards.append({'idx': i, 'avail': avail})
                
            # 分配逻辑：构造一个长度为 size 的数组，总和等于 qty
            response = [0] * size
            remaining = qty
            
            # 简单的随机分配算法
            loop_limit = 1000
            while remaining > 0 and loop_limit > 0:
                idx = random.randint(0, size - 1)
                if cards[idx]['avail'] > 0:
                    cards[idx]['avail'] -= 1
                    response[idx] += 1
                    remaining -= 1
                loop_limit -= 1
            
            # --- 构造返回包 ---
            # C++ 期望读取的是 int16 (svalue)，所以每个数字占 2 字节
            resp_buf = bytearray()
            for count_val in response:
                resp_buf.extend(struct.pack('<H', count_val))
                
            decision = bytes(resp_buf)
        
        # =================================================================
        # [新增] 12. 猜拳与手牌/先攻选择 (Type 132, 133)
        # =================================================================
        elif msg_type == 132: # MSG_ROCK_PAPER_SCISSORS
            stream.read(1) # Player
            decision = random.choice([1, 2, 3]) # 1:剪刀, 2:石头, 3:布

        elif msg_type == 133: # MSG_HAND_RES (选先攻/后攻)
            stream.read(1) # Player
            # 1: 先攻, 2: 后攻 (通常)
            decision = random.choice([1, 2])

        # =================================================================
        # [新增] 13. 硬币与骰子 (Type 130, 131)
        # =================================================================
        elif msg_type == 130: # MSG_TOSS_COIN
            stream.read(1) # Player
            count = struct.unpack('<B', stream.read(1))[0]
            # 这里的逻辑通常是 Core 告诉客户端结果，或者是客户端确认
            # 如果需要回复，通常是发 0 或 1 (猜正反)
            # 简单起见，发 0 (Heads) 或 1 (Tails) * Count
            resp_buf = bytearray([random.choice([0, 1]) for _ in range(count)])
            decision = bytes(resp_buf)

        elif msg_type == 131: # MSG_TOSS_DICE
            stream.read(1) # Player
            count = struct.unpack('<B', stream.read(1))[0]
            # 同上，回复占位符
            resp_buf = bytearray([0] * count)
            decision = bytes(resp_buf)
            
    except CoreMessageProtocolError:
        # 固定 Core 解码失败必须交由上层回滚，不能伪造取消
        raise
    except Exception as e:
        # 万一解析崩了，也要保证返回合法格式，避免 Bot 卡死
        print(f"[RuleBot] 解析出现错误 {msg_type}: {e}")
        decision = -1

    if decision is None:
        print(f"🚨 [RuleBot 警报] 消息类型 {msg_type} 的解析逻辑彻底失效，没有任何分支返回 decision！")
        print(f"   -> Payload 长度: {len(payload)}")
        decision = -1
    
    if msg_type in SELECTION_RESPONSE_MSGS:
        validate_selection_response(msg_type, payload, decision)
    return decision


def _macro_equivalence_key(card, brain, actor, *prompt_values):
    """仅合并公开状态与提示参数完全等价的隐藏区同名副本"""
    controller, location, sequence, position = LocationInfo.decode(card['loc'])
    # 场、墓地与除外区中的卡片个体可能保留不同关系或正规出场状态，禁止折叠
    if (location not in (Zone.HAND, Zone.DECK, Zone.EXTRA)
            or controller != actor or not card['code']):
        return ('INSTANCE', card.get('idx', card.get('index', sequence)))

    state_signature = ()
    if brain is not None and controller in (0, 1):
        field_map = getattr(brain, 'field_map', {})
        player_map = field_map.get(controller, {}) if hasattr(field_map, 'get') else {}
        info = player_map.get(location, {}).get(sequence, {})
        state_signature = (
            int(info.get('current_code', 0) or 0),
            int(info.get('reason', 0) or 0),
            int(info.get('status_mask', 0) or 0),
            int(info.get('equip_target', 0) or 0),
            tuple(info.get('target_locations', ())),
            tuple(info.get('overlay_codes', info.get('overlays', ()))),
            tuple(info.get('counter_items', ())),
            int(info.get('original_owner', -1)),
        )
    return (
        'EQUIVALENT_HIDDEN',
        controller,
        int(card.get('code', 0)),
        location,
        position,
        tuple(prompt_values),
        state_signature,
    )


def _material_options(msg_type, payload, brain, limit, preferences, index_preferences, trigger):
    """分张数生成规范的完整响应，只让不同且合法的组合占用预算"""
    visible = project_selection_payload(msg_type, payload, getattr(brain, 'selection_candidate_code', None))
    prompt = parse_selection_prompt(msg_type, visible)
    is_sum = msg_type == MSG_SELECT_SUM
    exact = is_sum and prompt['mode'] == 0
    tribute = msg_type == MSG_SELECT_TRIBUTE
    required = prompt.get('mandatory', ())
    target = prompt['target'] if is_sum else prompt['minimum'] if tribute else 0

    def material(row, index):
        raw = row.get('sum_value', row.get('release_value', 0))
        low, high = raw & 0xffff, raw >> 16
        smallest = min(low, high) if high else low
        if exact and not smallest:
            smallest = max(low, high)
        return {'idx': index, 'code': row['code'], 'value': raw,
                'loc': LocationInfo.encode(row['controller'], row['location'], row['sequence'], row.get('position', 0)),
                'lo': smallest, 'hi': max(low, high)}

    fixed = [material(row, i) for i, row in enumerate(required)]
    cards = [material(row, i) for i, row in enumerate(prompt['candidates'])]
    if exact:
        if any(not c['hi'] for c in fixed):
            return []
        cards = [c for c in cards if c['hi']]
    cards.sort(key=lambda c: (index_preferences[c['idx']] if index_preferences is not None
                              else preferences.get(c['code'], 0)), reverse=True)
    grouped = {}
    for c in cards:
        key = _macro_equivalence_key(c, brain, prompt['player'], c['value'])
        grouped.setdefault(key, []).append(c)
    groups = list(grouped.values())
    n = len(groups)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + len(groups[i])
    maximum = min(suffix[0], 255 - len(fixed))
    if not is_sum or exact:
        maximum = min(maximum, prompt['maximum'])
    # 精确凑数的每张素材必须贡献正值，超过目标值的张数不可能有解
    if exact:
        maximum = min(maximum, target - len(fixed))
    minimum = prompt['minimum'] if exact or msg_type == MSG_SELECT_CARD else 0
    if tribute and target:
        minimum = 1
    fixed_lo = sum(c['lo'] for c in fixed)
    fixed_hi = sum(c['hi'] for c in fixed)
    fixed_min = min((c['lo'] for c in fixed), default=0xffff)
    # 每个后缀按指定张数可能获得的上下界，只用于排除数学上无解的分支
    bounds = []
    if is_sum or tribute:
        for i in range(n + 1):
            tail = [c for group in groups[i:] for c in group]
            lows = sorted(c['lo'] for c in tail)
            highs = sorted((c['hi'] for c in tail), reverse=True)
            bounds.append((list(itertools.accumulate(lows, initial=0)), list(itertools.accumulate(highs, initial=0))))

    # 大精确和询问可在数值上下界内仍完全无解，用后缀位集排除这些分支
    # 随包内核目标为低16位；按大小稀疏保留后缀表，超出该范围仍走完整的原生校验
    checkpoints = {}
    residual = 0
    if exact and len(cards) >= 12 and 0 <= maximum and target <= 0xffff:
        residual = 1 << target
        for c in fixed:
            residual = (residual >> c['lo']) | (residual >> c['hi'])
        if not residual:
            return []
        mask = (1 << (target + 1)) - 1
        ways = [1] + [0] * maximum
        table_bytes = (maximum + 1) * (40 + 4 * (target // 30 + 1))
        table_count = max(1, (8 * 1024**2) // table_bytes)
        stride = max(1, (n + table_count - 1) // table_count)
        processed = 0
        for i in range(n - 1, -1, -1):
            for c in groups[i]:
                processed += 1
                for k in range(min(processed, maximum), 0, -1):
                    prior = ways[k - 1]
                    ways[k] |= ((prior << c['lo']) | (prior << c['hi'])) & mask
            if i % stride == 0:
                checkpoints[i] = tuple(ways)

    def solutions(size):
        path = []

        def search(i, need, lo, hi, smallest, debt):
            if need > suffix[i]:
                return
            if checkpoints and (not debt or (i in checkpoints and not (checkpoints[i][need] & debt))):
                return
            if bounds:
                lower, upper = bounds[i]
                if hi + upper[need] < target:
                    return
                if exact and lo + lower[need] > target:
                    return
                if is_sum and not exact and (fixed or path or need):
                    # 去掉最小值后仍不低于目标时必非法；未知的最小值只取安全上界
                    if lo + lower[need] - smallest >= target:
                        return
                    if need and not (fixed or path) and lower[need] - lower[1] >= target:
                        return
            if need == 0:
                if checkpoints and not (debt & 1):
                    return
                chosen = sorted(path, key=lambda c: c['idx']) if is_sum else list(path)
                response = bytes([len(fixed) + size]) + bytes(len(fixed)) + bytes(c['idx'] for c in chosen)
                try:
                    validate_parsed_selection_response(msg_type, prompt, response)
                except ValueError:
                    return
                all_cards = fixed + chosen
                yield {'bytes': response, 'locs': [c['loc'] for c in all_cards],
                       'codes': [c['code'] for c in all_cards], 'values': [c['value'] for c in all_cards]}
                return
            group = groups[i]
            first = group[0]
            for take in range(min(need, len(group)), max(0, need - suffix[i + 1]) - 1, -1):
                next_debt = debt
                if take:
                    path.extend(group[:take])
                    if checkpoints:
                        for _ in range(take):
                            next_debt = (next_debt >> first['lo']) | (next_debt >> first['hi'])
                yield from search(i + 1, need - take, lo + take * first['lo'], hi + take * first['hi'],
                                  min(smallest, first['lo']) if take else smallest, next_debt)
                if take:
                    del path[-take:]

        yield from search(0, size, fixed_lo, fixed_hi, fixed_min, residual)

    buckets = []
    capped = False
    for size in range(minimum, maximum + 1):
        source = solutions(size)
        first = next(source, None)
        if first is not None:
            buckets.append((first, source))
    if len(buckets) > limit:
        # 极小的自定义预算连每种张数一项也放不下，优先保留代表项单卡倾向更高的张数
        def first_score(bucket):
            option = bucket[0]
            if index_preferences is not None:
                return sum(index_preferences[i] for i in option['bytes'][1 + len(fixed):])
            return sum(preferences.get(code, 0) for code in option['codes'][len(fixed):])
        buckets.sort(key=first_score, reverse=True)
        buckets = buckets[:limit]
        capped = True
    # 对小池维持按张数及偏好顺序；预算紧张时先为每个有效张数留一个位置
    remaining = limit - len(buckets)
    rows = []
    for first, source in buckets:
        rows.append(first)
        if capped:
            continue
        for row in source:
            if remaining == 0:
                capped = True
                break
            rows.append(row)
            remaining -= 1
    if capped:
        print(f"📡 [RuleBot] Type {msg_type} 不同合法组合超过 {limit}，按可选张数保留代表 | {trigger}")
    if prompt['cancelable']:
        rows.append({'bytes': CANCEL_RESPONSE, 'locs': [], 'codes': [], 'values': []})
    return rows


def get_macro_options(msg_type, msg_payload, brain, limit=5000, pref_weights=None, *, index_preferences=None):
    """
    [AI 参谋部] 后台穷举合法素材组合，打包成“套餐”供 AI 挑选
    返回格式: [{'bytes': b'\x01...', 'locs': [loc_raw1, loc_raw2]}, ...]
    """
    if pref_weights is None: pref_weights = {}
    stream = io.BytesIO(msg_payload)
    options = []

    # 1. 抓取犯罪嫌疑卡 (用于雷达日志溯源)
    trigger_card = "未知机制/阶段动作"
    if brain and brain.chain_stack:
        trigger_card = f"【{card_db.get_card_name(brain.chain_stack[-1]['code'])}】"
    elif brain and brain.history_stack:
        trigger_card = f"【{card_db.get_card_name(brain.history_stack[0]['code'])}】"

    if msg_type in MATERIAL_SELECTION_MSGS:
        if limit < 1:
            raise ValueError("素材组合生成预算必须为正数")
        return _material_options(msg_type, msg_payload, brain, limit, pref_weights, index_preferences, trigger_card)

    try:
        if msg_type == MSG_SELECT_COUNTER:
            stream.read(1) # Player
            stream.read(2) # Type
            qty = struct.unpack('<H', stream.read(2))[0]
            size = struct.unpack('<B', stream.read(1))[0]
            
            cards = []
            for i in range(size):
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('<B', stream.read(1))[0]
                l = struct.unpack('<B', stream.read(1))[0]
                s = struct.unpack('<B', stream.read(1))[0]
                avail = struct.unpack('<H', stream.read(2))[0]
                cards.append({'idx': i, 'avail': avail, 'code': code, 'loc': LocationInfo.encode(c, l, s, 0)})
            
            # DFS 分配指示物
            def distribute_counters(card_idx, remaining_qty, current_distribution):
                if len(options) >= limit: return
                if card_idx == size:
                    if remaining_qty == 0:
                        resp_buf = bytearray()
                        locs = []
                        codes = []
                        values = []
                        for i, count_val in enumerate(current_distribution):
                            resp_buf.extend(struct.pack('<H', count_val))
                            if count_val > 0:
                                locs.append(cards[i]['loc'])
                                codes.append(cards[i]['code'])
                                values.append(count_val)
                        options.append({
                            'bytes': bytes(resp_buf),
                            'locs': locs,
                            'codes': codes,
                            'values': values,
                        })
                    return
                
                max_take = min(cards[card_idx]['avail'], remaining_qty)
                for take in range(max_take, -1, -1):
                    current_distribution.append(take)
                    distribute_counters(card_idx + 1, remaining_qty - take, current_distribution)
                    current_distribution.pop()
            
            distribute_counters(0, qty, [])
            
            if len(options) >= limit:
                sample_names = [card_db.get_card_name(c['code']) for c in cards[:4]]
                print(f"📡 [RuleBot 截断雷达] Type 22 (移除指示物) 组合超 {limit}，安全阻断。")
                print(f"   -> 🎯 发动源头: {trigger_card}")
                print(f"   -> 📊 引擎要求: 从 {size} 张卡中移除总计 {qty} 个指示物")
                print(f"   -> 🃏 候选样本: {sample_names}...")

        # 4. 排序卡片
        elif msg_type in (21, 25): # MSG_SORT_CHAIN / MSG_SORT_CARD
            stream.read(1) # P
            count = struct.unpack('<B', stream.read(1))[0]
            cards = []
            for i in range(count):
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('<B', stream.read(1))[0]
                l = struct.unpack('<B', stream.read(1))[0]
                s = struct.unpack('<B', stream.read(1))[0]
                cards.append({'idx': i, 'loc': LocationInfo.encode(c, l, s, 0), 'code': code})
                
            valid_solutions = []
            for i, sol in enumerate(itertools.permutations(cards)):
                valid_solutions.append(sol)
                if i >= limit - 1:
                    sample_names = [card_db.get_card_name(c['code']) for c in cards[:4]]
                    print(
                        f"📡 [RuleBot 截断雷达] Type {msg_type} "
                        "(排序) 选项组达到上限，触发安全阻断"
                    )
                    print(f"   -> 🎯 发动源头: {trigger_card}")
                    print(f"   -> 📊 需要排序的卡片总数: {count}")
                    print(f"   -> 🃏 涉案卡片: {sample_names}...")
                    break
            
            for sol in valid_solutions:
                resp_buf = bytearray()
                locs = []
                codes = []
                for cd in sol:
                    resp_buf.append(cd['idx'])
                    locs.append(cd['loc'])
                    codes.append(cd['code'])
                options.append({
                    'bytes': bytes(resp_buf),
                    'locs': locs,
                    'codes': codes,
                    'values': list(range(1, len(codes) + 1)),
                })

        # 5. 格子/区域封锁
        elif msg_type in [18, 24]:
            stream.read(1) # P
            count = struct.unpack('<B', stream.read(1))[0]
            mask = struct.unpack('<I', stream.read(4))[0]
            req_player = msg_payload[0] 
            
            if count == 0:
                options.append({'bytes': bytes([0, 0, 0]), 'places': []})
            
            avail_zones = []
            for i in range(32):
                if not (mask & (1 << i)): avail_zones.append(i)
            
            calc_count = max(1, count)
            all_combos = []
            for i, combo in enumerate(itertools.combinations(avail_zones, calc_count)):
                all_combos.append(combo)
                if i >= limit - 1:
                    print(f"📡 [RuleBot 截断雷达] Type {msg_type} (格子选择) 达到上限阻断。")
                    print(f"   -> 🎯 发动源头: {trigger_card}")
                    print(f"   -> 📊 可用空余格子数: {len(avail_zones)}, 需选: {calc_count}")
                    break
            
            for combo in all_combos:
                resp_buf = bytearray()
                places = []
                for i in combo:
                    p_flag = 1 if i >= 16 else 0
                    l = 0x08 if (i % 16) >= 8 else 0x04
                    s = i % 8
                    raw_p = req_player if p_flag == 0 else (1 - req_player)
                    final_p = 1 if raw_p == 1 else 0
                    resp_buf.extend([final_p, l, s])
                    places.append(i)
                options.append({'bytes': bytes(resp_buf), 'places': places})

        # 6. 种族/属性宣言必须一次返回恰好 count 个 bit 的按位或结果
        elif msg_type in [MSG_ANNOUNCE_RACE, MSG_ANNOUNCE_ATTRIB]:
            stream.read(1) # P
            required_count = struct.unpack('<B', stream.read(1))[0]
            available_mask = struct.unpack('<I', stream.read(4))[0]
            available_bits = [
                bit_index
                for bit_index in range(32)
                if available_mask & (1 << bit_index)
            ]
            for option_index, bit_indices in enumerate(
                itertools.combinations(available_bits, required_count)
            ):
                if option_index >= limit:
                    print(
                        f"📡 [RuleBot 截断雷达] Type {msg_type} 宣言组合达到 {limit}，安全截断。"
                    )
                    break
                response_value = 0
                for bit_index in bit_indices:
                    response_value |= 1 << bit_index
                options.append({
                    'value': response_value,
                    'indices': list(bit_indices),
                    'values': [bit_index + 1 for bit_index in bit_indices],
                })

    except Exception as e:
        print(f"⚠️ [参谋部] get_macro_options 解析错误: {e}")
        import traceback
        traceback.print_exc()

    return options
