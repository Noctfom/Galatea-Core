'''
gamestate模块
用于维护和更新决斗状态，并生成全息数据快照
'''

import struct
import io
import traceback # <--- 新增
from game_constants import LocationInfo, Zone, Phases
from collections import defaultdict
from card_reader import card_db
from data_types import GameSnapshot, GlobalFeature, CardEntity, GameAction

class MessageParser:
    # 基于源码的精确长度定义 (Payload长度)
    # -1: 变长消息，进入 calculate_dynamic_length
    MSG_LEN = {
        0:-1, 1: 0, 2: 6, 3: 0, 4: 0, 5: 2, 
        
        # --- 交互类 ---
        10: -1, 11: -1, 12: 13, 13: 5, 14: -1, 15: -1, 16: -1, 
        18: 6,  # PLACE (1+1+4) [已验证]
        19: 6, 20: -1, 22: -1, 23: -1, 
        24: 6,  # DISFIELD (同18) [已验证]
        25: -1, 26: -1, 
        
        # --- 确认/展示类 ---
        30: -1, 31: -1, 32: 1, 33: -1, 34: -1, 
        35: 1, # SWAP_GRAVE_DECK
        36: -1, 37: 0, 38: 6, 39: -1,
        
        # --- 流程/数值 ---
        40: 1, 41: 2, 42: -1,
        
        # --- 动作类 ---
        50: 16, 53: 9, 54: 8, 55: 16, 56: 4, 
        
        # --- 召唤/连锁 ---
        60: 8, 
        61: 0, 63: 0, 65: 0, # SUMMONED [已验证 0字节]
        62: 8, 64: 8, 
        
        70: 16, 
        71: 1, 72: 1, 73: 1, # CHAINED [已验证 1字节]
        74: 0, # CHAIN_END [已验证 0字节]
        75: 1, 76: 1, # NEGATED [已验证 1字节]
        
        # --- 对象/指示物 ---
        81: -1, 83: -1, # BECOME_TARGET [已验证 变长]
        
        # --- 伤害/数值 ---
        90: -1, 91: 5, 92: 5, 93: 8, 94: 5, 
        96: 8, 97: 8, # CARD_TARGET [已验证 4+4]
        100: 5, # PAY_LPCOST [已验证 1+4]
        101: 7, 102: 7, # COUNTER [已验证 2+1+1+1+2]
        
        # --- 战斗 ---
        110: 8, 111: 26, 112: 0, 113: 0, 114: 0, 
        
        # --- 杂项 ---
        120: 8,
        130: -1, 131: -1, 132: 1, 133: 1,
        140: 6, 141: 6, 142: -1, 
        143: -1, # ANNOUNCE_NUMBER [已验证]
        160: 9, 
        161: -1, 162: -1, 
        163: -1, 164: -1, # AI_NAME / SHOW_HINT [已验证 字符串]
        165: 6, # PLAYER_HINT [已验证 1+1+4]
        170: 4
    }

    @staticmethod
    def calculate_dynamic_length(msg_type, stream):
        start_pos = stream.tell()
        length = 0
        
        try:
            # 11: IDLECMD
            if msg_type == 11: 
                stream.read(1); length += 1 # P
                for i in range(6): 
                    b = stream.read(1); length += 1
                    count = struct.unpack('B', b)[0]
                    item_len = 11 if i == 5 else 7
                    stream.read(count * item_len); length += count * item_len
                stream.read(3); length += 3

            # 10: BATTLECMD
            elif msg_type == 10:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1
                c = struct.unpack('B', b)[0]
                stream.read(c * 11); length += c * 11
                b = stream.read(1); length += 1
                c = struct.unpack('B', b)[0]
                stream.read(c * 8); length += c * 8
                stream.read(2); length += 2

            # 14: SELECT_OPTION 
            elif msg_type == 14:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 4); length += count * 4

            # 15/20: SELECT_CARD / TRIBUTE
            elif msg_type in [15, 20]:
                stream.read(4); length += 4 # P, Cancel, Min, Max
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 8); length += count * 8

            # 16: SELECT_CHAIN
            elif msg_type == 16:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(10); length += 10 # Spe, forced, h1, h2
                
                # 🌟 [额外修复] 元素之间有 1 个字节的定界符 (共 count-1 个)
                extra_bytes = max(0, count - 1)
                stream.read(count * 13 + extra_bytes); length += count * 13 + extra_bytes

            # 18/24: PLACE / DISFIELD
            elif msg_type in [18, 24]:
                stream.read(6); length = 6

            # 21: SORT_CHAIN
            elif msg_type == 21:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 7); length += count * 7

            # 22: SELECT_COUNTER
            elif msg_type == 22:
                stream.read(5); length += 5 # P, type(2), qty(2)
                b = stream.read(1); length += 1 # Size
                size = struct.unpack('B', b)[0]
                stream.read(size * 9); length += size * 9

            # 23: SELECT_SUM 
            elif msg_type == 23:
                stream.read(8); length += 8 # Header
                b = stream.read(1); length += 1 # Must_count
                must_c = struct.unpack('B', b)[0]
                stream.read(must_c * 11); length += must_c * 11
                b = stream.read(1); length += 1 # Sel_count
                sel_c = struct.unpack('B', b)[0]
                stream.read(sel_c * 11); length += sel_c * 11

            # 25: SORT_CARD
            elif msg_type == 25:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 7); length += count * 7

            # 26: SELECT_UNSELECT
            elif msg_type == 26:
                stream.read(5); length += 5 # Header 5 bytes
                b = stream.read(1); length += 1 # Size_A
                size_a = struct.unpack('B', b)[0]
                stream.read(size_a * 8); length += size_a * 8
                b = stream.read(1); length += 1 # Size_B
                size_b = struct.unpack('B', b)[0]
                stream.read(size_b * 8); length += size_b * 8

            # 30/34/42: CONFIRM
            elif msg_type in [30, 34, 42]:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 7); length += count * 7

            # 31: CONFIRM_CARDS
            elif msg_type == 31:
                stream.read(1); length += 1 # P
                # 🌟 [额外修复] 吞掉强制插入的未知幽灵字节
                stream.read(1); length += 1 
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 7); length += count * 7

            # 36: SHUFFLE_SET_CARD
            elif msg_type == 36:
                stream.read(1); length += 1 # Loc
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 8); length += count * 8

            # 33/39/81/90/142/143: 1P + 1Count + Count*4
            elif msg_type in [33, 39, 81, 90, 142, 143]:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 4); length += count * 4

            # 83: BECOME_TARGET
            elif msg_type == 83:
                b = stream.read(1); length += 1 # Count (无 P 字节)
                count = struct.unpack('B', b)[0]
                stream.read(count * 4); length += count * 4

            # 130/131: TOSS_COIN/DICE
            elif msg_type in [130, 131]:
                stream.read(1); length += 1 # P
                b = stream.read(1); length += 1 # Count
                count = struct.unpack('B', b)[0]
                stream.read(count * 1); length += count * 1

            # 161: TAG_SWAP
            elif msg_type == 161:
                stream.read(1); length += 1 # P
                b = stream.read(4); length += 4 # main, extra, extra_p, hand
                _, extra_len, _, hand_len = struct.unpack('BBBB', b)
                stream.read(4); length += 4 # Deck top
                stream.read(hand_len * 4); length += hand_len * 4
                stream.read(extra_len * 4); length += extra_len * 4

            # 162: RELOAD_FIELD
            elif msg_type == 162:
                b = stream.read(1); length += 1
                rule = struct.unpack('B', b)[0]
                mzone_size = 7 if rule >= 4 else 5
                for _ in range(2):
                    stream.read(4); length += 4 
                    for _ in range(mzone_size):
                        b = stream.read(1); length += 1
                        if struct.unpack('B', b)[0] != 0:
                            stream.read(2); length += 2 
                    for _ in range(8):
                        b = stream.read(1); length += 1
                        if struct.unpack('B', b)[0] != 0:
                            stream.read(1); length += 1 
                    stream.read(6); length += 6 
                b = stream.read(1); length += 1
                chain_size = struct.unpack('B', b)[0]
                stream.read(chain_size * 15); length += chain_size * 15

            # 163/164: STRING MESSAGES
            elif msg_type in [163, 164]:
                b = stream.read(2); length += 2
                str_len = struct.unpack('H', b)[0]
                stream.read(str_len + 1); length += str_len + 1

            else:
                stream.read()
                length = stream.tell() - start_pos

        except Exception as e:
            stream.seek(start_pos)
            return -1
            
        stream.seek(start_pos)
        return length

    @staticmethod
    def parse(data):
        msgs = []
        stream = io.BytesIO(data)
        data_len = len(data)
        
        # 1. 严格白名单 (移除 0)
        VALID_MSGS = {
            1, 2, 3, 4, 5, 
            10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42,
            50, 53, 54, 55, 56, 
            60, 61, 62, 63, 64, 65,
            70, 71, 72, 73, 74, 75, 76, 
            81, 83, 
            90,
            91, 92, 93, 94, 96, 97, 
            100, 101, 102, 
            110, 111, 112, 113, 114, 
            120,
            130, 131, 132, 133, 
            140, 141, 142, 143, 
            160, 163, 164, 165, 170
        }
        
        # 🌟 [探针] 记录最近 15 个成功解析的指令，看清乱码源头！
        recent_msgs = []

        while stream.tell() < data_len:
            start_pos = stream.tell()
            b = stream.read(1)
            if not b: break
            msg_type = struct.unpack('B', b)[0]

            # [异常拦截]
            if msg_type not in VALID_MSGS:
                print(f"\n👻 [Parser] 抓到幽灵: {msg_type} (Hex: {hex(msg_type)})!")
                print(f"🔍 坠机前15个指令: {recent_msgs}")
                
                # 🌟 终极杀手锏：打印整个数据包的十六进制字节流
                hex_dump = " ".join([f"{b:02X}" for b in data])
                print(f"📦 完整数据包Hex (总长度 {data_len}):\n{hex_dump}")
                
                # 计算当前出错在第几个字节
                error_pos = stream.tell() - 1
                print(f"📍 错位发生位置: 字节 {error_pos}")
                break

            # [虚假胜利拦截]
            if msg_type == 5:
                if stream.tell() < data_len:
                    wb = stream.read(1)
                    winner = struct.unpack('B', wb)[0]
                    
                    reason = 0
                    if stream.tell() < data_len:
                        rb = stream.read(1)
                        reason = struct.unpack('B', rb)[0]
                        
                    stream.seek(start_pos + 1)
                    
                    if winner > 2 or reason > 0x30: 
                        print(f"\n🛡️ [Parser] 拦截到残存胜利! \n🔍 坠机前15个指令: {recent_msgs}")
                        break 

            # 计算长度
            length = MessageParser.MSG_LEN.get(msg_type, -1)
            
            if length == -1:
                try:
                    length = MessageParser.calculate_dynamic_length(msg_type, stream)
                except Exception as e:
                    print(f"\n👻 [Parser] 变长解析崩溃 (msg: {msg_type})! \n🔍 坠机前15个指令: {recent_msgs}")
                    break
            
            if length >= 0:
                if stream.tell() + length > data_len: break 
                payload = stream.read(length)
                msgs.append(bytes([msg_type]) + payload)
                
                # 🌟 [探针记录] 保留 15 个
                recent_msgs.append(msg_type)
                if len(recent_msgs) > 15:
                    recent_msgs.pop(0)
            else:
                break
                
        return msgs
    
# ==================================================================================
#  DuelState V2.1 - 包含动作解析
# ==================================================================================

class DuelState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn = 0
        self.phase = 0
        self.my_lp = 8000
        self.op_lp = 8000
        self.active_player = 0
        self.field_map = {0: defaultdict(dict), 1: defaultdict(dict)}
        
        # [新增] 当前挂起的合法动作列表
        # 每次收到交互消息 (IDLE, CHAIN, CARD...) 时更新
        self.current_valid_actions = [] 

    def update(self, msg_type, msg_payload):
        """解析消息，更新状态 + 解析合法动作"""
        try:
            stream = io.BytesIO(msg_payload)
            
            # --- 状态维护 (与 V2.0 相同) ---
            if msg_type == 50: # MSG_MOVE
                code, old_raw, new_raw, reason = struct.unpack('<IIII', stream.read(16))
                old_c, old_l, old_s, _ = LocationInfo.decode(old_raw)
                new_c, new_l, new_s, new_pos = LocationInfo.decode(new_raw)
                # 🛡️ [防弹衣] 防止 new_c = 39 导致的 KeyError
                if old_c in [0, 1] and old_l != 0 and old_s in self.field_map[old_c][old_l]:
                    del self.field_map[old_c][old_l][old_s]
                if new_c in [0, 1] and new_l != 0:
                    self.field_map[new_c][new_l][new_s] = {'code': code, 'pos': new_pos, 'owner': new_c}

            elif msg_type == 53: # POS_CHANGE
                code, c, l, s, prev, new_pos = struct.unpack('<IBBBB B', stream.read(9))
                if s in self.field_map[c][l]: self.field_map[c][l][s]['pos'] = new_pos

            elif msg_type == 90: # DRAW (抽卡)
                # 🌟 [录像修复 B] 记录抽卡到手牌！
                p = struct.unpack('B', stream.read(1))[0]
                count = struct.unpack('B', stream.read(1))[0]
                for _ in range(count):
                    raw_code = struct.unpack('<I', stream.read(4))[0]
                    code = raw_code & 0x7FFFFFFF
                    seq = 0
                    while seq in self.field_map[p][Zone.HAND]: seq += 1
                    self.field_map[p][Zone.HAND][seq] = {'code': code, 'pos': 0, 'owner': p}

            elif msg_type in [91, 92, 94]: # 伤害 / 回复 / LP直接更新
                # 🌟 [录像修复 A] 加上 '<' 强制对齐，并监听 91 和 92！
                p, val = struct.unpack('<BI', stream.read(5))
                if p == 0:
                    if msg_type == 91: self.my_lp = max(0, self.my_lp - val)
                    elif msg_type == 92: self.my_lp += val
                    else: self.my_lp = val
                else:
                    if msg_type == 91: self.op_lp = max(0, self.op_lp - val)
                    elif msg_type == 92: self.op_lp += val
                    else: self.op_lp = val

            elif msg_type == 94: # LP
                p, lp = struct.unpack('<BI', stream.read(5))
                if p == 0: self.my_lp = lp
                else: self.op_lp = lp
            
            elif msg_type == 40: self.turn += 1
            elif msg_type == 41: self.phase = struct.unpack('H', stream.read(2))[0]

            # --- [新增] 动作空间解析 (Action Parsing) ---
            # 如果是交互消息，解析出 valid_actions
            if msg_type in [10, 11, 12, 13, 14, 15, 16, 18, 19, 24]:
                self.active_player = struct.unpack('B', msg_payload[0:1])[0]
                self._parse_valid_actions(msg_type, stream)
            else:
                # 收到其他消息（如状态更新），清空动作列表
                # 只有在等待用户输入时，这个列表才不为空
                self.current_valid_actions = []

        except Exception as e:
            pass

    def _parse_valid_actions(self, msg_type, stream):
        """
        解析交互消息，生成 GameAction 列表
        """
        self.current_valid_actions = []
        stream.seek(0)
        
        try:
            # 1. MSG_SELECT_IDLECMD (11)
            if msg_type == 11:
                # 尝试读取 Player，读不到就直接退出
                b = stream.read(1)
                if not b: return 
                
                action_types = [0, 1, 2, 3, 4, 5]
                
                for at in action_types:
                    b = stream.read(1)
                    if not b: break # <--- [修改] 读不到就停，别报错
                    count = struct.unpack('B', b)[0]
                    
                    need_bytes = 11 if at == 5 else 7
                    for i in range(count):
                        # 预读检查
                        raw_bytes = stream.read(need_bytes)
                        if len(raw_bytes) < need_bytes: break # <--- [修改] 数据不够也停
                        
                        # 手动解包
                        if at == 5:
                            code = struct.unpack('<I', raw_bytes[0:4])[0]
                            c = struct.unpack('B', raw_bytes[4:5])[0]
                            l = struct.unpack('B', raw_bytes[5:6])[0]
                            s = struct.unpack('B', raw_bytes[6:7])[0]
                            desc = struct.unpack('<I', raw_bytes[7:11])[0]
                        else:
                            code = struct.unpack('<I', raw_bytes[0:4])[0]
                            c = struct.unpack('B', raw_bytes[4:5])[0]
                            l = struct.unpack('B', raw_bytes[5:6])[0]
                            s = struct.unpack('B', raw_bytes[6:7])[0]
                            desc = 0
                        
                        loc_raw = LocationInfo.encode(c, l, s, 0)
                        self.current_valid_actions.append(
                            GameAction(action_type=at, index=i, target_entity_idx=loc_raw, desc_id=desc)
                        )
                
                # Phase Buttons (尝试读取)
                # 能读几个是几个，绝对不报错
                bp = 0; ep = 0
                b = stream.read(1)
                if b: bp = struct.unpack('B', b)[0]
                
                b = stream.read(1)
                if b: ep = struct.unpack('B', b)[0]
                
                # shuf 读不读无所谓
                
                if bp: self.current_valid_actions.append(GameAction(action_type=6, index=0, desc_str="To BP"))
                if ep: self.current_valid_actions.append(GameAction(action_type=7, index=0, desc_str="To EP"))

            # 2. MSG_SELECT_CHAIN (16)
            elif msg_type == 16:
                stream.read(1) # P
                count = struct.unpack('B', stream.read(1))[0]
                stream.read(10) # Spe, Forced, H1, H2
                
                for i in range(count):
                    stream.read(1) # Flag
                    code = struct.unpack('<I', stream.read(4))[0]
                    loc_val = struct.unpack('<I', stream.read(4))[0]
                    desc = struct.unpack('<I', stream.read(4))[0]
                    
                    # 🌟 [额外修复] 吞掉元素之间的 1 字节定界符
                    if i < count - 1:
                        stream.read(1)
                    
                    self.current_valid_actions.append(
                        GameAction(action_type=16, index=i, target_entity_idx=loc_val, desc_id=desc)
                    )
                
                # 连锁允许取消 (-1)
                self.current_valid_actions.append(GameAction(action_type=16, index=-1, desc_str="Cancel"))

            # 3. MSG_SELECT_CARD (15)
            # 结构: P + Cancelable + Min + Max + Count + List(Code4+Loc4)
            elif msg_type == 15:
                stream.read(1) # P
                can_cancel = struct.unpack('B', stream.read(1))[0]
                stream.read(2) # Min, Max
                count = struct.unpack('B', stream.read(1))[0]
                
                for i in range(count):
                    code = struct.unpack('<I', stream.read(4))[0]
                    loc_val = struct.unpack('<I', stream.read(4))[0]
                    
                    self.current_valid_actions.append(
                        GameAction(action_type=15, index=i, target_entity_idx=loc_val)
                    )
                
                if can_cancel or count == 0:
                    self.current_valid_actions.append(GameAction(action_type=15, index=-1, desc_str="Cancel"))
            
            # 4. MSG_SELECT_BATTLECMD (10)
            elif msg_type == 10:
                stream.read(1) # Player
                
                # --- A. Activatable (发动效果) ---
                # C++: write_buffer8(core.select_chains.size())
                count = struct.unpack('B', stream.read(1))[0]
                
                for i in range(count):
                    # C++ 结构 (11 字节): 
                    # Code(4) + Controler(1) + Location(1) + Sequence(1) + Desc(4)
                    code = struct.unpack('<I', stream.read(4))[0]
                    c = struct.unpack('B', stream.read(1))[0]
                    l = struct.unpack('B', stream.read(1))[0]
                    s = struct.unpack('B', stream.read(1))[0]
                    desc = struct.unpack('<I', stream.read(4))[0]
                    
                    # 编码位置 -> Entity ID
                    loc_raw = LocationInfo.encode(c, l, s, 0)
                    
                    # 🟢 [修正] action_type 必须是 0 (C++ t=0)
                    self.current_valid_actions.append(
                        GameAction(action_type=0, index=i, target_entity_idx=loc_raw, desc_id=desc)
                    )

                # --- B. Attackable (攻击宣言) ---
                # C++: write_buffer8(core.attackable_cards.size())
                count_atk = struct.unpack('B', stream.read(1))[0]
                
                for i in range(count_atk):
                    # C++ 结构 (8 字节):
                    # Code(4) + Controler(1) + Location(1) + Sequence(1) + Direct(1)
                    code = struct.unpack('<I', stream.read(4))[0]
                    c = struct.unpack('B', stream.read(1))[0]
                    l = struct.unpack('B', stream.read(1))[0]
                    s = struct.unpack('B', stream.read(1))[0]
                    direct = struct.unpack('B', stream.read(1))[0]
                    
                    # 编码位置 -> Entity ID
                    loc_raw = LocationInfo.encode(c, l, s, 0)
                    
                    # 🟢 [修正] action_type 必须是 1 (C++ t=1)
                    desc_str = "Direct Attack" if direct else f"Attack {code}"
                    self.current_valid_actions.append(
                        GameAction(action_type=1, index=i, target_entity_idx=loc_raw, desc_str=desc_str)
                    )

                # --- C. Phase Transition ---
                m2 = struct.unpack('B', stream.read(1))[0]
                ep = struct.unpack('B', stream.read(1))[0]
                
                # 🟢 [修正] M2=2, EP=3 (C++ t=2, t=3)
                if m2: self.current_valid_actions.append(GameAction(action_type=2, index=0, desc_str="To M2"))
                if ep: self.current_valid_actions.append(GameAction(action_type=3, index=0, desc_str="To EP"))

            # 5. MSG_SELECT_YESNO (13) / EFFECTYN (12)
            elif msg_type == 12:
                # [C++源码证实] 有卡片实体
                # 结构: P(1) + Code(4) + C(1) + L(1) + S(1) + Desc(4)
                stream.read(1) # P
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                desc = struct.unpack('<I', stream.read(4))[0]
                
                # 🟢 关键：绑定卡片位置！让 AI 知道是哪张卡在问
                loc_raw = LocationInfo.encode(c, l, s, 0)
                
                # Index 1=Yes, 0=No
                self.current_valid_actions.append(
                    GameAction(action_type=msg_type, index=1, target_entity_idx=loc_raw, desc_id=desc, desc_str="Yes")
                )
                self.current_valid_actions.append(
                    GameAction(action_type=msg_type, index=0, target_entity_idx=loc_raw, desc_id=desc, desc_str="No")
                )

            elif msg_type == 13:
                # [C++源码证实] 无卡片实体，通用询问
                # 结构: P(1) + Desc(4)
                stream.read(1)
                desc = struct.unpack('<I', stream.read(4))[0]
                # target_entity_idx = -1
                self.current_valid_actions.append(GameAction(action_type=msg_type, index=1, desc_id=desc, desc_str="Yes"))
                self.current_valid_actions.append(GameAction(action_type=msg_type, index=0, desc_id=desc, desc_str="No"))

            # 6. MSG_SELECT_OPTION (14)
            elif msg_type == 14:
                stream.read(1) # P
                count = struct.unpack('B', stream.read(1))[0]
                for i in range(count):
                    self.current_valid_actions.append(GameAction(action_type=14, index=i, desc_str=f"Option {i}"))

            # 7. MSG_SELECT_POSITION (19)
            elif msg_type == 19:
                stream.read(5) # P + Code
                mask = struct.unpack('B', stream.read(1))[0]
                # 0x1:ATK, 0x2:ATK_down(N/A), 0x4:DEF, 0x8:DEF_down
                if mask & 0x1: self.current_valid_actions.append(GameAction(action_type=19, index=1, desc_str="ATK"))
                if mask & 0x2: self.current_valid_actions.append(GameAction(action_type=19, index=2, desc_str="ATK_Down"))
                if mask & 0x4: self.current_valid_actions.append(GameAction(action_type=19, index=4, desc_str="DEF"))
                if mask & 0x8: self.current_valid_actions.append(GameAction(action_type=19, index=8, desc_str="Set"))

            # 8. MSG_SELECT_PLACE (18) / DISFIELD (24) - [攻克难点！]
            elif msg_type in [18, 24]:
                stream.read(1) # P
                count = struct.unpack('B', stream.read(1))[0]
                mask = struct.unpack('<I', stream.read(4))[0]
                # 把 mask 解压成具体的格子位置
                for i in range(32):
                    if not (mask & (1 << i)):
                        # 这是一个可用的格子
                        # 我们把 i 作为 index 传给 AI，翻译时再转回 locations
                        self.current_valid_actions.append(GameAction(action_type=18, index=i, desc_str=f"Zone {i}"))

        except Exception:
            '''
            print(f"❌ [Parser Error] Failed to parse msg_type {msg_type}")
            traceback.print_exc() # <--- 关键！打印完整堆栈
            print(f"📦 Payload (Hex): {stream.getvalue().hex()}") # 打印原始数据
            '''
            pass

    def get_snapshot(self) -> GameSnapshot:
        """
        生成快照 + 填充 Actions
        """
        def count_zone(p, loc): return len(self.field_map[p].get(loc, {}))
        
        global_feat = GlobalFeature(
            turn_count=self.turn, phase_id=self.phase, to_play=self.active_player,
            my_lp=self.my_lp, op_lp=self.op_lp,
            my_hand_len=count_zone(0, Zone.HAND), op_hand_len=count_zone(1, Zone.HAND),
            my_deck_len=count_zone(0, Zone.DECK), op_deck_len=count_zone(1, Zone.DECK),
            my_grave_len=count_zone(0, Zone.GRAVE), op_grave_len=count_zone(1, Zone.GRAVE),
            my_removed_len=count_zone(0, Zone.REMOVED), op_removed_len=count_zone(1, Zone.REMOVED),
            my_extra_len=count_zone(0, Zone.EXTRA), op_extra_len=count_zone(1, Zone.EXTRA)
        )

        entities = []
        # 构建查找表: (p, l, s) -> entity_index
        # 用于把 Action 里的 Loc 转换成 Entity Index
        loc_to_idx_map = {} 
        
        idx_counter = 0
        zones_order = [Zone.MZONE, Zone.SZONE, Zone.HAND, Zone.GRAVE, Zone.REMOVED, Zone.EXTRA]
        
        for player in [0, 1]:
            for zone in zones_order:
                card_dict = self.field_map[player].get(zone, {})
                sorted_seqs = sorted(card_dict.keys())
                
                for seq in sorted_seqs:
                    info = card_dict[seq]
                    code = info['code']
                    pos = info['pos']
                    # 🛡️ [防弹衣] 防止脏数据 code=39 导致的 KeyError
                    try:
                        stats = card_db.get_full_stats(code)
                    except Exception:
                        stats = [0]*10
                    
                    # 记录位置映射
                    # LocationInfo: C | L<<8 | S<<16 | P<<24
                    # 这里的 Key 我们做一个简化的 Hash
                    loc_key = (player, zone, seq)
                    loc_to_idx_map[loc_key] = idx_counter

                    type_mask = stats[0]
                    race = stats[1]
                    attr = stats[2]
                    level = stats[3]
                    lscale = stats[4]
                    rscale = stats[5]
                    base_atk = stats[8]
                    base_def = stats[9]
                    setcodes = stats[10] # 接收字段集合
                    
                    link_marker = 0
                    # TYPE_LINK 的掩码是 0x4000000
                    if type_mask & 0x4000000: 
                        link_marker = base_def # 连接怪兽的 DEF 其实是箭头掩码
                        base_def = 0           # 连接怪兽没有防御力
                    
                    entities.append(CardEntity(
                        code=code, owner=player, location=zone, sequence=seq, position=pos,
                        current_atk=stats[8], current_def=stats[9],
                        type_mask=type_mask, race=race, attribute=attr, level=level,
                        base_atk=base_atk, base_def=base_def,
                        lscale=lscale, rscale=rscale, link_marker=link_marker, # 传入新参数
                        setcodes=setcodes, # 写入实体
                        is_public=(pos & 0x1 or pos & 0x4)
                    ))
                    idx_counter += 1

        # --- [核心步骤] 匹配 Action 指针 ---
        # 把 Action 里的 "Loc数值" 翻译成 "实体列表第几项"
        final_actions = []
        for act in self.current_valid_actions:
            # 深拷贝一下，因为要修改
            new_act = GameAction(act.action_type, act.index, act.target_entity_idx, act.desc_str)
            
            if new_act.target_entity_idx > 0 and new_act.index != -1:
                c, l, s, _ = LocationInfo.decode(new_act.target_entity_idx)
                # 查找 Map
                if (c, l, s) in loc_to_idx_map:
                    new_act.target_entity_idx = loc_to_idx_map[(c, l, s)]
                else:
                    new_act.target_entity_idx = -1 # 没找到实体
            else:
                new_act.target_entity_idx = -1
            
            final_actions.append(new_act)

        return GameSnapshot(
            global_data=global_feat,
            entities=entities,
            valid_actions=final_actions # [NEW] 填充动作列表
        )