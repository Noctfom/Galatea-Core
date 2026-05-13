'''
规则型 Bot 的核心决策逻辑
保证在所有交互请求中，100% 返回合法格式的数据，防止超时
'''


import struct
import io
import random
import itertools
from game_constants import LocationInfo
from card_reader import card_db

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
MSG_SELECT_COUNTER = 22
MSG_SELECT_SUM = 23
MSG_SELECT_DISFIELD = 24
MSG_SORT_CARD = 25
MSG_SELECT_UNSELECT_CARD = 26
MSG_ANNOUNCE_RACE = 140
MSG_ANNOUNCE_ATTRIB = 141
MSG_ANNOUNCE_CARD = 142
MSG_ANNOUNCE_NUMBER = 143

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
        count = struct.unpack('B', b)[0]
        
        for i in range(count):
            stream.read(7) # code(4) + con(1) + loc(1) + seq(1)
            if cat == 5: stream.read(4) # desc
            legal_actions.append({'cat': cat, 'idx': i})          

    b = stream.read(1)
    if b and struct.unpack('B', b)[0]: legal_actions.append({'cat': 6, 'idx': 0}) # BP
    b = stream.read(1)
    if b and struct.unpack('B', b)[0]: legal_actions.append({'cat': 7, 'idx': 0}) # EP
    return legal_actions

def parse_battle_cmd(msg_data):
    """解析 BATTLE 消息"""
    stream = io.BytesIO(msg_data)
    stream.read(1) # player
    legal_actions = []
    
    # Activatable (发动效果)
    b = stream.read(1) 
    if b:
        count = struct.unpack('B', b)[0]
        for i in range(count): 
            stream.read(11) # Skip details
            legal_actions.append({'cat': 0, 'idx': i})
            
    # Attackable
    b = stream.read(1) 
    if b:
        count = struct.unpack('B', b)[0]
        for i in range(count):
            code = struct.unpack('<I', stream.read(4))[0]
            stream.read(4) 
            legal_actions.append({'cat': 1, 'idx': i, 'code': code})
            
    # M2 / EP
    b = stream.read(1); 
    if b and struct.unpack('B', b)[0]: legal_actions.append({'cat': 2, 'idx': 0}) 
    b = stream.read(1); 
    if b and struct.unpack('B', b)[0]: legal_actions.append({'cat': 3, 'idx': 0}) 
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
                count = struct.unpack('B', stream.read(1))[0]
            except:
                return bytes([0])
            possible_choices = list(range(count))
            
            valid_choices = [i for i in possible_choices if i not in ignore_actions]
            
            if valid_choices:
                decision = random.choice(valid_choices)
            else:
                decision = -1 # 尝试取消

        elif msg_type == MSG_SELECT_CHAIN:
            try:
                # 1. 精确读取头部的 4 个字节
                header_start = stream.read(4)
                if len(header_start) < 4: raise Exception("Header incomplete")
                
                count = header_start[1]
                forced = header_start[3] # 强制标志位
                
                # 2. 构造选项：无论是谁，只能从 0 到 count-1 里选
                candidates = list(range(count))
                
                # 如果不是强制发动 (forced == 0)，才可以追加 Cancel (-1)
                if forced == 0:
                    candidates.append(-1)
                    
                # 3. 过滤掉已经被核心拒绝过的操作
                valid_choices = [c for c in candidates if c not in ignore_actions]
                
                # 4. 安全决策
                if valid_choices:
                    decision = random.choice(valid_choices)
                else:
                    # 终极兜底：如果所有合法选项都被拒绝了，说明 C++ 陷入了必须发动的死角
                    # 我们强行返回 0（发动第一个效果），绝不返回 -1
                    decision = 0 
                    
            except Exception:
                decision = 0
            
            return decision

        # ==================== 3. 宣言与竞猜 (新增与补全) ====================
        # 这类消息如果不处理，遇到《抹杀之指名者》等卡会直接卡死
        
        # =================================================================
        # [终极修正] 10. 种族/属性宣言 (位掩码处理)
        # =================================================================
        elif msg_type in [MSG_ANNOUNCE_RACE, MSG_ANNOUNCE_ATTRIB]:
            stream.read(1) # Player
            count = struct.unpack('B', stream.read(1))[0] # 需要选几个
            available = struct.unpack('<I', stream.read(4))[0] # 可选掩码
            
            # 1. 解析掩码，找出所有可用的位 (Races/Attributes)
            options = []
            for i in range(32): # 遍历 32 位
                bit = 1 << i
                if available & bit:
                    options.append(bit)
            
            # 2. 随机选择 count 个
            # 如果 options 不够选，就全选
            if len(options) <= count:
                selected = options
            else:
                selected = random.sample(options, count)
            
            # 3. 计算结果掩码 (求和/按位或)
            result_mask = 0
            for bit in selected:
                result_mask |= bit
            
            # 4. 发送 4 字节整数
            decision = struct.pack('<I', result_mask)

        # =================================================================
        #  卡片/数字宣言 
        # =================================================================
        elif msg_type == MSG_ANNOUNCE_CARD: # 142: 必须返回真实卡密
            stream.read(1)
            count = struct.unpack('B', stream.read(1))[0]
            for _ in range(count): stream.read(4)
            
            # 彻底抛弃原来愚蠢的“盲猜灰流丽”逻辑
            # 直接从 gamestate 传过来的完美选项池里提取合法的卡密
            valid_codes = [act.desc_id for act in _shared_valid_actions if act.action_type == 142]
            
            # 过滤掉已经被引擎拒绝的卡（防死锁保护）
            safe_codes = []
            for code in valid_codes:
                packed = struct.pack('<I', code)
                if packed not in ignore_actions:
                    safe_codes.append(packed)
            
            if safe_codes:
                # 完美过关
                decision = random.choice(safe_codes)
            else:
                print("🚨 [RuleBot 警报] Type 142 (宣言卡片) 没有合法选项可选！")

        elif msg_type == MSG_ANNOUNCE_NUMBER: # 143: 必须返回索引
            stream.read(1)
            count = struct.unpack('B', stream.read(1))[0]
            for _ in range(count): stream.read(4) 
            
            ignored_set = set(b for b in ignore_actions if isinstance(b, bytes))
            decision = struct.pack('<I', 0)
            
            if count > 0:
                valid_indices = []
                for i in range(count):
                    cand = struct.pack('<I', i)
                    if cand not in ignored_set: valid_indices.append(cand)
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
                    mask = struct.unpack('B', mask_byte)[0]
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

        # [RuleBot 修正 1] 选卡/素材：优先凑满 Max (为了连接召唤)
        elif msg_type in [MSG_SELECT_CARD, MSG_SELECT_TRIBUTE]:
            # 真正的 Payload 去掉 Type 后，至少包含 P, Cancel, Min, Max, Count 5个字节
            if len(payload) < 5: 
                return bytes([0])

            stream = io.BytesIO(payload)
            stream.read(1) # 跳过 player_id
            
            try:
                cancelable = struct.unpack('B', stream.read(1))[0]
                min_c = struct.unpack('B', stream.read(1))[0]
                max_c = struct.unpack('B', stream.read(1))[0]
                list_len = struct.unpack('B', stream.read(1))[0]
            except Exception as e:
                print(f"⚠️ [RuleBot] 处理 MSG_SELECT_CARD 时发生异常: {e}")
                return bytes([0]) 
            
            stream.read(list_len * 8) # 跳过卡片数据

            ignored_set = set(b for b in ignore_actions if isinstance(b, bytes))
            decision = None
            
            for _ in range(50):
                real_max = min(max_c, list_len)
                real_min = min(min_c, list_len)
                
                # 🛡️ 强制纠正大小关系，防崩溃
                if real_min > real_max: 
                    real_min = real_max
                
                rand_val = random.random()
                if rand_val < 0.5: count = real_max
                elif rand_val < 0.8: count = real_min
                else: count = random.randint(real_min, real_max)
                
                if count == 0 and min_c > 0: count = min_c
                
                indices = list(range(list_len))
                random.shuffle(indices)
                selected_indices = indices[:count]
                selected_indices.sort()
                
                resp_buf = bytearray()
                resp_buf.append(count)
                for idx in selected_indices:
                    resp_buf.append(idx)
                
                candidate = bytes(resp_buf)
                if candidate not in ignored_set:
                    decision = candidate
                    break
            
            if decision is None:
                # 如果代码走到这里，说明能选的所有组合，全被引擎 RETRY 拒绝了！
                # [新增报警]
                print(f"🚨 [RuleBot 警报] Type {msg_type} (选卡/祭品) 算法穷尽！所有生成组合全在黑名单！")
                print(f"   -> 引擎要求: Min={min_c}, Max={max_c}, 可选列表长度={list_len}")
                print(f"   -> 当前黑名单: {ignore_actions}")
                if cancelable: 
                    decision = struct.pack('<i', -1)
                else: 
                    # 清空黑名单强行选最初的，总比发错格式好
                    decision = candidate if 'candidate' in locals() else bytes([0])

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
                    finishable = struct.unpack('B', stream.read(1))[0]
                    cancelable = struct.unpack('B', stream.read(1))[0]
                    min_c = struct.unpack('B', stream.read(1))[0]
                    max_c = struct.unpack('B', stream.read(1))[0]
                    
                    size_a = struct.unpack('B', stream.read(1))[0]
                    stream.read(size_a * 8) # 跳过 List A
                    
                    size_b = struct.unpack('B', stream.read(1))[0]
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

        # =================================================================
        # 严格修正：MSG_SELECT_SUM (23) - 支持动态双重数值与 DFS 提取
        # =================================================================
        elif msg_type == MSG_SELECT_SUM:
            if len(payload) < 10: return bytes([0])
            try:
                stream = io.BytesIO(payload)
                mode = struct.unpack('B', stream.read(1))[0]
                stream.read(1) # 跳过 player_id
                total_acc = struct.unpack('<I', stream.read(4))[0]
                min_c = struct.unpack('B', stream.read(1))[0]
                max_c = struct.unpack('B', stream.read(1))[0]
                
                must_count = struct.unpack('B', stream.read(1))[0]
                must_vals = []
                for _ in range(must_count):
                    stream.read(7)
                    v = struct.unpack('<I', stream.read(4))[0]
                    must_vals.append(v)
                
                count_b = stream.read(1)
                if not count_b: return bytes([0])
                count = struct.unpack('B', count_b)[0]

                candidates = []
                for i in range(count):
                    stream.read(7)
                    val = struct.unpack('<I', stream.read(4))[0]
                    candidates.append({'index': i, 'val': val})
                
                valid_solutions = []
                real_max = max_c if max_c > 0 else count
                
                def check_sum(vals, current_idx, current_sum, current_min):
                    if current_idx == len(vals):
                        if mode == 0: return current_sum == total_acc
                        else: return current_sum >= total_acc and (current_sum - current_min) < total_acc
                    
                    v = vals[current_idx]
                    v1 = v & 0xffff
                    v2 = v >> 16
                    
                    n_min1 = min(current_min, v1) if current_min != -1 else v1
                    if check_sum(vals, current_idx + 1, current_sum + v1, n_min1): return True
                    
                    if v2 > 0 and v2 != v1:
                        n_min2 = min(current_min, v2) if current_min != -1 else v2
                        if check_sum(vals, current_idx + 1, current_sum + v2, n_min2): return True
                    return False

                def backtrack(start, k, path):
                    if len(valid_solutions) >= 200: return # 规则机器人不需要穷尽，找200个够用了
                    if k >= min_c:
                        combined_vals = must_vals + [x['val'] for x in path]
                        if check_sum(combined_vals, 0, 0, -1):
                            valid_solutions.append(list(path))
                    
                    if k == real_max or start == count: return
                        
                    for i in range(start, count):
                        path.append(candidates[i])
                        backtrack(i + 1, k + 1, path)
                        path.pop()

                backtrack(0, 0, [])
                
                ignored_set = set(b for b in ignore_actions if isinstance(b, bytes))
                decision = bytes([0])
                
                if valid_solutions:
                    random.shuffle(valid_solutions) # 洗牌增加盲打多样性
                    for sol in valid_solutions:
                        sol_sorted = sorted(sol, key=lambda x: x['index'])
                        resp_buf = bytearray([must_count + len(sol_sorted)])
                        for _ in range(must_count): resp_buf.append(0)
                        for cd in sol_sorted: resp_buf.append(cd['index'])
                        
                        candidate_bytes = bytes(resp_buf)
                        if candidate_bytes not in ignored_set:
                            decision = candidate_bytes
                            break
                else:
                    print("🚨 [RuleBot 警报] Type 23 (凑星计算) DFS 算法在必须选取的情况下无解！")
                    print(f"   -> 目标={total_acc}, Mode={mode}, Must={must_vals}, 候选池={[c['val'] for c in candidates]}")
                    decision = struct.pack('<i', -1) # 只能尝试发送取消指令
                            
                return decision
            except Exception as e:
                print(f"⚠️ [RuleBot] 处理 MSG_SELECT_SUM 时发生异常: {e}")
                return bytes([0])

        # ==================== 6. 排序与位置 (MSG_SORT_CARD) ====================
        elif msg_type == MSG_SORT_CARD:
            stream.read(1) # Player
            count = struct.unpack('B', stream.read(1))[0]
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
            req_player = struct.unpack('B', stream.read(1))[0]
            count = struct.unpack('B', stream.read(1))[0]
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
            qty = struct.unpack('H', stream.read(2))[0] # 需要移除的总数
            size = struct.unpack('B', stream.read(1))[0] # 列表长度
            
            cards = []
            for i in range(size):
                # 9 Bytes: Code(4)+C(1)+L(1)+S(1)+Avail(2)
                stream.read(7)
                avail = struct.unpack('H', stream.read(2))[0]
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
                resp_buf.extend(struct.pack('H', count_val))
                
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
            count = struct.unpack('B', stream.read(1))[0]
            # 这里的逻辑通常是 Core 告诉客户端结果，或者是客户端确认
            # 如果需要回复，通常是发 0 或 1 (猜正反)
            # 简单起见，发 0 (Heads) 或 1 (Tails) * Count
            resp_buf = bytearray([random.choice([0, 1]) for _ in range(count)])
            decision = bytes(resp_buf)

        elif msg_type == 131: # MSG_TOSS_DICE
            stream.read(1) # Player
            count = struct.unpack('B', stream.read(1))[0]
            # 同上，回复占位符
            resp_buf = bytearray([0] * count)
            decision = bytes(resp_buf)
            
    except Exception as e:
        # 万一解析崩了，也要保证返回合法格式，避免 Bot 卡死
        print(f"[RuleBot] 解析出现错误 {msg_type}: {e}")
        decision = -1

    if decision is None:
        print(f"🚨 [RuleBot 警报] 消息类型 {msg_type} 的解析逻辑彻底失效，没有任何分支返回 decision！")
        print(f"   -> Payload 长度: {len(payload)}")
        decision = -1
    
    return decision


def get_macro_options(msg_type, msg_payload, brain, limit=5000, pref_weights=None):
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
    
    try:
        # 1. 普通选卡 / 祭品 (Link, Xyz, 融合)
        if msg_type in [MSG_SELECT_CARD, MSG_SELECT_TRIBUTE]:
            stream.read(1) # P
            cancelable = struct.unpack('B', stream.read(1))[0]
            min_c = struct.unpack('B', stream.read(1))[0]
            max_c = struct.unpack('B', stream.read(1))[0]
            count = struct.unpack('B', stream.read(1))[0]
            
            cards = []
            for i in range(count):
                code = struct.unpack('<I', stream.read(4))[0] # 提取真实卡密
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                stream.read(1) # Skip desc
                loc_raw = LocationInfo.encode(c, l, s, 0)
                cards.append({'idx': i, 'loc': loc_raw, 'code': code})
            
            # 权重越高的卡片，在 DFS 穷举时越先被组合，完美确保前 5000 个必定包含最优解
            cards.sort(key=lambda x: pref_weights.get(x['code'], 0.0), reverse=True)

            real_max = min(max_c, count)
            real_min = min(min_c, count)
            if real_min > real_max: real_min = real_max
            
            # [聚类 DFS 算法] 先区域去重，后按需分配，零内存泄漏
            groups = {}
            for cd in cards:
                c, l, s, _ = LocationInfo.decode(cd['loc'])
                if l == 0x04 or l == 0x08: 
                    key = ('FIELD', cd['idx']) # 场上卡片绝对不去重
                else: 
                    key = ('NON_FIELD', cd['code'], l) # 区域 + 卡密 独立去重
                    
                if key not in groups: groups[key] = []
                groups[key].append(cd)
                
            group_lists = list(groups.values())
            all_combos = []
            
            def dfs(group_idx, current_combo, needed):
                if needed == 0:
                    all_combos.append(current_combo)
                    return
                if group_idx >= len(group_lists): return
                if len(all_combos) >= limit: return
                
                group = group_lists[group_idx]
                max_pick = min(needed, len(group))
                
                # 完美覆盖挑选 0 到 N 张同名卡的情况
                for i in range(max_pick, -1, -1):
                    dfs(group_idx + 1, current_combo + group[:i], needed - i)
                    if len(all_combos) >= limit: return

            for r in range(real_min, real_max + 1):
                dfs(0, [], r)
                if len(all_combos) >= limit: 
                    sample_names = [card_db.get_card_name(c['code']) for c in cards[:4]]
                    # 高级版雷达日志
                    print(f"📡 [RuleBot 截断雷达] Type {msg_type} (选卡/祭品) 组合超 {limit}，安全阻断。")
                    print(f"   -> 🎯 发动源头: {trigger_card}")
                    print(f"   -> 📊 引擎要求: 从 {len(cards)} 张备选卡中挑选 {real_min} ~ {real_max} 张")
                    print(f"   -> 🃏 候选样本: {sample_names}...")
                    break
            
            # 打包组合
            for combo in all_combos:
                resp_buf = bytearray([len(combo)])
                locs = []
                for cd in combo:
                    resp_buf.append(cd['idx'])
                    locs.append(cd['loc'])
                options.append({'bytes': bytes(resp_buf), 'locs': locs})
                
            if cancelable:
                options.append({'bytes': struct.pack('<i', -1), 'locs': []})
                
        # 2. 星级凑数求和 (同调, 仪式, 以及械刀等特殊SumEqual卡)
        elif msg_type == MSG_SELECT_SUM:
            mode = struct.unpack('B', stream.read(1))[0]
            stream.read(1) # P
            total_acc = struct.unpack('<I', stream.read(4))[0]
            min_c = struct.unpack('B', stream.read(1))[0]
            max_c = struct.unpack('B', stream.read(1))[0]
            must_count = struct.unpack('B', stream.read(1))[0]
            
            must_vals = []
            must_locs = []
            for _ in range(must_count):
                stream.read(4) # Code
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                must_locs.append(LocationInfo.encode(c, l, s, 0))
                v = struct.unpack('<I', stream.read(4))[0]
                must_vals.append(v)
            
            count = struct.unpack('B', stream.read(1))[0]
            
            candidates = []
            for i in range(count):
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                val = struct.unpack('<I', stream.read(4))[0]
                candidates.append({'index': i, 'val': val, 'code': code, 'loc': LocationInfo.encode(c, l, s, 0)})
            
            # 神经网络动态赋权排序 (最聪明的选择排前面)
            candidates.sort(key=lambda x: pref_weights.get(x['code'], 0.0), reverse=True)
            
            # 引入 Type 15 同名卡聚类去重算法
            groups = {}
            for cd in candidates:
                c, l, s, _ = LocationInfo.decode(cd['loc'])
                if l == 0x04 or l == 0x08: 
                    key = ('FIELD', cd['index']) # 场上怪兽牵涉到具体格子，绝对不去重
                else: 
                    key = ('NON_FIELD', cd['code'], l) # 手牌/墓地/额外里的同名卡直接打包合并
                    
                if key not in groups: groups[key] = []
                groups[key].append(cd)
                
            group_lists = list(groups.values())
            
            valid_solutions = []
            real_max = max_c if max_c > 0 else count
            
            # 完美兼容双重数值的算法
            def check_sum(vals, current_idx, current_sum, current_min):
                if current_idx == len(vals):
                    if mode == 0: return current_sum == total_acc
                    else: return current_sum >= total_acc and (current_sum - current_min) < total_acc
                
                v = vals[current_idx]
                v1 = v & 0xffff
                v2 = v >> 16
                
                n_min1 = min(current_min, v1) if current_min != -1 else v1
                if check_sum(vals, current_idx + 1, current_sum + v1, n_min1): return True
                
                # 如果真的是多重星级（v2>0），才走第二个分支
                if v2 > 0 and v2 != v1:
                    n_min2 = min(current_min, v2) if current_min != -1 else v2
                    if check_sum(vals, current_idx + 1, current_sum + v2, n_min2): return True
                return False

            # 分组 DFS 遍历 (彻底根除 5000 溢出)
            def backtrack(group_idx, k, path):
                if len(valid_solutions) >= limit: 
                    if len(valid_solutions) == limit: # 只报一次警
                        sample_names = [card_db.get_card_name(cd['code']) for cd in candidates[:4]]
                        print(f"📡 [RuleBot 截断雷达] Type 23 (凑星/分值计算) 组合超 {limit}，安全阻断。")
                        print(f"   -> 🎯 发动源头: {trigger_card}")
                        print(f"   -> 📊 目标总值: {total_acc}, 必选={len(must_vals)}张, 候选池: {len(candidates)} 张")
                        print(f"   -> 🃏 候选样本: {sample_names}...")
                        valid_solutions.append([]) # 占位防狂刷
                    return
                
                if k >= min_c:
                    combined_vals = must_vals + [x['val'] for x in path]
                    if check_sum(combined_vals, 0, 0, -1):
                        valid_solutions.append(list(path))
                        
                if k == real_max or group_idx >= len(group_lists): return
                
                # 从聚类的同名卡堆里，一次性抽走 i 张，极大地修剪递归分支
                group = group_lists[group_idx]
                max_pick = min(len(group), real_max - k)
                
                for i in range(max_pick, -1, -1):
                    backtrack(group_idx + 1, k + i, path + group[:i])
            
            backtrack(0, 0, [])
            valid_solutions = [sol for sol in valid_solutions if sol] # 移除报警占位符
            
            for sol in valid_solutions:
                # 必须严格恢复原始 Index 的大小顺序，否则 C++ 引擎会抛错
                sol_sorted = sorted(sol, key=lambda x: x['index'])
                resp_buf = bytearray([must_count + len(sol_sorted)])
                for _ in range(must_count): resp_buf.append(0)
                locs = list(must_locs)
                for cd in sol_sorted:
                    resp_buf.append(cd['index'])
                    locs.append(cd['loc'])
                options.append({'bytes': bytes(resp_buf), 'locs': locs})
                
            options.append({'bytes': struct.pack('<i', -1), 'locs': []})

        # 3. 移除指示物大一统解析
        elif msg_type == MSG_SELECT_COUNTER:
            stream.read(1) # Player
            stream.read(2) # Type
            qty = struct.unpack('H', stream.read(2))[0]
            size = struct.unpack('B', stream.read(1))[0]
            
            cards = []
            for i in range(size):
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                avail = struct.unpack('H', stream.read(2))[0]
                cards.append({'idx': i, 'avail': avail, 'code': code, 'loc': LocationInfo.encode(c, l, s, 0)})
            
            # DFS 分配指示物
            def distribute_counters(card_idx, remaining_qty, current_distribution):
                if len(options) >= limit: return
                if card_idx == size:
                    if remaining_qty == 0:
                        resp_buf = bytearray()
                        locs = []
                        for i, count_val in enumerate(current_distribution):
                            resp_buf.extend(struct.pack('H', count_val))
                            if count_val > 0: locs.append(cards[i]['loc'])
                        options.append({'bytes': bytes(resp_buf), 'locs': locs})
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
        elif msg_type == 25: # MSG_SORT_CARD
            stream.read(1) # P
            count = struct.unpack('B', stream.read(1))[0]
            cards = []
            for i in range(count):
                code = struct.unpack('<I', stream.read(4))[0]
                c = struct.unpack('B', stream.read(1))[0]
                l = struct.unpack('B', stream.read(1))[0]
                s = struct.unpack('B', stream.read(1))[0]
                cards.append({'idx': i, 'loc': LocationInfo.encode(c, l, s, 0), 'code': code})
                
            valid_solutions = []
            for i, sol in enumerate(itertools.permutations(cards)):
                valid_solutions.append(sol)
                if i >= limit - 1:
                    sample_names = [card_db.get_card_name(c['code']) for c in cards[:4]]
                    print(f"📡 [RuleBot 截断雷达] Type 25 (排序) 选项组达到上限，触发安全阻断")
                    print(f"   -> 🎯 发动源头: {trigger_card}")
                    print(f"   -> 📊 需要排序的卡片总数: {count}")
                    print(f"   -> 🃏 涉案卡片: {sample_names}...")
                    break
            
            for sol in valid_solutions:
                resp_buf = bytearray()
                locs = []
                for cd in sol:
                    resp_buf.append(cd['idx'])   
                    locs.append(cd['loc'])
                options.append({'bytes': bytes(resp_buf), 'locs': locs})

        # 5. 格子/区域封锁
        elif msg_type in [18, 24]: 
            stream.read(1) # P
            count = struct.unpack('B', stream.read(1))[0]
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
                
    except Exception as e:
        print(f"⚠️ [参谋部] get_macro_options 解析错误: {e}")
        import traceback
        traceback.print_exc()
        
    return options