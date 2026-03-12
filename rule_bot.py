'''
规则型 Bot 的核心决策逻辑
保证在所有交互请求中，100% 返回合法格式的数据，防止超时
'''


import struct
import io
import random
import itertools

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
    
    # Activatable
    b = stream.read(1) 
    if b:
        count = struct.unpack('B', b)[0]
        for i in range(count): stream.read(11) # Skip details
            
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
            if not actions: 
                decision = (0 << 16) | 3 # 强制结束战斗阶段
            else:
                choice = random.choice(actions)
                if choice['cat'] == 1: 
                    decision = (choice['idx'] << 16) | 1
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
                # 1. 尝试读取前 4 个关键字节
                # P(1) + Count(1) + Spe(1) + Forced(1)
                # 只要读到这 4 个，我们就掌握了主动权
                header_start = stream.read(4)
                if len(header_start) < 4: raise Exception("Header incomplete")
                
                count = header_start[1]
                forced = header_start[3] # 第4个字节是强制标志
                
                # 2. 尝试吞掉剩下的头部 (Hint)
                # 标准是 12，我们读了 4，还剩 8。但考虑到那个怪异的 11 字节包，我们宽松一点
                # 只要流里还有数据，就尽量读，但不强求读满
                stream.read(8) 
                
                # 3. 构造选项
                candidates = list(range(count)) + [-1]
                valid_choices = [c for c in candidates if c not in ignore_actions]
                
                # [核心修复] 决策逻辑
                if valid_choices:
                    activation_choices = [c for c in valid_choices if c != -1]
                    
                    # 如果是【强制发动】(forced != 0) 且有可发动的选项
                    # 我们绝对不能选 -1，必须随机选一个发动的
                    if forced and activation_choices:
                        decision = random.choice(activation_choices)
                    
                    # 如果不是强制的，或者没得选，再考虑 -1
                    elif activation_choices:
                        decision = random.choice(activation_choices + [-1]) # 纯随机
                    else:
                        decision = -1
                else:
                    decision = -1
                    
            except Exception:
                # 解析彻底失败时的兜底
                # 如果我们连 Forced 都没读出来，为了防止死循环，我们盲猜 0 (发动第一个)
                # 这是一个赌博：如果是必发，0 可能蒙对；如果不是必发，0 也只是发动效果而已
                decision = 0

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
        # [终极修正] 11. 卡片/数字宣言 (列表选择)
        # =================================================================
        elif msg_type in [MSG_ANNOUNCE_CARD, MSG_ANNOUNCE_NUMBER]:
            stream.read(1) # Player
            count = struct.unpack('B', stream.read(1))[0] # 选项数量
            
            options = []
            for _ in range(count):
                options.append(struct.unpack('<I', stream.read(4))[0])
            
            # 这里的规则通常是选 1 个 (如宣言一个卡名)
            # 源码: int32_t code = returns.ivalue[0];
            if options:
                choice = random.choice(options)
                decision = struct.pack('<I', choice)
            else:
                decision = struct.pack('<I', 0)

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
            except:
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
            except:
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
                # 🌟 [终极防死锁机制]
                # 如果代码走到这里，说明我们能选的所有组合，全被引擎 RETRY 拒绝了！
                # 此时必须尝试壮士断腕：能取消就强制取消，不能取消就发随机字节强行引发熔断重置！
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
                except Exception:
                    # 解析中途失败（如数据包截断），默认选0
                    decision = struct.pack('<i', 0)

        # =================================================================
        # 严格修正：MSG_SELECT_SUM (23)
        # =================================================================
        elif msg_type == MSG_SELECT_SUM:
            if len(payload) < 10:
                return bytes([0])
            try:
                stream = io.BytesIO(payload)
                mode = struct.unpack('B', stream.read(1))[0]
                stream.read(1) # 跳过 player_id
                total_acc = struct.unpack('<I', stream.read(4))[0]
                min_c = struct.unpack('B', stream.read(1))[0]
                max_c = struct.unpack('B', stream.read(1))[0]
                
                must_count = struct.unpack('B', stream.read(1))[0]
                must_sum = 0
                for _ in range(must_count):
                    stream.read(7)
                    v = struct.unpack('<I', stream.read(4))[0]
                    must_sum += (v & 0xffff) # 提取低16位基础星级
                
                target_val = total_acc - must_sum
                
                count_b = stream.read(1)
                if not count_b: return bytes([0])
                count = struct.unpack('B', count_b)[0]

                candidates = []
                for i in range(count):
                    stream.read(7)
                    val = struct.unpack('<I', stream.read(4))[0]
                    candidates.append({'index': i, 'val': val})
                
                # 🌟 [终极数学求解器] DFS 完美拆解双重星级，必定求出正确解！
                valid_solutions = []
                real_max = max_c if max_c > 0 else count
                
                def backtrack(start, k, current_sum, path, min_v):
                    if k >= min_c:
                        # Mode 0: 同步召唤 (绝对相等)
                        if mode == 0 and current_sum == target_val:
                            valid_solutions.append(list(path))
                        # Mode 1: 仪式召唤等 (大于等于，但去掉最小者必须小于目标)
                        elif mode == 1 and current_sum >= target_val and (current_sum - min_v) < target_val:
                            valid_solutions.append(list(path))
                    
                    if k == real_max or start == count:
                        return
                        
                    for i in range(start, count):
                        c = candidates[i]
                        # 核心：拆解 YGOPro 的双重星级参数
                        v1 = c['val'] & 0xffff
                        v2 = c['val'] >> 16
                        
                        path.append(c['index'])
                        
                        # 尝试使用第一星级
                        n_min1 = min(min_v, v1) if min_v != -1 else v1
                        backtrack(i + 1, k + 1, current_sum + v1, path, n_min1)
                        
                        # 如果存在第二星级，尝试使用第二星级
                        if v2 > 0 and v2 != v1:
                            n_min2 = min(min_v, v2) if min_v != -1 else v2
                            backtrack(i + 1, k + 1, current_sum + v2, path, n_min2)
                            
                        path.pop()

                backtrack(0, 0, 0, [], -1)
                
                ignored_set = set(b for b in ignore_actions if isinstance(b, bytes))
                decision = bytes([0])
                
                if valid_solutions:
                    random.shuffle(valid_solutions) # 洗牌以增加 AI 对局的多样性
                    for sol in valid_solutions:
                        resp_buf = bytearray([must_count + len(sol)])
                        for _ in range(must_count): resp_buf.append(0)
                        for idx in sol: resp_buf.append(idx)
                        
                        candidate_bytes = bytes(resp_buf)
                        if candidate_bytes not in ignored_set:
                            decision = candidate_bytes
                            break
                            
                return decision
            except Exception as e:
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
            expected_len = count * 3
            
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
        # 万一解析崩了，返回 0 或空字节作为最后的倔强
        print(f"[RuleBot] Parsing Error for Msg {msg_type}: {e}")
        decision = 0

    if decision is None:
        decision = 0

    # ---------------------------------------------------------
    # [数据记录点]
    # 在这里，你应该将 (gamestate, msg_type, decision) 保存下来。
    # 比如：data_recorder.save(gamestate, msg_type, decision)
    # ---------------------------------------------------------
    
    return decision