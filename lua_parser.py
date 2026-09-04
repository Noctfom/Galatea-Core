import os
import re
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from semantic_assets import (
    CODE_SEMANTIC_FILENAMES,
    HASH_MAPPING_FILENAME,
    download_remote_semantic_bundle,
)


def _stable_unique(values):
    """按首次出现顺序去重，避免集合随机顺序污染固定槽位"""
    return list(dict.fromkeys(values))

class YGOProLuaParser:
    def __init__(self, script_dir='./script'):
        self.script_dir = script_dir
        self.hash_registry = defaultdict(lambda: {"cards": [], "sample_code": ""})

    def _rebuild_hash_registry(self, knowledge_base):
        """从知识库中的自定义标签重建可供增量解析接续的 Hash 索引"""
        for card_id, card_data in knowledge_base.items():
            for effect in card_data.get("effects", []):
                slot = int(effect.get("slot", 1) or 1)
                card_label = f"{card_id}_E{slot}"
                for category in effect.get("categories", []):
                    if not str(category).startswith("CUSTOM_HASH_"):
                        continue
                    record = self.hash_registry[str(category)]
                    if card_label not in record["cards"]:
                        record["cards"].append(card_label)

    def _hash_code_block(self, code_block, card_id, slot_idx):
        """将特殊的代码块转化为统一的 Hash 标签，使用深度词法规范化榨干冗余变种"""
        if not code_block: return "CUSTOM_HASH_EMPTY", {"numbers": [], "hexes": []}
        
        # 1. 提取真实参数 (保留，供神经网络查阅)
        extracted_numbers = re.findall(r'\b\d+\b', code_block)
        extracted_hexes = re.findall(r'0x[0-9a-fA-F]+', code_block)
        extracted_constants = re.findall(r'(RACE|ATTRIBUTE|CATEGORY|LOCATION|TYPE|PHASE|POS)_[A-Z_]+', code_block)
        
        clean_code = code_block

        # ==============================================================
        # 2. 深度词法规范化 (Lexical Normalization) 
        # 核心目的：消除不同写脚本的人带来的“语法个性”差异
        # ==============================================================
        
        # A. 玩家指针统一
        clean_code = re.sub(r'\b1\s*-\s*tp\b', '<OPPO>', clean_code) # 对手
        clean_code = re.sub(r'\b(tp|ep|rp)\b', '<PLAYER>', clean_code) # 己方
        
        # B. 屏蔽本卡专属的私有函数名 (如 s.filter, c12345678.tg, s.thfilter)
        clean_code = re.sub(r'\b(?:c\d+|s)\.[a-zA-Z0-9_]+\b', '<FUNC>', clean_code)
        
        # C. 统一常用的自我指代
        clean_code = re.sub(r'e:GetHandler\(\)', '<CARD>', clean_code)
        
        # D. 消灭 Lua 变量声明的语法糖 (彻底删掉 local a, b, c =)
        clean_code = re.sub(r'local\s+[a-zA-Z0-9_,\s]+\s*=', '=', clean_code)
        
        # E. 统一判空与布尔逻辑
        clean_code = re.sub(r'~=\s*nil', '', clean_code)  # if tc~=nil 和 if tc 语义完全一致
        clean_code = re.sub(r'==\s*true', '', clean_code) 
        clean_code = re.sub(r'>\s*0', '', clean_code)     # 统一数字判定
        clean_code = re.sub(r'\b(true|false)\b', '<BOOL>', clean_code)
        
        # F. 屏蔽 YGOPro 标准常用变量名 (避免不同人命名习惯导致 Hash 突变)
        clean_code = re.sub(r'\b(tc|g|e|c|eg|ev|re|r|chk|chkc|mat|tg)\b', '<VAR>', clean_code)
        
        # G. 魔法数值脱敏 (掩码处理)
        clean_code = re.sub(r'\b\d+\b', '<NUM>', clean_code)
        clean_code = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', clean_code)
        clean_code = re.sub(r'(RACE|ATTRIBUTE|CATEGORY|LOCATION|TYPE|PHASE|POS)_[A-Z_]+', '<CONST>', clean_code)
        
        # H. 最终去除所有排版格式和空白字符 
        # (必须放在最后，否则前面的 \b 单词边界匹配会失效)
        clean_code = re.sub(r'\s+', '', clean_code)

        # ==============================================================
        
        # 3. 计算 MD5
        hash_val = hashlib.md5(clean_code.encode('utf-8')).hexdigest()[:8]
        tag_name = f"CUSTOM_HASH_{hash_val.upper()}"
        
        # 4. 登记到对照表
        card_label = f"{card_id}_E{slot_idx}"
        self.hash_registry[tag_name]["cards"].append(card_label)
        if not self.hash_registry[tag_name]["sample_code"]:
            # 存下被极致压缩后的机器码，方便你在 JSON 里查阅它为什么碰撞
            self.hash_registry[tag_name]["sample_code"] = clean_code 
            
        return tag_name, {"numbers": extracted_numbers, "hexes": extracted_hexes, "constants": extracted_constants}

    def parse_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        filename = os.path.basename(filepath)
        match = re.search(r'c(\d+)\.lua', filename)
        if not match: return None
        card_id = int(match.group(1))
        
        card_data = {
            "id": card_id,
            "summon_conditions": [],
            "effects": []
        }
        
        # ==========================================
        # 1. 提取召唤条件 (Procedures) [修复嵌套括号截断]
        # ==========================================
        # 把 .*? 改成 .*，利用贪婪匹配直接吃到最外层的右括号
        proc_matches = re.finditer(r'aux\.Add(Fusion|Synchro|Xyz|Link|Ritual|Pendulum)[A-Za-z0-9_]*\((.*)\)', content)
        for m in proc_matches:
            proc_type = m.group(1) 
            proc_args = m.group(2)
            card_data["summon_conditions"].append({
                "type": proc_type.upper(),
                "raw_args": proc_args.strip()
            })

        # ==========================================
        # 核心防夺舍屏障：安全提取 initial_effect 作用域
        # ==========================================
        start_idx = content.find('.initial_effect(c)')
        if start_idx == -1: return card_data
        
        # YGOPro 脚本极其规范，下一个 function 定义就是 initial_effect 的结束边界
        next_func_idx = content.find('\nfunction ', start_idx)
        if next_func_idx == -1:
            init_body = content[start_idx:]
        else:
            init_body = content[start_idx:next_func_idx]

        # ==========================================
        # 2. 提取效果槽位 (Effect Slots)
        # ==========================================
        # 必须且只能在 init_body 里找 Effect.CreateEffect(c)
        effect_creations = re.finditer(r'local\s+(e\d*)\s*=\s*Effect\.CreateEffect\(c\)', init_body)
        
        slot_idx = 1
        for m in effect_creations:
            e_name = m.group(1)
            effect_slot = {
                "slot": slot_idx,
                "type": [],
                "code": [],
                "range": [],
                "categories": [],
                "requirements": {"setcodes": [], "races": [], "attributes": [], "types": [], 
                                 "summon_types": [], "locations": [], "phases": [], 
                                 "reasons": [], "positions": []},
                "ref_codes": []
            }
            
            # --- A. 扫描该效果的直接属性配置 ---
            # 必须且只能在 init_body 里搜索 e1:SetXXX，杜绝同名变量夺舍
            prop_pattern = rf'{e_name}:Set([A-Za-z0-9_]+)\((.*?)\)'
            for p_match in re.finditer(prop_pattern, init_body):
                prop_name = p_match.group(1)
                prop_val = p_match.group(2)
                
                if prop_name == 'Type': effect_slot['type'] = re.findall(r'EFFECT_TYPE_[A-Z0-9_]+', prop_val)
                elif prop_name == 'Code': effect_slot['code'] = [prop_val.strip()]
                elif prop_name == 'Range': effect_slot['range'] = re.findall(r'LOCATION_[A-Z_]+', prop_val)
                elif prop_name == 'Category': effect_slot['categories'].extend(re.findall(r'CATEGORY_[A-Z_]+', prop_val))
            
            # --- B. 顺藤摸瓜：追踪绑定的函数 (Condition, Target, Operation) ---
            bound_funcs = []
            for func_type in ['Condition', 'Cost', 'Target', 'Operation']:
                func_match = re.search(rf'{e_name}:Set{func_type}\((.*?)\)', init_body)
                if func_match:
                    func_name = func_match.group(1).strip()
                    bound_funcs.append(func_name)
            
            # 使用 BFS (广度优先) 队列，确保所有深层嵌套子函数被 100% 提取
            funcs_to_process = list(bound_funcs)
            processed_funcs = set()
            func_bodies_text = ""
            op_code_block = ""
            
            while funcs_to_process:
                func_name = funcs_to_process.pop(0)
                if func_name in processed_funcs: continue
                processed_funcs.add(func_name)
                
                # 用寻找下一个 function 声明来划分边界
                # 彻底解决 Lua 中 if...end 导致的提前截断问题
                func_def_str = f"function {func_name}("
                start_idx = content.find(func_def_str)
                
                if start_idx != -1:
                    next_func_idx = content.find('\nfunction ', start_idx + 10)
                    if next_func_idx == -1:
                        body = content[start_idx:]
                    else:
                        body = content[start_idx:next_func_idx]
                        
                    func_bodies_text += body + "\n"
                    if 'Operation' in func_name or 'op' in func_name.lower():
                        op_code_block += body + "\n"
                    
                    # 查找这个函数内部有没有调用其他的本地函数 (比如 s.thfilter 或 c56532353.filter)
                    sub_funcs = re.findall(r'(?:c\d+|s)\.[a-zA-Z0-9_]+', body)
                    for sf in sub_funcs:
                        if sf not in processed_funcs and sf not in funcs_to_process:
                            funcs_to_process.append(sf) # 加入队列，确保一定会被处理到

            # 开始地毯式提取
            effect_slot['categories'].extend(re.findall(r'CATEGORY_[A-Z_]+', func_bodies_text))
            effect_slot['requirements']['setcodes'].extend(re.findall(r'IsSetCard\((0x[0-9a-fA-F]+|[0-9]+)\)', func_bodies_text))
            effect_slot['requirements']['races'].extend(re.findall(r'RACE_[A-Z_]+', func_bodies_text))
            effect_slot['requirements']['attributes'].extend(re.findall(r'ATTRIBUTE_[A-Z_]+', func_bodies_text))
            
            if 'types' not in effect_slot['requirements']: effect_slot['requirements']['types'] = []
            effect_slot['requirements']['types'].extend(re.findall(r'TYPE_[A-Z_]+', func_bodies_text))
            
            if 'summon_types' not in effect_slot['requirements']: effect_slot['requirements']['summon_types'] = []
            effect_slot['requirements']['summon_types'].extend(re.findall(r'SUMMON_TYPE_[A-Z_]+', func_bodies_text))
            
            if 'locations' not in effect_slot['requirements']: effect_slot['requirements']['locations'] = []
            effect_slot['requirements']['locations'].extend(re.findall(r'LOCATION_[A-Z_]+', func_bodies_text))
            
            # [新增] 解析 common.h 里的 Phase, Reason, Position
            if 'phases' not in effect_slot['requirements']: effect_slot['requirements']['phases'] = []
            effect_slot['requirements']['phases'].extend(re.findall(r'PHASE_[A-Z0-9_]+', func_bodies_text))
            
            if 'reasons' not in effect_slot['requirements']: effect_slot['requirements']['reasons'] = []
            effect_slot['requirements']['reasons'].extend(re.findall(r'REASON_[A-Z_]+', func_bodies_text))
            
            if 'positions' not in effect_slot['requirements']: effect_slot['requirements']['positions'] = []
            effect_slot['requirements']['positions'].extend(re.findall(r'POS_[A-Z_]+', func_bodies_text))

            # 去重清洗
            effect_slot['categories'] = _stable_unique(effect_slot['categories'])
            effect_slot['requirements']['setcodes'] = _stable_unique(effect_slot['requirements']['setcodes'])
            effect_slot['requirements']['races'] = _stable_unique(effect_slot['requirements']['races'])
            effect_slot['requirements']['attributes'] = _stable_unique(effect_slot['requirements']['attributes'])
            effect_slot['requirements']['types'] = _stable_unique(effect_slot['requirements']['types'])
            effect_slot['requirements']['summon_types'] = _stable_unique(effect_slot['requirements']['summon_types'])
            effect_slot['requirements']['locations'] = _stable_unique(effect_slot['requirements']['locations'])
            effect_slot['requirements']['phases'] = _stable_unique(effect_slot['requirements']['phases'])
            effect_slot['requirements']['reasons'] = _stable_unique(effect_slot['requirements']['reasons'])
            effect_slot['requirements']['positions'] = _stable_unique(effect_slot['requirements']['positions'])
            
            # --- C. 特殊效果兜底机制 (Hash 聚类) ---
            # 如果找遍了属性和函数，都没有官方的 Category，触发聚类机制
            if not effect_slot['categories']:
                hash_tag, custom_params = self._hash_code_block(op_code_block, card_id, slot_idx)
                effect_slot['categories'].append(hash_tag)
                
                # 把按顺序提取出来的独立性质标签，贴在这个特殊效果的后面
                effect_slot['requirements']['custom_numbers'] = custom_params['numbers']
                effect_slot['requirements']['custom_hexes'] = custom_params['hexes']
            else:
                # 常规效果也给个空列表，保证 JSON 结构统一
                effect_slot['requirements']['custom_numbers'] = []
                effect_slot['requirements']['custom_hexes'] = []
            
            raw_text = func_bodies_text + "\n" + op_code_block
            effect_slot['raw_code'] = raw_text.strip()

            card_data["effects"].append(effect_slot)
            slot_idx += 1
            
        return card_data

    def run_batch(self, output_file='knowledge_base.json', clear_existing=False, remote_url=None):
        """批量处理所有脚本并导出 (支持 Github 同步、断点续传与物理清空)"""
        print(f"🚀 开始知识库构建任务...")
        knowledge_base = {}
        output_path = Path(output_file).resolve()
        mapping_file = output_path.with_name(HASH_MAPPING_FILENAME)
        
        # =======================================================
        # 1. 物理清空逻辑 (--clear)
        # =======================================================
        if clear_existing:
            print("🧨 [--clear] 清空指令触发，正在物理删除本地旧数据...")
            if output_path.exists(): output_path.unlink()
            if mapping_file.exists(): mapping_file.unlink()
            for filename in CODE_SEMANTIC_FILENAMES:
                semantic_path = output_path.with_name(filename)
                if semantic_path.exists():
                    semantic_path.unlink()
            knowledge_base = {}
            self.hash_registry.clear()
        else:
            # =======================================================
            # 2. Github 远程同步逻辑 (--sync)
            # =======================================================
            if remote_url:
                print(f"🌐 正在从 Github 获取完整语义基座: {remote_url}")
                try:
                    bundle = download_remote_semantic_bundle(
                        remote_url,
                        output_path.parent,
                    )
                    knowledge_base = bundle["knowledge_base"]
                    remote_mapping = bundle["hash_mapping"]
                    if remote_mapping is not None:
                        for key, value in remote_mapping.items():
                            self.hash_registry[key] = value
                    else:
                        self._rebuild_hash_registry(knowledge_base)
                        print("⚠️ 远程仓库缺少 Hash 映射，已从知识库重建接续索引。")
                    if bundle["installed_code_semantics"]:
                        print("✅ 代码语义向量与索引已同步，可继续增量提取。")
                    else:
                        print("⚠️ 远程代码语义向量不完整，本次仅接续结构化语义。")
                    print(f"✅ 成功合并远程仓库数据: 包含 {len(knowledge_base)} 张卡片语义！")
                except Exception as e:
                    print(f"❌ 远程拉取失败: {e}，将退回本地模式...")

            # =======================================================
            # 3. 本地断点续传读取
            # =======================================================
            if not knowledge_base and output_path.exists():
                print(f"📂 检测到本地知识库 {output_path}，开启增量更新模式...")
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        knowledge_base = json.load(f)
                    print(f"✅ 已加载本地 {len(knowledge_base)} 张卡片数据，将仅解析新脚本。")
                    
                    if os.path.exists(mapping_file):
                        with open(mapping_file, 'r', encoding='utf-8') as f:
                            old_registry = json.load(f)
                            for k, v in old_registry.items():
                                self.hash_registry[k] = v
                except Exception as e:
                    print(f"⚠️ 本地读取失败: {e}，将重新解析...")

        # [新增] 拍快照：记录解析前的 Hash 状态，用于计算“完美碰撞率”
        old_hashes = set(self.hash_registry.keys())
        old_hash_card_counts = {k: len(v["cards"]) for k, v in self.hash_registry.items()}

        if not os.path.exists(self.script_dir):
            print(f"❌ 找不到脚本目录 {self.script_dir}")
            return
            
        # =======================================================
        # 4. 增量扫描与解析
        # =======================================================
        print(f"🔍 扫描 {self.script_dir} 目录下的增量更新...")
        count = 0
        skip_count = 0
        
        for filename in sorted(os.listdir(self.script_dir)):
            if filename.endswith('.lua'):
                match = re.search(r'c(\d+)\.lua', filename)
                if not match: continue
                card_id = str(match.group(1)) 
                
                # 跳过已有卡片
                if card_id in knowledge_base:
                    skip_count += 1
                    continue
                
                filepath = os.path.join(self.script_dir, filename)
                res = self.parse_file(filepath)
                if res and res['effects']:
                    knowledge_base[card_id] = res
                    count += 1
                    if count % 1000 == 0:
                        print(f"   ... 新增解析 {count} 张卡片")

        # =======================================================
        # [新增] 核心价值结算：计算碰撞与合并数据
        # =======================================================
        new_hashes = set(self.hash_registry.keys()) - old_hashes
        new_hash_count = len(new_hashes)
        
        # 统计有多少个新增效果完美贴合到了老 Hash 里
        merged_into_old_count = sum(len(self.hash_registry[k]["cards"]) - old_hash_card_counts[k] for k in old_hashes)
        
        # 统计在这次前所未见的新 Hash 里，有多少次内部合并 (新卡A和新卡B代码不同，但被算法压成了同一个Hash)
        new_hash_total_cards = sum(len(self.hash_registry[k]["cards"]) for k in new_hashes)
        merged_into_new_count = new_hash_total_cards - new_hash_count if new_hash_total_cards > 0 else 0
        
        # 本次提取出来的特殊效果总数 (槽位总数)
        total_custom_extracted = merged_into_old_count + new_hash_total_cards

        # =======================================================
        # 5. 覆写输出结果
        # =======================================================
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.hash_registry, f, indent=2, ensure_ascii=False)
            
        # =======================================================
        # 6. 终极清晰报告
        # =======================================================
        print("\n==============================================")
        print("🏁 语义知识库 (Semantic Knowledge Base) 构建完成！")
        print("==============================================")
        print(f"📄 [卡片统计] 继承/跳过 {skip_count} 张，本次新增解析 {count} 张。")
        print(f"📁 [卡片库总容量] 当前知识库共计 {len(knowledge_base)} 张卡片。")
        print("----------------------------------------------")
        
        if total_custom_extracted > 0:
            print(f"🧩 [特殊效果降维统计] 本次扫描共提取了 {total_custom_extracted} 个自定义特殊效果槽位：")
            if old_hashes and merged_into_old_count > 0:
                print(f"   ├── 🎯 完美吸收：成功合并至已有的老效果组 {merged_into_old_count} 次！")
            if merged_into_new_count > 0:
                print(f"   ├── 🔄 内部碰撞：这批新卡相互之间合并了 {merged_into_new_count} 次！")
            print(f"   └── 🆕 知识盲区：产生全新的独立底层逻辑 {new_hash_count} 种。")
            
            # 计算压缩率 = (合并掉的次数 / 提取总数)
            compression_rate = (merged_into_old_count + merged_into_new_count) / total_custom_extracted * 100
            print(f"   📉 算法带来的记忆减负率: {compression_rate:.1f}%")
        else:
            print("🧩 [特殊效果统计] 本次未提取到新的自定义特殊效果。")
            
        print(f"📚 [特殊效果总容量] 对照表目前共收录 {len(self.hash_registry)} 种独立逻辑。")
        print("==============================================\n")

if __name__ == "__main__":
    parser = YGOProLuaParser(script_dir='./script')
    
    # [测试模式] 先拿一张卡做手术
    test_file = './script/全量提取.lua'
    if os.path.exists(test_file):
        print("🔍 [单一测试]:")
        res = parser.parse_file(test_file)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print("⚠️ 找不到单卡测试文件，直接运行全量提取...")
        parser.run_batch()
