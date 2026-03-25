import os
import re
import json
import hashlib
from collections import defaultdict

class YGOProLuaParser:
    def __init__(self, script_dir='./script'):
        self.script_dir = script_dir
        self.hash_registry = defaultdict(lambda: {"cards": [], "sample_code": ""})
        
    def _hash_code_block(self, code_block, card_id, slot_idx):
        """将特殊的代码块转化为统一的 Hash 标签，并顺序提取参数"""
        if not code_block: return "CUSTOM_HASH_EMPTY", {"numbers": [], "hexes": []}
        
        # 1. 基础清洗
        clean_code = re.sub(r'\s+', '', code_block)
        clean_code = re.sub(r'local\w+=', '', clean_code)
        
        # 2. 🌟 提取真实参数 (按在代码中出现的顺序提取！)
        extracted_numbers = re.findall(r'\b\d+\b', clean_code)
        extracted_hexes = re.findall(r'0x[0-9a-fA-F]+', clean_code)
        
        # 3. 魔法数值脱敏 (掩码处理)
        clean_code = re.sub(r'\b\d+\b', '<NUM>', clean_code)
        clean_code = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', clean_code)
        
        # 4. 计算 MD5
        hash_val = hashlib.md5(clean_code.encode('utf-8')).hexdigest()[:8]
        tag_name = f"CUSTOM_HASH_{hash_val.upper()}"
        
        # 5. 登记到对照表 (保留占位后的脱敏代码作为分析样本)
        card_label = f"{card_id}_E{slot_idx}"
        self.hash_registry[tag_name]["cards"].append(card_label)
        if not self.hash_registry[tag_name]["sample_code"]:
            self.hash_registry[tag_name]["sample_code"] = clean_code # 存脱敏后的代码，便于你看出它的通用结构
            
        return tag_name, {"numbers": extracted_numbers, "hexes": extracted_hexes}

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
        # 1. 提取召唤条件 (Procedures) [🌟 修复嵌套括号截断]
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
        # 🌟 核心防夺舍屏障：安全提取 initial_effect 作用域
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
        # 🌟 必须且只能在 init_body 里找 Effect.CreateEffect(c)
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
            # 🌟 必须且只能在 init_body 里搜索 e1:SetXXX，杜绝同名变量夺舍！
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
            
            # 🌟 [史诗级修复] 使用 BFS (广度优先) 队列，确保所有深层嵌套子函数被 100% 提取
            funcs_to_process = list(bound_funcs)
            processed_funcs = set()
            func_bodies_text = ""
            op_code_block = ""
            
            while funcs_to_process:
                func_name = funcs_to_process.pop(0)
                if func_name in processed_funcs: continue
                processed_funcs.add(func_name)
                
                # 🌟 抛弃正则匹配 end！用寻找下一个 function 声明来划分边界
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
                            funcs_to_process.append(sf) # 加入队列，确保一定会被处理到！

            # 现在，所有的相关代码都浓缩在 func_bodies_text 里了，开始地毯式提取！
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
            effect_slot['categories'] = list(set(effect_slot['categories']))
            effect_slot['requirements']['setcodes'] = list(set(effect_slot['requirements']['setcodes']))
            effect_slot['requirements']['races'] = list(set(effect_slot['requirements']['races']))
            effect_slot['requirements']['attributes'] = list(set(effect_slot['requirements']['attributes']))
            effect_slot['requirements']['types'] = list(set(effect_slot['requirements']['types']))
            effect_slot['requirements']['summon_types'] = list(set(effect_slot['requirements']['summon_types']))
            effect_slot['requirements']['locations'] = list(set(effect_slot['requirements']['locations']))
            effect_slot['requirements']['phases'] = list(set(effect_slot['requirements']['phases']))
            effect_slot['requirements']['reasons'] = list(set(effect_slot['requirements']['reasons']))
            effect_slot['requirements']['positions'] = list(set(effect_slot['requirements']['positions']))
            
            # --- C. 特殊效果兜底机制 (Hash 聚类) ---
            # 如果找遍了属性和函数，都没有官方的 Category，触发聚类机制！
            if not effect_slot['categories']:
                hash_tag, custom_params = self._hash_code_block(op_code_block, card_id, slot_idx)
                effect_slot['categories'].append(hash_tag)
                
                # 🌟 把按顺序提取出来的独立性质标签，贴在这个特殊效果的后面！
                effect_slot['requirements']['custom_numbers'] = custom_params['numbers']
                effect_slot['requirements']['custom_hexes'] = custom_params['hexes']
            else:
                # 常规效果也给个空列表，保证 JSON 结构统一
                effect_slot['requirements']['custom_numbers'] = []
                effect_slot['requirements']['custom_hexes'] = []

            card_data["effects"].append(effect_slot)
            slot_idx += 1
            
        return card_data

    def run_batch(self, output_file='knowledge_base.json'):
        """批量处理所有脚本并导出"""
        print(f"🚀 开始扫描 {self.script_dir} 下的 Lua 脚本...")
        knowledge_base = {}
        count = 0
        
        if not os.path.exists(self.script_dir):
            print(f"❌ 找不到脚本目录 {self.script_dir}")
            return
            
        for filename in os.listdir(self.script_dir):
            if filename.endswith('.lua'):
                filepath = os.path.join(self.script_dir, filename)
                res = self.parse_file(filepath)
                if res and res['effects']: # 只保留有效果的卡
                    knowledge_base[res['id']] = res
                    count += 1
                    if count % 1000 == 0:
                        print(f"   ... 已解析 {count} 张卡片")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        # 🌟 [新增] 导出人工排查对照表
        mapping_file = 'hash_mapping_report.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.hash_registry, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 解析完成！共提取 {count} 张卡的语义结构，已保存至 {output_file}")
        print(f"📊 特殊效果对照表已生成，保存至 {mapping_file} (共 {len(self.hash_registry)} 种独立特殊效果)")


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