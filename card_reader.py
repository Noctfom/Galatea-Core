'''
CardReader 模块
用于读取 cards.cdb 数据库，提供卡片名称和属性查询功能
'''


import sqlite3
import os

class CardReader:
    def __init__(self, db_path='cards.cdb'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.cache = {} 
        self.stats_cache = {}
        
        if os.path.exists(db_path):
            try:
                self.conn = sqlite3.connect(db_path)
                self.cursor = self.conn.cursor()
            except:
                print("⚠️ 无法连接 cards.cdb")
        else:
            print("⚠️ 未找到 cards.cdb")

    def get_base_code(self, code):
        """
        [新增] 异画卡/马甲卡归一化
        查询 cards.cdb，如果该卡有 alias（异画），则返回原版卡密；否则返回自身。
        """
        if not self.cursor: 
            return code
            
        try:
            # 在 YGOPro 数据库中，alias 字段如果不为 0，就代表它指向原版卡密
            self.cursor.execute("SELECT alias FROM datas WHERE id=?", (code,))
            row = self.cursor.fetchone()
            if row and row[0] != 0:
                return row[0] # 返回原版卡密
            return code
        except Exception:
            return code

    def get_card_name(self, code):
        # ... (保持不变) ...
        if not self.cursor: return f"Code {code}"
        if code in self.cache: return self.cache[code]
        try:
            self.cursor.execute("SELECT name FROM texts WHERE id=?", (code,))
            row = self.cursor.fetchone()
            name = row[0] if row else f"Code {code}"
            self.cache[code] = name
            return name
        except: return f"Code {code}"

    def get_card_type(self, code):
        # ... (保持不变) ...
        if not self.cursor: return 0
        try:
            self.cursor.execute("SELECT type FROM datas WHERE id=?", (code,))
            row = self.cursor.fetchone()
            return row[0] if row else 0
        except: return 0

    def get_full_stats(self, code):
        """
        [V17.0 修复版] 严格对齐 gamestate.py 的索引期望！
        返回格式必须是 11 元素：
        0:type, 1:race, 2:attr, 3:level, 4:lscale, 5:rscale, 6:link_marker, 7:rank, 8:atk, 9:def, 10:tuple(setcodes)
        """
        # 绝对安全的兜底返回值
        safe_fallback = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0, 0, 0, 0))
        
        if code == 0: return safe_fallback
        if code in self.stats_cache: return self.stats_cache[code]
        if not self.cursor: return safe_fallback
        
        try:
            self.cursor.execute("SELECT type, race, attribute, level, atk, def, setcode FROM datas WHERE id=?", (code,))
            row = self.cursor.fetchone()
            if not row: return safe_fallback
            
            raw_type, race, attr, raw_level, atk, defense, raw_setcode = row

            if atk < 0: atk = 0
            if defense < 0: defense = 0
            
            level = raw_level & 0xFFFF
            rank = 0; link = 0; link_marker = 0; lscale = 0; rscale = 0
            
            if raw_type & 0x4000000: # TYPE_LINK
                link = raw_level & 0xFFFF
                link_marker = defense 
                defense = 0 
            elif raw_type & 0x800000: # TYPE_XYZ
                rank = raw_level & 0xFFFF
            else:
                level = raw_level & 0xFFFF
                
            if raw_type & 0x1000000: # TYPE_PENDULUM
                lscale = (raw_level >> 24) & 0xFF
                rscale = (raw_level >> 16) & 0xFF
                
            setcodes = []
            val = raw_setcode
            for _ in range(4):
                if val & 0xFFFF: setcodes.append(val & 0xFFFF)
                val >>= 16
            setcodes = (setcodes + [0]*4)[:4]
                
            # 🌟 核心：严格按照 gamestate.py 的索引需求组装！
            # [0]raw_type, [1]race, [2]attr, [3]level, [4]lscale, [5]rscale, 
            # [6]link_marker, [7]rank, [8]atk, [9]defense, [10]setcodes
            stats = (raw_type, race, attr, level, lscale, rscale, link_marker, rank, atk, defense, tuple(setcodes))
            self.stats_cache[code] = stats
            return stats
        
        except Exception as e:
            print(f"⚠️ get_full_stats 解析异常: {e} (code={code})")
            return safe_fallback

# 单例
card_db = CardReader()