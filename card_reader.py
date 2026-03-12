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
        [V17.0] 获取全量静态数据
        返回: (type, race, attr, level, rank, link, link_marker, scale, atk, def)
        """
        if code == 0: return (0,)*10
        if code in self.stats_cache: return self.stats_cache[code]
        
        if not self.cursor: return (0,)*10
        
        try:
            # datas: id, ot, alias, setcode, type, atk, def, level, race, attribute, category
            self.cursor.execute("SELECT type, race, attribute, level, atk, def FROM datas WHERE id=?", (code,))
            row = self.cursor.fetchone()
            if not row: return (0,)*10
            
            raw_type, race, attr, raw_level, atk, defense = row
            
            # --- OCG Level 字段解码 ---
            # Level 字段在数据库里是个大杂烩，存储了 Level, Rank, Link, Scales
            # 字节结构: [24-31: Left Scale] [16-23: Right Scale] [0-15: Level/Rank/LinkRating]
            
            level = 0
            rank = 0
            link = 0
            link_marker = 0
            scale = 0
            
            # 判断类型
            if raw_type & 0x4000000: # TYPE_LINK
                link = raw_level & 0xFFFF
                # Link Marker 实际上存在 def 字段里，或者是 level 的高位？
                # 修正：在 cards.cdb 中，Link 怪兽的 DEF 字段存储的是 Link Arrows！
                link_marker = defense 
                defense = 0 # Link 怪兽没有防御力
            elif raw_type & 0x800000: # TYPE_XYZ
                rank = raw_level & 0xFFFF
            else:
                level = raw_level & 0xFFFF
                
            # 灵摆刻度 (Pendulum Scale)
            if raw_type & 0x1000000: # TYPE_PENDULUM
                # Scale 存储在 Level 的高位
                # (raw_level >> 24) & 0xFF 是左刻度
                # (raw_level >> 16) & 0xFF 是右刻度
                scale = (raw_level >> 24) & 0xFF
                
            stats = (raw_type, race, attr, level, rank, link, link_marker, scale, atk, defense)
            self.stats_cache[code] = stats
            return stats
        except:
            return (0,)*10

# 单例
card_db = CardReader()