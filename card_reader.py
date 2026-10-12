'''
CardReader 模块
用于读取 cards.cdb 数据库，提供卡片名称和属性查询功能
'''


import sqlite3
import os
from card_announcement import AnnouncementCard

class CardReader:
    def __init__(self, db_path='cards.cdb'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.cache = {}
        self.stats_cache = {}
        self.exists_cache = {}
        self.announcement_cache = {}
        self.announcement_catalog = None
        
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

    def has_card(self, code):
        """检查当前 cards.cdb 是否能为卡密提供完整数据"""
        try:
            normalized = int(code)
        except (TypeError, ValueError):
            return False
        if normalized in self.exists_cache:
            return self.exists_cache[normalized]
        if not self.cursor:
            return False
        try:
            self.cursor.execute("SELECT 1 FROM datas WHERE id=?", (normalized,))
            exists = self.cursor.fetchone() is not None
        except Exception:
            exists = False
        self.exists_cache[normalized] = exists
        return exists

    def _announcement_from_row(self, row):
        """缓存原始宣言字段，保留 alias 和完整类型位供 Core 合法性判断"""
        code, alias, card_type, race, attribute, packed_setcodes = row
        card = AnnouncementCard(
            int(code), int(alias or 0), int(card_type or 0), int(race or 0),
            int(attribute or 0),
            tuple((int(packed_setcodes or 0) >> shift) & 0xFFFF
                  for shift in (0, 16, 32, 48)),
        )
        self.announcement_cache[card.code] = card
        return card

    def get_announcement_card(self, code):
        """读取宣言专用元数据；不存在的卡返回 None，不伪造成合法零属性卡"""
        code = int(code)
        if code not in self.announcement_cache:
            row = self.cursor.execute(
                'SELECT id, alias, type, race, attribute, setcode FROM datas WHERE id=?',
                (code,),
            ).fetchone() if self.cursor else None
            self.announcement_cache[code] = self._announcement_from_row(row) if row else None
        return self.announcement_cache[code]

    def get_announcement_catalog(self):
        """仅在常识池无合法项时缓存完整卡库，仍须经过相同 RPN 过滤"""
        if self.announcement_catalog is None:
            rows = self.cursor.execute(
                'SELECT id, alias, type, race, attribute, setcode FROM datas ORDER BY id'
            ).fetchall() if self.cursor else ()
            self.announcement_catalog = tuple(self._announcement_from_row(row) for row in rows)
        return self.announcement_catalog

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

# 单例始终绑定项目资产，避免从其他工作目录启动时误读 CDB。
card_db = CardReader(
    db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards.cdb")
)
