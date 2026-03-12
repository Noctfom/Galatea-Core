'''
GalateaEnv 模块
封装 OCGCore DLL 的交互逻辑
'''

import ctypes
import os
import sqlite3
import struct
import time
import random

# --- OCGCore 常量 ---
LOCATION_DECK = 0x01
LOCATION_HAND = 0x02
LOCATION_MZONE = 0x04
LOCATION_SZONE = 0x08
LOCATION_GRAVE = 0x10
LOCATION_REMOVED = 0x20
LOCATION_EXTRA = 0x40

# --- 结构体对齐 (基于 card_data.h 源码) ---
class CardData(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint32),
        ("alias", ctypes.c_uint32),
        ("setcode", ctypes.c_uint16 * 16), # 源码确认: uint16 setcode[16]
        ("type", ctypes.c_uint32),
        ("level", ctypes.c_uint32),
        ("attribute", ctypes.c_uint32),
        ("race", ctypes.c_uint32),
        ("attack", ctypes.c_int32),
        ("defense", ctypes.c_int32),
        ("lscale", ctypes.c_uint32),
        ("rscale", ctypes.c_uint32),
        ("link_marker", ctypes.c_uint32),
    ]

SCRIPT_READER_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
CARD_READER_FUNC = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(CardData))
MSG_HANDLER_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)

class GalateaEnv:
    def __init__(self, dll_path='./ocgcore.dll', cdb_path='./cards.cdb', script_path='./script'):
        # 路径标准化
        self.dll_path = os.path.abspath(dll_path)
        self.cdb_path = os.path.abspath(cdb_path)
        self.script_path = os.path.abspath(script_path)

        # Windows DLL 加载优化
        if os.name == 'nt':
            try: os.add_dll_directory(os.path.dirname(self.dll_path))
            except: pass
        
        self.lib = ctypes.cdll.LoadLibrary(self.dll_path)
        self.cdb = sqlite3.connect(self.cdb_path)
        self.pduel = None
        
        # [核心修复] 内存保活容器
        # 这一步至关重要：C++ 的 load_script 假设 buffer 一直有效
        # 如果这里不存，Python GC 会回收内存，导致 C++ 读到垃圾 -> RETRY 死循环
        self.script_buffers = {} 
        
        self.cb_script_reader = SCRIPT_READER_FUNC(self._on_read_script)
        self.cb_card_reader = CARD_READER_FUNC(self._on_read_card)
        self.cb_msg_handler = MSG_HANDLER_FUNC(self._on_message)
        
        self._setup_lib()
        
        # 注册回调
        self.lib.set_script_reader(self.cb_script_reader)
        self.lib.set_card_reader(self.cb_card_reader)
        if hasattr(self.lib, 'set_message_handler'):
            self.lib.set_message_handler(self.cb_msg_handler)

    def _setup_lib(self):
        # API 签名映射
        self.lib.create_duel.argtypes = [ctypes.c_uint32]; self.lib.create_duel.restype = ctypes.c_void_p
        self.lib.start_duel.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.lib.end_duel.argtypes = [ctypes.c_void_p]
        self.lib.set_player_info.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        # new_card 签名验证无误
        self.lib.new_card.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
        self.lib.process.argtypes = [ctypes.c_void_p]; self.lib.process.restype = ctypes.c_int32
        # set_responseb 签名验证
        self.lib.set_responseb.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        if hasattr(self.lib, 'get_message'):
            self.lib.get_message.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte)]
            self.lib.get_message.restype = ctypes.c_uint32
            
        # 确认使用 set_responsei
        self.lib.set_responsei.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    def _on_read_script(self, name_ptr, len_ptr):
        try:
            # C++ 传过来可能是 "./script/c123.lua"
            raw_name = name_ptr.decode('utf-8')
            # 剥离路径，只取文件名 "c123.lua"
            basename = os.path.basename(raw_name)
            
            # 1. 检查缓存 (内存保活)
            if basename in self.script_buffers:
                buf = self.script_buffers[basename]
                len_ptr[0] = len(buf)
                return ctypes.addressof(buf)
            
            # 2. 从磁盘读取
            content = None
            # 尝试多个常见路径
            paths_to_try = [
                os.path.join(self.script_path, basename),
                os.path.join(self.script_path, "official", basename),
                os.path.join(self.script_path, "c" + basename) # 应对只传ID的情况
            ]
            
            for p in paths_to_try:
                if os.path.exists(p):
                    with open(p, 'rb') as f: content = f.read()
                    break
            
            if content:
                # [核心修复] 创建 ctypes buffer 并永久保存引用
                # 这一行阻止了 Python 垃圾回收销毁这块内存
                buf = (ctypes.c_byte * len(content)).from_buffer_copy(content)
                self.script_buffers[basename] = buf 
                
                len_ptr[0] = len(content)
                # print(f"📜 Loaded: {basename}") # 调试：确认脚本加载成功
                return ctypes.addressof(buf)
            
            # 如果找不到 constant.lua，打印红色警告
            if "constant" in basename or "utility" in basename:
                print(f"\033[91m❌ [Script Error] 缺少核心脚本: {basename} (在 {self.script_path} 下找不到)\033[0m")
            
            return 0
        except: return 0

    def _on_read_card(self, code, data_ptr):
        try:
            cursor = self.cdb.cursor()
            cursor.execute("SELECT * FROM datas WHERE id=?", (code,))
            row = cursor.fetchone()
            if not row: return 0 
            
            data = data_ptr.contents
            data.code = row[0]; data.alias = row[2]
            
            # MDPro3 setcode 填充逻辑 (uint16 array)
            setcode_val = row[3]
            for i in range(16): data.setcode[i] = 0
            ctr = 0
            while setcode_val and ctr < 16:
                if (setcode_val & 0xffff):
                    data.setcode[ctr] = setcode_val & 0xffff
                    ctr += 1
                setcode_val >>= 16
            
            data.type = row[4]; data.attack = row[5]; data.defense = row[6]
            data.level = row[7] & 0xFF; data.race = row[8]; data.attribute = row[9]
            data.lscale = (row[7] >> 24) & 0xFF; data.rscale = (row[7] >> 16) & 0xFF
            return 1 
        except: return 0

    def _on_message(self, pduel, msg_type): 
        # MDPro3 可能会通过 msg_type=1 发送 lua 错误信息
        # 如果有需要可以在这里 hook 错误日志
        return 0

    # --- Callbacks ---
    def dummy_script_reader(self, ptr, name): return 0
    def dummy_card_reader(self, code, data): return 0
    def dummy_message_handler(self, ptr, msg_type): return 0

    def reset(self, deck0, deck1):
        if self.pduel:
            self.lib.end_duel(self.pduel)
            self.pduel = None
        
        # 每次重置时不必清空 script_buffers，常用脚本常驻内存更好
        
        seed = int(time.time()) & 0xFFFFFFFF
        self.pduel = self.lib.create_duel(seed)
        
        self.lib.set_player_info(self.pduel, 0, 8000, 5, 1)
        self.lib.set_player_info(self.pduel, 1, 8000, 5, 1)
        
        def inject_deck(player_id, deck_obj):
            # 主卡组洗牌
            main_cards = deck_obj.main[:]
            random.shuffle(main_cards) 
            for code in main_cards:
                self.lib.new_card(self.pduel, code, player_id, player_id, LOCATION_DECK, 0, 0)
            
            # 额外卡组加载
            extra_cards = deck_obj.extra[:]
            random.shuffle(extra_cards)
            for code in extra_cards:
                self.lib.new_card(self.pduel, code, player_id, player_id, LOCATION_EXTRA, 0, 0)

        inject_deck(0, deck0)
        inject_deck(1, deck1)
        
        self.lib.start_duel(self.pduel, 0)
        return self.step()

    def step(self):
        msg_buf = (ctypes.c_byte * 65536)() 
        for _ in range(1000):
            res = self.lib.process(self.pduel)
            if hasattr(self.lib, 'get_message'):
                msg_len = self.lib.get_message(self.pduel, ctypes.cast(msg_buf, ctypes.POINTER(ctypes.c_byte)))
                if msg_len > 0: return bytearray(msg_buf)[:msg_len]
            if res == 0: return None 
        return None

    # galatea_env.py 修改 send_action 方法
    def send_action(self, response):
        if isinstance(response, int):
            # 简单交互用整数
            self.lib.set_responsei(self.pduel, ctypes.c_uint32(response))
        elif isinstance(response, (bytes, bytearray)):
            # 复杂交互用 Buffer (如选位置)
            # 确保 set_responseb 已在 _setup_lib 中定义
            # self.lib.set_responseb.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            buf = (ctypes.c_byte * len(response)).from_buffer_copy(response)
            self.lib.set_responseb(self.pduel, ctypes.cast(buf, ctypes.c_void_p))