'''
GalateaEnv 模块
封装 OCGCore DLL 的交互逻辑
'''

import ctypes
import os
import sqlite3
import time
import random
import sys

from core_query import PUBLIC_CARD_QUERY_FLAGS, parse_legacy_card_query
from selection_protocol import build_core_response_buffer

# --- OCGCore 常量 ---
LOCATION_DECK = 0x01
LOCATION_HAND = 0x02
LOCATION_MZONE = 0x04
LOCATION_SZONE = 0x08
LOCATION_GRAVE = 0x10
LOCATION_REMOVED = 0x20
LOCATION_EXTRA = 0x40

# 全局纯内存卡片数据库缓存
_GLOBAL_CARD_CACHE = {}
_GLOBAL_CACHE_INIT = False


def _copy_ctypes_bytes(buffer, length):
    """按原始无符号字节复制 ctypes 缓冲区，避免 c_byte 负值破坏 bytes 转换"""
    size = ctypes.sizeof(buffer)
    normalized_length = int(length)
    if normalized_length < 0 or normalized_length > size:
        raise ValueError(
            f"ctypes buffer length {normalized_length} is outside 0..{size}"
        )
    return ctypes.string_at(ctypes.addressof(buffer), normalized_length)

def _init_card_cache():
    global _GLOBAL_CACHE_INIT, _GLOBAL_CARD_CACHE
    if _GLOBAL_CACHE_INIT: return
    try:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cards.cdb'))
        # 🚀 修复 Windows URI 报错：直接使用原生连接，并给一个 10 秒的锁等待防争抢
        conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM datas")
        for row in cursor.fetchall():
            _GLOBAL_CARD_CACHE[row[0]] = row
        conn.close()
        _GLOBAL_CACHE_INIT = True
    except Exception as e:
        # 换成这个醒目的打印，如果以后连不上数据库，终端会直接炸红字！
        print(f"\n❌ [致命错误] 无法将 cards.cdb 载入内存: {e}\n")
        

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
    # 动态判断默认核心文件名
    default_core = './ocgcore.so' if sys.platform.startswith('linux') else './ocgcore.dll'
    
    def __init__(self, dll_path=default_core, cdb_path='./cards.cdb', script_path='./script'):
        # 路径标准化
        self.dll_path = os.path.abspath(dll_path)
        self.cdb_path = os.path.abspath(cdb_path)
        self.script_path = os.path.abspath(script_path)

        # Windows DLL 加载优化
        if os.name == 'nt':
            try: os.add_dll_directory(os.path.dirname(self.dll_path))
            except Exception as e: 
                print(f"[galatea_env]⚠️ 无法添加 DLL 目录: {e}")
        
        self.lib = ctypes.cdll.LoadLibrary(self.dll_path)
        self.cdb = sqlite3.connect(self.cdb_path)
        self.pduel = None
        self.query_card_success_count = 0
        self.query_card_error_count = 0
        
        # [核心修复] 内存保活容器
        # 这一步至关重要：C++ 的 load_script 假设 buffer 一直有效
        # 如果这里不存，Python GC 会回收内存，导致 C++ 读到垃圾 -> RETRY 死循环
        self.script_buffers = {} 
        self._preload_all_scripts()
        
        self.cb_script_reader = SCRIPT_READER_FUNC(self._on_read_script)
        self.cb_card_reader = CARD_READER_FUNC(self._on_read_card)
        self.cb_msg_handler = MSG_HANDLER_FUNC(self._on_message)
        
        self._setup_lib()

        self.msg_buf = (ctypes.c_byte * 65536)()
        self.query_buf = (ctypes.c_byte * 8192)()
        
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
        self.lib.set_responseb.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        if hasattr(self.lib, 'get_message'):
            self.lib.get_message.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte)]
            self.lib.get_message.restype = ctypes.c_uint32

        # V4 只使用公开 Core 查询接口，不依赖本地私有内核扩展
        if not hasattr(self.lib, 'query_card'):
            raise RuntimeError("当前公开 Core 未导出 query_card，无法构造 V4 完整观测")
        self.lib.query_card.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint32, ctypes.POINTER(ctypes.c_byte), ctypes.c_int32]
        self.lib.query_card.restype = ctypes.c_int32
        if hasattr(self.lib, 'query_field_count'):
            self.lib.query_field_count.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8]
            self.lib.query_field_count.restype = ctypes.c_int32
            
        # 确认使用 set_responsei
        self.lib.set_responsei.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    def _preload_all_scripts(self):
        """[极致优化] 在环境启动时，将所有脚本一次性载入内存，彻底斩断磁盘 I/O"""
        import glob
        print(f"🚀 正在将 {self.script_path} 下的 Lua 脚本全量装载至内存...")
        
        # 扫描 script 目录及其 official 子目录下的所有 lua 文件
        search_paths = [
            os.path.join(self.script_path, "*.lua"),
            os.path.join(self.script_path, "official", "*.lua")
        ]
        
        count = 0
        for pattern in search_paths:
            for filepath in glob.glob(pattern):
                basename = os.path.basename(filepath)
                if basename not in self.script_buffers:
                    with open(filepath, 'rb') as f:
                        content = f.read()
                        
                    # 加上防弹 \0 截断
                    content_with_null = content + b'\0'
                    buf = (ctypes.c_byte * len(content_with_null)).from_buffer_copy(content_with_null)
                    self.script_buffers[basename] = (buf, len(content)) # 存 buf 和真实长度
                    count += 1
                    
        print(f"✅ 预加载完成，共吸入 {count} 个脚本，彻底告别磁盘 I/O！")

    # 对应的 _on_read_script 修改为极速模式：
    def _on_read_script(self, name_ptr, len_ptr):
        raw_name = name_ptr.decode('utf-8')
        basename = os.path.basename(raw_name)
        
        # O(1) 极速内存读取，如果没有就返回 0 (通常不可能，因为已经全量加载)
        if basename in self.script_buffers:
            buf, real_len = self.script_buffers[basename]
            len_ptr[0] = real_len
            return ctypes.addressof(buf)
        print(f"⚠️ [致命警告] 内存未命中，引擎试图索要 {basename} 失败！AI 正在打白板牌！", flush=True)
        return 0

    def _on_read_card(self, code, data_ptr):
        try:
            _init_card_cache()
            row = _GLOBAL_CARD_CACHE.get(code)
            if not row: return 0
            
            data = data_ptr.contents
            
            # 强转防呆，预防极端情况下的数据库 NULL 值
            data.code = int(row[0] or 0)
            data.alias = int(row[2] or 0)
            
            setcode_val = int(row[3] or 0)
            for i in range(16): data.setcode[i] = 0
            ctr = 0
            while setcode_val and ctr < 16:
                if (setcode_val & 0xffff):
                    data.setcode[ctr] = setcode_val & 0xffff
                    ctr += 1
                setcode_val >>= 16
            
            data.type = int(row[4] or 0)
            data.attack = int(row[5] or 0)
            data.defense = int(row[6] or 0)
            
            level_val = int(row[7] or 0)
            data.level = level_val & 0xFF
            data.race = int(row[8] or 0)
            data.attribute = int(row[9] or 0)
            data.lscale = (level_val >> 24) & 0xFF
            data.rscale = (level_val >> 16) & 0xFF
            
            # 兼容 Link 怪兽的额外字段填充 (确保 Link 值不串台)
            if data.type & 0x4000000: # TYPE_LINK
                data.link_marker = int(row[6] or 0)
                data.defense = 0
            
            return 1
            
        except Exception as e:
            print(f"⚠️ _on_read_card (C++底盘读卡) 处理异常: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def query_card_state(self, player_id, location, sequence):
        """通过公开旧版 query_card 读取一张卡的标准动态状态"""
        length = self.lib.query_card(
            self.pduel,
            player_id,
            location,
            sequence,
            PUBLIC_CARD_QUERY_FLAGS,
            self.query_buf,
            0,
        )
        if length <= 8:
            return None
        try:
            result = parse_legacy_card_query(
                _copy_ctypes_bytes(self.query_buf, length)
            )
            result['code'] &= 0x7FFFFFFF
            result['pos'] = result['position']
            result['owner'] = result['controller']
            result['current_atk'] = max(0, result['current_atk'])
            result['current_def'] = max(0, result['current_def'])
            result['base_atk'] = max(0, result['base_atk'])
            result['base_def'] = max(0, result['base_def'])
            # 保留旧字段名供现有状态机平滑接线。
            result['overlays'] = list(result['overlay_codes'])
            result['counters'] = result['counter_count']
            result['is_equipped'] = False
            self.query_card_success_count += 1
            return result
        except Exception as e:
            self.query_card_error_count += 1
            print(
                "⚠️ query_card 解析异常: "
                f"P{player_id} loc={location:#x} seq={sequence} len={length}: {e}"
            )
            return None

    def query_zone_count(self, player_id, location):
        """通过公开 query_field_count 获取连续区域的实际卡片数量"""
        if not hasattr(self.lib, 'query_field_count'):
            return None
        count = self.lib.query_field_count(self.pduel, player_id, location)
        return max(0, int(count))

    def _on_message(self, pduel, msg_type): 
        # 可能会通过 msg_type=1 发送 lua 错误信息
        # 如果有需要可以在这里 hook 错误日志
        return 0

    # --- Callbacks ---
    def dummy_script_reader(self, ptr, name): return 0
    def dummy_card_reader(self, code, data): return 0
    def dummy_message_handler(self, ptr, msg_type): return 0

    def __del__(self):
        """析构函数：确保 Python 对象销毁时，C++ 端的内存也被彻底释放"""
        self._close_duel()

    def _close_duel(self):
        """内部清理函数"""
        if hasattr(self, 'pduel') and self.pduel is not None:
            try:
                # 显式通知 ocgcore 销毁这个决斗实例
                self.lib.end_duel(self.pduel)
                self.pduel = None
            except Exception:
                pass

    # 修改参数，增加 seed=None
    def reset(self, deck0, deck1, seed=None):
        self._close_duel()
        if self.pduel:
            self.lib.end_duel(self.pduel)
            self.pduel = None
        
        # 修复1：动态处理 Seed
        if seed is None:
            duel_seed = int(time.time()) & 0xFFFFFFFF
            is_replay = False # AI 左右互搏模式
        else:
            duel_seed = seed
            is_replay = True  # 录像放映模式
            
        self.pduel = self.lib.create_duel(duel_seed)
        
        self.lib.set_player_info(self.pduel, 0, 8000, 5, 1)
        self.lib.set_player_info(self.pduel, 1, 8000, 5, 1)
        
        def inject_deck(player_id, deck_obj):
            # 主卡组加载
            main_cards = deck_obj.main[:]
            # 修复2：录像模式下不能在 Python 层洗牌
            if not is_replay:
                random.shuffle(main_cards) 
            for code in main_cards:
                self.lib.new_card(self.pduel, code, player_id, player_id, LOCATION_DECK, 0, 0)
            
            # 额外卡组加载
            extra_cards = deck_obj.extra[:]
            if not is_replay:
                random.shuffle(extra_cards)
            for code in extra_cards:
                self.lib.new_card(self.pduel, code, player_id, player_id, LOCATION_EXTRA, 0, 0)

        inject_deck(0, deck0)
        inject_deck(1, deck1)
        
        self.lib.start_duel(self.pduel, 0)
        return self.step()

    def step(self):
        for i in range(100000):
            res = self.lib.process(self.pduel)
            
            msg_len = self.lib.get_message(self.pduel, ctypes.cast(self.msg_buf, ctypes.POINTER(ctypes.c_byte)))
            if msg_len > 0:
                return bytearray(self.msg_buf)[:msg_len]
            
            if res == 0:
                return None 
            
            # 🌟 [新增] 心跳打印：如果它循环了 5万次还没出结果，强制发声！
            if i == 50000:
                print(f"   [底层心跳] C++ 正在疯狂运算中 (当前 res={res})...")
                
        print("⚠️ 警告：引擎运算超时（10万次循环未响应）")
        return None

    def send_action(self, response):
        """向 Core 提交整数或固定 512 字节响应，并在调用期间保持缓冲存活"""
        if isinstance(response, int):
            self.lib.set_responsei(self.pduel, ctypes.c_uint32(response))
        elif isinstance(response, (bytes, bytearray)):
            resp_bytes = build_core_response_buffer(response)

            self._lifeline_response = resp_bytes
            self.lib.set_responseb(self.pduel, self._lifeline_response)
