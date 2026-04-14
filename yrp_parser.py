import struct
import lzma
import os
import glob

class YrpParser:
    def __init__(self, filepath):
        self.filepath = filepath

    def parse(self):
        if not os.path.exists(self.filepath):
            print(f"❌ 找不到文件: {self.filepath}")
            return None

        with open(self.filepath, 'rb') as f:
            file_bytes = f.read()

        # ==============================================================
        # 🌟 终极剥壳技术：直接全文扫描 YRP 魔法数字特征码！
        # YRP1 = 0x31707279 (79 72 70 31), YRP2 = 0x32707279 (79 72 70 32)
        # ==============================================================
        idx = file_bytes.rfind(b'\x79\x72\x70\x31')
        if idx == -1:
            idx = file_bytes.rfind(b'\x79\x72\x70\x32')
            
        if idx == -1:
            print("❌ 无效的录像文件，找不到 YRP 核心数据！")
            return None
            
        if idx > 0:
            print("✨ 检测到 MDPro3 套壳录像，已暴力切除 3D 外壳！")
        
        # 🗡️ 极致暴力：直接从特征码切到文件末尾！
        # 抛弃 TCP 包长度猜测，LZMA 解压器会自动忽略尾部的冗余网络残渣！
        yrp_bytes = file_bytes[idx:]

        # 1. 动态判断 YRP1 还是 YRP2，确定头部长度
        magic = struct.unpack('<I', yrp_bytes[0:4])[0]
        header_size = 80 if magic == 0x32707279 else 32
        
        header_data = yrp_bytes[:header_size]
        _, version, flag, seed, datasize, hash_val = struct.unpack('<IIIIII', header_data[:24])
        props = header_data[24:29]
        
        # 2. 提取压缩数据
        comp_data = yrp_bytes[header_size:]

        # 3. 解压 (使用 LZMADecompressor 容忍并忽略文件尾部的冗余网络数据)
        lzma_header = props + struct.pack('<Q', datasize)
        try:
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
            uncompressed_data = decompressor.decompress(lzma_header + comp_data)
        except Exception as e:
            print(f"❌ LZMA 解压失败: {e}")
            return None

        offset = 0
        
        # 4. 解析玩家名字
        is_tag = flag & 0x2
        player_count = 4 if is_tag else 2
        players = []
        for _ in range(player_count):
            name_bytes = uncompressed_data[offset:offset+40]
            name = name_bytes.decode('utf-16-le', errors='ignore').rstrip('\x00')
            players.append(name)
            offset += 40

        # 5. 解析游戏参数 (DuelParameters, 16 字节)
        start_lp, start_hand, draw_count, duel_flag = struct.unpack('<IIII', uncompressed_data[offset:offset+16])
        offset += 16

        # 6. 解析双方卡组
        decks = []
        for p_idx in range(player_count):
            main_count = struct.unpack('<I', uncompressed_data[offset:offset+4])[0]
            offset += 4
            if main_count > 100:
                print(f"⚠️ 玩家 {p_idx} 的主卡组数量异常 ({main_count})")
                return None
                
            main_deck = struct.unpack(f'<{main_count}I', uncompressed_data[offset:offset+main_count*4])
            offset += main_count * 4
            
            extra_count = struct.unpack('<I', uncompressed_data[offset:offset+4])[0]
            offset += 4
            extra_deck = struct.unpack(f'<{extra_count}I', uncompressed_data[offset:offset+extra_count*4])
            offset += extra_count * 4
            
            decks.append({'main': list(main_deck), 'extra': list(extra_deck)})

        # 7. 提取操作流
        responses = []
        while offset < len(uncompressed_data):
            resp_len = uncompressed_data[offset]
            offset += 1
            if resp_len == 0: continue
            resp_data = uncompressed_data[offset:offset+resp_len]
            responses.append(resp_data)
            offset += resp_len

        return {
            'seed': seed, 'flag': flag, 'players': players,
            'duel_flag': duel_flag, 
            'start_lp': start_lp, 'start_hand': start_hand, 'draw_count': draw_count,
            'decks': decks, 'responses': responses
        }

if __name__ == "__main__":
    replay_dir = "./replays"
    if not os.path.exists(replay_dir): os.makedirs(replay_dir)
    
    # 🌟 同扫两种格式
    yrp_files = glob.glob(os.path.join(replay_dir, "*.yrp")) + glob.glob(os.path.join(replay_dir, "*.yrp3d"))
    
    if not yrp_files:
        print("❌ 找不到任何录像！")
    else:
        test_file = yrp_files[0]
        print(f"🚀 开始解析录像: {test_file}")
        parser = YrpParser(test_file)
        data = parser.parse()
        
        if data:
            print("✅ 解析成功！")
            print(f"👥 玩家: {data['players']}")
            print(f"🎲 种子: {data['seed']}")
            print(f"🖱️ 步数: {len(data['responses'])}")
            
            print("\n前5步操作预览:")
            for i, resp in enumerate(data['responses'][:5]):
                print(f"   [Step {i+1}] {' '.join([f'{b:02X}' for b in resp])}")