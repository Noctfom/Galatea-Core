# 本文件安全读取标准 YRP1/YRP2 与明确分帧的 YRP3D，保留原始响应并拒绝猜测格式
import hashlib
import lzma
import struct

MAX_YRP_FILE_BYTES = 16 * 1024 * 1024
MAX_YRP_DATA_BYTES = 8 * 1024 * 1024
MAX_YRP_RESPONSES = 100_000
YRP1 = 0x31707279
YRP2 = 0x32707279


class UnsupportedReplayError(ValueError):
    """表示已识别但当前公开 Core 接口不能可靠还原的录像格式"""


def _unwrap_yrp(data):
    """只从完整 YRP 或长度正确的 YRP3D replay 包提取数据，不全文搜索魔数"""
    if data[:4] in (b'yrp1', b'yrp2'):
        return data, 'yrp', []
    offset, embedded, packet_types = 0, [], []
    while offset < len(data):
        if len(data) - offset < 5:
            raise ValueError('truncated YRP3D packet header')
        kind, size = struct.unpack_from('<BI', data, offset)
        offset += 5
        if size > len(data) - offset:
            raise ValueError('truncated YRP3D packet payload')
        packet_types.append(kind)
        if len(packet_types) > MAX_YRP_RESPONSES:
            raise ValueError('too many YRP3D packets')
        if kind == 231:
            embedded.append(data[offset:offset + size])
        offset += size
    if len(embedded) != 1:
        raise UnsupportedReplayError('YRP3D must contain exactly one complete replay packet (231)')
    if embedded[0][:4] not in (b'yrp1', b'yrp2'):
        raise ValueError('invalid embedded YRP header')
    return embedded[0], 'yrp3d', sorted(set(packet_types))


def read_yrp(filepath):
    """有界读取录像、解压与响应流；仅解析数据，不加载模型、不调用 Core"""
    with open(filepath, 'rb') as stream:
        file_data = stream.read(MAX_YRP_FILE_BYTES + 1)
    if len(file_data) > MAX_YRP_FILE_BYTES:
        raise ValueError('YRP file size limit exceeded')
    data, container, packet_types = _unwrap_yrp(file_data)
    if len(data) < 32:
        raise ValueError('truncated YRP header')
    magic, version, flags, seed, datasize, start_time, props = struct.unpack_from('<6I8s', data)
    header_size = 32
    seed_sequence = None
    if magic == YRP2:
        if len(data) < 80:
            raise ValueError('truncated YRP2 extended header')
        seed_sequence = list(struct.unpack_from('<8I', data, 32))
        extended = struct.unpack_from('<4I', data, 64)
        if extended != (1, 0, 0, 0):
            raise UnsupportedReplayError('unsupported YRP2 extended header version/parameters')
        header_size = 80
    elif magic != YRP1:
        raise ValueError('invalid YRP magic')
    if version < 0x12D0 or flags & ~0x1F:
        raise UnsupportedReplayError('unsupported YRP version or flags')
    if flags & (0x2 | 0x8):
        raise UnsupportedReplayError('Tag/single-script YRP is not a two-player constructed duel')
    if not 0 < datasize <= MAX_YRP_DATA_BYTES:
        raise ValueError('invalid YRP decompressed size')
    if flags & 1:
        # 限制声明字典及输出，防止小压缩文件耗尽内存。
        dictionary = struct.unpack_from('<I', props, 1)[0]
        if dictionary > 64 * 1024 * 1024:
            raise ValueError('YRP LZMA dictionary limit exceeded')
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE, memlimit=96 * 1024 * 1024)
        try:
            body = decoder.decompress(props[:5] + struct.pack('<Q', datasize) + data[header_size:], max_length=datasize + 1)
        except lzma.LZMAError:
            # 公开编码器允许有/无流结束标记；第二种必须真的读到结束标记，输出仍有界。
            del decoder
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE, memlimit=96 * 1024 * 1024)
            body = decoder.decompress(props[:5] + b'\xff' * 8 + data[header_size:], max_length=datasize + 1)
        if len(body) != datasize or not decoder.eof or decoder.unused_data:
            raise ValueError('YRP compressed size/trailing data mismatch')
    else:
        body = data[header_size:]
        if len(body) != datasize:
            raise ValueError('YRP uncompressed size mismatch')
    offset = 0

    def take(size):
        """按确定长度消费字段，截断时拒绝继续猜测"""
        nonlocal offset
        if size < 0 or offset + size > len(body):
            raise ValueError('truncated YRP field')
        value = body[offset:offset + size]
        offset += size
        return value

    def read_name():
        """仅解码首个 UTF16 终止符前的昵称；尾部未初始化填充不作为玩家身份"""
        raw = take(40)
        end = next((index for index in range(0, 40, 2) if raw[index:index + 2] == b'\x00\x00'), 40)
        return raw[:end].decode('utf-16-le', errors='replace')

    players = [read_name() for _ in range(2)]
    start_lp, start_hand, draw_count, duel_flag = struct.unpack('<4I', take(16))
    decks = []
    for _ in range(2):
        deck = {}
        for part in ('main', 'extra'):
            count = struct.unpack('<I', take(4))[0]
            if count > 256:
                raise ValueError('YRP deck count limit exceeded')
            codes = list(struct.unpack(f'<{count}I', take(count * 4)))
            if any(code == 0 for code in codes):
                raise ValueError('YRP contains invalid card code')
            deck[part] = codes
        decks.append(deck)
    responses = []
    while offset < len(body):
        size = take(1)[0]
        if not size:
            raise ValueError('empty YRP response cannot be safely replayed')
        responses.append(take(size))
        if len(responses) > MAX_YRP_RESPONSES:
            raise ValueError('YRP response count limit exceeded')
    return {
        'seed': seed, 'seed_sequence': seed_sequence, 'flag': flags, 'players': players, 'duel_flag': duel_flag,
        'start_lp': start_lp, 'start_hand': start_hand, 'draw_count': draw_count,
        'decks': decks, 'responses': responses, 'version': version, 'start_time': start_time,
        'container': container, 'packet_types': packet_types,
        'file_sha256': hashlib.sha256(file_data).hexdigest(),
        'replay_sha256': hashlib.sha256(data).hexdigest(),
        'historical_assets_known': False, 'original_core_output_available': False,
        'replay_mode_supported': bool(flags & 0x10),
    }


def inspect_yrp(filepath):
    """返回不含长响应数组的格式诊断，明确历史资产及训练资格边界"""
    data = read_yrp(filepath)
    limitations = ['historical_assets_unknown', 'original_core_output_unavailable']
    if not data['replay_mode_supported']:
        limitations.append('legacy_non_uniform_replay_mode_unsupported')
    return {key: value for key, value in data.items() if key != 'responses'} | {
        'response_count': len(data['responses']), 'behavior_clone_eligible': False,
        'limitations': limitations,
    }
