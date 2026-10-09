"""按随包公开旧式 Core 的固定布局解码连锁询问、卡片确认和场面快照"""

import io
import struct


CORE_MESSAGE_PROTOCOL = 'fluorohydride_legacy_per_candidate_forced_v1'


class CoreMessageProtocolError(ValueError):
    """表示报文不符合当前随包 Core 的固定协议"""


def _read_exact(stream, size, label):
    """读取完整字段，拒绝把截断报文静默当成空消息"""
    value = stream.read(size)
    if len(value) != size:
        raise CoreMessageProtocolError(
            f'{label} truncated: expected {size} bytes, got {len(value)}'
        )
    return value


def read_chain_prompt(stream):
    """读取 Type 16；强制标志属于逐候选字段，而非消息头"""
    player, count, special_count, timing, opponent_timing = struct.unpack(
        '<BBBII', _read_exact(stream, 11, 'Type 16 header')
    )
    if player not in (0, 1):
        raise CoreMessageProtocolError('Type 16 player must be 0 or 1')
    candidates = []
    for index in range(count):
        mode, forced, code, location, desc = struct.unpack(
            '<BBIII', _read_exact(stream, 14, f'Type 16 candidate {index}')
        )
        if mode not in (0, 1, 2) or forced not in (0, 1):
            raise CoreMessageProtocolError(
                f'Type 16 candidate {index} has invalid mode/forced flags'
            )
        candidates.append({
            'effect_mode': mode, 'forced': bool(forced), 'code': code,
            'location': location, 'desc': desc,
        })
    return {
        'player': player, 'special_count': special_count,
        'hint_timing': timing, 'opponent_hint_timing': opponent_timing,
        'forced': any(candidate['forced'] for candidate in candidates),
        'candidates': candidates,
    }


def parse_chain_prompt(payload):
    """按固定布局解析完整连锁负载，拒绝截断及多余尾部字节"""
    stream = io.BytesIO(bytes(payload))
    prompt = read_chain_prompt(stream)
    if stream.read(1):
        raise CoreMessageProtocolError('Type 16 has unexpected trailing bytes')
    return prompt


def read_confirm_cards(stream):
    """读取 Type 31 的玩家、跳过展示面板标志及完整确认卡列表"""
    player, skip_panel, count = _read_exact(stream, 3, 'Type 31 header')
    if player not in (0, 1) or skip_panel not in (0, 1):
        raise CoreMessageProtocolError('Type 31 has invalid player/skip_panel')
    cards = []
    for index in range(count):
        code, controller, location, sequence = struct.unpack(
            '<IBBB', _read_exact(stream, 7, f'Type 31 card {index}')
        )
        cards.append({
            'code': code, 'controller': controller,
            'location': location, 'sequence': sequence,
        })
    return {'player': player, 'skip_panel': bool(skip_panel), 'cards': cards}


def parse_confirm_cards(payload):
    """解析完整确认负载；skip_panel 只控制界面，不改变卡片公开结果"""
    stream = io.BytesIO(bytes(payload))
    prompt = read_confirm_cards(stream)
    if stream.read(1):
        raise CoreMessageProtocolError('Type 31 has unexpected trailing bytes')
    return prompt


def is_empty_chain_prompt(msg_type, payload):
    """识别已校验的空连锁询问，避免为唯一自动响应调用模型"""
    return msg_type == 16 and len(payload) == 11 and payload[1] == 0


def read_reload_field(stream):
    """按随包 Core 固定7/8槽读取162，不按规则号猜测怪兽区长度"""
    rule = _read_exact(stream, 1, 'Type 162 rule')[0]
    players = []
    for player in (0, 1):
        lp = struct.unpack('<I', _read_exact(stream, 4, 'Type 162 LP'))[0]
        zones = []
        for capacity, width in ((7, 2), (8, 1)):
            cards = []
            for sequence in range(capacity):
                occupied = _read_exact(stream, 1, 'Type 162 occupied')[0]
                if occupied not in (0, 1):
                    raise CoreMessageProtocolError('Type 162 invalid occupied flag')
                if occupied:
                    values = _read_exact(stream, width, 'Type 162 card')
                    cards.append((sequence, values[0], values[1] if width == 2 else 0))
            zones.append(tuple(cards))
        counts = tuple(_read_exact(stream, 6, 'Type 162 zone counts'))
        if counts[5] > counts[4]:
            raise CoreMessageProtocolError('Type 162 face-up Extra count is invalid')
        players.append({'lp': lp, 'monsters': zones[0], 'spells': zones[1], 'counts': counts})
    count = _read_exact(stream, 1, 'Type 162 chain count')[0]
    chains = []
    for index in range(count):
        code, raw, controller, location, sequence, desc = struct.unpack(
            '<IIBBBI', _read_exact(stream, 15, 'Type 162 chain')
        )
        if controller not in (0, 1):
            raise CoreMessageProtocolError('Type 162 invalid chain player')
        chains.append((code, raw, controller, location, sequence, desc))
    return {'rule': rule, 'players': tuple(players), 'chains': tuple(chains)}


def parse_reload_field(payload):
    """完整验证快照后再交给状态层，拒绝截断或多余字节"""
    stream = io.BytesIO(bytes(payload))
    result = read_reload_field(stream)
    if stream.read(1):
        raise CoreMessageProtocolError('Type 162 has unexpected trailing bytes')
    return result
