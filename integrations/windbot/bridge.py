"""Loopback YGOPro adapter for the bundled public Core, not a new game engine.

    Original Python implementation. Wire layouts are checked against the pinned
    IceYGO client; no YGOSharp host source is copied into this module.
"""
import selectors
import secrets
import socket
import struct
import time

from integrations.duel_session import SessionProtocolError, finite_timeout, response_bytes
from integrations.player_view import (
    DECISIONS, PUBLIC_MESSAGES, encode_player_start, encode_zone_refresh, visible_core_message, public_query_updates,
)



def selection_label_bytes(response, prompt_type, payload):
    """WindBot Type15 label equivalence, not a rewrite of its Core response.

    The public Core consumes SELECT_CARD as a set of physical pool indices.
    Validate first, then compare only the used count/indices. Candidate and
    expert use the same key, including when RuleBot preference order differs.
    Never merge different copies by code or apply this to ordered selections.
    """
    raw = response_bytes(response)
    if prompt_type != 15:
        return raw
    from selection_protocol import CANCEL_RESPONSE, validate_selection_response
    payload = bytes(payload)
    if (len(payload) < 5 or len(payload) != 5 + 8 * payload[4]
            or payload[0] not in (0, 1)
            or payload[2] > payload[3] or payload[2] > payload[4]):
        raise SessionProtocolError('invalid Type15 label prompt')
    validate_selection_response(15, payload, raw)
    if raw[:4] == CANCEL_RESPONSE:
        return CANCEL_RESPONSE
    return bytes((raw[0],)) + bytes(sorted(raw[1:1 + raw[0]]))


def prompt_actor(message):
    from core_message_protocol import decision_player_from_message
    return decision_player_from_message(message[0], message[1:])


def frame(packet):
    if not isinstance(packet, bytes) or not 1 <= len(packet) <= 65535:
        raise SessionProtocolError('invalid YGOPro packet size')
    return struct.pack('<H', len(packet)) + packet


def visible_message(message, perspective):
    """Reuse Core's unchanged conservative BO1 visibility profile."""
    return visible_core_message(message, perspective)


def query_updates(env, perspective):
    """复用Core定义的原有派生查询刷新，封包字节和隐藏策略保持一致"""
    yield from public_query_updates(env, perspective)


class YGOProPeer:
    """Bounded loopback endpoint. The owner alone advances the native Core."""

    def __init__(self, seat, timeout=30):
        if type(seat) is not int or seat not in (0, 1):
            raise SessionProtocolError('invalid peer seat')
        self.seat = seat
        self.timeout = finite_timeout(timeout)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.connection = None
        self._read_selector = None
        self.port = self.listener.getsockname()[1]
        self.responses = 0
        self.room_token = secrets.token_hex(8)

    def send(self, kind, payload=b''):
        self.connection.settimeout(self.timeout)
        self.connection.sendall(frame(bytes((kind,)) + payload))

    def game_message(self, payload):
        self.send(1, payload)

    def receive(self, deadline):
        def read(size):
            result = bytearray()
            while len(result) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('external client response deadline exceeded')
                self.connection.settimeout(remaining)
                block = self.connection.recv(size - len(result))
                if not block:
                    raise ConnectionError('external client disconnected')
                result.extend(block)
            return bytes(result)
        length = struct.unpack('<H', read(2))[0]
        if not 1 <= length <= 65535:
            raise SessionProtocolError('invalid incoming YGOPro frame length')
        return read(length)

    def accept(self, expected_deck):
        self.listener.settimeout(self.timeout)
        self.connection, _ = self.listener.accept()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        deadline = time.monotonic() + self.timeout
        joined = deck_seen = ready = named = False
        for _ in range(32):
            packet = self.receive(deadline)
            kind, data = packet[0], packet[1:]
            if kind == 0x17:  # EXTERNAL_ADDRESS, not trusted as player identity.
                continue
            if kind == 0x10 and not named and len(data) == 40:
                named = True
            elif kind == 0x12 and named and not joined and len(data) == 48:
                if struct.unpack_from('<H', data)[0] != 0x1362:
                    raise SessionProtocolError('YGOPro client version mismatch')
                if data[8:48].decode('utf-16le').rstrip('\0') != self.room_token:
                    raise SessionProtocolError('client does not belong to this controlled session')
                joined = True
                self.send(0x12, struct.pack('<IBBBBB3xIBBH', 0, 0, 0, 5, 1, 1, 8000, 5, 1, 0))
                self.send(0x13, bytes((self.seat,)))
                for actor, name in ((0, 'P0'), (1, 'P1')):
                    self.send(0x20, name.encode('utf-16le').ljust(40, b'\0') + bytes((actor,)))
            elif kind == 2 and joined and not deck_seen and len(data) >= 8:
                main, side = struct.unpack_from('<ii', data)
                if not 0 <= main <= 512 or not 0 <= side <= 256 or len(data) != 8 + 4 * (main + side):
                    raise SessionProtocolError('invalid external deck update')
                cards = list(struct.unpack_from('<' + 'I' * (main + side), data, 8))
                if (sorted(cards[:main]) != sorted(expected_deck['main'] + expected_deck['extra'])
                        or sorted(cards[main:]) != sorted(expected_deck['side'])):
                    raise SessionProtocolError('client submitted a different deck than the controlled session')
                deck_seen = True
            elif kind == 0x22 and joined and not ready and not data:
                ready = True
            elif kind in (0x15, 0x16):
                continue
            else:
                raise SessionProtocolError(f'unexpected or repeated lobby packet {kind}')
            if joined and deck_seen and ready:
                return
        raise SessionProtocolError('too many lobby packets')

    def start(self, decks):
        self.send(0x15)
        self.game_message(encode_player_start(self.seat, decks))

    def check_idle(self):
        if self.connection is None:
            return
        if self._read_selector is None:
            # Linux使用epoll等后端，避免共享张量使socket编号超过select的1023上限
            reader = selectors.DefaultSelector()
            try:
                reader.register(self.connection, selectors.EVENT_READ)
            except BaseException:
                reader.close()
                raise
            self._read_selector = reader
        if self._read_selector.select(0):
            for _ in range(32):
                packet = self.receive(time.monotonic() + self.timeout)
                if packet[0] not in (0x15, 0x16):
                    raise SessionProtocolError('unsolicited external packet while no response is pending')
                if not self._read_selector.select(0):
                    return
            raise SessionProtocolError('external idle packet flood')

    def respond(self, prompt, env):
        # 保留每条查询和询问的独立帧及原顺序，只合并一次socket写入
        packets = [frame(b'\x01' + update) for update in query_updates(env, self.seat)]
        packets.append(frame(b'\x01' + visible_message(prompt, self.seat)))
        self.connection.settimeout(self.timeout)
        self.connection.sendall(b''.join(packets))
        deadline = time.monotonic() + self.timeout
        for _ in range(32):
            packet = self.receive(deadline)
            if packet[0] == 1 and 1 <= len(packet[1:]) <= 512:
                self.responses += 1
                return packet[1:]
            if packet[0] not in (0x15, 0x16):
                raise SessionProtocolError(f'unexpected packet {packet[0]} while waiting for response')
        raise SessionProtocolError('external response packet flood')

    def close(self):
        # 选择器每个连接只创建一次，正常与异常结束均释放其底层句柄
        try:
            if self._read_selector is not None:
                self._read_selector.close()
        finally:
            self._read_selector = None
            try:
                if self.connection is not None:
                    self.connection.close()
            finally:
                self.connection = None
                self.listener.close()
