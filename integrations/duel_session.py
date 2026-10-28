"""Core-owned external-session contract; independent of model and transport versions."""
from dataclasses import dataclass
import math
import secrets
import uuid

SESSION_CONTRACT_VERSION = 1
MAX_RESPONSE_BYTES = 512
WINDBOT_SELECTION_LABEL_V1 = 'windbot_select_card_set_v1'


def resolve_core_seed(seed, games=1):
    """Randomize a new schedule by default, or retain an explicit uint32 seed."""
    if type(games) is not int or not 1 <= games <= 2**32:
        raise ValueError('invalid Core seed schedule length')
    if seed is None:
        return secrets.randbelow(2**32 - games + 1)
    if type(seed) is not int or not 0 <= seed <= 2**32 - games:
        raise ValueError('Core seed schedule must stay within uint32')
    return seed


class SessionProtocolError(ValueError):
    """Reject mismatched, stale or unsolicited submissions without advancing Core."""


def response_bytes(response):
    """Preserve response order and significant zero bytes; never deserialize objects."""
    if type(response) is int and -(2**31) <= response < 2**32:
        return (response & 0xFFFFFFFF).to_bytes(4, 'little')
    if isinstance(response, (bytes, bytearray)) and 0 < len(response) <= MAX_RESPONSE_BYTES:
        return bytes(response)
    raise SessionProtocolError('response must be an int32/uint32 or 1..512 bytes')


def validate_response_label_adapters(adapters, participants):
    """Validate explicit per-seat label contracts, never infer them from wire bytes.

    These identifiers describe label comparison only, not a transport/model
    protocol migration. An adapter is imported only when explicitly requested.
    Missing/unknown contracts must fail closed, not revert to guessed matching.
    """
    if (not isinstance(adapters, dict) or not set(adapters) <= {'0', '1'}
            or not isinstance(participants, dict)):
        raise SessionProtocolError('invalid response label adapters/participants')
    for actor, name in adapters.items():
        participant = participants.get(actor)
        if (name != WINDBOT_SELECTION_LABEL_V1 or not isinstance(participant, dict)
                or participant.get('kind') != 'windbot'):
            raise SessionProtocolError('unsupported response label adapter or wrong participant kind')
    return dict(adapters)


def response_label_bytes(adapter, response, prompt_type, payload):
    """Return comparison-only bytes; callers still submit/archive the raw response.

    Core owns this closed dispatch contract so future Link ingress can reuse it.
    No module names/callables are loaded from capture content. Ordinary Core,
    RuleBot and PPO paths do not opt into an external adapter.
    """
    if adapter != WINDBOT_SELECTION_LABEL_V1:
        raise SessionProtocolError('unsupported response label adapter')
    from integrations.windbot.bridge import selection_label_bytes
    return selection_label_bytes(response, prompt_type, payload)


def finite_timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
        raise SessionProtocolError('timeout must be finite and in (0, 3600] seconds')
    return float(value)


@dataclass(frozen=True)
class DecisionRequest:
    session_id: str
    request_id: int
    actor: int
    prompt_type: int


@dataclass(frozen=True)
class ResponseSubmission:
    session_id: str
    request_id: int
    actor: int
    response: object


class SessionGate:
    """One outstanding request per authoritative duel, not one per network socket.

    Transports must submit actual responses against this token. Legacy YGOPro has
    no token on the wire: its adapter must serialize requests and close the duel
    on timeout/disconnect, never reconnect and reuse a pending response.
    """

    def __init__(self, session_id=None):
        self.session_id = str(uuid.UUID(session_id)) if session_id is not None else str(uuid.uuid4())
        self.request_id = 0
        self.pending = None
        self.closed = False

    def request(self, actor, prompt_type):
        if self.closed or self.pending is not None:
            raise SessionProtocolError('session closed or an outstanding request already exists')
        if type(actor) is not int or actor not in (0, 1) or type(prompt_type) is not int or not 0 < prompt_type < 256:
            raise SessionProtocolError('invalid physical actor or prompt type')
        self.request_id += 1
        self.pending = DecisionRequest(self.session_id, self.request_id, actor, prompt_type)
        return self.pending

    def submit(self, submission):
        if (self.closed or self.pending is None or not isinstance(submission, ResponseSubmission)
                or type(submission.request_id) is not int or type(submission.actor) is not int
                or (submission.session_id, submission.request_id, submission.actor)
                != (self.pending.session_id, self.pending.request_id, self.pending.actor)):
            raise SessionProtocolError('unsolicited, stale, cross-session or wrong-seat response')
        raw = response_bytes(submission.response)
        self.pending = None
        return raw

    def close(self):
        self.pending = None
        self.closed = True
