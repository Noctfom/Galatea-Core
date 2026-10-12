"""本文件按随包公开 Core 的 RPN 规则筛选合法卡名宣言，不扩大模型输入协议"""

from dataclasses import dataclass

from core_message_protocol import CoreMessageProtocolError


OP_ADD, OP_SUB, OP_MUL, OP_DIV = range(0x40000000, 0x40000004)
OP_AND, OP_OR, OP_NEG, OP_NOT = range(0x40000004, 0x40000008)
OP_ISCODE, OP_ISSETCARD, OP_ISTYPE, OP_ISRACE, OP_ISATTRIBUTE = range(
    0x40000100, 0x40000105
)
_BINARY_OPS = {OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_AND, OP_OR}
_UNARY_OPS = {OP_NEG, OP_NOT, OP_ISCODE, OP_ISSETCARD, OP_ISTYPE,
              OP_ISRACE, OP_ISATTRIBUTE}
TYPE_TOKEN = 0x4000


@dataclass(frozen=True)
class AnnouncementCard:
    """保存宣言校验所需的真实卡库字段，不使用缺卡时的零属性兜底"""

    code: int
    alias: int
    type: int
    race: int
    attribute: int
    setcodes: tuple


def _signed32(value):
    """把 RPN 字面量和算术结果还原为 Core 的有符号32位栈值。"""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


class CardAnnouncementPredicate:
    """一次校验表达式栈布局，再对候选逐张执行同一合法性条件"""

    def __init__(self, opcodes):
        """拒绝空、截断或栈不平衡表达式；单个1表示恒真而非卡密白名单"""
        self.opcodes = tuple(opcodes)
        depth = 0
        for op in self.opcodes:
            if not isinstance(op, int) or not 0 <= op <= 0xFFFFFFFF:
                raise CoreMessageProtocolError('Type 142 opcode must be uint32')
            required = 2 if op in _BINARY_OPS else 1 if op in _UNARY_OPS else 0
            if depth < required:
                raise CoreMessageProtocolError('Type 142 RPN stack underflow')
            depth += 1 - required
        if depth != 1:
            raise CoreMessageProtocolError('Type 142 RPN must leave exactly one value')

    def literal_card_codes(self):
        """仅提取紧接 ISCODE 的字面卡密，避免把属性掩码当成卡片"""
        return tuple(
            self.opcodes[index - 1]
            for index, op in enumerate(self.opcodes)
            if op == OP_ISCODE and index > 0
            and self.opcodes[index - 1] not in _BINARY_OPS | _UNARY_OPS
        )

    def matches(self, card):
        """按 Core 执行算术、布尔和位交集，并排除不存在、异画及衍生物"""
        if card is None or card.alias or card.type & TYPE_TOKEN:
            return False
        stack = []
        for op in self.opcodes:
            if op in _BINARY_OPS:
                rhs, lhs = stack.pop(), stack.pop()
                if op == OP_ADD:
                    value = lhs + rhs
                elif op == OP_SUB:
                    value = lhs - rhs
                elif op == OP_MUL:
                    value = lhs * rhs
                elif op == OP_DIV:
                    # C++ 整数除法向零截断，Python // 对负数会向下取整
                    quotient = abs(lhs) // abs(rhs) if rhs else 0
                    value = -quotient if (lhs < 0) != (rhs < 0) else quotient
                elif op == OP_AND:
                    value = int(bool(lhs) and bool(rhs))
                else:
                    value = int(bool(lhs) or bool(rhs))
                stack.append(_signed32(value))
            elif op in _UNARY_OPS:
                operand = stack.pop()
                if op == OP_NEG:
                    value = -operand
                elif op == OP_NOT:
                    value = int(not operand)
                elif op == OP_ISCODE:
                    value = int(card.code == (operand & 0xFFFFFFFF))
                elif op == OP_ISSETCARD:
                    value = int(any(
                        code and (code & 0xFFF) == (operand & 0xFFF)
                        and (code & operand & 0xF000) == (operand & 0xF000)
                        for code in card.setcodes
                    ))
                elif op == OP_ISTYPE:
                    value = card.type & operand
                elif op == OP_ISRACE:
                    value = card.race & operand
                else:
                    value = card.attribute & operand
                stack.append(_signed32(value))
            else:
                # Core 未定义的数值也是字面量，包括负数的 uint32 编码
                stack.append(_signed32(op))
        return bool(stack[0])
