# 本文件定义中心推理的 ZMQ 请求协议，并校验共享内存结果是否属于当前请求。


class InferenceProtocolError(RuntimeError):
    """表示中心推理响应或共享结果与当前请求不匹配"""


def encode_inference_request(worker_id, request_id):
    """将 Worker 编号和请求号编码为服务端可解析的字节串"""
    return f"{int(worker_id)}:{int(request_id)}".encode("ascii")


def decode_inference_request(payload):
    """解析中心推理请求并拒绝格式错误或负数编号"""
    try:
        worker_text, request_text = payload.decode("ascii").split(":", 1)
        worker_id = int(worker_text)
        request_id = int(request_text)
    except (AttributeError, UnicodeDecodeError, ValueError) as exc:
        raise InferenceProtocolError("invalid inference request payload") from exc

    if worker_id < 0 or request_id < 0:
        raise InferenceProtocolError("inference request ids must be non-negative")
    return worker_id, request_id


def encode_inference_success(request_id):
    """编码带请求号的成功响应"""
    return f"OK:{int(request_id)}".encode("ascii")


def encode_inference_error(request_id, reason):
    """编码服务端拒绝请求时的错误响应"""
    return f"ERR:{int(request_id)}:{reason}".encode("ascii", errors="replace")


def validate_inference_success(reply, request_id):
    """确认 ZMQ 响应确实对应当前请求"""
    expected = encode_inference_success(request_id)
    if reply != expected:
        raise InferenceProtocolError(
            f"unexpected inference reply: expected {expected!r}, got {reply!r}"
        )


def request_shared_inference_result(
    socket,
    worker_id,
    request_id,
    response_id_slot,
    result_slot,
):
    """请求中心推理，并仅在响应号和共享槽完成号一致时复制结果"""
    if response_id_slot is None:
        raise InferenceProtocolError("missing shared inference response id slot")

    socket.send(encode_inference_request(worker_id, request_id))
    reply = socket.recv()
    validate_inference_success(reply, request_id)

    completed_request_id = int(response_id_slot.item())
    if completed_request_id != request_id:
        raise InferenceProtocolError(
            "shared inference result id mismatch: "
            f"expected {request_id}, got {completed_request_id}"
        )

    # 返回独立快照，避免后续请求覆盖共享槽时影响当前决策
    return result_slot.detach().clone()
