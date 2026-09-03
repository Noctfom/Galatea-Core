'''全息回放辅助工具：统一录像帧、时间轴游标、箭头和协议语义展示。'''

from collections import Counter


def get_replay_frames(replay_data):
    """优先读取 V2 帧列表，并兼容只含 decisions 的早期录像。"""
    frames = replay_data.get("frames")
    if isinstance(frames, list) and frames:
        return frames
    decisions = replay_data.get("decisions", [])
    return decisions if isinstance(decisions, list) else []


def get_replay_frame_state(replay_data, frame):
    """解析 V2 状态编号，并兼容旧帧内直接保存的完整状态。"""
    direct_state = frame.get("state")
    if isinstance(direct_state, dict):
        return direct_state
    state_id = frame.get("state_id")
    states = replay_data.get("states", [])
    if isinstance(state_id, int) and 0 <= state_id < len(states):
        state = states[state_id]
        return state if isinstance(state, dict) else {}
    return {}


def get_replay_decklists(replay_data):
    """读取录像内一次性保存的双方初始卡组，并过滤损坏字段。"""
    raw_decklists = replay_data.get("decklists", {})
    if not isinstance(raw_decklists, dict):
        return {}
    result = {}
    for player in ("0", "1"):
        raw_entry = raw_decklists.get(player)
        if not isinstance(raw_entry, dict):
            continue
        entry = {
            "name": str(raw_entry.get("name", "") or ""),
            "main": [],
            "extra": [],
        }
        for section in ("main", "extra"):
            values = raw_entry.get(section, [])
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    code = int(value) & 0x7FFFFFFF
                except (TypeError, ValueError):
                    continue
                if code > 0:
                    entry[section].append(code)
        result[player] = entry
    return result


def group_replay_card_codes(codes):
    """按首次出现顺序合并同名卡，供完整卡组视图显示数量。"""
    normalized = []
    for value in codes or []:
        try:
            code = int(value) & 0x7FFFFFFF
        except (TypeError, ValueError):
            continue
        if code > 0:
            normalized.append(code)
    counts = Counter(normalized)
    return [(code, counts[code]) for code in dict.fromkeys(normalized)]


def set_replay_cursor(session_state, step, max_steps):
    """同步按钮游标与 Streamlit 滑块状态。"""
    bounded_step = max(0, min(int(max_steps), int(step)))
    session_state["replay_step"] = bounded_step
    session_state["step_slider_widget"] = bounded_step
    return bounded_step


def queue_replay_cursor(session_state, step, max_steps):
    """安排下次重绘的回放游标，避免渲染后再次改写滑块控件状态。"""
    bounded_step = max(0, min(int(max_steps), int(step)))
    session_state["replay_step"] = bounded_step
    return bounded_step


def sync_replay_session(session_state, selected_file, max_steps):
    """切换录像时归零时间轴，并在同一录像内安全限制游标。"""
    if session_state.get("replay_loaded_file") != selected_file:
        session_state["replay_loaded_file"] = selected_file
        session_state["is_playing"] = False
        return set_replay_cursor(session_state, 0, max_steps)
    session_state.setdefault("is_playing", False)
    return set_replay_cursor(
        session_state,
        session_state.get("replay_step", 0),
        max_steps,
    )


def get_selected_replay_option_index(selection_state, option_count):
    """从 Streamlit 表格选择状态中提取安全的单行候选索引。"""
    if not selection_state:
        return None
    if isinstance(selection_state, dict):
        selection = selection_state.get("selection", {})
    else:
        selection = getattr(selection_state, "selection", {})
    if isinstance(selection, dict):
        rows = selection.get("rows", [])
    else:
        rows = getattr(selection, "rows", [])
    if not rows:
        return None
    index = int(rows[0])
    if 0 <= index < int(option_count):
        return index
    return None


def get_frame_visuals(frame, preview_option_index=None):
    """提取当前帧或点选候选的发起者、目标、多重移动和有向箭头。"""
    event = frame.get("event") or {}
    options = frame.get("options", [])
    chosen = next(
        (option for option in options if option.get("is_chosen")),
        {},
    )
    preview = chosen
    if (
        isinstance(preview_option_index, int)
        and 0 <= preview_option_index < len(options)
    ):
        preview = options[preview_option_index]
    actor = event.get("actor") or preview.get("actor")
    targets = list(event.get("targets") or preview.get("targets") or [])
    primary_target = event.get("target") or preview.get("target")
    if primary_target and primary_target not in targets:
        targets.insert(0, primary_target)

    arrows = []
    movements = list(event.get("movements") or [])
    for movement in movements:
        source = movement.get("from")
        target = movement.get("to")
        if source and target:
            arrows.append({"from": source, "to": target, "kind": "move"})
    if not arrows and actor:
        for target in targets:
            if target and target != actor:
                arrows.append({
                    "from": actor,
                    "to": target,
                    "kind": event.get("kind", "action"),
                })
    return {
        "event": event,
        "chosen": chosen,
        "preview": preview,
        "actor": actor,
        "targets": targets,
        "arrows": arrows,
    }


def format_action_semantics(semantic):
    """把动作协议 V2 的关键字段整理为紧凑的人类可读标签。"""
    if not isinstance(semantic, dict) or not semantic:
        return []
    details = []
    operation = semantic.get("operation")
    if operation:
        details.append(f"语义: {operation}")
    summon_method = semantic.get("summon_method")
    if summon_method:
        details.append(f"召唤方式: {summon_method}")
    card_name = semantic.get("card_name")
    if card_name:
        details.append(f"目标卡: {card_name}")
    selection_min = int(semantic.get("selection_min", 0) or 0)
    selection_max = int(semantic.get("selection_max", 0) or 0)
    selection_count = int(semantic.get("selection_count", 0) or 0)
    if selection_min or selection_max or selection_count:
        details.append(f"选择: {selection_count}（要求 {selection_min}~{selection_max}）")
    if semantic.get("finishable"):
        details.append("允许完成")
    if semantic.get("cancelable"):
        details.append("允许取消")
    target_codes = semantic.get("target_codes") or []
    if target_codes:
        details.append(f"结果集合: {', '.join(str(code) for code in target_codes)}")
    target_values = semantic.get("target_values") or []
    if target_values:
        details.append(f"素材值: {', '.join(str(value) for value in target_values)}")
    prompt_flags = int(semantic.get("prompt_flags", 0) or 0)
    prompt_value = int(semantic.get("prompt_value", 0) or 0)
    prompt_value2 = int(semantic.get("prompt_value2", 0) or 0)
    if prompt_flags or prompt_value or prompt_value2:
        details.append(
            f"提示: flags=0x{prompt_flags:X}, value={prompt_value}, value2={prompt_value2}"
        )
    return details
