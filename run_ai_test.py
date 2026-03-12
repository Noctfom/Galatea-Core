# run_ai_test.py
import deck_utils
from galatea_env import GalateaEnv
from run_self_play import run_single_game # 复用你的运行逻辑
from ai_bot import AiBot
from rule_bot import RuleBot # 假设你保留了原来的 RuleBot 类以便对比

def main():
    # 1. 初始化 AI
    print("🧠 正在初始化 AI Bot...")
    ai = AiBot() # 随机权重
    
    # 2. 准备卡组
    env = GalateaEnv()
    d1_name, d1, d2_name, d2 = deck_utils.get_random_deck_pair()
    
    if not d1: return

    print(f"⚔️ 实验对局: AI (P0, {d1_name}) vs AI (P1, {d2_name})")
    
    # 3. 运行对局
    # 我们这里用两个随机 AI 互殴，看看能不能跑通流程
    # 这里的 run_single_game 需要稍微改一下接口来接受 AiBot 对象
    # 但我们可以利用 Python 的鸭子类型：
    # AiBot 和 RuleBot 都有 .get_decision() 吗？
    # RuleBot 是静态方法，AiBot 是实例方法。
    
    # 为了兼容，我们写一个 wrapper
    class BotWrapper:
        def __init__(self, bot_instance):
            self.bot = bot_instance
        def get_decision(self, msgs): # 兼容旧接口签名，虽然我们现在用 snapshot
            # run_self_play 里需要改成传入 snapshot
            # 但为了不改动 run_self_play 太多，我们建议直接去修改 run_self_play
            pass

    # --- 更加直接的方法 ---
    # 我们直接修改 run_self_play.py，让它支持 AiBot
    # 请手动去 run_self_play.py 里，把:
    # resp = rule_bot.get_rule_decision(...)
    # 改成:
    # if player == 0: resp = ai_bot_0.get_decision(snapshot)
    # else: resp = rule_bot.get_rule_decision(...)
    
    # 但为了演示，我这里写一个极简的 loop
    raw_data = env.reset(d1, d2)
    from gamestate import MessageParser, DuelState
    
    msg_queue = MessageParser.parse(raw_data)
    brain = DuelState()
    
    step = 0
    while step < 1000:
        if not msg_queue:
            raw_data = env.step()
            if not raw_data: break
            msg_queue = MessageParser.parse(raw_data)
            continue
            
        msg = msg_queue.pop(0)
        brain.update(msg[0], msg[1:])
        
        # 如果需要决策
        if msg[0] in [11, 16, 15]:
            # 获取快照
            snap = brain.get_snapshot()
            
            # AI 决策
            # 注意: 这里不管是 P0 还是 P1 都用 AI 决策
            # 实际根据 msg[1] (Player ID) 来决定用哪个脑子
            resp = ai.get_decision(snap)
            
            if resp:
                print(f"🤖 AI 执行操作: {resp.hex()}")
                env.send_action(resp)
                msg_queue = []
        
        if msg[0] == 5:
            print("🏆 决斗结束")
            break
            
        step += 1

if __name__ == "__main__":
    main()