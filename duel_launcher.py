# duel_manager.py
import time
from collections import defaultdict
from rich.console import Console
from rich.table import Table

# 引入模块
from galatea_env import GalateaEnv
import deck_utils
import run_self_play # 引入刚才改好的 worker

class DuelManager:
    def __init__(self, core_dir, deck_dir, standard_core=False):
        self.deck_dir = deck_dir
        self.standard_core = standard_core

        import gamestate
        if standard_core:
            gamestate.CORE_HAS_GHOST_BYTE = False
            print("🔧 [DuelManager] 协议自适应：已关闭幽灵定界符 (Standard Core Mode)")
        else:
            gamestate.CORE_HAS_GHOST_BYTE = True

        print(f"🔧 初始化核心环境: {core_dir}")
        # 如果是相对路径，转换为绝对路径防止 cdb 加载错位
        import os
        self.env = GalateaEnv(os.path.abspath(core_dir))

    def run_tournament(self, n_games):
        """运行 N 场决斗并统计"""
        stats = defaultdict(lambda: {
            'matches': 0, 'wins': 0, 
            'first': 0, 'first_wins': 0, 
            'second': 0, 'second_wins': 0
        })

        print(f"🚀 开始 {n_games} 场自动对战...\n")
        start_t = time.time()

        for i in range(1, n_games + 1):
            # 1. 选卡组 (使用增强版 deck_utils)
            deck_pair = deck_utils.get_random_deck_pair(self.deck_dir)
            if deck_pair is None:
                print("⚠️ 无法加载卡组，跳过")
                continue
            env_name, name1, d1, name2, d2 = deck_pair

            print(f"\n{'='*20} 第 {i} / {n_games} 局 {'='*20}")
            print(f"⚔️ 对阵: 【{name1}】(先手) VS 【{name2}】(后手)")

            # 2. 调用 run_self_play 执行单局
            # 这里传入名字 purely for display inside the function (if needed)
            winner, turns, err = run_self_play.run_single_game(self.env, d1, d2, name1, name2)

            # 3. 统计结果
            stats[name1]['matches'] += 1
            stats[name1]['first'] += 1
            stats[name2]['matches'] += 1
            stats[name2]['second'] += 1

            if winner == 0: # P0 (name1) 胜
                stats[name1]['wins'] += 1
                stats[name1]['first_wins'] += 1
            elif winner == 1: # P1 (name2) 胜
                stats[name2]['wins'] += 1
                stats[name2]['second_wins'] += 1
            else:
                print(f"❌ 异常结束: {err}")

        # 4. 打印报表
        self._print_report(stats, time.time() - start_t)

    def _print_report(self, stats, duration):
        console = Console()
        table = Table(title=f"📊 决斗统计报告 (耗时 {duration:.1f}s)")
        
        table.add_column("卡组名", style="cyan")
        table.add_column("场次")
        table.add_column("总胜率", style="green")
        table.add_column("先手胜率", style="yellow")
        table.add_column("后手胜率", style="blue")

        # 按胜率排序
        for name, d in sorted(stats.items(), key=lambda x: x[1]['wins']/x[1]['matches'] if x[1]['matches'] else 0, reverse=True):
            total = d['matches']
            if total == 0: continue
            
            win_rate = d['wins'] / total
            f_rate = d['first_wins'] / d['first'] if d['first'] else 0
            s_rate = d['second_wins'] / d['second'] if d['second'] else 0
            
            table.add_row(
                name, str(total),
                f"{win_rate:.1%}",
                f"{f_rate:.1%} ({d['first_wins']}/{d['first']})",
                f"{s_rate:.1%} ({d['second_wins']}/{d['second']})"
            )
        
        console.print(table)
