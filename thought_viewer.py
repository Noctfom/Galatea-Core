#录像回放器

import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def view_thoughts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    console.clear()
    console.print(Panel(f"🧠 AI 决策思维回放\n模型: [cyan]{data['model_name']}[/] | 赢家: P{data['winner']}", border_style="magenta"))
    
    for step_idx, decision in enumerate(data["decisions"]):
        # 打印当前状态栏
        state = decision.get("state", {})
        p0_lp = state.get('p0_lp', 8000)
        p1_lp = state.get('p1_lp', 8000)
        
        console.print(f"\n[bold yellow]👉 决策点 {step_idx + 1} | Turn {decision['turn']} - {decision['phase']}[/]")
        
        # 战况播报版
        status_text = (
            f"👑 [cyan]AI (P0)[/]  LP: [red]{p0_lp}[/] | 手牌: {state.get('p0_hand', 0)} | 怪兽: {state.get('p0_mzone', 0)}  🆚  "
            f"🤖 [magenta]对手 (P1)[/] LP: [red]{p1_lp}[/] | 手牌: {state.get('p1_hand', 0)} | 怪兽: {state.get('p1_mzone', 0)}"
        )
        console.print(Panel(status_text, border_style="blue"))
        
        table = Table(box=None, header_style="bold green")
        table.add_column("最终决定", justify="center", width=10)
        table.add_column("AI 信心 (概率)", justify="right", width=20)
        table.add_column("动作描述", style="white")
        
        for opt in decision["options"]:
            conf_str = f"{opt['confidence'] * 100:.2f}%"
            
            if opt["is_chosen"]:
                mark = "[bold green]✅ 选取[/]"
                conf_str = f"[bold green]{conf_str}[/]"
                desc = f"[bold green]{opt['desc']}[/]"
            else:
                mark = ""
                desc = opt["desc"]
                if opt['confidence'] < 0.05: 
                    conf_str = f"[dim]{conf_str}[/]"
                    desc = f"[dim]{desc}[/]"
            
            table.add_row(mark, conf_str, desc)
            
        console.print(table)
        input("按 [Enter] 查看下一步 AI 的思考...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSON 心声文件路径")
    args = parser.parse_args()
    view_thoughts(args.file)