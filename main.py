import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import sys

import torch

# 引入功能模块
import run_self_play
from trainer import PPOTrainer
from model_versus import ModelArena

# [必须] Windows多进程入口保护
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ==============================================================================
# [Galatea 核心架构参数备忘录]
# 这些参数定义了 AI 的"脑容量"和"思考方式"，在训练开始前决定，且无法中途更改。
# ==============================================================================
#
# 1. d_model (默认 256) -> [思维维度/特征丰富度]
#    - 含义: 将一张卡片转化为向量时，这个向量的长度。
#    - 类比: 类似于一张卡的"详细属性栏"。
#      - d_model=64: AI 只能记住"这是怪兽，攻击力3000"。
#      - d_model=256: AI 能记住"这是龙族、光属性、配合青眼白龙、能检索..."。
#    - 调整建议: 
#      - 越高越聪明，但计算量成倍增加。
#      - 必须能被 n_heads 整除 (例如 256/4=64 OK, 256/5=51.2 报错)。
#
# 2. n_heads (默认 4) -> [注意力头数/多线程视角]
#    - 含义: Transformer 同时关注不同特征子空间的能力。
#    - 类比: AI 做决策时有几只"眼睛"在看场面。
#      - Head 1: 盯着"攻击力数值"。
#      - Head 2: 盯着"卡片种族配合"。
#      - Head 3: 盯着"对手后场盖牌"。
#      - Head 4: 盯着"墓地资源"。
#    - 调整建议: 通常设为 4 或 8。头越多，处理复杂局面的关系网能力越强。
#
# 3. n_layers (默认 2) -> [思考深度/推理步数]
#    - 含义: Transformer Encoder 堆叠的层数。
#    - 类比: AI 在出牌前对自己进行"预判的预判"的次数。
#      - Layer 1: 直觉反应 (这张亮了，点它)。
#      - Layer 2: 简单连招 (先发A检索B，再发B)。
#      - Layer 6+: 深度博弈 。
#    - 调整建议: 
#      - 2层适合快速实验和简单卡组。
#      - 4-6层适合主流竞技卡组。
#      - 层数太深会导致训练极慢，且容易难以收敛(梯度消失)。
#
# 4. vocab_size (默认 20000) -> [识字量/卡池大小]
#    - 含义: Embedding 层的词表大小。
#    - 类比: AI 认识多少张不同的游戏王卡。
#    - 调整建议: 只要比实际出现的卡片ID总数大即可。游戏王目前约1.2万张卡，设2万足够。
#
# 5. batch_size (默认 4096) -> [采集批量/经验池大小]
#    - 含义: 一次采集的总步数。
#    - 类比: AI 在一次训练中收集的经验数量。
#    - 调整建议: 越大越稳定，但需要更多内存。通常设为 4096 或 8192。
# 6. mini_batch (默认 512) -> [训练批量/GPU更新大小]
#    - 含义: 每次模型更新时使用的样本数量。
#    - 类比: AI 在每次学习时看的经验数量。
#    - 调整建议: 越大更新越快，但可能不稳定。通常设为 512 或 1024。
# 7. workers (默认 4) -> [采集进程数/数据工人数量]针对 CPU 多线程采集
#    - 含义: 同时运行的环境采集进程数量。
#    - 类比: AI 有几个"实习生"在帮它收集对局经验。
#    - 调整建议: 越多采集越快，但CPU占用也越高。通常设为 4 或 8。
# 8. worker_device (默认 'cuda') -> [采集设备/实习生工作站]
#    - 含义: 采集进程中模型推理使用的设备。
#    - 类比: 实习生是用高性能GPU还是普通CPU在帮忙。
#    - 调整建议: 如果有多块GPU，可以设为 'cuda' 来加速采集；如果资源有限或兼容性问题，可以设为 'cpu'。
# 9. async_infer (默认 False) -> [异步推断/独立推理服务器]
#    - 含义: 是否启用独立的推断服务器来处理模型推理请求。
#    - 类比: AI 是否有一个专门的"顾问"在负责分析局面，实习生们只负责收集数据。
#    - 调整建议: 启用后可以大幅节省采集进程的显存占用，并且在GPU环境下能显著提升采集速度。建议有条件的用户启用。

#  tensorboard --logdir=runs    查看训练过程

#  训练示例命令:
#  python main.py train --dir ./models --batch_size 32768 --mini_batch 1024 --workers 12 --steps 1000 --d_model 512 --n_heads 8 --n_layers 6 --async_infer --no_compile

#  （例）从第 100 轮存档继续，目标是练到第 5000 轮
#  python main.py train --resume ./models/galatea_iter_100.pth --batch_size 32768 --mini_batch 1024 --workers 12 --steps 5000 --async_infer --no_compile

#  测试示例命令:
#  python main.py duel --p0 ./models/galatea_iter_100.pth --num 100       

# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Galatea AI 主控程序")
    
    subparsers = parser.add_subparsers(dest='command', help='可用指令')
    
    # --- 1. 训练模式 (Train) ---
    train_parser = subparsers.add_parser('train', help='开始强化学习训练')
    train_parser.add_argument('--dir', type=str, default='./models', help='模型保存路径')
    train_parser.add_argument('--steps', type=int, default=1000, help='训练总迭代轮数')
    # [修正] 默认路径改为 ./decks
    train_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    # === 新增：模型架构参数 (就像 duel 那样) ===
    train_parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    train_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    train_parser.add_argument("--n_layers", type=int, default=2, help="Transformer layers")
    # 训练参数
    train_parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    train_parser.add_argument('--batch_size', type=int, default=4096, help='采集总步数')
    train_parser.add_argument('--mini_batch', type=int, default=512, help='GPU训练Batch')
    train_parser.add_argument('--workers', type=int, default=4, help='CPU进程数')
    train_parser.add_argument('--worker_device', type=str, default='cuda', choices=['cpu', 'cuda'], help="Worker 推理使用的设备 (cpu 或 cuda)")
    # [新增] 异步推断开关
    train_parser.add_argument('--async_infer', action='store_true', help="启用异步推断服务器(大幅节省显存并提速)")
    # [新增] 添加禁用编译的开关 (防止win/老旧环境报错)
    train_parser.add_argument('--no_compile', action='store_true', help='禁用 torch.compile (兼容性模式)')

    # ==========================================
    
    # --- 2. 验证模式 (Play/Test) ---
    play_parser = subparsers.add_parser('play', help='运行自我博弈测试')
    play_parser.add_argument('-n', '--num', type=int, default=10, help='对局数量')
    play_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    
    # --- 3. 竞技场模式 (Duel) ---
    duel_parser = subparsers.add_parser('duel', help='模型竞技场')
    duel_parser.add_argument('--p0', type=str, default=None, help='P0 模型路径')
    duel_parser.add_argument('--p1', type=str, default=None, help='P1 模型路径')
    duel_parser.add_argument('-n', '--num', type=int, default=100, help='对战局数')
    duel_parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    duel_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    duel_parser.add_argument('--thought_freq', type=int, default=0, help='每隔几局保存一次AI心声 (0为不保存)')
    # === 新增模型参数 ===
    duel_parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    duel_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    duel_parser.add_argument("--n_layers", type=int, default=2, help="Transformer layers")

    args = parser.parse_args()

    # --- 检查卡组路径 ---
    if hasattr(args, 'deck_dir'):
        if not os.path.exists(args.deck_dir):
            try:
                os.makedirs(args.deck_dir)
                print(f"⚠️ 警告: 卡组目录 '{args.deck_dir}' 不存在，已自动创建。")
                print(f"👉 请务必将 .ydk 卡组文件放入该文件夹！")
            except:
                print(f"❌ 错误: 无法访问卡组目录 '{args.deck_dir}'")
                # 不强制退出，因为可能 deck_utils 内部有处理
                
    # --- 调度逻辑 ---
    if args.command == 'train':
        # 1. 组装配置字典
        net_config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'vocab_size': 20000
        }
        print(f"🚀 启动训练模式 (保存至 {args.dir})...")
        print(f"📂 读取卡组: {args.deck_dir}")
        print(f"⚙️ 模型架构: {net_config}")
        # [修改] 传入 resume 参数
        trainer = PPOTrainer(
            save_dir=args.dir, 
            deck_dir=args.deck_dir, 
            net_config=net_config,
            resume_path=args.resume,  # <--- 关键：把命令行参数传进去
            update_timesteps=args.batch_size,  # [新增] 传参
            mini_batch_size=args.mini_batch,
            num_workers=args.workers,
            worker_device=args.worker_device,
            async_infer=args.async_infer,
            compile_model=not args.no_compile # [新增] 如果用户输入 --no_compile，这里就是 False
        )
        trainer.run_training_loop(max_iterations=args.steps)
        
    elif args.command == 'play':
        print(f"⚔️ 启动测试模式...")
        run_self_play.main(total_games=args.num, deck_dir=args.deck_dir)
        
    elif args.command == 'duel':
        print(f"🏟️ 启动竞技场模式...")
        # 组装配置
        config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'vocab_size': 20000, # 这个通常不变，不需要传参
            'thought_freq': args.thought_freq  # [新增] 把参数传给竞技场
        }

        arena = ModelArena(
            model_p0_path=args.p0, 
            model_p1_path=args.p1, 
            device=args.device,
            deck_dir=args.deck_dir,
            config=config
        )
        arena.run_tournament(n_games=args.num)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()