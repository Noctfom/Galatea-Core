import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import sys

import torch

# 引入功能模块
import run_self_play
from trainer import PPOTrainer, resolve_training_device
from model_versus import ModelArena
from system_logger import setup_global_logger
from checkpoint_utils import load_training_checkpoint
from training_lock import TrainerAlreadyRunningError, TrainerProcessLock
from training_validation import resolve_training_target

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
#
# 6. mini_batch (默认 512) -> [训练批量/单次更新大小]
#    - 含义: 每次模型更新时使用的样本数量。
#    - 类比: AI 在每次学习时看的经验数量。
#    - 调整建议: 越大PPO环节计算越快，但占用显存更大。通常设为 512 或 1024。
#
# 7. workers (默认 4) -> [采集进程数/数据工人数量]针对 CPU 多线程采集
#    - 含义: 同时运行的环境采集进程数量。
#    - 类比: AI 有几个"实习生"在帮它收集对局经验。
#    - 调整建议: 越多采集越快，但CPU占用也越高。通常设为 4 或 8。
#          中央批量推理固定启用，进程过多时通信与 CPU 争抢会增大，建议按物理核心数调整。
#          注意:进程数应当小于等于 CPU 核心数，否则会报错。
#
# 8. device (默认 'auto') -> [训练主设备]
#    - auto: 有可用 CUDA 时使用 CUDA，否则自动使用 CPU。
#    - cpu: 中央批量推理和 PPO 更新都只使用 CPU。
#    - cuda: 中央批量推理和 PPO 更新使用 CUDA；不可用时启动前报错。
#    - 所有采集 Worker 固定使用 CPU，中央批量推理服务固定启用。
#
# 9. no_compile (默认 False) -> [禁用编译/兼容模式]
#    - 含义: 是否禁用 PyTorch 的 torch.compile 功能。
#    - 调整建议: 如果你遇到了与 torch.compile 相关的兼容性问题，可以启用这个选项来禁用模型编译。虽然会牺牲一部分性能，但能确保程序正常运行。
#          注意:windows用户请务必启用此选项。
# 10. use_onnx (默认 False) -> [ONNX 极速推理/算力剥离]
#    - 含义: 是否在保存点同步导出 ONNX，并供历史对手在 Worker 本地推理。
#    - 类比: AI 是否有一个专门的"引擎剥离器"，在后台将最新的模型转换成一个轻量级的推理引擎，供实习生们快速使用。
#    - 调整建议: 启用后可以显著提升采集速度，尤其是在 CPU 上，强烈建议开启。需要安装 onnxruntime 包


#  tensorboard --logdir=runs    查看训练过程

#  训练示例命令:
#  python main.py train --dir ./models --additional-iterations 1000 --model-prefix galatea --batch_size 16384 --mini_batch 256 --workers 6 --device auto --d_model 512 --n_heads 8 --n_layers 6 --no_compile --use_onnx

#  恢复训练命令示例:  从第 100 轮存档继续，目标是练到第 5000 轮
#  python main.py train --resume ./models/galatea_iter_100.pth --target-iteration 5000 --batch_size 16384 --mini_batch 256 --workers 6 --device auto --no_compile --use_onnx

#  测试示例命令(每隔 5 局保存一次心声):
#  python main.py duel --p0 ./models/galatea_iter_100.pth --thought_freq 5 --num 100
# 
#  回放示例命令:
#  python thought_viewer.py ./ai_thoughts/xxx.json    
#
#  更新示例命令:
#  python main.py update --core --data
#
#  语义化提取示例命令:
#  python main.py parse

# ==============================================================================


def run_training_command(args, parser):
    """在单 Trainer 互斥锁内完成训练配置解析、初始化和训练循环。"""
    try:
        training_lock = TrainerProcessLock().acquire()
    except TrainerAlreadyRunningError as error:
        parser.error(str(error))

    try:
        try:
            resolve_training_device(args.device)
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))

        if args.target_iteration is None and args.additional_iterations is None:
            if args.resume:
                parser.error(
                    "恢复训练必须指定 --target-iteration 或 --additional-iterations"
                )
            args.additional_iterations = 1000

        resume_checkpoint = None
        current_iteration = 0
        if args.resume:
            resume_checkpoint = load_training_checkpoint(
                args.resume,
                map_location="cpu",
            )
            current_iteration = int(resume_checkpoint['iteration'])
        resolved_target_iteration = resolve_training_target(
            current_iteration,
            target_iteration=args.target_iteration,
            additional_iterations=args.additional_iterations,
        )

        net_config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'vocab_size': 20000,
        }
        print(f"🚀 启动训练模式 (保存至 {args.dir})...")
        print(f"📂 读取卡组: {args.deck_dir}")
        print(f"⚙️ 模型架构: {net_config}")
        trainer = PPOTrainer(
            save_dir=args.dir,
            deck_dir=args.deck_dir,
            net_config=net_config,
            resume_path=args.resume,
            update_timesteps=args.batch_size,
            mini_batch_size=args.mini_batch,
            num_workers=args.workers,
            training_device=args.device,
            compile_model=not args.no_compile,
            worker_timeout=args.timeout,
            gamma=args.gamma,
            lr=args.lr,
            entropy=args.entropy,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            use_onnx=args.use_onnx,
            standard_core=args.standard_core,
            model_prefix=args.model_prefix,
            preloaded_resume_checkpoint=resume_checkpoint,
            protocol_audit=args.protocol_audit,
        )
        try:
            training_lock.set_run_id(trainer.run_id)
            trainer.run_training_loop(target_iteration=resolved_target_iteration)
        finally:
            trainer.close()
    finally:
        training_lock.release()


def main():
    # 修改：根据输入命令动态切换日志前缀
    cmd_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else "Main"
    prefix_mapping = {'train': 'Trainer', 'duel': 'Arena', 'play': 'SelfCheck', 'parse': 'Parser', 'update': 'Updater'}
    log_prefix = prefix_mapping.get(cmd_name, 'System')
    
    setup_global_logger(prefix=log_prefix)
    
    parser = argparse.ArgumentParser(description="Galatea AI 主控程序")
    subparsers = parser.add_subparsers(dest='command', help='可用指令')
    
    # --- 1. 训练模式 (Train) ---
    train_parser = subparsers.add_parser('train', help='开始强化学习训练')
    train_parser.add_argument('--dir', type=str, default='./models', help='模型保存路径')
    iteration_group = train_parser.add_mutually_exclusive_group()
    iteration_group.add_argument(
        '--target-iteration',
        type=int,
        default=None,
        help='训练停止时的绝对轮次',
    )
    iteration_group.add_argument(
        '--additional-iterations',
        type=int,
        default=None,
        help='从当前检查点开始追加的轮数',
    )
    train_parser.add_argument(
        '--model-prefix',
        type=str,
        default=None,
        help='新模型文件前缀，默认 galatea；恢复训练时自动读取且不得改写',
    )
    # [修正] 默认路径改为 ./decks
    train_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    # === 新增：模型架构参数 (就像 duel 那样) ===
    train_parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    train_parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    train_parser.add_argument("--n_layers", type=int, default=2, help="Transformer layers")
    # 训练参数
    train_parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    train_parser.add_argument('--batch_size', type=int, default=4096, help='采集总步数')
    train_parser.add_argument('--mini_batch', type=int, default=512, help='PPO 单次训练 Batch')
    train_parser.add_argument('--workers', type=int, default=4, help='CPU进程数')
    train_parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="训练主设备；Worker 始终使用 CPU (默认 auto)",
    )
    train_parser.add_argument('--timeout', type=int, default=300, help='Worker 数据采集的最高容忍时间(秒)')
    # 添加禁用编译的开关 (防止win/老旧环境报错)
    train_parser.add_argument('--no_compile', action='store_true', help='禁用 torch.compile (兼容性模式)')

    train_parser.add_argument('--use_onnx', action='store_true', help='在保存点同步导出 ONNX，并加速历史对手本地推理')
    train_parser.add_argument('--standard_core', action='store_true', help='使用自己编译的标准内核（无幽灵定界符）时请开启此项')
    train_parser.add_argument(
        '--protocol-audit',
        action='store_true',
        help='采集 Model Protocol V3 诊断报告（默认关闭）',
    )

    # RL 灵魂超参数
    train_parser.add_argument('--gamma', type=float, default=0.998, help='目光长远度 (推荐0.998)')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (大脑神经元重塑速度)')
    train_parser.add_argument('--entropy', type=float, default=0.03, help='探索欲/好奇心系数')
    train_parser.add_argument('--gae_lambda', type=float, default=0.95, help='经验平滑度')
    train_parser.add_argument('--clip_eps', type=float, default=0.2, help='单次顿悟的上限')

    # ==========================================
    
    # --- 2. 验证模式 (Play/Test) ---
    play_parser = subparsers.add_parser('play', help='运行自我博弈测试')
    play_parser.add_argument('-n', '--num', type=int, default=10, help='对局数量')
    play_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    play_parser.add_argument('--standard_core', action='store_true', help='使用自己编译的标准内核（无幽灵定界符）时请开启此项')


    # --- 3. 竞技场模式 (Duel) ---
    duel_parser = subparsers.add_parser('duel', help='模型竞技场')
    duel_parser.add_argument('--p0', type=str, default=None, help='P0 模型路径')
    duel_parser.add_argument('--p1', type=str, default=None, help='P1 模型路径')
    duel_parser.add_argument('-n', '--num', type=int, default=100, help='对战局数')
    duel_parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    duel_parser.add_argument('--deck_dir', type=str, default='./decks', help='YGOPro卡组文件夹路径')
    duel_parser.add_argument('--thought_freq', type=int, default=0, help='每隔几局保存一次AI心声 (0为不保存)')
    duel_parser.add_argument(
        '--arena-mode',
        choices=('normal', 'benchmark'),
        default='normal',
        help='normal=普通随机竞技；benchmark=固定赛程并交替先后手',
    )
    duel_parser.add_argument(
        '--p0-deck-source', '--p0_deck_source',
        default='weighted',
        help='P0 卡组来源：weighted、physical:<池>、virtual:<池> 或 deck:<池>/<卡组>',
    )
    duel_parser.add_argument(
        '--p1-deck-source', '--p1_deck_source',
        default='same_range',
        help='P1 卡组来源：same_range、same_deck，或与 P0 相同的四类独立来源',
    )
    duel_parser.add_argument(
        '--benchmark-seed',
        type=int,
        default=20260906,
        help='新建竞技场基准计划时使用的 uint32 随机种子',
    )
    duel_parser.add_argument(
        '--benchmark-name',
        default='baseline',
        help='新建竞技场基准计划的名称',
    )
    duel_parser.add_argument(
        '--benchmark-plan',
        default=None,
        help='复用既有竞技场基准计划 JSON；计划中的局数优先于 --num',
    )
    # 兼容既有命令行但不再参与竞技场构网；模型架构始终读取检查点内置配置。
    duel_parser.add_argument("--d_model", type=int, default=256, help=argparse.SUPPRESS)
    duel_parser.add_argument("--n_heads", type=int, default=4, help=argparse.SUPPRESS)
    duel_parser.add_argument("--n_layers", type=int, default=2, help=argparse.SUPPRESS)
    duel_parser.add_argument('--standard_core', action='store_true', help='使用自己编译的标准内核（无幽灵定界符）时请开启此项')
    duel_parser.add_argument(
        '--protocol-audit',
        action='store_true',
        help='采集 Model Protocol V3 诊断报告（默认关闭）',
    )
    # --- 4. 语义化提取模式 (Parse) ---
    parse_parser = subparsers.add_parser('parse', help='提取并更新卡片Lua脚本语义知识库')
    parse_parser.add_argument('--script_dir', type=str, default='./script', help='Lua脚本所在目录')
    parse_parser.add_argument('--output', type=str, default='knowledge_base.json', help='输出的知识库文件路径')
    parse_parser.add_argument('--clear', action='store_true', help='清空本地知识库、Hash 映射和代码语义向量后重新解析')
    
    parse_parser.add_argument('--sync', action='store_true', help='从主仓库拉取完整语义资产组作为基座')
    parse_parser.add_argument('--remote_url', type=str, 
                              default='https://raw.githubusercontent.com/Noctfom/Galatea-Core/main/knowledge_base.json', 
                              help='指定其他的 Github Raw URL')
    parse_parser.add_argument('--embed', action='store_true', help='为新增效果槽接续生成代码语义向量，必要时全量重建')

    # --- 5. 更新同步模式 (Update) ---
    update_parser = subparsers.add_parser('update', help='更新本地代码、卡片数据库(CDB)与脚本库')
    update_parser.add_argument('--core', action='store_true', help='仅更新 Galatea 核心代码 (从你的Github拉取)')
    update_parser.add_argument('--data', action='store_true', help='仅更新 cards.cdb 与 script 脚本库 (从萌卡与官方拉取)')
    update_parser.add_argument('--repo', type=str, default='default', help='指定脚本的来源仓库地址 (默认官方)')
    update_parser.add_argument('--force', action='store_true', help='覆盖更新：清空本地旧脚本，完全以远程为准')

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
        run_training_command(args, parser)
        
    elif args.command == 'play':
        print(f"⚔️ 启动规则系统自检测压测 (Self-Check)...")
        import platform
        from duel_launcher import DuelManager
        
        # 自动探测不同系统的引擎核心库后缀
        dll_name = "ocgcore.dll" if platform.system() == "Windows" else "ocgcore.so"
        core_path = os.path.abspath(os.path.join(".", dll_name))
        
        if not os.path.exists(core_path):
            print(f"❌ 致命错误: 找不到核心动态库 {core_path}。")
            sys.exit(1)
            
        manager = DuelManager(
            core_path,
            args.deck_dir,
            standard_core=args.standard_core,
        )
        manager.run_tournament(args.num)
        
    elif args.command == 'duel':
        print(f"🏟️ 启动竞技场模式...")
        # 竞技场控制参数与模型架构分离，模型结构由各自检查点决定。
        config = {
            'thought_freq': args.thought_freq,
        }

        arena = ModelArena(
            model_p0_path=args.p0, 
            model_p1_path=args.p1, 
            device=args.device,
            deck_dir=args.deck_dir,
            config=config,
            standard_core=args.standard_core,
            protocol_audit=args.protocol_audit,
            p0_deck_source=args.p0_deck_source,
            p1_deck_source=args.p1_deck_source,
            arena_mode=args.arena_mode,
            benchmark_seed=args.benchmark_seed,
            benchmark_name=args.benchmark_name,
            benchmark_plan=args.benchmark_plan,
        )
        arena.run_tournament(n_games=args.num)
        
    elif args.command == 'parse':
        print("🧠 启动语义知识库构建模块...")
        from lua_parser import YGOProLuaParser
        parser = YGOProLuaParser(script_dir=args.script_dir)
        
        # 逻辑判定：如果开启了 --sync，就使用默认的 remote_url，否则传入 None
        actual_remote_url = args.remote_url if args.sync else None
        
        parser.run_batch(output_file=args.output, clear_existing=args.clear, remote_url=actual_remote_url)
        if args.embed or args.sync:
            print("🧬 启动代码语义向量接续检查...")
            from code_embedder import CodeSemanticEmbedder
            embedder = CodeSemanticEmbedder()
            embedding_path = os.path.join(
                os.path.dirname(os.path.abspath(args.output)),
                'code_embeddings.npy',
            )
            embedder.generate_embeddings(
                kb_file=args.output,
                output_file=embedding_path,
                incremental=not args.clear,
            )
        
    elif args.command == 'update':
        print("🌐 启动自动同步更新模块...")
        import update_tools
        
        # 如果什么都没输入 (既没有 --core 也没有 --data)，提示用户
        if not args.core and not args.data:
            print("⚠️ 请指定更新目标！使用 '--core' 更新代码，或使用 '--data' 更新卡库和脚本。")
            print("💡 示例: python main.py update --core --data")
        else:
            if args.core:
                update_tools.update_core_code()
                
            if args.data:
                update_tools.update_data_and_scripts(repo_type=args.repo, force=args.force)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
