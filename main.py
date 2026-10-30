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
from semantic_assets import DEFAULT_SEMANTIC_REPOSITORY_URL
from training_lock import TrainerAlreadyRunningError, TrainerProcessLock
from training_validation import resolve_training_target
from ppo_control import DEFAULT_GAE_LAMBDA, DEFAULT_PPO_EPOCHS, DEFAULT_TARGET_KL
from update_tools import CARD_VOCAB_URL, MOCKA_CDB_URL

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
# 4. vocab_size (固定 20000) -> [精确卡片词表容量]
#    - 含义: V4 使用 card_vocab.json 将真实卡密唯一映射到 Embedding 行。
#    - 当前容量 20000 包含协议保留项与未来追加空间，不再使用取模 Hash。
#    - 此值属于模型协议，训练时不可手动修改。
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
#  语义资产同步 / 本地接续示例命令:
#  python main.py parse --sync
#  python main.py parse --local-update

# ==============================================================================


def run_training_command(args, parser):
    """在单 Trainer 互斥锁内完成训练配置解析、初始化和训练循环。"""
    try:
        training_lock = TrainerProcessLock().acquire()
    except TrainerAlreadyRunningError as error:
        parser.error(str(error))

    try:
        if not getattr(args, 'windbot', False) and (
                any(getattr(args, key, None) for key in ('windbot_executor', 'windbot_holdout', 'windbot_runtime', 'windbot_mono'))
                or getattr(args, 'windbot_rate', 0.2) != 0.2
                or getattr(args, 'windbot_deck_mode', 'mirror') != 'mirror'
                or getattr(args, 'windbot_response_timeout', 10) != 10):
            parser.error('WindBot对手参数需要同时显式指定 --windbot')
        try:
            resolve_training_device(args.device)
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))

        if args.target_iteration is None and args.additional_iterations is None:
            if args.resume or getattr(args, 'init_from', None):
                parser.error(
                    "恢复训练必须指定 --target-iteration 或 --additional-iterations"
                )
            args.additional_iterations = 1000

        resume_checkpoint = None
        current_iteration = 0
        source_path = args.resume or getattr(args, 'init_from', None)
        if source_path:
            resume_checkpoint = load_training_checkpoint(
                source_path,
                map_location="cpu",
            )
            from checkpoint_utils import validate_ppo_resume_stage
            if args.resume:
                validate_ppo_resume_stage(resume_checkpoint)
                current_iteration = int(resume_checkpoint['iteration'])
            else:
                from checkpoint_utils import get_training_lineage
                current_iteration = get_training_lineage(resume_checkpoint)['ppo_iteration']
        resolved_target_iteration = resolve_training_target(
            current_iteration,
            target_iteration=args.target_iteration,
            additional_iterations=args.additional_iterations,
        )
        if args.dir is None:
            args.dir = os.path.dirname(os.path.abspath(args.resume)) if args.resume else './models'
        training_resume_path = args.resume
        if getattr(args, 'init_from', None):
            from training_validation import validate_training_config
            validate_training_config(resume_checkpoint['net_config'], update_timesteps=args.batch_size,
                mini_batch_size=args.mini_batch, num_workers=args.workers, training_device=args.device,
                worker_timeout=args.timeout, gamma=args.gamma, learning_rate=args.lr, entropy=args.entropy,
                gae_lambda=args.gae_lambda, clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs, target_kl=args.target_kl)
            from checkpoint_utils import initialize_ppo_from_weights
            training_resume_path, resume_checkpoint = initialize_ppo_from_weights(args.init_from, args.dir,
                learning_rate=args.lr, model_prefix=args.model_prefix, checkpoint=resume_checkpoint)
            args.dir = os.path.dirname(training_resume_path)
            print(f'🧬 PPO权重初始化分支: {training_resume_path}；PPO轮次从{current_iteration}继续，优化器/更新步计数重建。')

        net_config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'vocab_size': 20000,
        }
        if resume_checkpoint is not None:
            net_config = dict(resume_checkpoint['net_config'])
        print(f"🚀 启动训练模式 (保存至 {args.dir})...")
        print(f"📂 读取卡组: {args.deck_dir}")
        print(f"⚙️ 模型架构: {net_config}")
        trainer = PPOTrainer(
            save_dir=args.dir,
            deck_dir=args.deck_dir,
            net_config=net_config,
            resume_path=training_resume_path,
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
            ppo_epochs=args.ppo_epochs,
            target_kl=args.target_kl,
            use_onnx=args.use_onnx,
            model_prefix=args.model_prefix,
            preloaded_resume_checkpoint=resume_checkpoint,
            protocol_audit=args.protocol_audit,
            **({'windbot_training': {
                'executors': args.windbot_executor, 'holdouts': args.windbot_holdout,
                'rate': args.windbot_rate, 'deck_mode': args.windbot_deck_mode,
                'runtime': args.windbot_runtime, 'mono': args.windbot_mono,
                'response_timeout': args.windbot_response_timeout,
            }} if getattr(args, 'windbot', False) else {}),
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
    prefix_mapping = {
        'train': 'Trainer',
        'duel': 'Arena',
        'play': 'SelfCheck',
        'parse': 'Parser',
        'update': 'Updater',
        'vocab': 'Vocabulary',
        'replay-ingest': 'ReplayIngest',
        'dataset-build': 'DatasetQuality',
        'expert-read': 'ExpertRead',
        'imitate': 'Imitation',
        'windbot': 'ExternalDuel',
    }
    log_prefix = prefix_mapping.get(cmd_name, 'System')
    
    if cmd_name not in {'trajectory-check', 'semantic-check'} and '--inspect' not in sys.argv:
        setup_global_logger(prefix=log_prefix)
    
    parser = argparse.ArgumentParser(description="Galatea AI 主控程序")
    subparsers = parser.add_subparsers(dest='command', help='可用指令')
    
    # --- 1. 训练模式 (Train) ---
    train_parser = subparsers.add_parser('train', help='开始强化学习训练')
    train_parser.add_argument('--dir', type=str, default=None, help='默认新训models；恢复沿用检查点目录；权重初始化在此目录下建立独立ppo_branches批次')
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
    ppo_source = train_parser.add_mutually_exclusive_group()
    ppo_source.add_argument('--resume', type=str, default=None, help='完整恢复PPO检查点与PPO优化器')
    ppo_source.add_argument('--init-from', type=str, default=None, help='从兼容V4/BC权重建立新PPO分支；重建PPO优化器，保留累计PPO/BC双计数')
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
    train_parser.add_argument(
        '--protocol-audit',
        action='store_true',
        help='采集 Model Protocol V3 诊断报告（默认关闭）',
    )

    # RL 灵魂超参数
    train_parser.add_argument('--gamma', type=float, default=0.998, help='目光长远度 (推荐0.998)')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (大脑神经元重塑速度)')
    train_parser.add_argument('--entropy', type=float, default=0.03, help='探索欲/好奇心系数')
    train_parser.add_argument('--gae_lambda', type=float, default=DEFAULT_GAE_LAMBDA, help='GAE Lambda，V4 默认 0.98，可对照 0.95/0.99')
    train_parser.add_argument('--clip_eps', type=float, default=0.2, help='单次顿悟的上限')
    train_parser.add_argument('--ppo-epochs', type=int, default=DEFAULT_PPO_EPOCHS, choices=range(1, 5), help='每轮 PPO 更新次数上限（1～4）')
    train_parser.add_argument('--target-kl', type=float, default=DEFAULT_TARGET_KL, help='目标 KL；超过 1.5 倍目标停止本轮共享更新，0 关闭')
    train_parser.add_argument('--windbot', action='store_true', help='显式启用WindBot主训练对手，关闭时不加载可选模块')
    train_parser.add_argument('--windbot-executor', action='append', default=None, help='可重复指定训练执行器；省略则自动按层级留出并选择兼容构筑')
    train_parser.add_argument('--windbot-holdout', action='append', default=None, help='可重复指定保留作独立评测的执行器，不进入WindBot训练抽样')
    train_parser.add_argument('--windbot-rate', type=float, default=0.2, help='WindBot Worker分配概率，范围(0,1]；不是保证的样本比例')
    train_parser.add_argument('--windbot-deck-mode', choices=['mirror', 'training-pool'], default='mirror', help='模型侧使用镜像或常规训练卡组池，WindBot始终固定构筑')
    train_parser.add_argument('--windbot-runtime', default=None, help='可选WindBot.exe路径，省略使用集成目录runtime')
    train_parser.add_argument('--windbot-mono', default=None, help='Linux可选Mono路径')
    train_parser.add_argument('--windbot-response-timeout', type=float, default=10, help='单次机器人响应预算秒数，范围(0,10]')

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
    duel_parser.add_argument('--record-trajectory', action='store_true', help='仅录像局额外保存原始 Core 规范轨迹')
    duel_parser.add_argument('--record-evaluations', action='store_true', help='仅录像局逐事件/决策快照保存模型侧状态预测，不影响训练或动作选择，会增加录像推理开销')
    duel_parser.add_argument('--trajectory-root', default='./replays/core_trajectories', help='规范轨迹保存目录')
    duel_parser.add_argument(
        '--policy-mode', '--policy_mode',
        choices=('greedy', 'training', 'deployment'),
        default='greedy',
        help='竞技场动作策略：greedy=贪心；training=训练同分布；deployment=可调温度采样',
    )
    duel_parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='deployment 模式采样温度，范围 0.05～5.0；training 固定为 1.0',
    )
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
    
    parse_parser.add_argument('--sync', action='store_true', help='仅从远程仓库同步完整语义资产组，不解析本地 Lua')
    parse_parser.add_argument('--remote_url', type=str,
                              default=DEFAULT_SEMANTIC_REPOSITORY_URL,
                              help='语义资产仓库 URL，也兼容旧式 knowledge_base.json Raw URL')
    parse_parser.add_argument('--local-update', action='store_true', help='解析本地 Lua 并自动接续结构语义与代码向量')
    parse_parser.add_argument('--embed', action='store_true', help=argparse.SUPPRESS)
    semantic_check_parser = subparsers.add_parser('semantic-check', help='只读检查代码语义来源、生成清单和向量重复率')
    semantic_check_parser.add_argument('--directory', default='.', help='语义资产所在目录')

    # --- 5. 更新同步模式 (Update) ---
    update_parser = subparsers.add_parser('update', help='更新本地代码、卡片数据库(CDB)与脚本库')
    update_parser.add_argument('--core', action='store_true', help='仅更新 Galatea 核心代码 (从你的Github拉取)')
    update_parser.add_argument('--data', action='store_true', help='仅更新 cards.cdb 与 script 脚本库 (从萌卡与官方拉取)')
    update_parser.add_argument('--repo', type=str, default='default', help='指定脚本的来源仓库地址 (默认官方)')
    update_parser.add_argument(
        '--cdb-url',
        type=str,
        default=MOCKA_CDB_URL,
        help='指定 cards.cdb 文件源 URL',
    )
    update_parser.add_argument(
        '--card-vocab-url',
        type=str,
        default=CARD_VOCAB_URL,
        help='指定词表仓库或 card_vocab.json 文件源 URL',
    )
    update_parser.add_argument('--force', action='store_true', help='覆盖更新：清空本地旧脚本，完全以远程为准')

    # --- 6. 本地精确卡片词表更新模式 (Vocabulary) ---
    vocab_parser = subparsers.add_parser('vocab', help='从本地 CDB 只追加更新精确卡片词表')
    vocab_parser.add_argument('--cdb', type=str, default='cards.cdb', help='本地 cards.cdb 或自制卡数据库路径')
    vocab_parser.add_argument('--output', type=str, default='card_vocab.json', help='目标词表路径')

    trajectory_parser = subparsers.add_parser('trajectory-check', help='只读检查规范轨迹格式/摘要，或在相同公开 Core 下重放验证')
    trajectory_parser.add_argument('path', help='*.core.jsonl.gz 文件')
    trajectory_parser.add_argument('--replay', action='store_true', help='额外验证 Core 原始输出、响应映射和观测摘要')

    ingest_parser = subparsers.add_parser('replay-ingest', help='检查或离线转换 YRP/Link 捕获；不启动学习')
    ingest_parser.add_argument('path', help='YRP/YRP3D 或 *.link.jsonl.gz 文件')
    ingest_parser.add_argument('--source', choices=('yrp', 'link'), required=True, help='明确指定来源，不按内容猜测')
    ingest_parser.add_argument('--inspect', action='store_true', help='只检查文件格式，不调用原生 Core')
    ingest_parser.add_argument('--output', default='./replays/imported_trajectories', help='诊断规范轨迹输出目录')
    ingest_parser.add_argument('--report', default=None, help='可选的新建报告 JSON 路径，不覆盖已有文件')
    ingest_parser.add_argument('--p0-player-id', default=None, help='YRP 来源命名空间内 P0 稳定匿名 ID，不是昵称')
    ingest_parser.add_argument('--p1-player-id', default=None, help='YRP 来源命名空间内 P1 稳定匿名 ID，不是昵称')
    ingest_parser.add_argument('--timeout', type=float, default=300, help='转换时间预算秒数，1～3600')

    dataset_parser = subparsers.add_parser('dataset-build', help='严格审查规范轨迹并按整局/构筑/玩家隔离划分')
    dataset_parser.add_argument('--directory', default='./replays/core_trajectories', help='递归扫描规范轨迹目录')
    dataset_parser.add_argument('--output', required=True, help='新建数据集清单 JSON，不覆盖已有文件')
    dataset_parser.add_argument('--seed', type=int, default=20261009, help='划分 uint32 种子')
    dataset_parser.add_argument('--validation-fraction', type=float, default=0.1, help='独立连通组验证抽样比例')
    dataset_parser.add_argument('--test-fraction', type=float, default=0.1, help='独立连通组测试抽样比例')

    expert_parser = subparsers.add_parser('expert-read', help='完整重放后只读专家 CPU 样本，不启动学习')
    expert_parser.add_argument('path', help='完整且可严格重放的 *.core.jsonl.gz')
    expert_parser.add_argument('--expert-seat', action='append', type=int, choices=(0, 1), default=None,
                               help='明确指定专家物理座位，可重复；不填时仅选来源声明的 WindBot')
    expert_parser.add_argument('--preview-rows', type=int, default=5, help='预览标签条数，0～20，不导出观测张量')
    expert_parser.add_argument('--expected-sha256', default=None, help='可选的清单文件摘要；不匹配时在重放前拒绝')
    expert_parser.add_argument('--report', default=None, help='新建只读样本审查报告 JSON，不覆盖现有文件')

    external_parser = subparsers.add_parser('windbot', help='可选受控 BO1 对局/完整采集，不加入 PPO 或联盟')
    external_parser.add_argument('--inspect', action='store_true', help='仅检查源码目录和运行依赖，不启动 Bot/Core')
    external_parser.add_argument('--executor', default='Blue-Eyes', help='上游执行器名称，--inspect 可列出')
    external_parser.add_argument('--opponent-executor', default=None, help='指定后运行两个 WindBot；默认另一侧为原生 RuleBot/模型')
    external_parser.add_argument('--runtime', default=None, help='显式 WindBot.exe 路径；默认 integrations/windbot/runtime/WindBot.exe')
    external_parser.add_argument('--mono', default=None, help='Linux Mono 可执行名称或路径；不自动安装')
    external_parser.add_argument('--windbot-seat', type=int, choices=(0, 1), default=1, help='物理座位；当前受控局 P0 固定先手')
    external_parser.add_argument('--model', default=None, help='可选原生 V4 PTH；未提供时另一侧为 RuleBot')
    external_parser.add_argument('--native-deck', default=None, help='原生侧 YDK；默认与指定 WindBot 构筑相同')
    external_parser.add_argument('--native-deck-source', default=None, help='原生侧使用竞技场选池：weighted、physical:<池>、virtual:<池>、deck:<池>/<卡组>；与 --native-deck 互斥')
    external_parser.add_argument('--deck_dir', default='./decks', help='原生侧卡组目录，不改变 WindBot 执行器的固定构筑')
    external_parser.add_argument('--seed', type=int, default=None, help='默认随机；指定后使用可复现 Core uint32 开局序列，不控制 WindBot 自身随机数')
    external_parser.add_argument('--games', type=int, default=1, help='受控局数，1～100；逐局隔离并记录不同种子')
    external_parser.add_argument('--timeout', type=float, default=600, help='单局机器会话时间预算秒数，最大3600')
    external_parser.add_argument('--response-timeout', type=float, default=30, help='单次握手/响应预算秒数')
    external_parser.add_argument('--max-decisions', type=int, default=2000, help='真实决策数上限；空连锁自动响应不计入')
    external_parser.add_argument('--capture-root', default='./replays/link_captures', help='完整原始采集输出目录')
    external_parser.add_argument('--verify', action='store_true', help='真实终局后独立转换和重放；不启动模仿学习')
    external_parser.add_argument('--record-replay', action='store_true', help='额外保存双方决策和 Core 事件的全息录像；原始捕获始终保留')
    external_parser.add_argument('--replay-root', default='./ai_thoughts', help='可选全息录像输出目录')
    external_parser.add_argument('--record-evaluations', action='store_true', help='全息录像补充原生模型侧逐状态预测；WindBot 不产生模型概率')
    external_parser.add_argument('--policy-mode', choices=('greedy', 'training', 'deployment'), default='training',
        help='仅控制原生 V4 模型：greedy取最高Logit，training固定T=1，deployment使用指定温度；不控制WindBot')
    external_parser.add_argument('--temperature', type=float, default=0.8,
        help='仅deployment生效，范围0.05～5.0；training固定1.0，greedy不使用温度')

    imitation_parser = subparsers.add_parser('imitate', help='独立模仿工作流；不加入 PPO 或真人在线学习')
    imitation_commands = imitation_parser.add_subparsers(dest='imitation_command', required=True)
    collect_parser = imitation_commands.add_parser('collect', help='独立 WindBot 镜像采集，直接生成专家缓存')
    collect_parser.add_argument('--train-executor', action='append', required=True, help='训练教师执行器，可重复')
    collect_parser.add_argument('--validation-executor', action='append', default=[], help='验证执行器，不进入学习')
    collect_parser.add_argument('--test-executor', action='append', default=[], help='最终测试执行器，不参与选模')
    collect_parser.add_argument('--games-per-executor', type=int, default=10)
    collect_parser.add_argument('--families', default=None, help='可选 JSON：执行器到家族标签的显式映射，不按名字猜测')
    collect_parser.add_argument('--split-mode', choices=('none', 'episode', 'deck', 'executor'), default=None,
        help='默认按同一教师的不同完整对局划分；另选执行器时为executor')
    collect_parser.add_argument('--split-seed', type=int, default=73, help='仅控制数据划分，不控制Core开局')
    collect_parser.add_argument('--validation-fraction', type=float, default=0.1)
    collect_parser.add_argument('--test-fraction', type=float, default=0.1)
    prepare_parser = imitation_commands.add_parser('prepare', help='一次准备完整旧轨迹/Link捕获，训练轮次不重放')
    prepare_parser.add_argument('source', help='完整规范轨迹、完整 Link 捕获或其目录；普通缺证据 YRP 不放行')
    prepare_parser.add_argument('--split-mode', choices=('none', 'episode', 'deck', 'executor'), default='episode')
    prepare_parser.add_argument('--expert-seat', action='append', type=int, choices=(0, 1), default=None)
    prepare_parser.add_argument('--validation-executor', action='append', default=[])
    prepare_parser.add_argument('--test-executor', action='append', default=[])
    prepare_parser.add_argument('--validation-fraction', type=float, default=0.1)
    prepare_parser.add_argument('--test-fraction', type=float, default=0.1)
    bc_parser = imitation_commands.add_parser('train', help='独立行为克隆：epoch预算、专用优化器和实验检查点')
    bc_parser.add_argument('--dataset', required=True, help='独立准备的 dataset.json')
    bc_source = bc_parser.add_mutually_exclusive_group(required=True)
    bc_source.add_argument('--init-from', help='兼容 V4 权重；生成新模型 UUID，不恢复来源优化器')
    bc_source.add_argument('--resume', help='恢复本工作流的 BC 检查点，不接受 PPO 优化器')
    bc_parser.add_argument('--epochs', type=int, default=5, help='本次追加的示范遍历轮次，不是 PPO iteration')
    bc_parser.add_argument('--batch-size', type=int, default=64)
    bc_parser.add_argument('--lr', type=float, default=1e-5)
    bc_parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    bc_parser.add_argument('--model-prefix', default=None, help='默认继承来源主前缀，文件使用<前缀>_iter_<PPO轮次>_bc_<累计BC轮次>')
    bc_parser.add_argument('--save-every', type=int, default=1)
    bc_parser.add_argument('--cpu-threads', type=int, default=1)
    eval_parser = imitation_commands.add_parser('evaluate', help='候选/基线固定对手与换边计划，不更新权重')
    eval_parser.add_argument('--dataset', required=True)
    eval_parser.add_argument('--checkpoint', required=True)
    eval_parser.add_argument('--baseline', default=None)
    eval_parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    eval_parser.add_argument('--executor', action='append', default=[], help='显式覆盖评测对手，属于回归测试而非自动未见声明')
    eval_parser.add_argument('--games-per-seat', type=int, default=5)
    eval_parser.add_argument('--policy-mode', choices=('greedy', 'training', 'deployment'), default='training')
    eval_parser.add_argument('--temperature', type=float, default=0.8)
    eval_parser.add_argument('--record-replay', action='store_true')
    eval_parser.add_argument('--record-evaluations', action=argparse.BooleanOptionalAction, default=None,
        help='录像内记录模型侧逐状态AI预测曲线；开启录像时默认启用，--no-record-evaluations可关闭')
    for child in (collect_parser, prepare_parser, bc_parser, eval_parser):
        child.add_argument('--output', required=child is not bc_parser,
                           default='./models/imitation' if child is bc_parser else None)
        child.add_argument('--seed', type=int, default=None if child in (collect_parser, eval_parser) else 73,
            help='采集/评测默认随机Core开局；准备/训练默认为73，分别用于划分/优化随机性')
    for child in (collect_parser, eval_parser):
        child.add_argument('--runtime', default=None)
        child.add_argument('--mono', default=None)
        child.add_argument('--timeout', type=float, default=600)
    collect_parser.add_argument('--max-decisions', type=int, default=4000)

    args = parser.parse_args()
    if args.command == 'duel' and (args.record_trajectory or args.record_evaluations) and args.thought_freq <= 0:
        parser.error('轨迹/预测记录需要 --thought_freq 大于 0；仅选定的录像局进行额外记录')



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
        )
        manager.run_tournament(args.num)
        
    elif args.command == 'duel':
        print(f"🏟️ 启动竞技场模式...")
        # 竞技场控制参数与模型架构分离，模型结构由各自检查点决定。
        config = {
            'thought_freq': args.thought_freq,
            'record_trajectory': args.record_trajectory,
            'record_evaluations': args.record_evaluations,
            'trajectory_root': args.trajectory_root,
        }

        arena = ModelArena(
            model_p0_path=args.p0, 
            model_p1_path=args.p1, 
            device=args.device,
            deck_dir=args.deck_dir,
            config=config,
            protocol_audit=args.protocol_audit,
            p0_deck_source=args.p0_deck_source,
            p1_deck_source=args.p1_deck_source,
            arena_mode=args.arena_mode,
            benchmark_seed=args.benchmark_seed,
            benchmark_name=args.benchmark_name,
            benchmark_plan=args.benchmark_plan,
            policy_mode=args.policy_mode,
            temperature=args.temperature,
        )
        arena.run_tournament(n_games=args.num)
        
    elif args.command == 'semantic-check':
        from semantic_assets import audit_code_semantic_quality
        import json
        print(json.dumps(audit_code_semantic_quality(args.directory), ensure_ascii=False, indent=2))

    elif args.command == 'parse':
        print("🧠 启动语义资产管理模块...")
        from semantic_assets import (
            clear_local_semantic_assets,
            invalidate_static_semantic_assets,
            synchronize_remote_semantic_bundle,
            audit_code_semantic_quality,
        )
        from semantic_lookup import ensure_static_semantic_assets

        output_path = os.path.abspath(args.output)
        output_directory = os.path.dirname(output_path)
        local_update = bool(args.local_update or args.embed)
        semantic_ready = False
        if args.clear:
            removed = clear_local_semantic_assets(
                output_directory,
                knowledge_base_filename=os.path.basename(output_path),
            )
            print(f"🧨 已物理清除 {len(removed)} 个本地语义资产。")

        if args.sync:
            print(f"🌐 正在仅同步远程语义资产: {args.remote_url}")
            sync_result = synchronize_remote_semantic_bundle(
                args.remote_url,
                output_directory,
                knowledge_base_filename=os.path.basename(output_path),
            )
            if sync_result["installed_code_semantics"]:
                print("✅ 远程结构语义、Hash 映射及代码向量已同步完成。")
                semantic_ready = True
            else:
                print("⚠️ 远程代码向量不完整，已仅安装结构语义；可启用 --local-update 重建向量。")

        if local_update:
            print('⚠️ 代码语义内容更新会改变语义资产身份；已有模型须保留原资产，新资产用于新训练。')
            print("🔍 启动本地 Lua 语义接续更新...")
            from lua_parser import YGOProLuaParser
            semantic_parser = YGOProLuaParser(script_dir=args.script_dir)
            semantic_parser.run_batch(
                output_file=args.output,
                clear_existing=False,
                remote_url=None,
            )
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
                incremental=True,
            )
            invalidate_static_semantic_assets(output_directory)
            semantic_ready = True
        elif not args.sync and not args.clear:
            print("⚠️ 未选择任何操作；请使用 --sync 和/或 --local-update。")
        if semantic_ready:
            lookup = ensure_static_semantic_assets(output_directory, knowledge_base_filename=os.path.basename(output_path))
            print(
                "📦 静态语义运行资产已生成: "
                f"{lookup.card_vocabulary.card_count} 张卡 / "
                f"{len(lookup.runtime_effect_bindings)} 个运行时效果绑定"
            )
            quality = audit_code_semantic_quality(output_directory, knowledge_base_filename=os.path.basename(output_path))
            print(f"🧪 代码语义质量：{quality['distinct_vectors']} 个不同向量 / {quality['effect_slots']} 槽")
            for warning in quality['warnings']:
                print(f'⚠️ {warning}')
        
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
                update_tools.update_data_and_scripts(
                    repo_type=args.repo,
                    force=args.force,
                    cdb_url=args.cdb_url,
                    card_vocab_url=args.card_vocab_url,
                )

    elif args.command == 'trajectory-check':
        import json
        from core_trajectory import inspect_core_trajectory, replay_core_trajectory
        try:
            report = (replay_core_trajectory(args.path) if args.replay
                      else inspect_core_trajectory(args.path))
        except Exception as error:
            parser.exit(1, f'规范轨迹检查失败: {error}\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == 'replay-ingest':
        import json
        import lzma
        from pathlib import Path
        from trajectory_ingest import import_link_capture, import_yrp, inspect_link_capture
        from yrp_ingest import inspect_yrp
        try:
            if args.report and Path(args.report).exists():
                raise FileExistsError('报告已存在，请选择新文件名')
            if args.source == 'link' and (args.p0_player_id or args.p1_player_id):
                raise ValueError('Link 玩家 ID 必须来自原捕获头部，不能在导入时覆写')
            if args.inspect:
                report = inspect_yrp(args.path) if args.source == 'yrp' else inspect_link_capture(args.path)
            elif args.source == 'yrp':
                report = import_yrp(args.path, args.output, timeout_seconds=args.timeout,
                                    player_ids={'0': args.p0_player_id, '1': args.p1_player_id})
            else:
                report = import_link_capture(args.path, args.output, timeout_seconds=args.timeout)
            if args.report:
                target = Path(args.report)
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, 'x', encoding='utf-8') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if report.get('conversion_error') or report.get('verification_error'):
                parser.exit(2, '已保存诊断失败报告，该轨迹不得进入训练。\n')
        except (ValueError, RuntimeError, OSError, KeyError, EOFError, RecursionError, lzma.LZMAError) as error:
            parser.exit(1, f'录像入口拒绝: {error}\n')

    elif args.command == 'dataset-build':
        import json
        from trajectory_dataset import build_trajectory_dataset
        try:
            report = build_trajectory_dataset(args.directory, args.output, seed=args.seed,
                validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
        except (ValueError, RuntimeError, OSError) as error:
            parser.exit(1, f'数据门禁失败: {error}\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'entries'},
                         ensure_ascii=False, indent=2))

    elif args.command == 'expert-read':
        import json
        from pathlib import Path
        from trajectory_dataset import preview_verified_demonstrations
        try:
            target = Path(args.report) if args.report else None
            if target is not None and target.exists():
                raise FileExistsError('expert report already exists; choose a new output name')
            report = preview_verified_demonstrations(args.path, expert_seats=args.expert_seat,
                preview_rows=args.preview_rows, expected_file_sha256=args.expected_sha256)
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, 'x', encoding='utf-8') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except (ValueError, RuntimeError, OSError, KeyError, EOFError, RecursionError) as error:
            parser.exit(1, f'专家样本读取拒绝: {error}\n')

    elif args.command == 'imitate':
        import json
        from integrations.imitation.workflow import (collect_demonstrations, prepare_demonstrations,
                                                    train_imitation, evaluate_imitation)
        try:
            if args.imitation_command == 'collect':
                from integrations.imitation.data import read_json
                report = collect_demonstrations(args.output, train_executors=args.train_executor,
                    validation_executors=args.validation_executor, test_executors=args.test_executor,
                    games_per_executor=args.games_per_executor, seed=args.seed, runtime=args.runtime,
                    mono=args.mono, timeout=args.timeout, max_decisions=args.max_decisions,
                    families=read_json(args.families) if args.families else None,
                    split_mode=args.split_mode, split_seed=args.split_seed,
                    validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
            elif args.imitation_command == 'prepare':
                report = prepare_demonstrations(args.source, args.output, mode=args.split_mode, seed=args.seed,
                    expert_seats=args.expert_seat, validation_executors=args.validation_executor,
                    test_executors=args.test_executor, validation_fraction=args.validation_fraction,
                    test_fraction=args.test_fraction)
            elif args.imitation_command == 'train':
                report = train_imitation(args.dataset, init_from=args.init_from, resume=args.resume,
                    output=args.output, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                    device=args.device, seed=args.seed, model_prefix=args.model_prefix,
                    save_every=args.save_every, cpu_threads=args.cpu_threads)
            else:
                report = evaluate_imitation(args.dataset, args.checkpoint, args.output, baseline=args.baseline,
                    split=args.split, executors=args.executor, games_per_seat=args.games_per_seat,
                    seed=args.seed, runtime=args.runtime, mono=args.mono, timeout=args.timeout,
                    policy_mode=args.policy_mode, temperature=args.temperature, record_replay=args.record_replay,
                    record_evaluations=args.record_evaluations)
            print(json.dumps({key: value for key, value in report.items() if key not in ('entries', 'history')},
                             ensure_ascii=False, indent=2))
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, EOFError) as error:
            parser.exit(1, f'独立模仿工作流停止: {error}\n')

    elif args.command == 'windbot':
        import json
        try:
            from integrations.windbot.runtime import inspect_runtime
            if args.inspect:
                print(json.dumps(inspect_runtime(args.runtime, args.mono), ensure_ascii=False, indent=2))
                return
            if not 1 <= args.games <= 100:
                parser.error('受控局数必须为1～100')
            from integrations.duel_session import resolve_core_seed
            args.seed = resolve_core_seed(args.seed, args.games)
            print(f'[WindBot] 本批 Core 起始种子: {args.seed}')
            if args.record_evaluations and (not args.record_replay or not args.model):
                parser.error('状态预测需要 --record-replay 和原生 --model')
            from integrations.windbot.session import run_controlled_duel
            for game in range(args.games):
                report = run_controlled_duel(executor=args.executor, runtime=args.runtime, mono=args.mono,
                    opponent_executor=args.opponent_executor, windbot_seat=args.windbot_seat,
                    model=args.model, native_deck=args.native_deck,
                    native_deck_source=args.native_deck_source, deck_dir=args.deck_dir, seed=args.seed + game,
                    timeout=args.timeout, response_timeout=args.response_timeout,
                    max_decisions=args.max_decisions, capture_root=args.capture_root,
                    verify=args.verify, policy_mode=args.policy_mode, temperature=args.temperature,
                    record_replay=args.record_replay, record_evaluations=args.record_evaluations,
                    game_index=game + 1, replay_root=args.replay_root)
                # Keep a concise result; the canonical conversion report contains
                # full assets/decks and is available through the existing ingress UI.
                if 'verification' in report:
                    verification = report['verification']
                    report['verification'] = {key: verification.get(key) for key in (
                        'trajectory_path', 'conversion_error', 'verification_error',
                        'replay_verified', 'raw_output_exact', 'behavior_clone_eligible')}
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if report['error']:
                    parser.exit(2, '受控局异常已隔离，未替换对手或生成学习样本。\n')
                verification = report.get('verification')
                if verification and (verification.get('conversion_error') or verification.get('verification_error')
                                     or not verification.get('replay_verified')):
                    parser.exit(2, '对局原始捕获已保存，但独立重放未通过，不具备学习资格。\n')
        except ImportError as error:
            parser.exit(1, f'WindBot 可选模块或依赖缺失，不影响其他命令: {error}\n')
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f'WindBot 接入失败: {error}\n')

    elif args.command == 'vocab':
        print("⚠️ 本地词表追加会形成模型协议身份；请备份并向所有训练/部署机器分发同一 card_vocab.json。")
        from card_vocab import load_card_vocabulary, update_card_vocabulary

        previous_count = 0
        if os.path.isfile(args.output):
            previous_count = load_card_vocabulary(args.output).card_count
        vocabulary = update_card_vocabulary(args.cdb, args.output)
        print(
            f"✅ 精确卡片词表已更新：新增 {vocabulary.card_count - previous_count} 张，"
            f"当前 {vocabulary.card_count}/{vocabulary.capacity}，"
            f"SHA-256 {vocabulary.vocabulary_hash}"
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
