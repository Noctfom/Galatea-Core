# 本文件验证检查点、训练模式、模型产物和竞技场决策链修复。

import ast
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_candidates import build_macro_action_pool
from ai_bot import AiBot
from checkpoint_utils import (
    canonical_model_state_dict,
    load_training_checkpoint,
    restore_model_state_strict,
)
import deck_utils
from data_types import GameAction
from game_constants import LocationInfo, Zone
from galatea_net import GalateaNet
from model_versus import ModelArena
from model_artifacts import (
    collect_model_artifact_files,
    describe_onnx_artifact,
    safe_extract_zip,
    write_checkpoint_artifact_manifest,
)
from trainer import PPOTrainer
from training_validation import validate_max_iterations, validate_training_config


class CheckpointResumeTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(model):
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scaler_state_dict": {},
            "net_config": {"d_model": 8},
            "iteration": 7,
            "train_step": 11,
            "global_step": 13,
        }

    def test_current_checkpoint_restores_exactly_before_compile(self):
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with torch.no_grad():
            target.weight.zero_()
            target.bias.zero_()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            torch.save(self._checkpoint(source), path)
            checkpoint = load_training_checkpoint(path)
            restore_model_state_strict(target, checkpoint)

        for source_value, target_value in zip(
            source.state_dict().values(), target.state_dict().values()
        ):
            self.assertTrue(torch.equal(source_value, target_value))

    def test_strict_restore_rejects_incomplete_state(self):
        model = nn.Linear(3, 2)
        checkpoint = self._checkpoint(model)
        del checkpoint["model_state_dict"]["bias"]

        with self.assertRaises(RuntimeError):
            restore_model_state_strict(nn.Linear(3, 2), checkpoint)

    def test_compiled_prefix_is_removed_only_when_exporting(self):
        class CompiledLikeWrapper(nn.Module):
            def __init__(self, original):
                super().__init__()
                self._orig_mod = original

        source = nn.Linear(3, 2)
        exported = canonical_model_state_dict(CompiledLikeWrapper(source))

        self.assertEqual(set(exported), set(source.state_dict()))
        self.assertFalse(any(key.startswith("_orig_mod.") for key in exported))

    def test_compiled_keys_inside_checkpoint_are_rejected(self):
        source = nn.Linear(3, 2)
        checkpoint = self._checkpoint(source)
        checkpoint["model_state_dict"] = {
            f"_orig_mod.{key}": value for key, value in source.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_checkpoint.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "canonical uncompiled keys"):
                load_training_checkpoint(path)


class PpoTrainingModeTests(unittest.TestCase):
    def test_update_uses_train_mode_and_restores_eval_mode(self):
        """PPO 更新期间必须启用训练模式，完成后恢复推理模式。"""
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.agent = SimpleNamespace(net=nn.Linear(2, 2).eval())
        observed_modes = []
        trainer._update_policy_training = lambda _steps: observed_modes.append(
            trainer.agent.net.training
        )

        trainer.update_policy(8)

        self.assertEqual(observed_modes, [True])
        self.assertFalse(trainer.agent.net.training)

    def test_update_exception_still_restores_eval_mode(self):
        """更新异常不得把共享网络遗留在训练模式。"""
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.agent = SimpleNamespace(net=nn.Linear(2, 2).eval())

        def fail_update(_steps):
            """模拟 PPO 更新中途失败。"""
            raise RuntimeError("boom")

        trainer._update_policy_training = fail_update
        with self.assertRaisesRegex(RuntimeError, "boom"):
            trainer.update_policy(8)
        self.assertFalse(trainer.agent.net.training)


class TrainingValidationTests(unittest.TestCase):
    BASE_CONFIG = {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 2,
        "vocab_size": 20000,
    }

    @classmethod
    def validate(cls, **overrides):
        """使用一组合法基线参数验证单项非法覆盖。"""
        values = {
            "net_config": dict(cls.BASE_CONFIG),
            "update_timesteps": 4096,
            "mini_batch_size": 512,
            "num_workers": 4,
            "worker_device": "cpu",
            "worker_timeout": 300,
            "gamma": 0.998,
            "learning_rate": 1e-4,
            "entropy": 0.03,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
        }
        values.update(overrides)
        validate_training_config(**values)

    def test_valid_configuration_is_accepted(self):
        """标准训练参数必须通过统一校验。"""
        self.validate()
        validate_max_iterations(10)

    def test_invalid_attention_shape_is_rejected(self):
        """注意力维度无法整除时必须在模型构建前拒绝。"""
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.validate(
                net_config={
                    "d_model": 250,
                    "n_heads": 4,
                    "n_layers": 2,
                    "vocab_size": 20000,
                }
            )

    def test_invalid_batch_and_timeout_are_rejected(self):
        """无效批量关系和过短 Worker 超时必须被拒绝。"""
        with self.assertRaisesRegex(ValueError, "mini_batch_size"):
            self.validate(update_timesteps=128, mini_batch_size=256)
        with self.assertRaisesRegex(ValueError, "greater than 30"):
            self.validate(worker_timeout=30)

    def test_non_finite_hyperparameter_is_rejected(self):
        """NaN 等非有限超参数不得进入优化器。"""
        with self.assertRaisesRegex(ValueError, "finite"):
            self.validate(learning_rate=float("nan"))


class ModelArtifactTests(unittest.TestCase):
    @staticmethod
    def _write_external_onnx(directory, iteration=10):
        """生成引用单个外置权重文件的最小 ONNX 模型。"""
        import onnx
        from onnx import helper, numpy_helper

        graph_name = f"galatea_iter_{iteration}.onnx"
        data_name = f"{graph_name}.data"
        weight = numpy_helper.from_array(
            np.asarray([1.0, 2.0], dtype=np.float32),
            name="weight",
        )
        graph = helper.make_graph([], "artifact_test", [], [], [weight])
        model = helper.make_model(graph)
        graph_path = Path(directory) / graph_name
        onnx.save_model(
            model,
            graph_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=0,
        )
        return graph_path, Path(directory) / data_name

    def test_onnx_external_data_is_collected_and_tagged(self):
        """ONNX 主图、外置权重和轮次标记必须形成完整产物集。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path, data_path = self._write_external_onnx(temp_dir)
            self.assertTrue(data_path.is_file())

            record = describe_onnx_artifact(graph_path)
            self.assertEqual(record["iteration"], 10)
            self.assertEqual(record["external_data"], [data_path.name])
            self.assertEqual(
                collect_model_artifact_files(temp_dir, [graph_path.name]),
                [graph_path.name, data_path.name],
            )

            checkpoint_path = Path(temp_dir) / "galatea_iter_10.pth"
            checkpoint_path.write_bytes(b"checkpoint")
            marker = write_checkpoint_artifact_manifest(
                checkpoint_path,
                10,
                onnx_record=record,
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["iteration"], 10)
            self.assertEqual(payload["onnx"]["status"], "complete")

    def test_missing_external_data_is_rejected(self):
        """外置权重缺失时不得把 ONNX 判定为可用。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path, data_path = self._write_external_onnx(temp_dir)
            data_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "external data"):
                describe_onnx_artifact(graph_path)

    def test_incomplete_iteration_marker_rejects_stale_onnx(self):
        """未完成标记必须阻止同名旧 ONNX 被误用。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path, _data_path = self._write_external_onnx(temp_dir)
            checkpoint_path = Path(temp_dir) / "galatea_iter_10.pth"
            checkpoint_path.write_bytes(b"checkpoint")
            write_checkpoint_artifact_manifest(
                checkpoint_path,
                10,
                onnx_error="export_in_progress",
            )
            with self.assertRaisesRegex(RuntimeError, "not marked complete"):
                describe_onnx_artifact(graph_path)

    def test_archive_path_traversal_is_rejected(self):
        """部署包不得通过父目录路径写出暂存区。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "unsafe.gkg"
            target = Path(temp_dir) / "target"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.pth", b"bad")
            with zipfile.ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    safe_extract_zip(archive, target)
            self.assertFalse((Path(temp_dir) / "escape.pth").exists())


class DeckPairContractTests(unittest.TestCase):
    def test_failure_returns_none_instead_of_short_tuple(self):
        """卡组不足时必须返回空值，避免调用方按五项解包崩溃。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(deck_utils.get_random_deck_pair(temp_dir))


class MacroActionCandidateTests(unittest.TestCase):
    class ArgmaxRng:
        def __init__(self):
            self.probabilities = None

        def choice(self, count, *, size, replace, p):
            self.probabilities = p
            return np.asarray([int(np.argmax(p))])

    def test_target_scoring_ignores_position_bits_and_preserves_response(self):
        target_a = LocationInfo.encode(0, Zone.MZONE, 1, position=1)
        target_b = LocationInfo.encode(0, Zone.MZONE, 2, position=4)
        option_a = LocationInfo.encode(0, Zone.MZONE, 1, position=0)
        option_b = LocationInfo.encode(0, Zone.MZONE, 2, position=0)

        actions = [
            GameAction(action_type=15, index=0, target_entity_idx=target_a),
            GameAction(action_type=15, index=1, target_entity_idx=target_b),
        ]
        options = [
            {"bytes": b"\x01\x00", "locs": [option_a]},
            {"bytes": b"\x01\x01", "locs": [option_b]},
        ]
        rng = self.ArgmaxRng()

        with patch("action_candidates.rule_bot.get_macro_options", return_value=options):
            pool = build_macro_action_pool(
                15,
                b"payload",
                object(),
                actions,
                [0.1, 0.9],
                max_actions=1,
                rng=rng,
            )

        self.assertGreater(rng.probabilities[1], rng.probabilities[0])
        self.assertTrue(np.all(rng.probabilities > 0))
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].decision_bytes, b"\x01\x01")
        self.assertEqual(pool[0].macro_targets, [option_b])


class AsyncInferenceWiringTests(unittest.TestCase):
    def test_async_server_receives_shared_logits(self):
        """确认异步推理线程能够写入宏动作第一遍推理的分数槽。"""
        source = (PROJECT_ROOT / "trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_start_inference_server"
        )
        thread_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Thread"
        )
        args_keyword = next(
            keyword for keyword in thread_call.keywords if keyword.arg == "args"
        )

        self.assertIsInstance(args_keyword.value, ast.Tuple)
        shared_arguments = args_keyword.value.elts[-2:]
        self.assertTrue(all(isinstance(arg, ast.Attribute) for arg in shared_arguments))
        self.assertEqual(
            [arg.attr for arg in shared_arguments],
            ["shared_logits", "shared_response_ids"],
        )

    def test_worker_receives_shared_response_ids(self):
        """确认 Worker 进程能够收到用于结果校验的共享完成号。"""
        source = (PROJECT_ROOT / "trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        worker_process_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Process"
            and any(
                keyword.arg == "target"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "worker_process"
                for keyword in node.keywords
            )
        )
        args_keyword = next(
            keyword for keyword in worker_process_call.keywords if keyword.arg == "args"
        )

        last_argument = args_keyword.value.elts[-1]
        self.assertIsInstance(last_argument, ast.Attribute)
        self.assertEqual(last_argument.attr, "shared_response_ids")


class ArenaCorrectnessTests(unittest.TestCase):
    def test_constructor_loads_a_current_checkpoint(self):
        config = {
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "vocab_size": 128,
        }
        network = GalateaNet(config)
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": {},
            "scaler_state_dict": {},
            "net_config": config,
            "iteration": 1,
            "train_step": 2,
            "global_step": 3,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "current_checkpoint.pt"
            torch.save(checkpoint, path)
            with patch("model_versus.GalateaEnv", return_value=object()), patch(
                "builtins.print"
            ):
                arena = ModelArena(str(path), device="cpu", config=config)

        self.assertIsNotNone(arena.p0_bot)

    def test_padding_sentinel_is_not_clamped_to_a_real_card(self):
        tensor_dict = {"act_card_idx": torch.tensor([[0, 120]])}

        ModelArena._validate_action_indices(tensor_dict)

        self.assertEqual(tensor_dict["act_card_idx"].tolist(), [[0, 120]])
        with self.assertRaises(ValueError):
            ModelArena._validate_action_indices(
                {"act_card_idx": torch.tensor([[0, 121]])}
            )

    def test_failed_model_load_is_fatal(self):
        class FailedBot:
            @staticmethod
            def load_model(path):
                return False

        with self.assertRaisesRegex(RuntimeError, "failed to load"):
            ModelArena._require_model_loaded(FailedBot(), "model.pt", "P0")

    def test_initialization_failure_always_returns_three_values(self):
        arena = object.__new__(ModelArena)
        arena.deck_dir = "unused"

        with patch("model_versus.deck_utils.get_random_deck_pair", return_value=None):
            result = arena.run_duel()

        self.assertEqual(result, (-1, -3, 0))

    def test_arena_uses_canonical_response_packer(self):
        bot = object.__new__(AiBot)
        declaration = GameAction(action_type=142, index=0, desc_id=123456)
        macro = GameAction(
            action_type=15,
            index=0,
            decision_bytes=b"\x02\x01\x03",
        )

        self.assertEqual(bot._pack_response(declaration, msg_type=142), 123456)
        self.assertEqual(
            bot._pack_response(macro, msg_type=15), b"\x02\x01\x03"
        )


if __name__ == "__main__":
    unittest.main()
