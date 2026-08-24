import sys
import tempfile
import unittest
from pathlib import Path
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
from data_types import GameAction
from game_constants import LocationInfo, Zone
from galatea_net import GalateaNet
from model_versus import ModelArena


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
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].decision_bytes, b"\x01\x01")
        self.assertEqual(pool[0].macro_targets, [option_b])


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
