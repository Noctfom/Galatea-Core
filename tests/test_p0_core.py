# 本文件验证采样游标、玩家视角编码和场面快照同步修复。

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_types import CardEntity, GlobalFeature
from feature_encoder import GalateaEncoder
from game_constants import Zone
from gamestate import DuelState
from rollout_cursor import RolloutCursor


class RolloutCursorTests(unittest.TestCase):
    def test_aborted_episode_reuses_tentative_rows(self):
        cursor = RolloutCursor()

        cursor.begin_episode()
        self.assertEqual(cursor.record_step(), 0)
        self.assertEqual(cursor.record_step(), 1)
        cursor.rollback_episode()

        self.assertEqual(cursor.write_pos, 0)
        self.assertEqual(cursor.committed_pos, 0)
        self.assertEqual(cursor.collected_steps, 0)

        cursor.begin_episode()
        self.assertEqual(cursor.record_step(), 0)
        self.assertEqual(cursor.record_step(), 1)
        cursor.commit_episode(2)

        self.assertEqual(cursor.committed_pos, 2)
        self.assertEqual(cursor.collected_steps, 2)

    def test_commit_rejects_observation_trajectory_mismatch(self):
        cursor = RolloutCursor()
        cursor.begin_episode()
        cursor.record_step()

        with self.assertRaisesRegex(RuntimeError, "observation/trajectory mismatch"):
            cursor.commit_episode(0)

        cursor.rollback_episode()


class PerspectiveEncodingTests(unittest.TestCase):
    def setUp(self):
        self.global_state = GlobalFeature(
            turn_count=7,
            phase_id=3,
            to_play=1,
            my_lp=7000,
            op_lp=3000,
            my_hand_len=2,
            op_hand_len=5,
            my_deck_len=31,
            op_deck_len=22,
            my_grave_len=4,
            op_grave_len=9,
            my_removed_len=1,
            op_removed_len=3,
            my_extra_len=10,
            op_extra_len=6,
        )

    def test_p1_global_resources_are_swapped(self):
        p0 = GalateaEncoder._encode_global_vector(self.global_state, 0)
        p1 = GalateaEncoder._encode_global_vector(self.global_state, 1)

        self.assertEqual(len(p0), 15)
        self.assertEqual(p0[2], 0.0)
        self.assertEqual(p1[2], 1.0)
        for offset in range(3, 15, 2):
            self.assertAlmostEqual(p0[offset], p1[offset + 1])
            self.assertAlmostEqual(p0[offset + 1], p1[offset])

    def test_invalid_player_id_is_rejected(self):
        with self.assertRaises(ValueError):
            GalateaEncoder._encode_global_vector(self.global_state, 2)


class SnapshotSynchronizationTests(unittest.TestCase):
    CARD_CODE = 89631139

    class FakeEnv:
        def __init__(self, responses):
            self.responses = responses

        def query_card_state(self, player, zone, sequence):
            value = self.responses.get((player, zone, sequence))
            return dict(value) if value else None

    @staticmethod
    def _field_card(code, position=1, used_effect_mask=0):
        return {
            "code": code,
            "pos": position,
            "owner": 0,
            "counters": 0,
            "overlays": [],
            "is_equipped": False,
            "used_effect_mask": used_effect_mask,
        }

    def test_snapshot_preserves_event_zones_and_effect_mask(self):
        state = DuelState([], [], [], [])
        state.field_map[0][Zone.MZONE][0] = self._field_card(
            self.CARD_CODE, used_effect_mask=4
        )
        state.field_map[0][Zone.GRAVE][0] = self._field_card(self.CARD_CODE)
        state.field_map[0][Zone.REMOVED][0] = self._field_card(self.CARD_CODE)
        state.field_map[0][Zone.EXTRA][0] = self._field_card(self.CARD_CODE)

        env = self.FakeEnv(
            {
                (0, Zone.MZONE, 0): self._field_card(self.CARD_CODE),
            }
        )
        fake_stats = [0, 0, 0, 8, 0, 0, 0, 0, 3000, 2500, (0, 0, 0, 0)]

        with patch("gamestate.card_db.get_full_stats", return_value=fake_stats):
            snapshot = state.get_snapshot(env)

        self.assertEqual(snapshot.global_data.my_grave_len, 1)
        self.assertEqual(snapshot.global_data.my_removed_len, 1)
        monster = next(
            entity
            for entity in snapshot.entities
            if entity.owner == 0 and entity.location == Zone.MZONE
        )
        self.assertEqual(monster.used_effect_mask, 4)

    def test_replacement_card_does_not_inherit_effect_mask(self):
        state = DuelState([], [], [], [])
        state.field_map[0][Zone.MZONE][0] = self._field_card(
            self.CARD_CODE, used_effect_mask=8
        )
        replacement_code = self.CARD_CODE + 1
        env = self.FakeEnv(
            {
                (0, Zone.MZONE, 0): self._field_card(replacement_code),
            }
        )

        state.sync_active_field(env)

        replacement = state.field_map[0][Zone.MZONE][0]
        self.assertEqual(replacement["code"], replacement_code)
        self.assertEqual(replacement.get("used_effect_mask", 0), 0)

    def test_hidden_extra_and_banished_cards_are_not_visible(self):
        for zone in (Zone.EXTRA, Zone.REMOVED):
            hidden = self._entity(owner=1, location=zone, is_public=False)
            public = self._entity(owner=1, location=zone, is_public=True)
            own = self._entity(owner=0, location=zone, is_public=False)

            self.assertFalse(
                GalateaEncoder._is_entity_visible_to_player(hidden, player_id=0)
            )
            self.assertTrue(
                GalateaEncoder._is_entity_visible_to_player(public, player_id=0)
            )
            self.assertTrue(
                GalateaEncoder._is_entity_visible_to_player(own, player_id=0)
            )

    @classmethod
    def _entity(cls, owner, location, is_public):
        return CardEntity(
            code=cls.CARD_CODE,
            owner=owner,
            location=location,
            sequence=0,
            position=1 if is_public else 8,
            current_atk=0,
            current_def=0,
            type_mask=0,
            race=0,
            attribute=0,
            level=0,
            base_atk=0,
            base_def=0,
            is_public=is_public,
        )


if __name__ == "__main__":
    unittest.main()
