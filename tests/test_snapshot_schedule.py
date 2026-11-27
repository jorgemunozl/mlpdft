import unittest

from mlpdft.snapshot_schedule import snapshot_schedule


class SnapshotScheduleTests(unittest.TestCase):
    def test_default_matches_the_schedule_before_burn_in(self):
        # exp-02 (150 epochs) and the 1,000-label node (51 epochs) without burn-in.
        s = snapshot_schedule(150, 3)
        self.assertEqual((s.cycle_epochs, s.total_epochs, s.member_epochs), (50, 150, (49, 99, 149)))
        s = snapshot_schedule(51, 3)
        self.assertEqual((s.cycle_epochs, s.total_epochs, s.member_epochs), (17, 51, (16, 33, 50)))
        s = snapshot_schedule(50, 3)  # not divisible: the last epochs run past the third cycle end
        self.assertEqual((s.cycle_epochs, s.total_epochs, s.member_epochs), (16, 50, (15, 31, 47)))

    def test_burn_in_adds_discarded_cycles_and_keeps_the_last_ends(self):
        s = snapshot_schedule(51, 3, burn_in_cycles=1)
        self.assertEqual((s.cycle_epochs, s.total_epochs, s.member_epochs), (17, 68, (33, 50, 67)))
        self.assertEqual(s.burn_in_cycles, 1)

    def test_cycle_length_can_be_set_per_rung(self):
        s = snapshot_schedule(51, 3, burn_in_cycles=1, cycle_epochs=68)
        self.assertEqual((s.total_epochs, s.member_epochs), (272, (135, 203, 271)))
        s = snapshot_schedule(51, 3, cycle_epochs=10)
        self.assertEqual((s.total_epochs, s.member_epochs), (30, (9, 19, 29)))

    def test_smoke_settings(self):
        s = snapshot_schedule(6, 3, burn_in_cycles=1)
        self.assertEqual((s.cycle_epochs, s.total_epochs, s.member_epochs), (2, 8, (3, 5, 7)))

    def test_rejects_impossible_schedules(self):
        with self.assertRaises(ValueError):
            snapshot_schedule(51, 0)
        with self.assertRaises(ValueError):
            snapshot_schedule(51, 3, burn_in_cycles=-1)
        with self.assertRaises(ValueError):
            snapshot_schedule(51, 3, cycle_epochs=1)
        with self.assertRaises(ValueError):
            snapshot_schedule(4, 3)  # 2-epoch cycles in a 4-epoch run: two members on epoch 3


if __name__ == "__main__":
    unittest.main()
