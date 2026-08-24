class RolloutCursor:
    """Tracks tentative and committed rollout rows for one worker.

    Observations are written directly into the worker's preallocated tensors.
    Rows produced by the current duel remain tentative until the duel is
    settled.  An aborted duel rewinds the cursor so those rows are overwritten
    by the next duel instead of being paired with unrelated labels.
    """

    def __init__(self):
        self.write_pos = 0
        self.committed_pos = 0
        self.collected_steps = 0
        self.episode_start_pos = 0
        self._episode_start_steps = 0
        self._episode_open = False

    @property
    def episode_open(self):
        return self._episode_open

    @property
    def next_write_pos(self):
        if not self._episode_open:
            raise RuntimeError("cannot write a rollout row outside an episode")
        return self.write_pos

    def begin_episode(self):
        if self._episode_open:
            raise RuntimeError("previous rollout episode is still open")
        if self.write_pos != self.committed_pos:
            raise RuntimeError(
                f"uncommitted rollout rows remain: write={self.write_pos}, "
                f"committed={self.committed_pos}"
            )

        self.episode_start_pos = self.write_pos
        self._episode_start_steps = self.collected_steps
        self._episode_open = True

    def record_step(self):
        """Commits one fully written observation row to the open episode."""
        index = self.next_write_pos
        self.write_pos += 1
        self.collected_steps += 1
        return index

    def validate_episode(self, trajectory_length):
        if not self._episode_open:
            raise RuntimeError("no rollout episode is open")

        written_rows = self.write_pos - self.episode_start_pos
        if trajectory_length != written_rows:
            raise RuntimeError(
                "rollout observation/trajectory mismatch: "
                f"observations={written_rows}, trajectory={trajectory_length}, "
                f"episode_start={self.episode_start_pos}, write={self.write_pos}"
            )

    def commit_episode(self, trajectory_length):
        self.validate_episode(trajectory_length)
        self.committed_pos = self.write_pos
        self._episode_open = False

    def rollback_episode(self):
        if not self._episode_open:
            raise RuntimeError("no rollout episode is open")

        self.write_pos = self.episode_start_pos
        self.collected_steps = self._episode_start_steps
        self._episode_open = False
