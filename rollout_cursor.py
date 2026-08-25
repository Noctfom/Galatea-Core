# 管理单个 Worker 中可回滚、可提交的采样数据写入游标


class RolloutCursor:
    """跟踪单个 Worker 的暂存行和已提交采样行"""

    def __init__(self):
        """初始化写入、提交和当前对局起点游标"""
        self.write_pos = 0
        self.committed_pos = 0
        self.collected_steps = 0
        self.episode_start_pos = 0
        self._episode_start_steps = 0
        self._episode_open = False

    @property
    def episode_open(self):
        """返回当前是否存在尚未提交或回滚的对局"""
        return self._episode_open

    @property
    def next_write_pos(self):
        """返回当前对局下一条观测应写入的位置"""
        if not self._episode_open:
            raise RuntimeError("cannot write a rollout row outside an episode")
        return self.write_pos

    def begin_episode(self):
        """从已提交位置开始一局新的暂存采样"""
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
        """记录一条已经完整写入的暂存观测"""
        index = self.next_write_pos
        self.write_pos += 1
        self.collected_steps += 1
        return index

    def validate_episode(self, trajectory_length):
        """校验观测写入行数与轨迹标签数量一致"""
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
        """校验并提交当前对局的全部采样"""
        self.validate_episode(trajectory_length)
        self.committed_pos = self.write_pos
        self._episode_open = False

    def rollback_episode(self):
        """回滚当前异常对局占用的全部暂存行"""
        if not self._episode_open:
            raise RuntimeError("no rollout episode is open")

        self.write_pos = self.episode_start_pos
        self.collected_steps = self._episode_start_steps
        self._episode_open = False
