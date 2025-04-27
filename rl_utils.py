import collections
import typing as T
import numpy.typing as np_typing

from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


def compute_reward(
    base_reward: float,
    info: T.Dict[str, T.Any],
    prev_info: T.Optional[T.Dict[str, T.Any]],
) -> float:
    if prev_info is None:
        return base_reward
    lives_reward = -100 if info["life"] < prev_info["life"] else 0
    score_reward = (info["score"] - prev_info["score"]) / 100
    if info["status"] != prev_info["status"]:
        status_reward = 100 if info["status"] != "small" else 0
    else:
        status_reward = 5 if info["status"] != "small" else 0
    total_reward = base_reward + lives_reward + score_reward + status_reward
    return total_reward


class ReplayBuffer:

    def __init__(self, max_size: int, batch_size: int):
        self.batch_size = batch_size
        self.states = collections.deque(maxlen=max_size)
        self.next_states = collections.deque(maxlen=max_size)
        self.actions = collections.deque(maxlen=max_size)
        self.rewards = collections.deque(maxlen=max_size)

    def store(
        self, state: np_typing.NDArray, action: int, reward: float, next_state: np_typing.NDArray
    ):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.states)

    def sample(self, n: int) -> DataLoader:
        # Create a dataset from this buffer
        dataset = ReplayDataset(self)

        # If n is larger than the buffer size, adjust it
        sample_size = min(n, len(self))

        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, sample_size),
            sampler=RandomSampler(data_source=dataset, replacement=False, num_samples=sample_size),
            drop_last=False,
        )


class ReplayDataset(Dataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        return {
            "state": self.replay_buffer.states[idx],
            "action": self.replay_buffer.actions[idx],
            "reward": self.replay_buffer.rewards[idx],
            "next_state": self.replay_buffer.next_states[idx],
        }
