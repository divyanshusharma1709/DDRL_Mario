import heapq
import typing as T
import numpy as np
import numpy.typing as np_typing

from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

NUM_LIVES = 3


def compute_reward(
    base_reward: float,
    info: T.Dict[str, T.Any],
    prev_info: T.Optional[T.Dict[str, T.Any]],
) -> float:
    if prev_info is None:
        return base_reward
    prev_lives_left = prev_info["life"]
    curr_lives_left = info["life"]
    if curr_lives_left < prev_lives_left:
        return -15
    return base_reward


class ReplayBuffer:

    def __init__(self, max_size: int, batch_size: int, weighted: bool = False):
        self.max_size = max_size
        self.batch_size = batch_size
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.errors = []
        self.weighted = weighted

    def store(
        self,
        state: np_typing.NDArray,
        action: int,
        reward: float,
        next_state: np_typing.NDArray,
        loss: float,
    ):
        key = -np.sqrt(loss)
        if len(self.states) < self.max_size:
            heapq.heappush(self.states, (key, state))
            heapq.heappush(self.next_states, (key, next_state))
            heapq.heappush(self.actions, (key, action))
            heapq.heappush(self.rewards, (key, reward))
        else:
            heapq.heappushpop(self.states, (key, state))
            heapq.heappushpop(self.next_states, (key, next_state))
            heapq.heappushpop(self.actions, (key, action))
            heapq.heappushpop(self.rewards, (key, reward))

    def __len__(self):
        return len(self.states)

    def sample(self, n: int, beta: float) -> DataLoader:
        # Create a dataset from this buffer
        dataset = ReplayDataset(self)

        # If n is larger than the buffer size, adjust it
        sample_size = min(n, len(self), self.batch_size)

        if self.weighted:
            neg_errors, _ = zip(*self.states)
            errors = -np.array(neg_errors)
            normalized_td_errors = errors / np.max(errors)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=WeightedRandomSampler(
                    weights=normalized_td_errors, replacement=False, num_samples=sample_size
                ),
            )
        # use uniform sampling
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
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
            "state": self.replay_buffer.states[idx][1],
            "action": self.replay_buffer.actions[idx][1],
            "reward": self.replay_buffer.rewards[idx][1],
            "next_state": self.replay_buffer.next_states[idx][1],
        }
