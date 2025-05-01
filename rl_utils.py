import collections
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
        self.batch_size = batch_size
        self.weighted = weighted
        self.states = collections.deque(maxlen=max_size)
        self.next_states = collections.deque(maxlen=max_size)
        self.actions = collections.deque(maxlen=max_size)
        self.rewards = collections.deque(maxlen=max_size)
        self.losses = collections.deque(maxlen=max_size)

    def store(
        self,
        state: np_typing.NDArray,
        action: int,
        reward: float,
        next_state: np_typing.NDArray,
        loss: float,
    ):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.losses.append(loss)

    def __len__(self):
        return len(self.states)

    def sample(self, n: int, beta: float) -> DataLoader:
        # Create a dataset from this buffer
        dataset = ReplayDataset(self)

        # If n is larger than the buffer size, adjust it
        sample_size = min(n, len(self))

        if self.weighted:
            normalized_td_errors = np.sqrt(self.losses) / np.max(self.losses)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=WeightedRandomSampler(
                    weights=normalized_td_errors, replacement=False, num_samples=sample_size
                ),
                drop_last=False,
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
            "state": self.replay_buffer.states[idx],
            "action": self.replay_buffer.actions[idx],
            "reward": self.replay_buffer.rewards[idx],
            "next_state": self.replay_buffer.next_states[idx],
        }
