import abc
import typing as T

import numpy as np
import numpy.typing as np_typing
import torch
from torch.utils.data.dataloader import DataLoader
from data_utils import ExperienceBuffer


class BaseAgent(abc.ABC):

    def __init__(self) -> None:
        raise ValueError("cannot instantiate a base agent")

    @abc.abstractmethod
    def train(self) -> None:
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def eval(self) -> None:
        raise ValueError("subclass must implement")

    def _get_dataloader(
        self,
        all_states: T.List[np_typing.NDArray],
        all_actions: T.List[int],
        all_rewards: T.List[float],
        all_next_states: T.List[np_typing.NDArray],
        num_train_examples: int,
        train_batch_size: int,
    ):
        stacked_states = np.vstack(
            [state[np.newaxis, :] for state in all_states]
        )
        stacked_next_states = np.vstack(
            [state[np.newaxis, :] for state in all_next_states]
        )

        states_t = torch.Tensor(stacked_states)
        next_states_t = torch.Tensor(stacked_next_states)
        actions_t = torch.Tensor(all_actions).int()
        rewards_t = torch.Tensor(all_rewards)

        perm = torch.randperm(len(all_states))[:num_train_examples]
        train_dataset = ExperienceBuffer(
            states_t[perm],
            actions_t[perm],
            rewards_t[perm],
            next_states_t[perm],
        )

        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )

    @abc.abstractmethod
    def learn(self, train_data: DataLoader) -> None:
        raise ValueError("subclass must implement")

    def update(
        self,
        states: T.List[np_typing.NDArray],
        actions: T.List[int],
        rewards: T.List[float],
        next_states: T.List[np_typing.NDArray],
        num_train_examples: int,
        train_batch_size: int,
    ) -> None:
        train_loader = self._get_dataloader(
            states,
            actions,
            rewards,
            next_states,
            num_train_examples,
            train_batch_size,
        )
        self.learn(train_loader)

    def act(self, state: torch.Tensor, ep: float = 0.0) -> int:
        raise ValueError("subclass must implement")
