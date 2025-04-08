import abc
import typing as T

import numpy as np
import numpy.typing as np_typing
import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
from data_utils import ExperienceBuffer


class BaseAgent(abc.ABC):

    def __init__(self, ep: float) -> None:
        self.ep = ep
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def _get_dataloader(
        self,
        all_states: T.List[np_typing.NDArray],
        all_actions: T.List[int],
        all_rewards: T.List[float],
        all_next_states: T.List[np_typing.NDArray],
        num_train_examples: int,
        train_batch_size: int,
    ):
        # First sample the indices
        idx = np.random.choice(
            len(all_states),
            min(len(all_states), num_train_examples),
            replace=False,
        )

        # Use the sampled indices to select the data
        sampled_states = [all_states[i] for i in idx]
        sampled_next_states = [all_next_states[i] for i in idx]
        sampled_actions = [all_actions[i] for i in idx]
        sampled_rewards = [all_rewards[i] for i in idx]

        # Stack the sampled states and next states
        stacked_states = np.vstack(
            [state[np.newaxis, :] for state in sampled_states]
        )
        stacked_next_states = np.vstack(
            [state[np.newaxis, :] for state in sampled_next_states]
        )

        # Convert to tensors
        states_t = torch.Tensor(stacked_states)
        next_states_t = torch.Tensor(stacked_next_states)
        actions_t = torch.Tensor(sampled_actions).int()
        rewards_t = torch.Tensor(sampled_rewards)

        train_dataset = ExperienceBuffer(
            states_t,
            actions_t,
            rewards_t,
            next_states_t,
        )

        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
        )

    def batch_learn(self, train_data: DataLoader) -> None:
        for batch in tqdm.tqdm(
            train_data, total=len(train_data), desc="Training"
        ):
            state, action, reward, next_state = batch
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)

            self.learn_one_step(state, action, reward, next_state)

    def batch_update(
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
        self.batch_learn(train_loader)

    @abc.abstractmethod
    def learn_one_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ) -> int:
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def act(self, state: torch.Tensor) -> int:
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def save(
        self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any]
    ) -> None:
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def load(self, checkpoint_dir: str, step: int) -> "BaseAgent":
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def train(self) -> None:
        raise ValueError("subclass must implement")

    @abc.abstractmethod
    def eval(self) -> None:
        raise ValueError("subclass must implement")
