import abc
import typing as T

import numpy as np
import numpy.typing as np_typing
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
import tqdm


class LinearEpsilonDecayScheduler:

    def __init__(
        self, ep_init: float, ep_final: float, num_steps_before_decay: int, num_decay_steps: int
    ):
        self.ep_init = ep_init
        self.ep_final = ep_final
        self.init_to_final_diff = ep_init - ep_final
        self.num_steps_before_decay = num_steps_before_decay
        self.num_decay_steps = num_decay_steps

    def get_epsilon(self, step: int) -> float:
        if step < self.num_steps_before_decay:
            return self.ep_init
        if step > self.num_steps_before_decay + self.num_decay_steps:
            return self.ep_final
        decay_steps_so_far = step - self.num_steps_before_decay
        return self.ep_init - (decay_steps_so_far / self.num_decay_steps) * self.init_to_final_diff


class BaseAgent(abc.ABC):

    def __init__(self, ep_sched_params: T.Dict[str, T.Any], optimizer: Optimizer) -> None:
        self.ep_sched = LinearEpsilonDecayScheduler(**ep_sched_params)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        self.device = device
        self.optim = optimizer

    def tensorize(
        self,
        state: np_typing.NDArray,
        action: int,
        reward: float,
        next_state: np_typing.NDArray,
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensor = torch.Tensor(state[np.newaxis, ...].copy()).to(self.device)
        next_state_tensor = torch.Tensor(next_state[np.newaxis, ...].copy()).to(self.device)
        action_tensor = torch.Tensor([[action]]).int().to(self.device)
        reward_tensor = torch.Tensor([[reward]]).to(self.device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor

    def learn_batch(self, step: int, dataloader: DataLoader) -> float:
        self.train()
        total_loss = 0.0
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            state, action, reward, next_state = (
                batch["state"].to(dtype=torch.float32).to(self.device),
                batch["action"].to(self.device),
                batch["reward"].to(dtype=torch.float32).to(self.device),
                batch["next_state"].to(dtype=torch.float32).to(self.device),
            )
            loss = self.compute_loss(step, state, action, reward, next_state)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @abc.abstractmethod
    def compute_loss(
        self,
        step: int,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        reward_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def act(
        self,
        step: int,
        state: torch.Tensor,
        greedy: bool = False,
        return_tensor: bool = False,
    ) -> int:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def update_state(self, step, state, action, reward, done):
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def save(self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any]) -> None:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def load(self, checkpoint_dir: str, step: int) -> "BaseAgent":
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError("subclass must implement")
