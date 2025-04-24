import abc
import typing as T

import numpy.typing as np_typing
import torch


class BaseAgent(abc.ABC):

    def __init__(self, ep: float) -> None:
        self.ep = ep
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    @abc.abstractmethod
    def learn_one_step(
        self,
        state: np_typing.NDArray,
        action: int,
        reward: float,
        next_state: np_typing.NDArray,
        done: bool,
    ) -> float:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def act(
        self,
        state: torch.Tensor,
        greedy: bool = False,
        return_tensor: bool = False,
    ) -> int:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def save(self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any]) -> None:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def load(self, checkpoint_dir: str, step: int) -> "BaseAgent":
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def train(self) -> None:
        raise NotImplementedError("subclass must implement")

    @abc.abstractmethod
    def eval(self) -> None:
        raise NotImplementedError("subclass must implement")
