import typing as T

import hashlib
import numpy as np
import numpy.typing as np_typing
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from agent.dqn_agent import BasicQAgent


class MemoryBank:

    def __init__(
        self, state_latent_dim: int, stack_size: int, num_actions: int, device: str
    ) -> None:
        self.bank = {i: {} for i in range(num_actions)}
        self.device = device
        np.random.seed(1)
        self.proj = np.random.randn(state_latent_dim, 84 * 84 * stack_size)
        self.num_actions = num_actions

    def _hash_state(self, state: np_typing.NDArray) -> str:
        encoded_state = self.proj @ state.flatten()
        state_bytes = encoded_state.tobytes()
        shape_bytes = str(encoded_state.shape).encode("utf-8")
        dtype_bytes = str(encoded_state.dtype).encode("utf-8")
        combined = state_bytes + shape_bytes + dtype_bytes
        return hashlib.sha256(combined).hexdigest()

    def state_value_update(
        self, state: np_typing.NDArray, action: int, reward_to_go: float
    ) -> None:
        state_hash = self._hash_state(state)
        current_entry = self.bank[action].get(state_hash, -np.inf)
        self.bank[action][state_hash] = max(current_entry, reward_to_go)

    def state_value_lookup(self, state: np_typing.NDArray, action: int) -> T.Optional[float]:
        state_hash = self._hash_state(state)
        return self.bank[action].get(state_hash, None)

    def add_episode(
        self,
        episode_states: T.List[np_typing.NDArray],
        episode_actions: T.List[int],
        episode_rewards: T.List[float],
        gamma: float,
    ) -> None:
        reward_to_go = 0.0
        for state, action, reward in zip(
            episode_states[::-1], episode_actions[::-1], episode_rewards[::-1]
        ):
            self.state_value_update(state, action, reward + gamma * reward_to_go)
            reward_to_go = reward + gamma * reward_to_go


class EMDQNAgent(BasicQAgent):

    def __init__(
        self,
        num_actions: int,
        q_params: T.Dict[str, T.Any],
        ep_sched_params: T.Dict[str, T.Any],
        lr: float,
        gamma: float,
        alpha_q: float = 1.0,
        alpha_mem: float = 1.0,
        memory_update_freq: int = 500,
        stack_size: int = 4,
        num_steps_between_target_updates: int = 1,
    ):
        super().__init__(
            num_actions=num_actions,
            ep_sched_params=ep_sched_params,
            q_params=q_params,
            lr=lr,
            gamma=gamma,
        )
        self.bank = MemoryBank(
            num_actions=num_actions,
            state_latent_dim=4,
            stack_size=stack_size,
            device=self.device,
        )
        self.alpha = alpha_mem / alpha_q

        self.reset_episode_memory()

        self.memory_hits = 0
        self.memory_update_freq = memory_update_freq
        self.num_steps_between_target_updates = num_steps_between_target_updates

    def reset_episode_memory(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def update_state(
        self, step: int, state: np_typing.NDArray, action: int, reward: float, done: bool
    ):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

        if done or len(self.episode_states) == self.memory_update_freq:
            self.bank.add_episode(
                self.episode_states, self.episode_actions, self.episode_rewards, self.gamma
            )
            self.reset_episode_memory()

        super().update_state(step, state, action, reward, done)

    def compute_loss(
        self,
        step: int,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        reward_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> torch.Tensor:

        predicted_reward = self.q(state_tensor, action_tensor)
        greedy_action = self.act(
            step,
            next_state_tensor,
            greedy=True,
            return_tensor=True,
        )
        target = (
            reward_tensor.view(-1, 1)
            + self.gamma * self.q_target(next_state_tensor, greedy_action).detach()
        )
        q_loss = F.mse_loss(predicted_reward, target)

        # look up each state in the batch in memory. if there, add loss for it
        batch_size = state_tensor.shape[0]
        mem_loss = 0.0
        for i in range(batch_size):
            state = state_tensor[i].cpu().numpy()
            action = int(action_tensor[i].item())
            best_remembered_reward = self.bank.state_value_lookup(state, action)
            if best_remembered_reward is not None:
                self.memory_hits += 1
                mem_loss += (best_remembered_reward - predicted_reward[i, 0]) ** 2 / batch_size

        return q_loss + self.alpha * mem_loss
