import typing as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy.typing as np_typing
import os
import random
import json

from agent.base_agent import BaseAgent
from agent.Q_DDQN import Qv2 as Q  # Your Q network

class DDQNAgent(BaseAgent):
    def __init__(self, num_actions, q_params, ep_sched_params: T.Dict[str, T.Any], lr=1e-4, gamma=0.95, target_update_freq=100, num_steps_between_target_updates: int = 1):
        # Initialize the Q-network first
        self.q = Q(**q_params)
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        super().__init__(
            ep_sched_params=ep_sched_params,
            optimizer=self.optimizer
        )

        self.q.to(self.device)
        self.target_q = Q(**q_params).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.gamma = gamma
        self.num_actions = num_actions
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        self.num_steps_between_target_updates = num_steps_between_target_updates

    def a2t(self, action: int, batch_size: int) -> torch.Tensor:
        return torch.Tensor([action])[None, :].int().repeat(batch_size, 1).to(self.device)

    def act(
        self,
        step: int,
        state: T.Union[torch.Tensor, np_typing.NDArray],
        greedy: bool = False,
        return_tensor: bool = False,
    ) -> T.Union[np_typing.NDArray, torch.Tensor, int]:
        if isinstance(state, torch.Tensor):
            s = state
        else:
            s = torch.Tensor(state.copy())[None, :].to(self.device)

        batch_size = s.shape[0]
        ep = 0.0 if greedy else self.ep_sched.get_epsilon(step)

        if random.random() < 1 - ep:
            q_values = self.q(s).detach()
            actions = torch.argmax(q_values, dim=1)
        else:
            actions = torch.randint(0, self.num_actions, (batch_size,), device=self.device)

        if batch_size == 1:
            return actions if return_tensor else int(actions.item())
        else:
            return actions if return_tensor else actions.cpu().numpy()

    def eval(self):
        self.q.eval()
        self.target_q.eval()

    def train(self):
        self.q.train()

    def _get_model_path(self, checkpoint_dir: str, step: int) -> str:
        return f"{checkpoint_dir}/step={step}"

    def load(self, checkpoint_dir: str, step: int) -> T.Dict[str, T.Any]:
        path = self._get_model_path(checkpoint_dir, step)
        state = torch.load(path, weights_only=True)
        self.q.load_state_dict(state)
        with open(os.path.join(path, "metrics_ddqn.json"), "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)

    def save(self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any]) -> None:
        path = self._get_model_path(checkpoint_dir, step)
        os.makedirs(path, exist_ok=True)
        torch.save(self.q.state_dict(), path + "/model.pt")
        with open(os.path.join(path, "metrics_ddqn.json"), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file)

    def compute_loss(
        self,
        step: int,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        reward_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.q(state_tensor)
        
        actions_long = action_tensor.long().view(-1).unsqueeze(1)

        q_values = q_values.to(self.device)
        actions_long = actions_long.to(self.device).long()

        q_values_selected = q_values.gather(dim=1, index=actions_long).squeeze(-1)

        next_q_values_online = self.q(next_state_tensor)
        next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

        next_q_values_target = self.target_q(next_state_tensor)
        next_q_value_selected = next_q_values_target.gather(1, next_actions).squeeze(1)

        expected_q_values = reward_tensor + self.gamma * next_q_value_selected
        expected_q_values = expected_q_values.squeeze(-1)

        loss = F.mse_loss(q_values_selected, expected_q_values)
        return loss

    def update_state(self, step, state, action, reward, done):
        self.step_counter += 1
        if step % self.num_steps_between_target_updates == 0:
            self.target_q.load_state_dict(self.q.state_dict())

    def learn_batch(self, step: int, batch) -> float:
        batch = next(iter(batch))
        states = batch["state"].to(self.device).float()
        actions = batch["action"].to(self.device).long()
        rewards = batch["reward"].to(self.device).float()
        next_states = batch["next_state"].to(self.device).float()

        loss = self.compute_loss(step, states, actions, rewards, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
