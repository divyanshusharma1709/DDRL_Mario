import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as np_typing
import typing as T

from agent.base_agent import BaseAgent
from agent.dueling_dqn import DuelingQ as Q
from agent.ddqn.replay import ReplayBuffer

class DDQNAgent(BaseAgent):
    def __init__(self, num_actions, q_params, lr=2.5e-4, gamma=0.95, ep=0.05, target_update_freq=100):
        """
        Initializes the DDQN agent.
        """
        super().__init__(ep)
        self.gamma = gamma
        self.num_actions = num_actions
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=7000)
        self.batch_size = 32
        self.start_training_after = 1000
        self.train_every = 4
        self.step_counter = 0

        # Networks
        self.q = Q(**q_params).to(self.device)
        self.target_q = Q(**q_params).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optim = torch.optim.AdamW(self.q.parameters(), lr=lr)

    def learn_one_step(self, states, action, reward, next_states, done):
        """
        Stores transition and periodically trains.
        """
        self.replay_buffer.push(states, action, reward, next_states, done)
        self.step_counter += 1

        loss = None

        if len(self.replay_buffer) > self.start_training_after and self.step_counter % self.train_every == 0:
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = self.replay_buffer.sample(self.batch_size)

            # Stack frames and transpose (H, W, C) -> (C, H, W)
            states_batch = np.stack(states_batch)  # (batch_size, C, H, W)
            next_states_batch = np.stack(next_states_batch)  # (batch_size, C, H, W)


            # Convert to tensors
            states_batch = torch.tensor(states_batch, dtype=torch.float32, device=self.device)
            next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32, device=self.device)
            actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
            dones_batch = torch.tensor(dones_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

            # Current Q-values
            q_values = self.q(states_batch).gather(1, actions_batch)

            # Target Q-values (Double DQN)
            next_actions = self.q(next_states_batch).argmax(1, keepdim=True)
            next_q_values = self.target_q(next_states_batch).gather(1, next_actions)

            targets = rewards_batch + self.gamma * (1 - dones_batch) * next_q_values

            loss = F.mse_loss(q_values, targets.detach())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Update target network
            if self.step_counter % self.target_update_freq == 0:
                self.target_q.load_state_dict(self.q.state_dict())

        return loss.item() if loss is not None else None

    def act(
        self,
        states: T.Union[torch.Tensor, T.List[np_typing.NDArray]],
        greedy: bool = False,
        return_tensor: bool = False,
    ) -> T.Union[np_typing.NDArray, torch.Tensor, int]:
        """
        Chooses an action using epsilon-greedy policy.
        """
        if isinstance(states, torch.Tensor):
            s = states
        else:
            s = torch.Tensor(states.copy())[None, :].to(self.device)

        batch_size = s.shape[0]
        ep = 0.0 if greedy else self.ep

        if random.random() < 1 - ep:
            q_values = self.q(s).detach()
            result = torch.argmax(q_values, dim=-1)
        else:
            result = (
                torch.randint(low=0, high=self.num_actions, size=(batch_size,), device=self.device)
            )

        if batch_size == 1:
            return result if return_tensor else int(result[0])
        return result if return_tensor else result.cpu().numpy()

    def _get_model_path(self, checkpoint_dir, step):
        return f"{checkpoint_dir}/step={step}"

    def save(self, checkpoint_dir, step, metrics):
        """
        Saves the model parameters and training metrics.
        """
        path = self._get_model_path(checkpoint_dir, step)
        os.makedirs(path, exist_ok=True)
        torch.save(self.q.state_dict(), os.path.join(path, "model.pt"))
        with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f)

    def load(self, checkpoint_dir, step):
        """
        Loads the model parameters and returns the training metrics.
        """
        path = self._get_model_path(checkpoint_dir, step)
        state = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.q.load_state_dict(state)
        self.target_q.load_state_dict(state)
        with open(os.path.join(path, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics

    def train(self):
        self.q.train()

    def eval(self):
        self.q.eval()
