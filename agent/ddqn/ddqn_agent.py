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
        eval_epsilon: float = None,
    ) -> T.Union[np_typing.NDArray, torch.Tensor, int]:
        if isinstance(state, torch.Tensor):
            s = state
        else:
            s = torch.Tensor(state.copy())[None, :].to(self.device)

        batch_size = s.shape[0]
        ep = 0.0 if greedy else self.ep_sched.get_epsilon(step)

        if eval_epsilon is not None:
            ep = eval_epsilon
        else:
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
        model_path = os.path.join(path, "model.pt")
        state = torch.load(model_path, map_location=self.device)
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
        done_tensor: torch.Tensor
    ) -> torch.Tensor:
        q_values = self.q(state_tensor)
        
        actions_long = action_tensor.long().view(-1, 1)

        q_values = q_values.to(self.device)
        actions_long = actions_long.to(self.device).long()

        q_values_selected = q_values.gather(dim=1, index=actions_long).squeeze(1)
        with torch.no_grad():
            next_q_values_online = self.q(next_state_tensor)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            next_q_values_target = self.target_q(next_state_tensor)
            next_q_value_selected = next_q_values_target.gather(1, next_actions).squeeze(1)
            next_q_value_selected = next_q_value_selected

        expected_q_values = reward_tensor + self.gamma * next_q_value_selected * (1.0 - done_tensor)
        # expected_q_values = expected_q_values.squeeze(-1)

        td_error = q_values_selected - expected_q_values
        td_error_np = td_error.detach().cpu().numpy()

        loss = F.mse_loss(q_values_selected, expected_q_values)
        return loss, td_error_np
    

    def save_training_state(self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any], replay_buffer) -> None:
        path = os.path.join(checkpoint_dir, "latest")
        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "step": step,
            "model_state_dict": self.q.state_dict(),
            "target_model_state_dict": self.target_q.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            # "replay_buffer": replay_buffer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    
    def load_training_state(self, checkpoint_dir: str, replay_buffer) -> T.Tuple[int, T.Dict[str, T.Any]]:
        path = os.path.join(checkpoint_dir, "latest", "checkpoint.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(checkpoint["model_state_dict"])
        self.target_q.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "replay_buffer" in checkpoint.keys():
            replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        return checkpoint["step"], checkpoint["metrics"]


    
    def update_state(self, step, state, action, reward, done):
        self.tau = 0.01
        self.step_counter += 1
        if step % self.num_steps_between_target_updates == 0:
            for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def learn_batch(self, step: int, batch, replay_buffer) -> float:

        self.train()                    # ⇦ switch to training mode
        total_loss, n_batches = 0.0, 0

        for minibatch in batch:
            states      = minibatch["state"].float().to(self.device)
            actions     = minibatch["action"].long().to(self.device)
            rewards     = minibatch["reward"].float().to(self.device)
            next_states = minibatch["next_state"].float().to(self.device)
            dones       = minibatch["done"].to(self.device).float()
            indices = minibatch["index"]

            loss, td_errors  = self.compute_loss(step, states, actions, rewards, next_states, dones)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            for idx, err in zip(indices, td_errors):
                replay_buffer.priorities[int(idx)] = abs(err) + 1e-3
            
            total_loss += loss.item()
            n_batches  += 1

        # Return the mean loss so callers can log it
        return total_loss / max(n_batches, 1)

