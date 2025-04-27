import collections
import json
import os
import random
import typing as T
import tqdm

import numpy.typing as np_typing
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from agent.base_agent import BaseAgent
from agent.backbone import ConvNetBackbone


class Q(nn.Module):

    def build_reward_predictor(self, input_dim: int, hidden_layer_dims: T.List[int]) -> nn.Module:
        layers = [
            nn.Linear(in_features=input_dim, out_features=hidden_layer_dims[0]),
            nn.ReLU(),
        ]
        for h1, h2 in zip(hidden_layer_dims, hidden_layer_dims[1:]):
            layers.append(nn.Linear(in_features=h1, out_features=h2))
            layers.append(nn.ReLU())
        layers += [
            nn.Linear(
                in_features=hidden_layer_dims[-1],
                out_features=1,
            )
        ]
        return nn.Sequential(*layers)

    def __init__(
        self,
        backbone_input_shape: T.Tuple[int, int],
        backbone_channels_per_image: int,
        backbone_conv_channels: T.List[int],
        backbone_output_dim: int,
        reward_predictor_hidden_layer_dims: T.List[int],
        action_emb_table_size: int,
        action_emb_dim: int,
        device: str,
    ) -> None:
        super().__init__()

        self.state_backbone = ConvNetBackbone(
            backbone_input_shape,
            backbone_channels_per_image,
            backbone_conv_channels,
            backbone_output_dim,
        )
        self.action_backbone = nn.Embedding(
            num_embeddings=action_emb_table_size,
            embedding_dim=action_emb_dim,
        )
        self.reward_predictor = self.build_reward_predictor(
            backbone_output_dim + action_emb_dim,
            reward_predictor_hidden_layer_dims,
        )

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x_state = self.state_backbone(state)
        x_action = self.action_backbone(action).view(action.shape[0], -1)
        return self.reward_predictor(torch.cat([x_state, x_action], dim=-1))


class BasicQAgent(BaseAgent):

    def __init__(
        self,
        num_actions: int,
        q_params: T.Dict[str, T.Any],
        ep_sched_params: T.Dict[str, T.Any],
        lr: float = 1e-4,
        gamma: float = 0.95,
        num_steps_between_target_updates: int = 1,
    ):
        super().__init__(ep_sched_params)

        self.gamma = gamma
        self.q = Q(device=self.device, **q_params)
        self.q_target = Q(device=self.device, **q_params)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.AdamW(self.q.parameters(), lr=lr)
        self.num_actions = num_actions
        self.target_network_update_steps = num_steps_between_target_updates

    def a2t(self, action: int, batch_size: int) -> torch.Tensor:
        return torch.Tensor([action])[None, :].int().repeat(batch_size, 1).to(self.device)

    def eval(self):
        self.q.eval()

    def train(self):
        self.q.train()

    def learn_one_step(
        self,
        step: int,
        state: np_typing.NDArray,
        action: int,
        reward: float,
        next_state: np_typing.NDArray,
        done: bool,
        update_target_network: bool = True,
    ) -> float:
        self.train()

        state_tensor, action_tensor, reward_tensor, next_state_tensor = self.tensorize(
            state, action, reward, next_state
        )

        loss = self.compute_loss(
            step, state_tensor, action_tensor, reward_tensor, next_state_tensor
        )

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if update_target_network and step % self.target_network_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss.item()

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
        bootstrapped_future_reward = (
            self.gamma * self.q_target(next_state_tensor, greedy_action).detach()
        )
        target = reward_tensor.view(-1, 1) + bootstrapped_future_reward
        loss = F.mse_loss(predicted_reward, target)
        return loss

    def learn_batch(self, step: int, dataloader: DataLoader) -> float:
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
            # compute the Q values for each action from the input state
            action_rewards = [
                self.q(s, self.a2t(a, batch_size)).detach() for a in range(self.num_actions)
            ]
            stacked_action_rewards = torch.cat(action_rewards, dim=-1)
            # compute the argmax over actions
            result = torch.argmax(stacked_action_rewards, dim=-1)
        else:
            result = (
                torch.Tensor(np.random.choice(self.num_actions, batch_size).reshape(-1, 1))
                .int()
                .to(self.device)
            )

        if batch_size == 1:
            return result if return_tensor else int(result[0])

        return result if return_tensor else result.cpu().numpy()

    def _get_model_path(self, checkpoint_dir: str, step: int) -> str:
        return f"{checkpoint_dir}/step={step}"

    def load(self, checkpoint_dir: str, step: int) -> T.Dict[str, T.Any]:
        path = self._get_model_path(checkpoint_dir, step)
        state = torch.load(path, weights_only=True)
        self.q.load_state_dict(state)
        with open(os.path.join(path, "metrics.json"), "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)

    def save(self, checkpoint_dir: str, step: int, metrics: T.Dict[str, T.Any]) -> None:
        path = self._get_model_path(checkpoint_dir, step)
        os.makedirs(path, exist_ok=True)
        torch.save(self.q.state_dict(), path + "/model.pt")
        with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file)
