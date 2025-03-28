import random
import typing as T

import numpy.typing as np_typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import tqdm

from base_agent import BaseAgent


class ConvNetBackbone(nn.Module):

    def __init__(
        self,
        input_shape: T.Tuple[int, int, int],
        conv_layer_channels: T.List[int],
        output_dim: int,
    ) -> None:
        super().__init__()

        conv_layers, batch_norms = [], []
        prev_in_channels = input_shape[-1]

        # Track the spatial dimensions after each conv layer
        height, width = input_shape[:2]

        for num_channels in conv_layer_channels:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=prev_in_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                )
            )
            batch_norms.append(nn.BatchNorm2d(num_channels))
            prev_in_channels = num_channels

            # Update spatial dimensions
            # For a Conv2d with kernel_size=3, padding=0, stride=1:
            # new_dim = old_dim - kernel_size + 1
            height = (height - 3 + 1) // 2
            width = (width - 3 + 1) // 2

        self.batch_norms = nn.ModuleList(batch_norms)
        self.conv_layers = nn.ModuleList(conv_layers)

        # Calculate the flattened tensor size
        flattened_size = conv_layer_channels[-1] * height * width
        self.fc = nn.Linear(
            in_features=flattened_size,
            out_features=output_dim,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # (batch_size, width, height, channels)
        x = state / 256
        # (batch_size, channels, width, height)
        x = x.permute(0, 3, 1, 2).float()
        for conv_layer, batch_norm in zip(self.conv_layers, self.batch_norms):
            x = conv_layer(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class Q(nn.Module):

    def build_reward_predictor(
        self, input_dim: int, hidden_layer_dims: T.List[int]
    ) -> nn.Module:
        layers = [
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_layer_dims[0],
            ),
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
        backbone_input_shape: T.Tuple[int, int, int],
        backbone_conv_channels: T.List[int],
        backbone_output_dim: int,
        reward_predictor_hidden_layer_dims: T.List[int],
        action_emb_table_size: int,
        action_emb_dim: int,
    ) -> None:
        super().__init__()

        self.state_backbone = ConvNetBackbone(
            backbone_input_shape,
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
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x_state = self.state_backbone(state)
        x_action = self.action_backbone(action).view(action.shape[0], -1)
        return self.reward_predictor(torch.cat([x_state, x_action], dim=-1))


class BasicQAgent(BaseAgent):

    def __init__(
        self, num_actions: int, q_params: T.Mapping[str, T.Any], lr: float = 1e-4
    ):
        self.q = Q(**q_params)
        self.optim = torch.optim.AdamW(self.q.parameters(), lr=lr)
        self.num_actions = num_actions
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def eval(self):
        self.q.eval()

    def train(self):
        self.q.train()

    def learn(self, train_data: DataLoader) -> None:
        self.train()
        for batch in tqdm.tqdm(train_data, total=len(train_data), desc="Training"):

            state, action, actual_reward = batch
            state = state.to(self.device)
            action = action.to(self.device)
            actual_reward = actual_reward.to(self.device)

            predicted_reward = self.q(state, action).flatten()
            loss = F.mse_loss(predicted_reward, actual_reward)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def act(self, state: np_typing.NDArray, ep: float = 0.0) -> int:
        s = torch.Tensor(state.copy())[None, :].to(self.device)
        as_ = [
            torch.Tensor([a])[None, :].int().to(self.device)
            for a in range(self.num_actions)
        ]
        if random.random() < 1 - ep:
            return int(np.argmax([self.q(s, a).item() for a in as_]))
        return random.choice(range(self.num_actions))
