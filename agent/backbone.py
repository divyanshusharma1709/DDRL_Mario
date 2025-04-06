import typing as T

import torch
from torch import nn
from torch.nn import functional as F


class ConvNetBackbone(nn.Module):

    def __init__(
        self,
        input_shape: T.Tuple[int, int, int],
        conv_layer_channels: T.List[int],
        output_dim: int,
        conv_kernel_size: int = 3,
        pooling_kernel_size: int = 2,
    ) -> None:
        super().__init__()

        conv_layers, batch_norms = [], []
        prev_in_channels = input_shape[-1]

        # Track the spatial dimensions after each conv layer
        height, width = input_shape[:2]

        self.pooling_kernel_size = pooling_kernel_size

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
            height = (height - conv_kernel_size + 1) // pooling_kernel_size
            width = (width - conv_kernel_size + 1) // pooling_kernel_size

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
            x = F.max_pool2d(x, kernel_size=self.pooling_kernel_size)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)
