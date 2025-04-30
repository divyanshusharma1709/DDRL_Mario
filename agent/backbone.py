import typing as T

import torch
from torch import nn
from torch.nn import functional as F


class ConvNetBackbone(nn.Module):

    def __init__(
        self,
        input_shape: T.Tuple[int, int],
        channels_per_image: int,
        conv_layer_channels: T.List[int],
        output_dim: int,
        conv_kernel_size: int = 3,
        pooling_kernel_size: int = 2,
    ) -> None:
        super().__init__()

        conv_layers, batch_norms = [], []
        prev_in_channels = channels_per_image

        # Track the spatial dimensions after each conv layer
        height, width = input_shape[:2]

        self.pooling_kernel_size = pooling_kernel_size

        for i, num_channels in enumerate(conv_layer_channels):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=prev_in_channels,
                    out_channels=num_channels,
                    kernel_size=conv_kernel_size,
                )
            )
            batch_norms.append(nn.BatchNorm2d(num_channels))
            prev_in_channels = num_channels

            # Update spatial dimensions
            # For a Conv2d with kernel_size=3, padding=0, stride=1:
            # new_dim = old_dim - kernel_size + 1
            height = height - conv_kernel_size + 1
            width = width - conv_kernel_size + 1
            if i % 2 == 0:
                height //= pooling_kernel_size
                width //= pooling_kernel_size

        self.batch_norms = nn.ModuleList(batch_norms)
        self.conv_layers = nn.ModuleList(conv_layers)

        # Calculate the flattened tensor size
        flattened_size = conv_layer_channels[-1] * height * width
        self.fc = nn.Linear(
            in_features=flattened_size,
            out_features=output_dim,
        )

        print("Backbone param count:", sum(p.numel() for p in self.parameters()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state  # (batch_size, channels, width, height)
        for i, (conv_layer, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv_layer(x)
            x = batch_norm(x)
            x = F.relu(x)
            if i % 2 == 0:
                x = F.max_pool2d(x, kernel_size=self.pooling_kernel_size)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)
