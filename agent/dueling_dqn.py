import torch
import torch.nn as nn

class DuelingQ(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingQ, self).__init__()
        
        c, h, w = input_shape  # Channels, Height, Width

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate size after convolutions
        dummy_input = torch.zeros(1, c, h, w)
        conv_out_size = self.features(dummy_input).view(1, -1).size(1)

        # Dueling heads
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Outputs a single value V(s)
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  # Outputs advantage A(s, a)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        # Combine value and advantage into Q-values
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
