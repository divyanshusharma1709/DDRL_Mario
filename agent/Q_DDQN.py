import torch
import torch.nn as nn

class Qv2(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Qv2, self).__init__()

        c, h, w = input_shape  # Example: (4, 84, 84)

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute fully connected size
        dummy_input = torch.zeros(1, c, h, w)
        fc_input_size = self.features(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state):
        x = self.features(state)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x  # shape: [batch_size, num_actions]
