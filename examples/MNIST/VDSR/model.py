import torch
import torch.nn as nn
import torch.optim as optim

class VDSR(nn.Module):
    def __init__(self, num_channels=1, num_filters=64, num_resblocks=20):
        super(VDSR, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.resblocks = self._make_resblocks(num_filters, num_resblocks)

        # Final convolution layer to map features to the output
        self.conv_final = nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1)

    def _make_resblocks(self, num_filters, num_resblocks):
        layers = []
        for _ in range(num_resblocks):
            layers.append(self._resblock(num_filters))
        return nn.Sequential(*layers)

    def _resblock(self, num_filters):
        """ A single residual block """
        block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        )
        return block

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        res = self.resblocks(x)
        x = res + residual
        x = self.conv_final(x)
        
        return x

#model = VDSR(num_channels=1, num_filters=64, num_resblocks=20) 
