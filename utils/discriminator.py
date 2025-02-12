import torch.nn as nn

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d( in_channels, out_channels, kernel_size, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2 )

    def forward(self, x):
        return self.conv(x)
