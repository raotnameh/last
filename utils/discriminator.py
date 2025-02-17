import torch.nn as nn
from torch.nn.utils import weight_norm


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Seqeuntial(
                                nn.LayerNorm(in_channels), 
                                nn.Conv1d( in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2 ),
                            )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=768, hidden_dim=256, kernel_size=7):
        super(Discriminator, self).__init__()
        
        # First convolutional layer: from input channels to hidden_dim filters.
        self.disc = nn.Sequential(
                Conv1d( in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size), 
                nn.GELU(),
                Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size),
                nn.GELU(),                    
                Conv1d( in_channels=hidden_dim, out_channels=1, kernel_size=kernel_size)
            )

    def forward(self, x, padding_mask=None):
        """
        x: Tensor of shape (B, in_channels, T) 
        padding_mask: Optional boolean Tensor of shape (B, T) where True indicates a padded (invalid) timestep.
        """
        
        # Pass input through the conv layers.
        x = self.disc(x) # (B, 1, T)

        # If a padding mask is provided, ignore padded positions when pooling.
        mask = padding_mask.bool().unsqueeze(1)  # (B, 1, T)
        # Set padded positions to 0 so they do not contribute to the sum.
        x = x.masked_fill(mask, 0)
        # Count the number of valid (non-padded) timesteps for each sample.
        valid_counts = (~padding_mask).sum(dim=1).unsqueeze(1).clamp(min=1).float()  # (B, 1)

        # Compute the sum over the time dimension.
        x_sum = x.sum(dim=-1)  # (B, 1)
        # Mean pooling: divide by the count of valid timesteps per sample.
        x_mean = x_sum / valid_counts  # (B, 1)
        # Remove the singleton channel dimension.
        x_mean = x_mean.squeeze(1)  # (B,)

        return x_mean
