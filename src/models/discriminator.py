import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = spectral_norm(nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            dilation=dilation, 
            padding=0,
            groups=groups,
        ))
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x, padding_mask=None):
        
        # x: (batch, time, channels)
        x = x.transpose(1, 2)
        # Apply causal (left) padding: (padding_left, padding_right)
        x = F.pad(x, (self.causal_padding, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        x = self.activation(x)
        
        x = x.masked_fill(padding_mask, 0)
        return x  

class Discriminator(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, num_layers=4, kernel_size=21, vocab_size=40):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(
            Conv1dBlock(in_channels, hidden_dim, kernel_size=1)
        )
        for i in range(num_layers):
            self.layers.append(Conv1dBlock(hidden_dim, hidden_dim, kernel_size=kernel_size))
        
        self.proj = spectral_norm(nn.Linear(hidden_dim, 1))
        
        self.lm = spectral_norm(nn.Linear(hidden_dim, vocab_size))
        
    def forward(self, x, padding_mask=None, labels=None):
        """
        x: (batch, time, channels)
        padding_mask: (batch, time, 1) where True indicates a padded timestep.
        """
        
        x = x.masked_fill(padding_mask, 0)
        x = self.layers[0](x, padding_mask)
        for layer in self.layers[1:]:
            x = x + layer(x, padding_mask)
        x = x.masked_fill(padding_mask, 0)
        
        # Compute mean pooling over valid timesteps
        valid_counts = (~padding_mask).sum(dim=1).float() # (batch, channels)
        x_mean = x.sum(dim=1) / valid_counts  # (batch, channels)
        # Apply the final projection
        x_mean = self.proj(x_mean) # (B, 1)
        x_mean = x_mean.squeeze(1)  # (B)
            
        if labels is not None: 
            targets = labels[:, 1:]  
            logits = self.lm(x[:, :-1, :])
            
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)   
            
            lm_loss = F.cross_entropy(logits, targets, ignore_index=0, reduction='mean') 
            return x_mean, lm_loss
        
        return x_mean

        
    

if __name__ == "__main__":
    # Test the Discriminator
    batch_size = 32
    seq_len = 1000
    channels = 2048
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -100:] = True

    x = torch.randn(batch_size, seq_len, channels)
    discriminator = Discriminator(in_channels=channels)
    print(discriminator)
    # calculate the parameters
    num_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Number of parameters: {num_params / 1e6}")
    out = discriminator(x, padding_mask)
    print(out.shape)  # (B,)