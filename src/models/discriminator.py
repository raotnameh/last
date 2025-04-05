import torch
import torch.nn as nn

# torch.nn.LayerNorm(input_dim)

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(in_channels)  # Normalize over channel dim
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            dilation=dilation, 
            padding=((kernel_size - 1) * dilation) // 2,
            groups=groups,
        )
        

    def forward(self, x, padding_mask=None):
        # x is expected to be of shape (batch, time, channels)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0)
        
        x = self.layernorm(x)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        return x

class StatisticalPooling(nn.Module):
    def __init__(self):
        super(StatisticalPooling, self).__init__()

    def forward(self, x, padding_mask):
        # Remove padded positions
        x = x.masked_fill(padding_mask, 0)
        
        # Compute statistics
        x_mean = x.mean(dim=1)  # Mean along time axis
        x_max = x.max(dim=1)[0]  # Max along time axis
        x_var = x.var(dim=1)  # Variance along time axis

        # You can add more statistics like min, std, etc.
        return torch.cat([x_mean, x_max, x_var], dim=1)  # Concatenate the statistics


class Discriminator(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, kernel_size=11, groups=1):
        super().__init__()
        
        self.mask_token_idx = 1
        self.mask_prob = 0.25
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, 256)) if self.mask_token_idx else None

        
        self.pre = nn.ModuleList([
            Conv1dBlock(in_channels, hidden_dim, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Dropout(0.1),
        ])
        self.disc_layers = nn.ModuleList([
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size),
            nn.GELU(),
            nn.Dropout(0.1),
            Conv1dBlock(hidden_dim, hidden_dim, kernel_size),
            nn.GELU(),
            nn.Dropout(0.1),
        ])
        
        self.pooling = StatisticalPooling()
        
        self.proj = Conv1dBlock(hidden_dim, 1, kernel_size)

    def forward(self, x, padding_mask=None):
        """
        x: (batch, time, channels)
        padding_mask: (batch, time, 1) where True indicates a padded timestep.
        """
        
        for layer in self.pre:
            if isinstance(layer, Conv1dBlock):
                x = layer(x, padding_mask)  # Pass padding_mask only to Conv1dBlock
            else:
                x = layer(x)  # GELU & Dropout don't need padding_mask
                
        if self.mask_token_idx:
            batch_size, seq_len, channels = x.shape
            
            # Step 1: Create mask with `mask_prob` probability
            rand = torch.rand(batch_size, seq_len, device=x.device)
            mask = rand < self.mask_prob

            # Step 2: Exclude positions that are padded
            mask = mask & ~padding_mask.squeeze(-1)
            
            # Step 3: Apply masking: replace with masked_embedding
            x = torch.where(mask.unsqueeze(-1), self.masked_embedding.expand(batch_size, seq_len, channels), x)


        for layer in self.disc_layers:
            if isinstance(layer, Conv1dBlock):
                x = layer(x, padding_mask)  # Pass padding_mask only to Conv1dBlock
            else:
                x = layer(x)  # GELU & Dropout don't need padding_mask
        x = x.masked_fill(padding_mask, 0)
        
        # Compute mean pooling over valid timesteps
        valid_counts = (~padding_mask).sum(dim=1).clamp(min=1).float()
        x_mean = x.sum(dim=1) / valid_counts  # (batch, channels)
        x_mean = x_mean.unsqueeze(1) # (batch, 1, channels)
        
        x_mean = self.proj(x_mean) # (batch, 1, 1)
        
        return x_mean.squeeze(1).squeeze(1)  # (batch,)
    

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