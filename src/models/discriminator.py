import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        
        self.causal_padding = (kernel_size - 1) * dilation
        self.layernorm = nn.LayerNorm(in_channels)  # Normalize over channel dim
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            dilation=dilation, 
            padding=0,
            groups=groups,
        )
        

    def forward(self, x, padding_mask=None):
        
        # x is expected to be of shape (batch, time, channels)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0)
        
        x = self.layernorm(x)
        
        x = x.transpose(1, 2)
        # Apply causal (left) padding: (padding_left, padding_right)
        x = F.pad(x, (self.causal_padding, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        return x  

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()

        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model//2)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # register as buffer (not a parameter)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]
    
class CausalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=2048)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        

    def forward(self, x, padding_mask=None):
        """
        x: (batch, seq_len, d_model)
        padding_mask: (batch, seq_len) - True for padding tokens
        """
        bsz, seq_len, _ = x.size()
        device = x.device

        # Add sinusoidal positional encoding
        x = self.pos_enc(x)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(
                src=x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )

        return x 


class Discriminator(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, num_layers=4):
        super().__init__()

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        self.pre = Conv1dBlock(in_channels, hidden_dim, kernel_size=1)
        
        self.norm = nn.LayerNorm(hidden_dim)  # Normalize over channel dim
        self.decoder = CausalTransformer(d_model=hidden_dim, nhead=8, num_layers=num_layers, dim_feedforward=hidden_dim*4, dropout=0.1)
        self.proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, padding_mask=None):
        """
        x: (batch, time, channels)
        padding_mask: (batch, time, 1) where True indicates a padded timestep.
        """
        
        x = self.pre(x, padding_mask) 
        
        x = self.norm(x)
        
        x = self.decoder(x, padding_mask.squeeze(-1))  # (batch, time, hidden_dim), (batch, time)
        
        # Extract the valid last timestep for each sequence in the batch
        B, T, C = x.size()
        padding_mask = padding_mask.squeeze(-1)
        time_indices = torch.arange(T).view(1, T, 1).expand(B, T, C).to(x.device)  # (1, T, 1)
        expanded_mask = padding_mask.unsqueeze(-1).expand(B, T, C)
        masked_indices = time_indices.masked_fill(expanded_mask, -1)
        last_valid_t = masked_indices.max(dim=1).values
        gather_index = last_valid_t.unsqueeze(1)  # (B, 1, C)
        x = torch.gather(x, dim=1, index=gather_index)  # (B, 1, C)
        x = x.squeeze(1)  # (B, C)
        
        # Apply the final projection
        x = self.proj(x)
        x = x.squeeze(-1)  # (B,)
        
        return x
        
    

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