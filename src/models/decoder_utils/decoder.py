import torch
import torch.nn as nn
import Models as models
import Layers as layers
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim=256, kernel_size=9, stride=2):
        super().__init__()
        
        self.norm = torch.nn.LayerNorm(input_dim)
        self.conv = nn.ConvTranspose1d( input_dim, output_dim, kernel_size, stride=stride,padding=((kernel_size - 1)) // 2, output_padding=stride - 1)
          
    def forward(self, x): # B x T x C 
        x = self.norm(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        return x # B x T x C 
    
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, stride=1):
        super().__init__()
        
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            dilation=dilation, 
            padding=((kernel_size - 1) * dilation) // 2,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        

    def forward(self, x, padding_mask=None):
        # x is expected to be of shape (batch, time, channels)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
            
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        return x
     
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.decoder = models.Decoder(config)
        self.proj = Conv1dBlock( config["transformer"]["decoder_hidden"], config["transformer"]["dac_hidden"])
        
        self.PostNet = layers.PostNet(n_mel_channels=config["transformer"]["dac_hidden"])

    
    def forward(self, x, mask): # b,t,c # mask should be b,t and 1 for masked position
   
        dec_output, mask = self.decoder(x, mask.bool())
        dec_output = self.proj(dec_output)
        
        dec_output2 = self.PostNet(dec_output) + dec_output
        
        return dec_output, dec_output2, mask
    
if __name__ == '__main__':
    config = {
        "transformer": {
            "decoder_layer": 4,
            "decoder_head": 2,
            "decoder_hidden": 256,
            "conv_filter_size": 2048,
            "conv_kernel_size": [3, 1],
            "decoder_dropout": 0.2,
            "dac_hidden": 1024
        },
        "max_seq_len": 1000
    }
    
    model = Decoder(config)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")
    x = torch.randn(2, 2, 256)
    mask = torch.zeros(2, 2) 
    dec_output, mask = model(x, mask)
    print(dec_output)