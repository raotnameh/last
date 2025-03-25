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
    
class conv1d(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=256, kernel_size=9, stride=1):
        super().__init__()
        
        self.norm = torch.nn.LayerNorm(input_dim)
        padding = kernel_size // 2
        self.conv = torch.nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x): # B x T x C 
        x = self.norm(x)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        return x # B x T x C 
     
class Decoder(nn.Module):
    def __init__(self, input_dim, config, stride=2):
        super().__init__()
        
        
        self.pre = conv1d(input_dim, output_dim=config["transformer"]["decoder_hidden"])
        self.stride = stride
        
        self.decoder = models.Decoder(config)
        
        self.proj = Upsample(config["transformer"]["decoder_hidden"], stride=stride, output_dim=768)
        
        self.PostNet = layers.PostNet()

    
    def forward(self, x, mask): # b,t,c # mask should be b,t and 1 for masked position
        
        x = self.pre(x)
        dec_output, mask = self.decoder(x, mask.bool())
        print(mask)
        mask = F.interpolate(mask.unsqueeze(1).float(), scale_factor=self.stride, mode="nearest").squeeze(1)
        print(mask)
        dec_output = self.proj(dec_output)
        psot_dec_output = self.PostNet(dec_output)
        
        
        return dec_output, mask
    
if __name__ == '__main__':
    config = {
        "transformer": {
            "decoder_layer": 4,
            "decoder_head": 2,
            "decoder_hidden": 256,
            "conv_filter_size": 2048,
            "conv_kernel_size": [3, 1],
            "decoder_dropout": 0.2
        },
        "max_seq_len": 1000
    }
    
    model = Decoder(2048, config)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")
    x = torch.randn(2, 2, 2048)
    mask = torch.zeros(2, 2) 
    dec_output, mask = model(x, mask)
    print(dec_output)