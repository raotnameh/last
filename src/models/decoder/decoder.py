import torch
import torch.nn as nn
import Models as models

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
        self.norm = nn.LayerNorm(out_channels)
        

    def forward(self, x):
        # x is expected to be of shape (batch, time, channels)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        return x
     
class Decoder(nn.Module): # fastspeech
    def __init__(self, config):
        super().__init__()
        
        # Global conditioning.
        self.use_s = config['speaker']['use_s']
        if self.use_s:
            self.speaker = Conv1dBlock( config["speaker"]["speaker_emb_dim"], config["transformer"]["decoder_hidden"])
        
        self.in_proj = Conv1dBlock( 768, config["transformer"]["decoder_hidden"])
        self.decoder = models.Decoder(config)
        self.out_proj = nn.Linear( config["transformer"]["decoder_hidden"], config["transformer"]["dac_hidden"])

    
    def forward(self, x, mask, spec, s=None, stride=1): # b,t,c # mask should be b,t and 1 for masked position and 0 for non-masked position # s is speaker embedding b,t',c
        if stride > 1: 
            x = x.repeat_interleave(stride, dim=1)  # Repeat each timestep s times
        
        x = x[:,:mask.shape[1]]
        x = self.in_proj(x)

        # speaker embedding
        if self.use_s:
            s = self.speaker(s) # b,t,c
            s = torch.mean(s, dim=1, keepdim=True) # b,1,c
            # Add speaker embedding with input at t dim
            x = x + s.expand(-1, x.size(1), -1) # b,t,c 
           
        dec_output, mask = self.decoder(x, mask)
        dec_output = self.out_proj(dec_output) # prediction
        
        return dec_output # b,t,c 
    
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