import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from torch.nn.utils import weight_norm

class Encoder(torch.nn.Module):
    def __init__(self,):
        super().__init__()

        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

        
    def forward(self, waveform):
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.input_values.squeeze(0)
        
        outputs = self.model(inputs)
        encoder = outputs.last_hidden_state.contiguous()
        encoder = encoder.transpose(1, 2)
        return encoder
    

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.Seqeuntial(
                                nn.LayerNorm(in_channels), 
                                nn.Conv1d( in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2 )
                            )

    def forward(self, x):
        return self.conv(x)
    
    
class Downsampling(nn.Module):
    def __init__(self,):
        super().__init__()
        self.downsampling = nn.Sequential(
            Conv1d(768, 768, 7, dilation=1, stride=2), 
            nn.Dropout(0.1),
        )
        
        
    def forward(self, x):
        x = self.downsampling(x)
        return x