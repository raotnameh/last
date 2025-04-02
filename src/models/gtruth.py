import dac
import torch
import torch.nn as nn


class Gtruth(nn.Module):
    def __init__(self, model_type="16khz"):
        super().__init__() 
        self.model_path = dac.utils.download(model_type)
        self.model = dac.DAC.load(self.model_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def encode(self, waveform, sample_rate=16000):
        # waveform [batch, 1, time]
        x = self.model.preprocess(waveform, sample_rate=sample_rate)
        return self.model.encoder(x).transpose(1,2) # [b,t,c]
    
    def decode(self, z):
        # z, _, _, _, _ = self.model.quantizer(z)
        return self.model.decoder(z).squeeze(1) # [b,t]

if __name__ == '__main__':
    gtruth = Gtruth()
    waveform = torch.randn(1, 16000)
    z = gtruth.encode(waveform)
    print(z.shape)
    waveform_ = gtruth.decode(z)
    print(waveform_.shape)