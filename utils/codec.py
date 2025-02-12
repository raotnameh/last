import dac
import torch


@torch.no_grad()
class Codec:
    def __init__(self, model_type="16khz"):
        self.model_path = dac.utils.download(model_type)
        self.model = dac.DAC.load(self.model_path)
               
    def encode(self, waveform, sample_rate=16000):
        # waveform [batch, 1, time]
        x = self.model.preprocess(waveform.unsqueeze(1), sample_rate=sample_rate)
        return self.model.encoder(x)
    
