import os
import logging


import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.gtruth import Gtruth
from tqdm import tqdm


class Dataset_speech(Dataset):
    def __init__(self, input_manifest, min_duration=0, max_duration=float("inf")):
        super().__init__()
        
        
        self.max_duration = max_duration
        
        paths = []
        min_dur, max_dur, tot_dur = float('inf'), 0, 0
        with open(input_manifest, "r") as infile:
            
            for line in infile:
                path, duration, txt = line.strip().split("\t")
                duration = int(duration)
                
                if min_duration <= duration <= max_duration:
                    paths.append([path, duration, txt])  
                    
                    min_dur = min(min_dur, duration)
                    max_dur = max(max_dur, duration)
                    tot_dur += duration
                    
        # Sort by duration
        paths.sort(key=lambda x: x[1])
        self.paths = paths
        # self.paths = paths[:2]  # For testing
        # print(f"Testing Mode: Using only {len(self.paths)} samples")
        
        logging.info(f"Speech dataset duration range in seconds: {min_dur/16000:.2f} - {max_dur/16000:.2f} | Total duration in hours: {tot_dur/16000/3600:.2f}")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path, duration, txt = self.paths[idx]
        
        waveform, sample_rate = sf.read(path)
        assert sample_rate == 16000, "Sampling rate must be 16000"
        waveform = torch.from_numpy(waveform).float()
        
        return waveform, duration, path, txt # (seq_len), (duration)
    
    # collate function to pad the waveforms to the same length wrt the maximum duration
    def collate_fn(self, batches):
        max_dur = max(batch[1] for batch in batches)
        
        waveforms = []
        padding_masks = []
        for batch in batches:
            pad_len = max_dur - batch[1]
            
            padded_waveform = F.pad(batch[0], (0, pad_len))
            padding_mask = torch.cat([
                                torch.zeros(batch[1], dtype=torch.bool), 
                                torch.ones(pad_len, dtype=torch.bool)
                            ])
            
            waveforms.append(padded_waveform)
            padding_masks.append(padding_mask) # 1 for masked position and 0 for non-masked position
        
        dur = [batch[1] / 16000.0 for batch in batches] # convert to seconds
        paths = [batch[2] for batch in batches]
        txts = [batch[3] for batch in batches]
        
        return torch.stack(waveforms), torch.stack(padding_masks), dur,  paths, txts # (batch_size, max_dur), (batch_size, max_dur), list of paths, (batch_size, max_dur, 1024)


if __name__ == "__main__":
    # Create the dataset and dataloader
    input_manifest = "/raid/home/rajivratn/hemant_rajivratn/librispeech/data/manifest/train-clean-100.tsv"
    input_manifest = "/raid/home/rajivratn/hemant_rajivratn/last/data/ljspeechmanifest.tsv"
    
    dataset = Dataset_speech(input_manifest=input_manifest, min_duration=32000, max_duration=250000)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i, (waveforms, padding_masks) in enumerate(dataloader):
        # print(waveforms.shape, padding_masks)
        break