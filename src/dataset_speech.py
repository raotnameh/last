import os

import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Dataset_speech(Dataset):
    def __init__(self, input_manifest, min_duration=0, max_duration=float("inf")):
        super().__init__()

        paths = []
        min_dur, max_dur, tot_dur = min_duration, max_duration, 0
        with open(input_manifest, "r") as infile:
            root_dir = infile.readline().strip()  # First line is the root directory
            
            for line in infile:
                file_name, duration = line.strip().split("\t")
                duration = int(duration)
                
                if min_duration <= duration <= max_duration:
                    path = os.path.join(root_dir, file_name)
                    paths.append((path, duration))
                    
                    min_dur = min(min_dur, duration)
                    max_dur = max(max_dur, duration)
                    tot_dur += duration
                    
        print(f"Speech dataset duration range in seconds: {min_dur/16000:.2f} - {max_dur/16000:.2f} | Total duration in hours: {tot_dur/16000/3600:.2f}")
        # Sort by duration
        paths.sort(key=lambda x: x[1])
        self.paths = paths
        # self.paths = paths[:32]  # For testing
        # print(f"Testing Mode: Using only {len(self.paths)} samples")
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path, duration = self.paths[idx]
        waveform, sample_rate = sf.read(path)
        assert sample_rate == 16000, "Sampling rate must be 16000"
        
        waveform = torch.from_numpy(waveform).float()
        return waveform, duration # (seq_len), (duration)
    
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
        
        return torch.stack(waveforms), torch.stack(padding_masks) # (bsz, seq_len) for waveforms and padding_masks


if __name__ == "__main__":
    # Create the dataset and dataloader
    input_manifest = "/raid/home/rajivratn/hemant_rajivratn/librispeech/data/manifest/train-clean-100.tsv"
    
    dataset = Dataset_speech(input_manifest=input_manifest, min_duration=32000, max_duration=250000)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i, (waveforms, padding_masks) in enumerate(dataloader):
        print(waveforms.shape, padding_masks)
        break