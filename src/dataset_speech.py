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
    def __init__(self, input_manifest, split = "train", min_duration=0, max_duration=float("inf"), CACHE_DIR="gtruth_cache"):
        super().__init__()
        
        paths = []
        min_dur, max_dur, tot_dur = min_duration, 0, 0
        with open(input_manifest, "r") as infile:
            root_dir = infile.readline().strip()  # First line is the root directory
            
            for line in infile:
                file_name, duration = line.strip().split("\t")
                duration = int(duration)
                
                if min_duration <= duration <= max_duration:
                    path = os.path.join(root_dir, file_name)
                    paths.append([path, duration, None])  # None for gtruth features
                    
                    min_dur = min(min_dur, duration)
                    max_dur = max(max_dur, duration)
                    tot_dur += duration
                    
        logging.info(f"Speech dataset duration range in seconds: {min_dur/16000:.2f} - {max_dur/16000:.2f} | Total duration in hours: {tot_dur/16000/3600:.2f}")
        # Sort by duration
        paths.sort(key=lambda x: x[1])
        self.paths = paths
        # self.paths = paths[:2]  # For testing
        # print(f"Testing Mode: Using only {len(self.paths)} samples")
        
        
        # Do the caching of the gtruth features    
        logging.info("Caching gtruth features...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.model = Gtruth()
        self.model.to("cuda")
        
        @torch.no_grad()
        def load_or_compute_gt_from_path( waveform, model, path):
            filename = os.path.basename(path).replace("/", "_").replace("\\", "_")
            cache_path = os.path.join(CACHE_DIR, f"{filename}.pt")
            
            if os.path.exists(cache_path):
                pass
            else:
                gt = model.encode(waveform.unsqueeze(0).unsqueeze(1))  # [1, T_enc, 1024]
                pad_sequence([gt.squeeze(0)], batch_first=True)
                torch.save(gt.squeeze(0), cache_path)
            return cache_path

        for path in tqdm(self.paths):
            p, _, _ = path
            waveform, _ = sf.read(p)
            waveform = torch.from_numpy(waveform).float().to("cuda")
            cache_path = load_or_compute_gt_from_path(waveform, self.model, p)
            path[-1] = cache_path      
            
        del self.model
        torch.cuda.empty_cache()
        logging.info("Gtruth features cached successfully.") 


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path, duration, gt_cache_path = self.paths[idx]
        
        waveform, sample_rate = sf.read(path)
        assert sample_rate == 16000, "Sampling rate must be 16000"
        waveform = torch.from_numpy(waveform).float()
        
        gt = torch.load(gt_cache_path, map_location=waveform.device) 
        
        return waveform, duration, path, gt # (seq_len), (duration)
    
    # collate function to pad the waveforms to the same length wrt the maximum duration
    def collate_fn(self, batches):
        max_dur = max(batch[1] for batch in batches)
        dur = [batch[1] / 16000.0 for batch in batches]
        
        gt_list = [batch[3] for batch in batches]
        gt = pad_sequence(gt_list, batch_first=True, padding_value=0)
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
        paths = [batch[2] for batch in batches]
        return torch.stack(waveforms), torch.stack(padding_masks), paths, gt, dur # (batch_size, max_dur), (batch_size, max_dur), list of paths, (batch_size, max_dur, 1024)


if __name__ == "__main__":
    # Create the dataset and dataloader
    input_manifest = "/raid/home/rajivratn/hemant_rajivratn/librispeech/data/manifest/train-clean-100.tsv"
    input_manifest = "/raid/home/rajivratn/hemant_rajivratn/last/data/ljspeechmanifest.tsv"
    
    dataset = Dataset_speech(input_manifest=input_manifest, min_duration=32000, max_duration=250000)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i, (waveforms, padding_masks) in enumerate(dataloader):
        # print(waveforms.shape, padding_masks)
        break