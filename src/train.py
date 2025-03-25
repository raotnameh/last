import os
import numpy as np
import yaml

import warnings
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchaudio
from torch.utils.data import DataLoader

# step 00 :- Prepare the Hyperparameters
config = yaml.load( open("config/try1.yaml", "r"), Loader=yaml.FullLoader)

# step 01 :- Prepare the speech dataset.
from dataset_speech import Dataset_speech
# Get dataset
sdataset = Dataset_speech(input_manifest=config['dataset_speech']['path'], min_duration=32000, max_duration=250000)
sdataloader = DataLoader(sdataset, batch_size=32, shuffle=False, num_workers=6, collate_fn=sdataset.collate_fn)






exit()



from models.encoder import Encoder, Downsample
# Prepare the model
encoder = Encoder()
print(f"Encoder loaded successfully")
# encoder.freeze(layers=[10, 11])
# encoder.eval()
encoder.train()

downsample = Downsample(input_dim=encoder.cfg['model']['encoder_embed_dim'], output_dim=256, kernel_size=9, stride=2)
print(f"Downsample loaded successfully")
downsample.train()

for i, (waveforms, padding_masks) in enumerate(dataloader):
    encoder_out = encoder(waveforms, padding_masks)
    print(encoder_out['encoder_out'].shape)
    downsample_out = downsample(encoder_out['encoder_out'])
    print(downsample_out.shape)
    break