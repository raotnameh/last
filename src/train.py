import os, sys
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
sdataset = Dataset_speech(input_manifest=config['dataset_speech']['path'], min_duration=32000, max_duration=320000)
sdataloader = DataLoader(sdataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=sdataset.collate_fn)
# Create an iterator from the dataloader
siter_data = iter(sdataloader)

# step 02 :- Prepare the text dataset.
from dataset_txt import Dataset_txt    
tdataset = Dataset_txt(data=config['dataset_txt']['path'])
vocab = tdataset.vocab
tdataloader = DataLoader(tdataset, batch_size=32, shuffle=True, collate_fn=tdataset.collate_fn)
# Create an iterator from the dataloader
titer_data = iter(tdataloader)



# step 03 :- Prepare the codebook
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models.codebook import Codebook
codebook = Codebook(vocab)
print(f"Size of codebook: {codebook.embedding.weight.shape[0]} x {codebook.embedding.weight.shape[1]}")

# step 11 :- Prepare the Encoder
from models.encoder import Encoder
encoder = Encoder(config['encoder']['ckpt_path'])
# step 12 :- Prepare the gloabl encoder

# step 2 :- Prepare the Downsample
from models.encoder import Downsample
downsample = Downsample(input_dim=encoder.cfg['model']['encoder_embed_dim'], output_dim=codebook.embedding.weight.shape[1], kernel_size=config['downsample']['kernel_size'], stride=config['downsample']['stride'])

# step 3 :- Prepare the quantizer
from models.quantizer import Quantizer
quantizer = Quantizer(config['quantizer']['beta'])

# step 4 :- Prepare the upsampler
sys.path.append(f"{os.getcwd()}/models/decoder_utils")
from models.decoder_utils.decoder import Upsample
upsample = Upsample(codebook.embedding.weight.shape[1], output_dim=config["decoder"]["transformer"]["decoder_hidden"], stride=config['upsample']['stride'])

# step 5 :- Prepare the decoder
from models.decoder_utils.decoder import Decoder
decoder = Decoder(config['decoder'])

# step 6 :- Prepare the discriminator
from models.discriminator import Discriminator
discriminator = Discriminator(codebook.embedding.weight.shape[1])





# ========================
# Training Setup
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to device
encoder = encoder.to(device)
downsample = downsample.to(device)
quantizer = quantizer.to(device)
upsample = upsample.to(device)
decoder = decoder.to(device)
discriminator = discriminator.to(device)

codebook = codebook.to(device)
# Freeze codebook embeddings (assuming they should remain fixed)
codebook.embedding.weight.requires_grad = False





# ========================
# Optimizers and Hyperparameters
# ========================
lr_gen = config['train']['lr']
lr_disc = config['train']['lr_disc']
num_steps = config['train']['num_steps']

# Group the generator parameters: encoder, downsample, quantizer, upsample, decoder.
gen_params = list(downsample.parameters()) + list(upsample.parameters()) + list(decoder.parameters())

optimizer_gen = optim.Adam(gen_params, lr=lr_gen)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)





# ========================
# Training Loop
# ========================
for epoch in range(num_steps):
    encoder.eval()
    downsample.train()
    upsample.train()
    decoder.train()
    discriminator.train()
    
    try: 
        batch = next(siter_data)
    except StopIteration:
        siter_data = iter(sdataloader)  # Reinitialize iterator
        batch = next(siter_data)  # Fetch the first batch again
    
    optimizer_gen.zero_grad()
    
    waveforms, padding_masks = batch
    waveforms = waveforms.to(device) # [B, T]
    padding_masks = padding_masks.to(device) # [B, T]
    
    # ===== Generator Forward Pass =====
    with torch.no_grad():
        enc_out = encoder(waveforms, padding_masks)  # [B, T, C]
    down_out = downsample(enc_out['encoder_out'])  # [B, T // 2, C]
    commitment_loss, z_q, encoding_indices = quantizer(down_out, codebook) # [B, T // 2, C], [B, T // 2, C], [B, T // 2]
    up_out = upsample(z_q)[:,:enc_out['encoder_out'].shape[1],:] # [B, T, C]
    dec_out, dec_out2, mask = decoder(up_out, padding_masks) # [B, T, C], [B, T, C], [B, T]
    
    pred_fake = discriminator(z_q) # discriminator fake output
    
    # loss_gen = loss_Gen(dec_out, dec_out2, waveforms, mask, pred_fake, z_q, commitment_loss)
    # ===== Loss Computation =====
    # 1. Reconstruction loss (dec_out with dac latent and dac_out2 with dac latent)
    # 2. Quantization loss (commitment loss)
    # 3. Smoothness penalty (z_q)
    # discriminator loss (pred_fake)
    # Total generator loss: reconstruction + quantization + smoothness + discriminator.

    # loss_gen.backward()
    # optimizer_gen.step()
    
    
    # ===== Discriminator Forward Pass =====
    try:
        tbatch = next(titer_data)
    except StopIteration:
        iter_data = iter(tdataloader)  # Reinitialize iterator
        tbatch = next(titer_data)  # Fetch the first batch again
    


































