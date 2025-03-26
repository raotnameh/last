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

# step 04 :- Prepare the ground truth using dac codec
from models.gtruth import Gtruth
gtruth = Gtruth()

# step 1 :- Prepare the Encoder
from models.encoder import Encoder
encoder = Encoder(config['encoder']['ckpt_path'])
# step 12 :- Prepare the gloabl encoder

# step 2 :- Prepare the Downsample
from models.encoder import Downsample
downsample = Downsample(input_dim=encoder.cfg['model']['encoder_embed_dim'], output_dim=codebook.embedding.weight.shape[1], kernel_size=config['downsample']['kernel_size'], stride=config['downsample']['stride'])

# step 3 :- Prepare the quantizer
from models.quantizer import Quantizer
quantizer = Quantizer(codebook.embedding.weight.shape[1])

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
gtruth = gtruth.to(device)
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
# Losses
# ========================
from loss import Loss
loss = Loss(config)



# ========================
# Training Loop
# ========================
for epoch in range(num_steps):
    encoder.eval()
    downsample.train()
    upsample.train()
    decoder.train()
    discriminator.train()
    
    for iter, batch in enumerate(sdataloader):
        disc = False
        output = {}
        
        # ===== Data Preparation =====
        waveforms, padding_masks = batch
        waveforms = waveforms.to(device) # [B, T]
        padding_masks = padding_masks.to(device) # [B, T] 1 (true) for masked, 0(false) for not masked means [0,0,0,1,1]
        
        # ===== Generator Forward Pass =====
        with torch.no_grad():
            enc_out = encoder(waveforms, padding_masks)  # [B, T, C] # step 1
        padding_masks = ~enc_out['padding_mask'] # [B, T // 320] # 0 for masked, 1 for not masked
        output["cnn_out"] = enc_out['cnn_out']
        output['encoder_out'] = enc_out['encoder_out']
        output['padding_mask'] = padding_masks
        
        
        with torch.no_grad():
            gt = gtruth.encode(waveforms.unsqueeze(1)) # [B, T, 1024] # step 04
            gt = gt[:,:padding_masks.shape[-1],:]
            gt = gt * padding_masks.unsqueeze(-1).float() # [B, T, 1024]
        output['gt'] = gt    
        
        down_out = downsample(enc_out['encoder_out'])  # [B, T // 2, C] # step 2
        dpadding_masks = padding_masks[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride']] 
        down_out = down_out[:,:dpadding_masks.shape[-1],:] # [B, T // 2, C]
        down_out = down_out * dpadding_masks.unsqueeze(-1).float() # [B, T // 2, C]
        output['down_out'] = down_out
        output['dpadding_masks'] = dpadding_masks
        
        commitment_loss, z_q, encoding_indices = quantizer(down_out, codebook) # [B, T // 2, C], [B, T // 2, C], [B, T // 2] # step 3
        z_q = z_q[:,:dpadding_masks.shape[-1],:] # [B, T // 2, C]
        z_q = z_q * dpadding_masks.unsqueeze(-1).float() # [B, T // 2, C]
        encoding_indices = encoding_indices[:,:dpadding_masks.shape[-1]] # [B, T // 2]
        encoding_indices = encoding_indices * dpadding_masks # [B, T // 2]
        output['commitment_loss'] = commitment_loss
        output['z_q'] = z_q
        output['encoding_indices'] = encoding_indices
        
        up_out = upsample(z_q)[:,:enc_out['padding_mask'].shape[1],:] # [B, T, C] # step 4
        up_out = up_out * enc_out['padding_mask'].float().unsqueeze(-1) # [B, T, C]       
        output['up_out'] = up_out 

        dec_out, dec_out2, mask = decoder(up_out, ~padding_masks) # [B, T, C], [B, T, C], [B, T] # step 5
        dec_out = dec_out[:,:padding_masks.shape[-1],:] # [B, T, C]
        dec_out = dec_out * padding_masks.unsqueeze(-1).float() # [B, T, C]
        dec_out2 = dec_out2[:,:padding_masks.shape[-1],:] # [B, T, C]
        dec_out2 = dec_out2 * padding_masks.unsqueeze(-1).float() # [B, T, C]
        output['dec_out'] = dec_out
        output['dec_out2'] = dec_out2
        
        
        # ===== Discriminator Forward Pass =====
        # if iter % 2 == 0:
        #     disc = True
        #     pred_fake = discriminator(z_q, ~dpadding_masks) # discriminator fake output # step 6
        #     print(pred_fake.shape)
        #     output['pred_fake'] = pred_fake
            
        #     try:
        #         tbatch = next(titer_data)
        #     except:
        #         iter_data = iter(tdataloader)  # Reinitialize iterator
        #         tbatch = next(titer_data)  # Fetch the first batch again
            
        #     text, mask = tbatch
        #     text = text.to(device)
        #     mask = mask.to(device)
        #     text = codebook(text)
        #     pred_real = discriminator(text, mask) # discriminator real output # step 6
        #     output['pred_real'] = pred_real

        # ===== Loss Computation =====
        total_loss = loss.step(output, disc)
        
        # ===== Backward Pass ===== 
        optimizer_gen.zero_grad()
        total_loss.backward()
        optimizer_gen.step()
        
        print(total_loss.item())
        
        
        
        # optimizer_gen.zero_grad()
        # loss_gen.backward()
        # optimizer_gen.step()
        
        
        

































