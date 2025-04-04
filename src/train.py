import os, sys
import numpy as np
import yaml, random


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchaudio
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler

# Ignore warnings
import warnings
warnings.simplefilter("ignore")
import logging
logging.getLogger('matplotlib').disabled = True
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
from datetime import datetime


# Configure logging
os.makedirs("logging", exist_ok=True)
log_filename = datetime.now().strftime("training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f"logging/{log_filename}"), logging.StreamHandler()])



class ShuffledBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)  

    def __iter__(self):
        batches = list(super().__iter__())  
        random.shuffle(batches)  # Shuffle batch order
        return iter(batches)



# step 00 :- Prepare the Hyperparameters
config = yaml.load( open("config/try1.yaml", "r"), Loader=yaml.FullLoader)
logging.info("Loaded Configuration:\n" + yaml.dump(config, default_flow_style=False))


# step 01 :- Prepare the speech dataset.
from dataset_speech import Dataset_speech
sdataset = Dataset_speech(input_manifest=config['dataset_speech']['path'], min_duration=32000, max_duration=320000)
ssampler = SequentialSampler(sdataset)
sbatch_sampler = ShuffledBatchSampler(ssampler, batch_size=32, drop_last=False)
sdataloader = DataLoader(
    sdataset,
    batch_sampler=sbatch_sampler,
    collate_fn=sdataset.collate_fn,
    pin_memory=True,
    num_workers=6,
)

# step 02 :- Prepare the text dataset.
from dataset_txt import Dataset_txt    
tdataset = Dataset_txt(data=config['dataset_txt']['path'])
vocab = tdataset.vocab
tsampler = SequentialSampler(tdataset)
tbatch_sampler = ShuffledBatchSampler(tsampler, batch_size=32, drop_last=False)
tdataloader = DataLoader(
    tdataset,
    batch_sampler=tbatch_sampler,
    collate_fn=tdataset.collate_fn,
    pin_memory=True,
    num_workers=6,
)
# Create an iterator from the dataloader
titer_data = iter(tdataloader)

# step 03 :- Prepare the codebook
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models.codebook import Codebook
codebook = Codebook(vocab)
logging.info(f"Size of codebook: {codebook.embedding.weight.shape[0]} x {codebook.embedding.weight.shape[1]}")

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

# step 3 :- Prepare the tokenizer
from models.tokenizer import Tokenizer
tokenizer = Tokenizer(codebook.embedding.weight.shape[0])

# step 4 :- Prepare the upsampler
sys.path.append(f"{os.getcwd()}/models/decoder")
from models.decoder.decoder import Upsample
upsample = Upsample(codebook.embedding.weight.shape[1], output_dim=config["decoder"]["transformer"]["decoder_hidden"], kernel_size=config['upsample']['kernel_size'], stride=config['upsample']['stride'])

# step 5 :- Prepare the decoder
from models.decoder.decoder import Decoder
decoder = Decoder(config['decoder'])

# step 6 :- Prepare the discriminator
from models.discriminator import Discriminator
discriminator = Discriminator(codebook.embedding.weight.shape[1], config['discriminator']['hidden_dim'], config['discriminator']['kernel_size'])

if __name__ == "__main__":



    # ========================
    # Training Setup
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move models to device
    codebook = codebook.to(device)
    gtruth = gtruth.to(device) # Always non trainable
    encoder = encoder.to(device)
    downsample = downsample.to(device)
    tokenizer = tokenizer.to(device)
    upsample = upsample.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)

    # Non trainable steps
    codebook.embedding.weight.requires_grad = False
    logging.info(f"Parameters in codebook are trainable: {codebook.embedding.weight.requires_grad}")
    for name, param in encoder.named_parameters():
        if "10" in name or "11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    logging.info(f"Parameters in encoder are trainable: {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6}M")

    # ========================
    # Parameters count in millions
    # ========================
    logging.info(f"Parameters in codebook: {sum(p.numel() for p in codebook.parameters()) / 1e6}M")
    logging.info(f"Parameters in encoder: {sum(p.numel() for p in encoder.parameters()) / 1e6}M")
    logging.info(f"Parameters in downsample: {sum(p.numel() for p in downsample.parameters()) / 1e6}M")
    logging.info(f"Parameters in tokenizer: {sum(p.numel() for p in tokenizer.parameters()) / 1e6}M")
    logging.info(f"Parameters in upsample: {sum(p.numel() for p in upsample.parameters()) / 1e6}M")
    logging.info(f"Parameters in decoder: {sum(p.numel() for p in decoder.parameters()) / 1e6}M")
    logging.info(f"Parameters in discriminator: {sum(p.numel() for p in discriminator.parameters()) / 1e6}M")
    


    # ========================
    # Optimizers and Hyperparameters
    # ========================
    num_steps = config['train']['num_steps']

    # Group the generator parameters: encoder, downsample, upsample, decoder. keeping the encoder frozen
    encoder_params = list([p for p in encoder.parameters() if p.requires_grad])
    downsample_params = list([p for p in downsample.parameters() if p.requires_grad])
    decoder_params = list([p for p in upsample.parameters() if p.requires_grad]) 
    decoder_params += list([p for p in decoder.parameters() if p.requires_grad])
    logging.info(f"Parameters in encoder: {sum(p.numel() for p in encoder_params) / 1e6}M")
    logging.info(f"Parameters in downsample: {sum(p.numel() for p in downsample_params) / 1e6}M")
    logging.info(f"Parameters in upsample: {sum(p.numel() for p in decoder_params) / 1e6}M")
    logging.info(f"Parameters in decoder: {sum(p.numel() for p in decoder_params) / 1e6}M")

    disc_params = list(discriminator.parameters())
    logging.info(f"Parameters in discriminator: {sum(p.numel() for p in disc_params) / 1e6}M")


    optimizer_enc = optim.Adam(encoder_params, lr=config['train']['lr_enc'])
    optimizer_down = optim.Adam(downsample_params, lr=config['train']['lr_down'])
    optimizer_dec = optim.Adam(decoder_params, lr=config['train']['lr_dec'])

    optimizer_disc = optim.Adam(disc_params, lr=config['train']['lr_disc'])




    # ========================
    # Losses
    # ========================
    from loss import Loss
    loss = Loss(config)
    
    

    # ========================
    # Training Loop
    # ========================
    step = 1
    while True:
        encoder.train()
        downsample.train()
        upsample.train()
        decoder.train()
        discriminator.train()
        
        for _, batch in enumerate(sdataloader):
            output = {}
            
            # ===== Data Preparation =====
            waveforms, padding_masks = batch
            waveforms = waveforms.to(device) # [B, T]
            padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
            
            # ===== Generator Forward Pass =====
            enc_out = encoder(waveforms, padding_masks)  # [B, T, C] # step 1
            padding_masks = enc_out['padding_mask'] # [B, T // 320] 
            output["cnn_out"] = enc_out['cnn_out'] # [B, T // 320] 
            output['encoder_out'] = enc_out['encoder_out'] # [B, T // 320] 
            output['padding_mask'] = padding_masks # [B, T // 320] ``
            
            mask = ~padding_masks # B,T//320,1 # 0 for masked positions.
            mask = mask.unsqueeze(-1).float()
            output['enc_mask'] = mask
            
            gt = gtruth.encode(waveforms.unsqueeze(1)) # [B, T//320, 1024] # step 04
            gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
            output['gt'] = gt    
            
            down_out = downsample(enc_out['encoder_out'])  # [B, T // 2, C] # step 2
            dmask = mask[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride'], 1]
            down_out = down_out[:,:dmask.shape[1],:] * dmask # [B, T // 2, C]
            output['down_out'] = down_out
            output['dmask'] = dmask
            
            commitment_loss, diversity_loss, z_q, z_q_disc, encoding_indices, non_repeated_min_encoding_indices, non_repeated_mask = tokenizer(down_out, codebook, dmask) # [B, T // 2, C], [B, T // 2, C], [B, T // 2] # step 3 -- all the necessary masks are already applied in the tokenizer
            output['commitment_loss'] = commitment_loss
            output['z_q'] = z_q
            output['encoding_indices'] = encoding_indices
            output['diversity_loss'] = diversity_loss
            
            
            # ===== Discriminator Forward Pass =====
            # pred_fake = discriminator(z_q, ~dmask.bool()) # discriminator fake output # step 6 # B 
            # output['dis_fake'] = pred_fake

            # if step % 2 == 0:
            #     try:
            #         tbatch = next(titer_data)
            #     except:
            #         iter_data = iter(tdataloader)  # Reinitialize iterator
            #         tbatch = next(titer_data)  # Fetch the first batch again
                
            #     text, tmask = tbatch
            #     text = text.to(device)
            #     tmask = tmask.to(device)
            #     text = codebook(text)
            #     pred_real = discriminator(text, tmask.unsqueeze(-1)) # discriminator real output # step 6 # B
            #     output['dis_real'] = pred_real
            #     output['dis_real_x'] = text
            #     output['tmask'] = tmask
                
                # # ===== Loss Computation =====
                # loss.gan_loss.discriminator = discriminator
                # total_loss = loss.step_disc(output, step, num_steps)
                # # ===== Backward Pass ===== 
                # optimizer_disc.zero_grad()
                # total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(disc_params, max_norm=5.0)
                # optimizer_disc.step()
                
                # step += 1
            #     continue
                 
            # ===== Generator Forward Pass Continued =====
            up_out = upsample(z_q) # [B, T, C] # step 4
            up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]       
            output['up_out'] = up_out 

            
            dec_out, dec_out2, dec_mask = decoder(up_out, padding_masks, output['cnn_out'], config['decoder']["speaker"]["use_s"]) # [B, T, C], [B, T, C], [B, T] # step 5
            dec_out = dec_out[:,:dec_mask.shape[1],:] * dec_mask
            dec_out2 = dec_out2[:,:dec_mask.shape[1],:] * dec_mask
            output['dec_out'] = dec_out
            output['dec_out2'] = dec_out2
            output['dec_mask'] = dec_mask
            
            # ===== Loss Computation =====
            loss_components = loss.step_gen(output)
            if step % 10 == 0:    
                logging.info(f"GEN-LOSS---step/total: {step}/{num_steps} rec_loss: {loss_components['rec_loss']}, commit_loss: {loss_components['commit_loss']}, smooth_loss: {loss_components['smooth_loss']}, gen_loss: {loss_components['gen_loss']}, diversity_loss: {loss_components['diversity_loss']}")
            total_loss = sum(loss_components.values())
            
            # ===== Backward Pass ===== 
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(downsample_params, max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=5.0)
            if (step) % 2 == 0:
                optimizer_down.step()
                optimizer_dec.step()
    
                optimizer_down.zero_grad()
                optimizer_dec.zero_grad()
                
                if step >= config['train']['freez_steps']:
                    optimizer_enc.step()
                    optimizer_enc.zero_grad()
            
            step +=1
            
        if step > num_steps:
            break
























