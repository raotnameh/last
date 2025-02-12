# This is the training script for the overall model. 
# The model has basically 6 components:
# 1. Encoder module
# 2. Downsampling module
# 3. Frozen Vocabulary module
# 4. Upsampling module 
# 5. Decoder module 
# 6. Frozen neural audio codec module used for generating the ground truth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

writer = SummaryWriter(log_dir="logs")  # Specify the log directory

# import models
import dac
from encoder import Encoder, Downsampling
from vocab import FrozenVocabulary, get_closest_vocab, merge_similar_indices
from decoder import Upsampling, Decoder, calculate_params
from codec import Codec

print("All imports are successful")

# params
hidden_dim = 256

# models 
encoder = Encoder() # frozen
downsampling = Downsampling()
vocab = FrozenVocabulary(path="model_checkpoint.pth") # frozen
upsampling = Upsampling(inp_dim=768, hidden_dim=hidden_dim)
decoder = Decoder(hidden_dim=hidden_dim, out_dim=1024, num_blocks=10, kernel_size=9)
codec = Codec() # frozen
vocab_embeddings, char_to_idx, idx_to_char = vocab.embeddings, vocab.char_to_idx, vocab.idx_to_char

try: 
    # load the model
    checkpoint = torch.load("model_0.pth")
    downsampling.load_state_dict(checkpoint['downsampling'])
    decoder.load_state_dict(checkpoint['decoder'])
    upsampling.load_state_dict(checkpoint['upsampling'])
    
    print("Model loaded successfully")  
except: 
    print("Model not found, starting from scratch")


# freeze the encoder, and codec
for param in encoder.named_parameters():
    param[1].requires_grad = False
    # param[1].requires_grad = True
for param in codec.model.parameters():
    param.requires_grad = False   
vocab_embeddings.requires_grad = False

print(f"Paraeters of downsampling: {calculate_params(downsampling)}")
print(f"Paraeters of upsampling: {calculate_params(upsampling)}")
print(f"Paraeters of decoder: {calculate_params(decoder)}")

print("Models are loaded")

# Training loop
criterion = nn.MSELoss()
optimizer = optim.Adam(
    list(downsampling.parameters()) + list(decoder.parameters()) + list(upsampling.parameters()),
    # list(downsampling.parameters()) + list(decoder.parameters()) + list(upsampling.parameters()) + list(encoder.parameters()),
    lr=0.001)

# Set the models to gpu
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
downsampling = downsampling.to(device)
vocab_embeddings = vocab_embeddings.to(device)
decoder = decoder.to(device)
upsampling = upsampling.to(device)
codec.model = codec.model.to(device)

# Set the models to training mode
encoder.train()
downsampling.train()
decoder.train()
upsampling.train()

print("Models are set to training mode")

class AudioDataset(Dataset):
    def __init__(self):
        input_manifest = "/raid/home/rajivratn/hemant_rajivratn/librispeech/data/manifest/train.tsv"

        # Read the first line to get the root directory
        with open(input_manifest, "r") as infile:
            root_dir = infile.readline().strip()  # First line is the root directory

        # Define valid duration range
        min_duration = 32000  # 2 seconds
        max_duration = 250000  # 15.625 seconds

        filtered_samples = []

        with open(input_manifest, "r") as infile:
            infile.readline()  # Skip header (already read root_dir)
            for line in infile:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                file_name, duration = parts
                duration = int(duration)

                if min_duration <= duration <= max_duration:
                    full_path = os.path.join(root_dir, file_name)
                    filtered_samples.append((full_path, duration))

        # Sort by duration
        filtered_samples.sort(key=lambda x: x[1])

        self.dataset = filtered_samples
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.dataset[idx][0])
        assert sample_rate == 16000, "Sampling rate must be 16000"
        return waveform 

# Create the dataset and dataloader
dataset = AudioDataset()

# create a collate function to truncate the audio files to minimum length
def collate_fn(batch):
    
    min_len = min([waveform.shape[1] for waveform in batch])
    batch = [waveform[:, :min_len] for waveform in batch]
    batch = torch.stack(batch)
    return batch.squeeze(1)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True, collate_fn=collate_fn)

# start training
num_epochs = 10
# Define linear decay function
def linear_lr_lambda(step):
    return max( 1e-6, 1 - step / (num_epochs*len(dataloader)))
scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_lambda)


for epoch in range(num_epochs):
    
    running_loss = 0.0
    for iteration, waveform in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        # data
        waveform = waveform.to(device) 
        
        # Forward pass
        with torch.no_grad():
            encoder_output = encoder(waveform)
        downsampling_output = downsampling(encoder_output)
        
        # Get the closest vocab embeddings
        vocab_output = get_closest_vocab(downsampling_output, vocab_embeddings)
        vocab_output = downsampling_output + (vocab_output - downsampling_output).detach()
        
        # Upsampling
        upsampling_output = upsampling(vocab_output)
        
        # Decoder
        decoder_output = decoder(upsampling_output)
        
        # Codec
        with torch.no_grad():
            codec_output = codec.encode(waveform).detach()
        
        # Ensure same sequence length for ground truth and output
        seq_len = min(codec_output.shape[-1], decoder_output.shape[-1])    
        codec_output = codec_output[:, :, :seq_len]
        decoder_output = decoder_output[:, :, :seq_len]    
        

        # Compute the loss
        loss = criterion(decoder_output, codec_output)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        scheduler.step()
        # empty cache
        torch.cuda.empty_cache()  
        
        running_loss += loss.item()
        
        # print for every 10 iterations
        if iteration % 10 == 0:
            print(f"Epoch: {epoch}, Iteration: {iteration}/{len(dataloader)}, Loss: {running_loss/(iteration+1)}")
            
            
            # save in tensorboard
            writer.add_scalar("Loss", loss.item(), epoch * len(dataloader) + iteration)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch * len(dataloader) + iteration)
                
    # save the model
    torch.save({
        'downsampling': downsampling.state_dict(),
        'decoder': decoder.state_dict(),
        'upsampling': upsampling.state_dict(),
        'running_loss': running_loss,
    }, f"model_{epoch}.pth")


writer.close()  # Close the writer when done