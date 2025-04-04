import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler


import sys
sys.path.append("..")  # Add parent directory to path

from dataset_txt import Dataset_txt
data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"    
data = "/raid/home/rajivratn/hemant_rajivratn/last/data/librispeech-lm-norm.txt"
dataset_txt = Dataset_txt(data=data)
print(F"Vocab: {dataset_txt.vocab}")
batch_size = 256
class ShuffledBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)  

    def __iter__(self):
        batches = list(super().__iter__())  
        random.shuffle(batches)  # Shuffle batch order
        return iter(batches)
sampler = SequentialSampler(dataset_txt)  # Keeps order within each batch
batch_sampler = ShuffledBatchSampler(sampler, batch_size=batch_size, drop_last=False)
dataloader = DataLoader(
    dataset_txt,
    batch_sampler=batch_sampler,
    collate_fn=dataset_txt.collate_fn,
    pin_memory=True,
    num_workers=6,
)
# dataloader = DataLoader(dataset_txt, batch_size=128, shuffle=False, collate_fn=dataset_txt.collate_fn, pin_memory=True, num_workers=6, persistent_workers=True)


from codebook import Codebook
vocab_size = len(dataset_txt.vocab)
emb_dim = 768  # Change as needed
codebook = Codebook(vocab_size, emb_dim)
print(f"Size of codebook: {vocab_size} x {emb_dim}")



class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(in_channels)  # Normalize over channel dim
        
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               dilation=dilation, padding=0)  # No automatic padding

    def forward(self, x):
        # x is expected to be of shape (batch, time, channels)
        
        x = self.layernorm(x)
        x = x.transpose(1, 2)
        
        pad_size = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_size, 0))  # Manual causal padding
    
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        return x  # Now it's properly causal
    
class ResidualBlock(nn.Module):
    """Residual Block with Causal Convolution."""
    def __init__(self, hidden_dim, kernel_size):
        super().__init__()
        self.conv = CausalConv1d(hidden_dim, hidden_dim, kernel_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=2)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=4)
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x):
        d = self.dropout(self.gelu(self.conv(x)))
        d = self.dropout2(self.gelu2(self.conv2(d)))
        d = self.dropout3(self.gelu3(self.conv3(d)))
        return x + d # Residual connection


class CausalCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, vocab_size):
        super().__init__()

        layers = []
        self.pre = CausalConv1d(input_dim, hidden_dim, 1)  # Input layer
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, kernel_size))
    
        self.model = nn.Sequential(*layers)
        self.proj = CausalConv1d(hidden_dim, vocab_size, 1)  # Output layer

    def forward(self, x):
        # x: (batch, time, channels)
        x = self.pre(x)  # Input layer
        x = self.model(x)
        x = self.proj(x)  # Output layer
        return x
    
# Initialize the model
input_dim = emb_dim
hidden_dim = 768

num_layers = 5
kernel_size = 7
num_epochs = 100
vocab_size = len(dataset_txt.vocab)
model = CausalCNN(input_dim, hidden_dim, num_layers, kernel_size, vocab_size)
# print(model)

criterion = nn.CrossEntropyLoss(ignore_index=dataset_txt.char_to_idx['p'])
optimizer = optim.Adam(list(model.parameters()) + list(codebook.parameters()), lr=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs, eta_min=1e-6)

print(f"Number of parameters in millions for model and codebook: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}, {sum(p.numel() for p in codebook.parameters()) / 1e6:.2f}")


try: 
    # Load the model
    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint["model"])
    codebook.load_state_dict(checkpoint["codebook"])
except:
    print("No checkpoint found!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
codebook = codebook.to(device)
criterion = criterion.to(device)




# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    codebook.train()
    
    running_loss = 0.0
    for iteration, batch in enumerate(dataloader):
        
        inputs = batch['inp'].to(device)
        targets = batch['out'].to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        x = codebook(inputs) # (b,t,c)
        outputs = model(x)
        # print(outputs.shape) # (b,t,v)
        # print(targets.shape) # (b,t)
        
        # Compute the loss
        loss = criterion(outputs.transpose(1, 2), targets)

        
        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()
        scheduler.step()
        
        # torch.cuda.empty_cache()
        running_loss += loss.detach().item()
        if iteration % 1000 == 0:
            print(f"Batch {iteration+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        if iteration % 10000 == 0:
            # print model predictions in char format and compare with ground truth
            with torch.no_grad():
                pred = outputs.argmax(dim=-1).squeeze().cpu().numpy()[0,:]
                gtruth = targets.squeeze().cpu().numpy()[0,:]
                
                def get_char(idx):
                    return dataset_txt.idx_to_char[int(idx)]
                
                gtruth = [get_char(idx) for idx in gtruth if get_char(idx) != "p"][:-1]
                pred = [get_char(idx) for idx in pred][:len(gtruth)]
                print(f"Ground truth: {''.join(gtruth)}")
                print(f"Predictions: {''.join(pred)}")
                print(f"Levenshtein distance (char) -- (lower is better): {Lev.distance(''.join(gtruth), ''.join(pred))}")
    
                torch.save({"model": model.state_dict(), "codebook": codebook.state_dict()}, "checkpoint.pt")
                print("Models saved!")
                
                
        
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
      
    

print("Training finished!")