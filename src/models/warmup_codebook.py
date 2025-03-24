import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import Levenshtein as Lev


import sys
sys.path.append("..")  # Add parent directory to path

from dataset_txt import Dataset_txt
data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"    
dataset_txt = Dataset_txt(data=data)
print(F"Vocab: {dataset_txt.vocab}")
dataloader = DataLoader(dataset_txt, batch_size=512, shuffle=False, collate_fn=dataset_txt.collate_fn, pin_memory=True, num_workers=6, persistent_workers=True)


from codebook import Codebook
vocab_size = len(dataset_txt.vocab)
emb_dim = 256  # Change as needed
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

    def forward(self, x):
        return x + self.dropout(self.gelu(self.conv(x)))  # Residual connection


class CausalCNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, kernel_size, vocab_size):
        super().__init__()

        layers = []
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, kernel_size))
    
        self.model = nn.Sequential(*layers)
        self.proj = nn.Conv1d(hidden_dim, vocab_size, 1)  # Output layer

    def forward(self, x):
        # x: (batch, time, channels)
        x = self.model(x)
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)  # Output layer
        return x
    
# Initialize the model
hidden_dim = emb_dim

num_layers = 20
kernel_size = 11
vocab_size = len(dataset_txt.vocab)
model = CausalCNN(hidden_dim, num_layers, kernel_size, vocab_size)
# print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")


criterion = nn.CrossEntropyLoss(ignore_index=dataset_txt.char_to_idx['p'])
optimizer = optim.Adam(model.parameters(), lr=0.0005)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
codebook = codebook.to(device)
criterion = criterion.to(device)



# Training loop
num_epochs = 100
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
        
        # torch.cuda.empty_cache()
        running_loss += loss.item()
        # if iteration % 10 == 0:
        #     print(f"Batch {iteration+1}/{len(dataloader)}, Loss: {loss.item():.4f}")


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
        
        
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
      
    torch.save(codebook, "codebook.pt")
    print("Models saved!")

print("Training finished!")