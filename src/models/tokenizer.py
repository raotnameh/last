import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class Tokenizer(nn.Module):
    def __init__(self, num_codebooks, save_dir="histogram_frames"):
        super(Tokenizer, self).__init__()
        self.frame_count = 0  
        self.save_dir = save_dir
        self.num_codebooks = num_codebooks
        self.codebook_usage = torch.zeros(self.num_codebooks)
           
    def forward(self, z, codebook, mask):
        """
        Tokenize encoder output z by finding the closest codebook vector.
        
        Args:
            z (torch.Tensor): Encoder output of shape (batch, time, channels).
            codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
            mask (torch.Tensor): Mask of shape (batch, time,1) with 1s for valid positions and 0s for padding.
            
        Returns:
            commitment_loss (torch.Tensor): Commitment loss, scaled by beta.
            z_q (torch.Tensor): Tokenize latent vectors with shape (batch, channels, seq_length).
            encoding_indices (torch.Tensor): Indices of the selected codebook vectors with shape (batch, seq_length).
        """
                
        e = codebook.embedding.weight.clone().detach() # (vocab_size, embed_dim)
        
        z_flattened = z.contiguous().view(-1, e.shape[1]) # (batch * time, channels)
        
        
        # distances from z to codebooks e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (torch.sum(z_flattened**2, dim=1, keepdim=True) \
            + torch.sum(e**2, dim=1) \
            - 2 * torch.matmul(z_flattened, e.t()))
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)    
        
        # get tokenized latent vectors
        z_q = torch.matmul(min_encodings, e).view(z.shape)

        # 
        # mask out padding positions in the latent vectors
        z_q = z_q[:,:mask.shape[1],:] * mask # b,t,c 
        # now replace the masked out positions with the padding embeddings
        pad_e = codebook.embedding(torch.tensor([0]).to(z.device)).unsqueeze(0) # 1,1,c
        z_q = torch.where(mask == 0, pad_e, z_q)  # Replace masked positions with padding embeddings
        
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none')  * mask 
        # MSE loss between z and z_q ignoring padding positions
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        z_q = z + (z_q - z).detach()
        
        min_encoding_indices = min_encoding_indices.view(z.shape[0], z.shape[1])
        
        codebook_prob = self.Codebook_usage(min_encoding_indices)
        
        return commitment_loss, z_q, min_encoding_indices # commitment_loss, z_q, encoding_indices
    
    def Codebook_usage(self, encoding_indices):
        """
        Computes the codebook usage as a histogram of codebook occurrences.
        Args:
            encoding_indices (Tensor): Tensor of shape (B, T) containing codebook indices.
        Returns:
            Tensor: Histogram of shape (num_codebooks,), representing the count of each codebook entry.
        """
        # Flatten indices across batch and time
        flattened_indices = encoding_indices.view(-1)  # Shape: (B*T,)

        # Compute histogram
        histogram = torch.bincount(flattened_indices, minlength=self.num_codebooks).float()
        # Accumulate usage
        self.codebook_usage += histogram.cpu()
        
        plt.clf() 
        plt.bar(range(self.num_codebooks), histogram.cpu().numpy())
        plt.xlabel("Codebook Index")
        plt.ylabel("Count")
        plt.title("Codebook Usage Histogram")
        
        # filename = os.path.join(self.save_dir, f"frame_{self.frame_count:04d}.png")  # Saves as frame_0000.png, frame_0001.png, ...
        filename = "codebook_usage_histogram.png"
        plt.savefig(filename)
        plt.close()  # Free memory

        self.frame_count += 1  # Increment frame count
        
        return histogram

    

if __name__ == "__main__":
    codebook = nn.Embedding(5, 1)
    z = torch.randn(1, 2, 1)
    print(z, codebook.weight)
    tokenizer = Tokenizer()
    commitment_loss, z_q, encoding_indices = tokenizer(z, codebook)
    
    print(f"Commitment loss: {commitment_loss}")
    print(f"Tokeized latent vectors: {z_q.shape}")
    print(f"Encoding indices: {encoding_indices.shape}")
    print(f"Encoding indices: {z_q}")
    print(f"Encoding indices: {encoding_indices}")
    