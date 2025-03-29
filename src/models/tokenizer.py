import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class Tokenizer(nn.Module):
    def __init__(self, num_codebooks, save_dir="histogram_frames"):
        super(Tokenizer, self).__init__()
        self.frame_count = 0  
        self.save_dir = save_dir
        self.num_codebooks = num_codebooks
        self.codebook_usage = torch.zeros(self.num_codebooks)
    
    def randomly_keep_one_with_indices(self, indices, pad_e, z_q):
        chosen_indices = []

        for row_idx, row in enumerate(indices):
            row_indices = []
            i = 0

            while i < len(row):
                value = row[i].item()
                
                # Detect consecutive duplicates
                j = i + 1
                while j < len(row) and row[j].item() == value:
                    j += 1
                
                # Randomly select one index from the range
                random_index = random.randint(i, j - 1)
                row_indices.append(random_index)  # Track chosen index
                
                # Move to next unique value
                i = j

            chosen_indices.append(torch.tensor(row_indices, device=indices.device))
        
        # extract the embedding from z_q and pad to the length of the longest row with pad_e
        max_length = max(len(ind) for ind in chosen_indices)
        
        dummy = []
        for r, ind in enumerate(chosen_indices):
            dummy.append(z_q[r, ind].unsqueeze(0))
            # pad the row with pad_e
            pad_len = max_length - len(ind)
            if pad_len > 0:
                pad = pad_e.repeat(1, pad_len,1)
                dummy[r] = torch.cat([dummy[r], pad], dim=1)
        
        # concatenate the rows
        z_q_disc = torch.cat(dummy, dim=0)

        return z_q_disc, chosen_indices

           
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


        # mask out padding positions in the latent vectors
        z_q = z_q[:,:mask.shape[1],:] * mask # b,t,c 
        # now replace the masked out positions with the padding embeddings
        pad_e = e[:1,:].unsqueeze(0) # 1,1,c
        z_q = torch.where(mask == 0, pad_e, z_q)  # Replace masked positions with padding embeddings
        
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none')  * mask 
        # MSE loss between z and z_q ignoring padding positions
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        z_q = z + (z_q - z).detach()
        
        min_encoding_indices = min_encoding_indices.view(z.shape[0], z.shape[1])


        z_q_disc, non_repeated_min_encoding_indices = self.randomly_keep_one_with_indices(min_encoding_indices, pad_e, z_q.clone())
        
        
        # ----- Soft Assignment for Diversity Loss with Masking -----
        # Compute soft assignments from distances using softmax (differentiable)
        soft_assignments = F.softmax(-d, dim=1)  # shape: (batch*time, num_codebooks)
        # Reshape mask to match soft_assignments shape
        mask_flattened = mask.contiguous().view(-1, 1)  # Shape: (batch*time, 1)
        # Apply masking to ignore padded positions
        masked_assignments = soft_assignments * mask_flattened  # Shape: (batch*time, num_codebooks)
        # Compute soft histogram using masked assignments
        soft_histogram = masked_assignments.sum(dim=0)  # Shape: (num_codebooks,)
        usage_sum = soft_histogram.sum() + 1e-8  # Avoid division by zero
        p = soft_histogram / usage_sum  # Normalize to get probability distribution
        # KL divergence from uniform distribution
        diversity_loss = torch.sum(p * torch.log(p * self.num_codebooks + 1e-8))
        
        
        codebook_prob = self.Codebook_usage(min_encoding_indices)
        
        
        return commitment_loss, diversity_loss, z_q, z_q_disc, min_encoding_indices, non_repeated_min_encoding_indices  # commitment_loss, z_q, encoding_indices
    
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
    
    
    