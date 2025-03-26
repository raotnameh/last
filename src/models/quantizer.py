import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Quantizer(nn.Module):
    def __init__(self, num_codebooks):
        super(Quantizer, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebook_usage = torch.zeros(self.num_codebooks)
           
    def forward(self, z, codebook):
        """
        Quantizes encoder output z by finding the closest codebook vector.
        
        Args:
            z (torch.Tensor): Encoder output of shape (batch, time, channels).
            codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
            
        Returns:
            commitment_loss (torch.Tensor): Commitment loss, scaled by beta.
            z_q (torch.Tensor): Quantized latent vectors with shape (batch, channels, seq_length).
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
        
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, e).view(z.shape)
        
        commitment_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        
        min_encoding_indices = min_encoding_indices.view(z.shape[0], z.shape[1])
        codebook_prob = self.Codebook_usage(min_encoding_indices)
        
        return commitment_loss, z_q, min_encoding_indices, codebook_prob # commitment_loss, z_q, encoding_indices, codebook_prob
    
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
        histogram = torch.bincount(flattened_indices, minlength=self.num_codebooks).float().cpu()
        # Normalize to probabilities
        histogram /= histogram.sum() + 1e-8  # Avoid division by zero
        
        self.codebook_usage += histogram
        
        
        plt.clf() 
        plt.bar(range(self.num_codebooks), histogram)
        plt.xlabel("Codebook Index")
        plt.ylabel("Count")
        plt.title("Codebook Usage Histogram")
        plt.savefig("codebook_usage_histogram.png")
        plt.close()
        
        
        return histogram

    

if __name__ == "__main__":
    codebook = nn.Embedding(5, 1)
    z = torch.randn(1, 2, 1)
    print(z, codebook.weight)
    quantizer = Quantizer()
    commitment_loss, z_q, encoding_indices = quantizer(z, codebook)
    
    print(f"Commitment loss: {commitment_loss}")
    print(f"Quantized latent vectors: {z_q.shape}")
    print(f"Encoding indices: {encoding_indices.shape}")
    print(f"Encoding indices: {z_q}")
    print(f"Encoding indices: {encoding_indices}")
    