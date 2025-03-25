import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quantizer(nn.Module):
    def __init__(self, beta=1.0):
        super(Quantizer, self).__init__()
        self.beta = beta
           
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
                
        e = codebook.weight.clone().detach() # (vocab_size, embed_dim)
        
        z_flattened = z.view(-1, e.shape[1]) # (batch * time, channels)
        
        
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
        
        commitment_loss = self.beta * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        
        return commitment_loss, z_q, min_encoding_indices.view(z.shape[0], z.shape[1]) # commitment_loss, z_q, encoding_indices
    

    def merge_similar_indices(self, indices):
        batch_size, seq = indices.shape
        merged_indices = []

        for b in range(batch_size):
            unique_indices = []
            prev_idx = None

            for t in range(seq):
                current_idx = indices[b, t].item()

                if prev_idx is None or current_idx != prev_idx:
                    unique_indices.append(current_idx)

                prev_idx = current_idx

            merged_indices.append(unique_indices)

        return merged_indices
    
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
    