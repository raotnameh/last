import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns


class Tokenizer(nn.Module):
    def __init__(self, vocab):
        super(Tokenizer, self).__init__()
        '''
        Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
        '''

        self.vocab = vocab[1:] # remove the padding token
        self.step = 0
    def codebook_usage(self, min_encodings, mask):
        
        # prob for each character
        mask_bool = (mask == 1).squeeze(1)  # shape: (B,), True where we keep
        valid_encodings = min_encodings[mask_bool]  # shape: (B', C)
        e_mean_np = valid_encodings.mean(dim=0).cpu().numpy()
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(self.vocab, e_mean_np, color='blue', alpha=0.7)
        plt.xlabel('Codebook Entry (Char)')
        plt.ylabel('Probability')
        plt.title('Codebook Usage Distribution')
        plt.grid(axis='y')
        plt.savefig('codebook_usage_distribution.png', bbox_inches='tight')
        # for every 1000 steps save the plot 
        if self.step % 100 == 0:
            plt.savefig(os.path.join('plots', f'codebook_usage_distribution_{self.step}.png'), bbox_inches='tight')
            plt.close()
        self.step += 1
        
        plt.close()
     
    def forward(self, z, codebook, mask):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """
        # Normalize z 
        z = F.normalize(z, p=2, dim=-1) # (batch, time, channels)
        
        
        # Using fixed codebook embeddings
        e = codebook.embedding.weight.clone().detach() # (vocab_size+1, embed_dim) 
        e = e[1:,:] # remove the padding embedding from the codebook # (vocab_size, embed_dim)        
        
        z_flattened = z.contiguous().view(-1, e.shape[1]) # (batch * time, channels==embed_dim)
        
        # distances from z to codebooks e_j ∥z−e∥**2 =∥z∥**2 +∥e∥**2 −2(z⋅e)
        d = (torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(e**2, dim=1) - 2 * torch.matmul(z_flattened, e.t())) # (batch*time, vocab_size)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # (batch*time, 1) This has 0 to vocab_size-1 except for padding token which was 0.
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device) # (batch*time, vocab_size)
        min_encodings.scatter_(1, min_encoding_indices, 1) # (batch*time, vocab_size)
        z_q = torch.matmul(min_encodings, e).view(z.shape) # (batch, time, channels) # get tokenized latent vectors
        z_q *= mask # mask out padding positions of index 0. REASON: our data is sequential compared to the original VQVAE paper where the data is not sequential (images) which did not need padding.
        
        # codebook usage Distribution
        self.codebook_usage(min_encodings, mask.contiguous().view(-1, 1))
        
        # commitment loss
        ################ no need to detach z_q since e is not trainable
        # Normalize z_flattened and e using p2 normalization 
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # (batch, time, channels) 
        # MSE loss between z and z_q ignoring padding positions
        valid_count = mask.sum() # * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
                
        # preserve gradients
        z_q = z + z_q - z.detach()
        
        # Smoothness loss
        smoothness_loss = F.mse_loss(z_q[:, :-1, :], z_q[:, 1:, :], reduction='none') * mask[:, 1:, :] # (batch, time-1, channels)
        # MSE loss between z_q[t] and z_q[t+1] ignoring padding positions
        valid_count = mask[:, 1:, :].sum() #* z.shape[-1]
        smoothness_loss = smoothness_loss.sum() / valid_count
        
        ##### Discriminator codebooks without repeated indices #####
        encodings = min_encoding_indices.view(z.shape[0], z.shape[1])
        n_z_q, n_mask, selected_encodings_list = self.remove_consecutive_repeated_indices( encodings, mask.squeeze(-1), z_q) # randomly pick one index from each group of consecutive repeating elements # shape (B,T) and also returns the mask 
    
        
        return smoothness_loss, commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list # commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list

        
    def remove_consecutive_repeated_indices(self, min_encoding_indices, mask, z_q):
        B, T = min_encoding_indices.shape
        selected_indices_list = []
        selected_encodings_list = []
        max_len = 0

        for b in range(B):
            indices = min_encoding_indices[b]
            selected_indices = []
            selected_encodings = []
            start = 0
            for i in range(1, T):
                if mask[b, i] == 0:
                    break
                if indices[i] != indices[i - 1]:
                    ii = random.randint(start, i - 1)
                    selected_indices.append(ii)
                    selected_encodings.append(indices[ii].item()+1) # +1 to avoid padding token
                    start = i
            ii = random.randint(start, T - 1)
            selected_indices.append(ii)
            selected_encodings.append(indices[ii].item()+1) # +1 to avoid padding token

            selected_encodings_list.append(selected_encodings)
            selected_indices_list.append(torch.tensor(selected_indices))
            max_len = max(max_len, len(selected_indices))

        # Pad and create mask in a vectorized way
        padded_indices = torch.zeros((B, max_len), dtype=min_encoding_indices.dtype, device=min_encoding_indices.device)
        masks = torch.zeros((B, max_len), dtype=torch.float, device=min_encoding_indices.device)

        for b, indices in enumerate(selected_indices_list):
            length = len(indices)
            padded_indices[b, :length] = indices
            masks[b, :length] = 1
        masks = masks.unsqueeze(-1)
        # Gather and apply mask
        out = padded_indices.unsqueeze(-1).expand(-1, -1, z_q.shape[-1])
        n_z_q = z_q.gather(dim=1, index=out)
        n_z_q *= masks

        return n_z_q, masks, selected_encodings_list # shape (B, max_len, channels), mask shape (B, max_len, 1), list of selected encodings