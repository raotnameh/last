import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch.nn.utils.rnn import pad_sequence
import random

class Tokenizer(nn.Module):
    def __init__(self, config, vocab, rot=True):
        super(Tokenizer, self).__init__()
        '''
        Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
        '''

        self.vocab = vocab[1:] # remove the padding token
        self.rot = rot
        self.save_dir = config['logging']['dir']
        os.makedirs(f"{self.save_dir}/plots/", exist_ok=True)
    
    def codebook_usage(self, min_encodings, mask, step):
        
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
        if step % 1000 == 0:
            plt.savefig(os.path.join(f'{self.save_dir}/plots', f'codebook_usage_distribution_{step}.png'), bbox_inches='tight')
            plt.close()

        plt.close()
   
    @staticmethod
    def get_very_efficient_rotation( u, q, x):
        w = F.normalize(u + q, dim=1).detach()
        return x - 2*torch.bmm(torch.bmm(x, w.unsqueeze(-1)), w.unsqueeze(1)) + 2*torch.bmm( torch.bmm(x, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
          
    def forward(self, z, codebook, mask, writer=None, step=1):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """  
        
        
        # 1. Prepare codebook embeddings (detach to avoid training update)
        e = codebook.embedding.weight.clone().detach() # (vocab_size+1, embed_dim) 
        e = e[1:,:] # remove the padding embedding from the codebook # (vocab_size, embed_dim)       
        # 2. Flatten z for distance computation: (b*t, c)
        b, t, c = z.shape
        z_flat = z.contiguous().view(-1, c) # (batch * time, channels==embed_dim)

        # 3. distances from z to codebooks e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (torch.sum(z_flat**2, dim=1, keepdim=True) \
            - 2 * z_flat @ e.t() \
                + torch.sum(e**2, dim=1, keepdim=True).t() 
        )
        # 4. find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # 5. Quantized latents via direct indexing
        z_q = torch.matmul(min_encodings, e).view(b,t,c) # (batch, time, channels) 
        
        # Angle between the z and z_q
        x = z_flat # (batch*time, channels)
        quantized = z_q.contiguous().view(-1, c) # (batch*time, channels)
        theta = torch.sum(x * quantized, dim=1) # (batch*time)
        theta_mask = theta > 0.8 # (batch*time) # Limitation of the roation trick. It avoids the rotation trick when the angle is too small. which results in opposite direction of gradeints for codebook  and  encoder output.
        theta_mask = theta_mask.float().unsqueeze(1) # (batch*time, 1)
        # count of theta_mask of value 1 
        if writer and step % 1000 == 0:
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
            writer.add_scalar('tokenizer/theta_mask_mean', theta_mask.sum().item(), step)
        
        # 6. Apply rotation trick on already normalized vectors           
        r_z_q = self.get_very_efficient_rotation(x , quantized, x.unsqueeze(1)).squeeze() 
        # 7. Straight-through estimator and mask padding
        s_z_q = z_flat + (quantized - z_flat).detach() # btc  
        
        z_q = theta_mask * r_z_q + (1 - theta_mask) * s_z_q # btc   
        z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        z_q = z_q * mask
        
        # # Straight-through estimator and mask padding
        # z_q = z_flat + (quantized - z_flat).detach() # btc  
        # z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        # z_q = z_q * mask
        
        # 8. commitment loss;  MSE loss between z and z_q ignoring padding positions
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # btc
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        # 9. Smoothness loss
        smoothness_loss = F.mse_loss(z[:, :-1, :], z[:, 1:, :], reduction='none') * mask[:, 1:, :] 
        smoothness_loss = smoothness_loss.sum() / valid_count 
        
        # 10. Discriminator codebooks without repeated indices #####
        encodings = min_encoding_indices.view(z.shape[0], z.shape[1]) # ( batch, time ) # (B, T)
        n_z_q, n_mask, selected_encodings_list, selected_encodings_repeated_list = self.remove_consecutive_repeated_indices( encodings, mask.squeeze(-1), z_q.clone()) # randomly pick one index from each group of consecutive repeating elements # shape (B,T) and also returns the mask 

        # codebook usage Distribution
        self.codebook_usage(min_encodings, mask.contiguous().view(-1, 1), step)

        return smoothness_loss, commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list, selected_encodings_repeated_list # commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list<=












    def remove_consecutive_repeated_indices(self, min_encoding_indices, mask, z_q):
        B, T = min_encoding_indices.shape
        selected_indices_list = []
        selected_encodings_list = []
        max_len = 0
        
        selected_encodings_repeated_list = []
        

        for b in range(B):
            indices = min_encoding_indices[b]
            selected_indices = []
            selected_encodings = []
            selected_encodings_repeated = []
            start = 0
            for i in range(1, T):
                if mask[b, i] == 0:
                    break
                
                selected_encodings_repeated.append(indices[i]+1) # +1 to avoid padding token
                                
                if indices[i] != indices[i - 1]:
                    ii = random.randint(start, i - 1)
                    selected_indices.append(ii)
                    selected_encodings.append(indices[ii]+1) # +1 to avoid padding token
                    start = i
            ii = random.randint(start, T - 1)
            selected_indices.append(ii)
            selected_encodings.append(indices[ii]+1) # +1 to avoid padding token
            selected_encodings_repeated.append(indices[ii]+1) # +1 to avoid padding token

            selected_encodings_list.append(selected_encodings)
            selected_encodings_repeated_list.append(selected_encodings_repeated)
            selected_indices_list.append(torch.tensor(selected_indices))
            max_len = max(max_len, len(selected_indices))
            
        # Pad and create mask in a vectorized way
        padded_indices = torch.zeros((B, max_len), dtype=min_encoding_indices.dtype, device=min_encoding_indices.device)
        masks = torch.zeros((B, max_len), dtype=torch.float, device=min_encoding_indices.device)

        for b, indices in enumerate(selected_indices_list):
            length = len(indices)
            padded_indices[b, :length] = indices
            masks[b, :length] = 1.0
        masks = masks.unsqueeze(-1) # shape (B, max_len, 1)
        padded_indices = padded_indices.unsqueeze(-1) # shape (B, max_len, 1)
        # Gather and apply mask
        padded_indices = padded_indices.expand(-1, -1, z_q.shape[-1]) # shape (B, max_len, channels)
        n_z_q = z_q.gather(dim=1, index=padded_indices)
        n_z_q *= masks

        return n_z_q, masks, selected_encodings_list, selected_encodings_repeated_list    # shape (B, max_len, channels), mask shape (B, max_len, 1), list of selected encodings
    
    
    
    
    



    # def remove_consecutive_repeated_indices(self, min_encoding_indices, mask, z_q):
    #     B, T = min_encoding_indices.shape # min encoding has 1,2,3,3,3,....
        
    #     selected_indices_list = []
    #     selected_encodings_list = []
    #     selected_encodings_repeated_list = []
    #     n_z_qs = []
    #     masks = []
        
    #     max_len = 0
    #     skip_non_speech = True
    #     masks_tensor = torch.zeros((B, T), dtype=torch.float, device=mask.device)
        
    #     for b in range(B):
    #         indices = min_encoding_indices[b]
    #         selected_indices = []
    #         selected_encodings = []
    #         selected_encodings_repeated = []
    #         n_z_q = []

    #         start = 0
    #         loop_broke = False
            
    #         for i in range(1, T):
    #             if mask[b, i] == 0: 
    #                 loop_broke = True
    #                 break
    #             selected_encodings_repeated.append(indices[i-1] + 1) # +1 to avoid padding token
                                
    #             if indices[i] != indices[i - 1]:
    #                 if indices[i-1] == 28 and skip_non_speech:     
    #                     continue
    #                 else:
    #                     seg = z_q[b, start:i, :]
    #                     if (i - 1) - start == 0: 
    #                         n_z_q.append(seg)
    #                     else: 
    #                         current_sum = seg.sum(dim=0, keepdim=True)
    #                         previous_sum = z_q[b, start:i-1, :].sum(dim=0, keepdim=True)
    #                         n_z_q.append(current_sum - previous_sum.clone().detach())
    #                         # n_z_q.append(seg.mean(dim=0, keepdim=True))
                            
                            
    #                     selected_indices.append(i-1)
    #                     selected_encodings.append(indices[i-1] + 1) # +1 to avoid padding token
    #                     start = i
            
    #         # Final flush: only if loop didn't break (i.e., full valid length was processed)
    #         if not loop_broke:
    #             selected_encodings_repeated.append(indices[i - 1] + 1)  # +1 to avoid padding token
    #             if (i - 1) - start == 0: 
    #                 n_z_q.append(z_q[b, start:i, :])
    #             else: 
    #                 current_sum = seg.sum(dim=0, keepdim=True)
    #                 previous_sum = z_q[b, start:i-1, :].sum(dim=0, keepdim=True)
    #                 n_z_q.append(current_sum - previous_sum.clone().detach())
    #                 # n_z_q.append(seg.mean(dim=0, keepdim=True))
                    
    #             selected_indices.append(i - 1)
    #             selected_encodings.append(indices[i - 1] + 1)


    #         selected_encodings_repeated_list.append(selected_encodings_repeated)
    #         n_z_qs.append(torch.cat(n_z_q, dim=0))
    #         selected_indices_list.append(selected_indices)
    #         selected_encodings_list.append(selected_encodings)
            
    #         masks_tensor[b, :len(selected_indices)] = 1.0  # Assigning mask directly
    #         max_len = max(max_len, len(selected_indices))
        
    #     masks = masks_tensor[:,:max_len].unsqueeze(-1) # shape (B, max_len, 1)
        
    #     n_z_qs = pad_sequence(n_z_qs, batch_first=True)
    #     n_z_qs *= masks
        
    #     return n_z_qs, masks, selected_encodings_list, selected_encodings_repeated_list    # shape (B, max_len, channels), mask shape (B, max_len, 1), list of selected encodings