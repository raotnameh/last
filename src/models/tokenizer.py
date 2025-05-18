import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch.nn.utils.rnn import pad_sequence


class ReinforceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, advantage):
        # Save advantage for backward; it should already be baseline-adjusted
        ctx.save_for_backward(advantage)
        return x  # identity on the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved advantage (shape b, t, c)
        (advantage,) = ctx.saved_tensors
        # Detach so we don't backprop into the advantage calculation
        adv = advantage.detach()
        # Scale the incoming gradient by the advantage
        return grad_output * adv, None  # no gradient w.r.t. advantage

# # Usage in your model:
# z = your_layer(x)                           # -> (b, t, c)
# # `advantage` must be shape (b, t, c), computed beforehand
# z_reinforced = ReinforceGrad.apply(z, advantage)
# # proceed with the rest of your network…

# advantage = self.rl_loss_from_logits()
# cur_z_q = ReinforceGrad.apply(cur_z_q, advantage)
    
    
class FirstTimeStepWithBroadcastGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x: (t, c)
        ctx.save_for_backward(x)
        return x[0:1, :]  # (1, c)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        t, c = x.shape
        # Broadcast grad_output (1, c) -> (t, c)
        grad_input = grad_output.expand(t, -1)
        return grad_input
    
class Tokenizer(nn.Module):
    def __init__(self, config, vocab, rot=True, beams=1):
        super(Tokenizer, self).__init__()
        '''
        Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
        '''

        self.beams = beams
        self.vocab = vocab[1:] # remove the padding token
        self.rot = rot
        self.save_dir = config['logging']['dir']
        os.makedirs(f"{self.save_dir}/plots/", exist_ok=True)
    
    def codebook_usage(self, min_encodings, mask, step):
        if step % 10 == 0:
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
            
            plt.savefig(os.path.join(f'{self.save_dir}/plots', f'codebook_usage_distribution_{step}.png'), bbox_inches='tight')
            plt.close()
   
    @staticmethod
    def get_very_efficient_rotation( u, q, x):
        w = F.normalize(u + q, dim=1).detach()
        return x - 2*torch.bmm(torch.bmm(x, w.unsqueeze(-1)), w.unsqueeze(1)) + 2*torch.bmm( torch.bmm(x, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
    
          
    def forward(self, z, codebook, mask, writer=None, step=1, skip_non_speech=False):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """  
        
        # use for reward modelling. 
        student_prob = z.clone()
        
        z = F.normalize(z, dim=-1)
        
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
        ) # (b*t, vocab_size) (10*1000, 29)
        
        # 3.1 converting distance to probs
        # log_probs = torch.nn.functional.log_softmax(-d, dim=1)  # shape: (b*t, vocab_size)
        # topk_log_probs, topk_indices = torch.topk(log_probs, k=self.beams, dim=1)  # both shape: (b*t, 5) beam is 5
        # # greedy_indices = topk_indices[:,:1]  # shape: (b * t,)
        # top_z_q = []
        # for beam in range(topk_indices.shape[1]): 
        #     cur_min_encoding_indices = topk_indices[:,beam].unsqueeze(1) # (b * t, 1)
        #     cur_min_encodings = torch.zeros(cur_min_encoding_indices.shape[0], e.shape[0], device=z.device)  # (b * t, vocab_size)
        #     cur_min_encodings.scatter_(1, cur_min_encoding_indices, 1)  # (b * t, vocab_size)
        #     # 5. Quantized latents via direct indexing
        #     cur_z_q = torch.matmul(cur_min_encodings, e).view(b,t,c) # (batch, time, channels) 
        #     top_z_q.append( cur_z_q)
          
        # 4. find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # (b * t, 1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)  # (b * t, vocab_size)
        min_encodings.scatter_(1, min_encoding_indices, 1)  # (b * t, vocab_size)
        # 5. Quantized latents via direct indexing
        z_q = torch.matmul(min_encodings, e).view(b,t,c) # (batch, time, channels) 
        
        
        # Angle between the z and z_q
        x = z_flat # (batch*time, channels)
        quantized = z_q.contiguous().view(-1, c) # (batch*time, channels)
        theta = torch.sum(x * quantized, dim=1) # (batch*time)
        theta_mask = theta > 0.5 # (batch*time) # Limitation of the roation trick. It avoids the rotation trick when the angle is too small. which results in opposite direction of gradeints for codebook  and  encoder output.
        theta_mask = theta_mask.float().unsqueeze(1) # (batch*time, 1)
        # count of theta_mask of value 1 
        if writer and step % 1000 == 0:
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
            writer.add_scalar('tokenizer/theta_mask_mean', theta_mask.sum().item(), step)
        
        # # 6. Apply rotation trick on already normalized vectors           
        # r_z_q = self.get_very_efficient_rotation(x , quantized, x.unsqueeze(1)).squeeze() 
        # # 7. Straight-through estimator and mask padding
        # s_z_q = z_flat + (quantized - z_flat).detach() # btc  
        
        # z_q = theta_mask * r_z_q + (1 - theta_mask) * s_z_q # btc   
        # z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        # z_q = z_q * mask
        
        # 7. Straight-through estimator and mask padding
        z_q = z_flat + (quantized - z_flat).detach() # b*t,c  
        z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        z_q = z_q * mask
        
        # 8. commitment loss;  MSE loss between z and z_q ignoring padding positions
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # btc
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        # 9. Smoothness loss
        smoothness_loss = F.mse_loss(z[:, :-1, :], z[:, 1:, :], reduction='none') * mask[:, 1:, :] 
        smoothness_loss = smoothness_loss.sum() / valid_count 
        
        # 10. Discriminator codebooks without repeated indices #####
        encodings = min_encoding_indices.view(z.shape[0], z.shape[1]) # ( batch, time ) # (B, T)
        n_student_probs, n_z_q, n_mask, selected_encodings_list, selected_encodings_repeated_list = self.remove_consecutive_repeated_indices( encodings, mask.squeeze(-1), z_q.clone(), student_prob, skip_non_speech) # randomly pick one index from each group of consecutive repeating elements # shape (B,T) and also returns the mask 

        # codebook usage Distribution
        self.codebook_usage(min_encodings, mask.contiguous().view(-1, 1), step)

        return n_student_probs, smoothness_loss, commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list, selected_encodings_repeated_list # commitment_loss, z_q, n_z_q, n_mask, selected_encodings_list<=


    def remove_consecutive_repeated_indices(self, min_encoding_indices, mask, z_q, student_prob, skip_non_speech=False):

        B, T, C = z_q.shape
        
        selected_encodings_list = []
        selected_encodings_repeated_list = []
        n_z_qs = []
        n_student_probs = []
        
        max_len = 0
        
        masks_tensor = torch.zeros((B, T), dtype=torch.float, device=mask.device)
        
        for b in range(B):
            indices = min_encoding_indices[b]    # (T,)
            mask_b = mask[b]                     # (T,)
            z_q_b = z_q[b]                       # (T, C)
            # student_prob_b = student_prob[b] 

            # how many valid frames until first zero in mask
            valid_len = int(mask_b.sum())

            # crop to valid region
            indices = indices[:valid_len]
            z_q_b = z_q_b[:valid_len]
            # student_prob_b = student_prob_b[:valid_len]
            
            # collapse runs of identical indices
            unique_vals, counts = torch.unique_consecutive(indices, return_counts=True)
            ends  = torch.cumsum(counts, dim=0)
            starts = torch.cat((torch.tensor([0], device=ends.device), ends[:-1]))
            lengths = ends - starts

            selected_encodings = []
            selected_encodings_repeated = []
            n_z_q = []
            # n_student_prob = []
            
            # iterate segments (far fewer than T if many repeats)
            for v, s, e, length in zip(unique_vals, starts, ends, lengths):
                # record each frame’s encoding+1
                selected_encodings_repeated.extend([v + 1] * length) # +1 to avoid padding token
                
                if v == 28 and skip_non_speech: continue # skip non-speech runs
                
                # mark the last frame of this segment
                selected_encodings.append(v + 1) # +1 to avoid padding token

                segment = z_q_b[s:e]  # shape (length, C)
                n_z_q.append( FirstTimeStepWithBroadcastGrad.apply(segment) )
                
                # student_segment = student_prob_b[s:e]
                # n_student_prob.append( FirstTimeStepWithBroadcastGrad.apply(student_segment) )
                
            # append per-batch results
            selected_encodings_list.append(selected_encodings)
            selected_encodings_repeated_list.append(selected_encodings_repeated)
            try:
                n_z_qs.append(torch.cat(n_z_q, dim=0))  # (num_segs, C)
                # n_student_probs.append(torch.cat(n_student_prob, dim=0))  # (num_segs, C)
            except: 
                print(n_z_q)
                n_z_qs.append(torch.zeros((1, C), device=z_q.device, dtype=z_q.dtype))
                # n_student_probs.append(torch.zeros((1, C), device=z_q.device, dtype=z_q.dtype))
            
            # build mask and track max length
            L = len(selected_encodings)
            masks_tensor[b, :L] = 1.0
            max_len = max(max_len, L)
        
        # finalize masks and pad n_z_qs
        masks = masks_tensor[:, :max_len].unsqueeze(-1)  # (B, max_len, 1)
        n_z_qs = pad_sequence(n_z_qs, batch_first=True)  # (B, max_len, C)
        n_z_qs *= masks
           
        # n_student_probs = pad_sequence(n_student_probs, batch_first=True)
        # n_student_probs *= masks
        n_student_probs = n_z_qs
          
        return n_student_probs, n_z_qs, masks, selected_encodings_list, selected_encodings_repeated_list    # shape (B, max_len, channels), mask shape (B, max_len, 1), list of selected encodings
    
        
        
        