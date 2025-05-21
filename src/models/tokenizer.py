import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

import re


# Precompile regex to remove blanks and collapse repeats
blank_char = re.escape('?')  # adjust if blank differs
remove_blanks = re.compile(blank_char)
collapse_repeats = re.compile(r'(.)\1+')

# Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
class Tokenizer(nn.Module):
    def __init__(self, config, codebook, groups=2, temp=1):
        super(Tokenizer, self).__init__()
        # temperature > 1.0: makes the distribution softer (more uniform).
        # temperature < 1.0: makes the distribution sharper (more confident).
        # temperature == 1.0: no change.
        
        self.codebook = codebook
        self.config = config
        self.temp = temp
        self.beam_size = groups
        
        self.vocab = np.array(codebook.vocab[1:]) # remove the padding token
        
        
        self.old_logits = None
        
        self.scorer = Scorer()
    
    # Convert token sequences to strings using regex merge
    def decode_string(self,arr):
        return [
            collapse_repeats.sub(r'\1',
                    remove_blanks.sub('',
                        ''.join(self.vocab[row])
                        )) 
                            for row in arr
            ]
    
    def decode(self, log_probs, mask, step, writer):
        """
        Args:
            log_probs: Tensor of shape [B, T, V]
            mask: Tensor of shape [B, T]
        Returns:
            loss: scalar tensor
            top_sent: list of best sentence strings per batch
        """
        B, T, V = log_probs.shape
        device = log_probs.device
        
        lengths = mask.sum(dim=1).long()  # [B]
        loss = 0.0
        top_sent = []

        for b in range(B):
            valid_len = lengths[b]
            g_log_prob = log_probs[b, :valid_len, :].unsqueeze(0)  # [1, T', V]

            # Beam search
            sequences, _ = beam_search(g_log_prob, self.beam_size)  # [1, beams, T']

            # Sample indices and select sequences
            random_beam_size = min(8, self.beam_size)
            sample_idxs = torch.randint(0, self.beam_size, (random_beam_size,), device=device)
            sampled_seqs = sequences[0, sample_idxs]  # [random_beam_size, T']
            
            sentences = self.decode_string(sampled_seqs.cpu())
            top_sent.append(sentences[0])
            
            # Compute advantages
            advantages_list = self.scorer.step(sentences).to(device)  # tensor[sentances,num_rewards]
            advantages = advantages_list.mean(dim=1)
            
            if step % self.config['logging']['step']== 0:
                for reward_count  in range(advantages_list.shape[1]): 
                    a = advantages_list[:,reward_count]
                    writer.add_scalar(f'advantages-min/{reward_count}', a.min().item(), step-1)
                    writer.add_scalar(f'advantages-mean/{reward_count}', a.mean().item(), step)
                    writer.add_scalar(f'advantages-max/{reward_count}', a.max().item(), step+1)
    
                writer.add_scalar(f'advantages/total-advantage', advantages.min().item(), step-1)
                writer.add_scalar(f'advantages/total-advantage', advantages.mean().item(), step)
                writer.add_scalar(f'advantages/total-advantage', advantages.max().item(), step+1)
                   
            # Gather per-token log-probs
            seq_idx = sampled_seqs.unsqueeze(-1)  # [random_beam_size, T', 1]
            per_token_logps = torch.gather(
                g_log_prob.expand(random_beam_size, -1, -1),
                2,
                seq_idx
            ).squeeze(-1)  # [random_beam_size, T']

            old_logps = per_token_logps.detach()
            coef = torch.exp(per_token_logps - old_logps)

            per_token_loss = -coef * advantages.unsqueeze(1)  # [random_beam_size, T']
            loss += per_token_loss.mean()

        loss /= B
        return loss, top_sent
        
    def forward(self, z, mask, writer=None, step=1):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, embed_dim).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """  
        
        e = self.codebook.embedding.weight.clone().detach() # (vocab_size+1, embed_dim) 
        e = e[1:,:] # remove the padding idx (vocab_size, embed_dim)       
        
        b, t, c = z.shape
        z = F.normalize(z, dim=-1) # normlaize
        z_flat = z.contiguous().view(-1, c) # (b * t, c)

        d = self.dist(z_flat, e) # (b*t, vocab_size)
        
        # converting distance to probs
        log_probs = torch.nn.functional.log_softmax( -d.view(b, t, -1) / self.temp, dim=-1) # shape: (b, t, vocab_size)
        reinforce_loss, top = self.decode(log_probs, mask.squeeze(-1), step, writer)

        # Quantized
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # (b * t, 1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)  # (b * t, vocab_size)
        min_encodings.scatter_(1, min_encoding_indices, 1)  # (b * t, vocab_size)
        # 5. Quantized latents via direct indexing
        z_q = torch.matmul(min_encodings, e).view(b,t,c) * mask # (batch, time, channels) 
        # Losses
        commitment_loss, smoothness_loss = self.loss(z, z_q, mask)
        # Angle between the z and z_q
        if step % self.config['logging']['step'] == 0:
            theta = torch.sum(z_flat * z_q.contiguous().view(-1, c), dim=1, keepdim=True) * mask.view(-1,1) # (batch*time, 1)
            
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
            
        e_mean_np = self.codebook_usage(min_encodings, mask.contiguous().view(-1, 1), step)
        
        return smoothness_loss, commitment_loss, reinforce_loss, top, self.vocab, e_mean_np # smoothness_loss, commitment_loss, reinforce_loss
    
    
    def dist(self, z_flat, e):
        # distances from z to codebooks e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (torch.sum(z_flat**2, dim=1, keepdim=True) \
            - 2 * z_flat @ e.t() \
                + torch.sum(e**2, dim=1, keepdim=True).t() 
        ) # (b*t, vocab_size)
        
        return d
    
    
    def loss(self, z, z_q, mask): 
        # commitment loss;  MSE loss between z and z_q ignoring padding positions
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # btc
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        # 9. Smoothness loss
        smoothness_loss = F.mse_loss(z[:, :-1, :], z[:, 1:, :], reduction='none') * mask[:, 1:, :] 
        smoothness_loss = smoothness_loss.sum() / valid_count 
    
        return commitment_loss, smoothness_loss
    
    def codebook_usage(self, min_encodings, mask, step):
        # prob for each character
        mask_bool = (mask == 1).squeeze(1)  # shape: (B,), True where we keep
        valid_encodings = min_encodings[mask_bool]  # shape: (B', C)
        e_mean_np = valid_encodings.mean(dim=0).cpu().numpy()
        return e_mean_np