import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

import time
from typing import List


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
        self.blank_index = len(self.vocab) - 1

        self.scorer = Scorer()
    

    def decode_seq(self, x, random_beam_size):  
        # Step 1: Create mask to keep first element and non-duplicates
        mask = np.ones_like(x, dtype=bool)
        mask[:, 1:] = x[:, 1:] != x[:, :-1]
        # Step 2: Remove a specific index (e.g., self.blank_index)
        mask &= x != self.blank_index  # Elementwise AND

        filtered = [ ''.join( self.vocab[ x[i][mask[i]] ] ) for i in range(random_beam_size) ]
        
        return filtered
    
    
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
            sequences = sequences.squeeze(0)  # [beams, T']
            
            # Convert sequences to strings
            sentences = self.decode_seq(sequences.cpu().numpy(), self.beam_size)
            top_sent.append(sentences[0])

            # Compute advantages
            advantages_list = self.scorer.step(sentences).to(device)  # tensor[sentances,num_rewards]
            advantages = advantages_list.sum(dim=1)
            
            if step % self.config['logging']['step']== 0:
                for reward_count  in range(advantages_list.shape[1]): 
                    a = advantages_list[:,reward_count]
                    writer.add_scalar(f'advantages-min/{reward_count}', a.min().item(), step-1)
                    writer.add_scalar(f'advantages-max/{reward_count}', a.max().item(), step+1)
    
                writer.add_scalar(f'total-advantage/min', advantages.min().item(), step-1)
                writer.add_scalar(f'total-advantage/max', advantages.max().item(), step+1)
                   
            # Gather per-token log-probs
            seq_idx = sequences.unsqueeze(-1)  # [self.beam_size, T', 1]
            per_token_logps = torch.gather(
                g_log_prob.expand(self.beam_size, -1, -1),
                2,
                seq_idx
            ).squeeze(-1)  # [self.beam_size, T']

            old_logps = per_token_logps.detach()
            coef = torch.exp(per_token_logps - old_logps)

            per_token_loss = -coef * advantages.unsqueeze(1)  # [self.beam_size, T']
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
        z_flat = z.contiguous().view(-1, c) # (b * t, c)
        
        log_probs, z_q, one_hot = self.sim(z_flat, e) 
        log_probs = log_probs.view(b, t, -1) # (b, t, vocab_size)
        
        # Reinforce loss
        reinforce_loss, top = self.decode(log_probs, mask.squeeze(-1), step, writer)

        # Quantized 
        commitment_loss, smoothness_loss = self.loss(z, z_q.view(b,t,c), mask) # losses
        # Angle between the z and z_q
        if step % self.config['logging']['step'] == 0:
            
            theta = torch.sum(z_flat * z_q, dim=1, keepdim=True) * mask.view(-1,1) # (batch*time, 1)
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
            
        # Calculate codebook usage 
        e_mean_np = self.codebook_usage(one_hot, mask.view(-1, 1))
        
        return smoothness_loss, commitment_loss, reinforce_loss, top, self.vocab, e_mean_np 

    
    def sim(self, z_flat, e):
        # cosine similarity between z and codebooks e_j
        cos_sim = torch.matmul(z_flat, e.t()) # (b*t, vocab_size)
        # converting distance to probs
        logprobs = F.log_softmax(cos_sim / self.temp, dim=1)  # (b * t, vocab_size)
        
        # Find the index of the max log probability for each token
        indices = torch.argmax(logprobs, dim=1) # (b * t,)
        one_hot = F.one_hot(indices, num_classes=logprobs.shape[1]).float() # (b * t, vocab_size)
        # Quantized latents via direct indexing
        z_q = torch.matmul(one_hot, e)# (b * t, embed_dim)
        
        return logprobs, z_q.contiguous(), one_hot
    
    
    def loss(self, z, z_q, mask): 
        # commitment loss;  MSE loss between z and z_q ignoring padding positions
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # btc
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        # 9. Smoothness loss
        smoothness_loss = F.mse_loss(z[:, :-1, :], z[:, 1:, :], reduction='none') * mask[:, 1:, :] 
        smoothness_loss = smoothness_loss.sum() / valid_count 
    
        return commitment_loss, smoothness_loss
    
    def codebook_usage(self, one_hot, mask):
        # prob for each characters
        one_hot *= mask  # (B * T, V), zero out masked positions
        prob = one_hot.sum(dim=0) / mask.sum()  # (V,) probabilities of each codebook token
        # Normalize the probabilities
        prob = prob / prob.sum()
        return prob.cpu().numpy()