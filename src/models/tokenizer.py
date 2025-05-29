import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence

# Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
class Tokenizer(nn.Module):
    def __init__(self, config, codebook, groups=2, temp=1.0):
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

        return [ ''.join( self.vocab[ x[i][mask[i]] ] ) for i in range(random_beam_size) ]
        
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
        g_log_probs = log_probs.detach()
        
        lengths = mask.sum(dim=1).long()  # [B]
        loss = 0.0
        top_ind = []
        top_seq = []

        for b in range(B):
            with torch.no_grad():
                g_log_prob = g_log_probs[b, :lengths[b], :].unsqueeze(0)  # [1, T', V]      
                # Beam search
                sequences = beam_search(g_log_prob, self.beam_size).squeeze(0)  # [beams, T']
                # Convert sequences to strings
                sentences = self.decode_seq(sequences.cpu().numpy(), self.beam_size)
                top_ind.append(sequences[0])
                top_seq.append(sentences[0])
                # Compute advantages
                rewards_dict = self.scorer.step(sentences)  # tensor[sentences,num_rewards]
                advantages = torch.stack( list(rewards_dict.values()), dim=1 ).sum(dim=1).to(device) # [sentences,] mean over all rewards
            
            # Gather per-token log-probs
            per_token_logps = torch.gather(
                log_probs[b, :lengths[b], :].unsqueeze(0).expand(self.beam_size, -1, -1),
                2,
                sequences.unsqueeze(-1)
            ).squeeze(-1)  # [self.beam_size, T']
            
            old_logps = per_token_logps.detach()
            coef = torch.exp(per_token_logps - old_logps)

            per_token_loss = -coef * advantages.unsqueeze(1)  # [self.beam_size, T']
            loss += per_token_loss.mean()
        loss /= B
        
        if step % self.config['logging']['step']== 0:
            for k,a in rewards_dict.items(): 
                writer.add_scalar(f'advantages-min/{k}', a.min().item(), step-1)
                writer.add_scalar(f'advantages-max/{k}', a.max().item(), step+1)

            writer.add_scalar(f'total-advantage/min', advantages.min().item(), step-1)
            writer.add_scalar(f'total-advantage/max', advantages.max().item(), step+1)
        
        return loss, top_ind, top_seq
    
    
    def forward(self, z, mask, writer=None, step=1, teacher=False):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, c).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """
        
        e = self.codebook.embedding.weight.clone().detach() # (vocab_size+1, c) 
        e = e[1:,:] # remove the padding idx (vocab_size, c)       
        
        b, t, c = z.shape
        z_flat = z.contiguous().view(-1, c) # (b * t, c)
        
        log_probs = self.sim(z_flat, e).view(b, t, -1) # (b, t, vocab_size) 
        if teacher: return log_probs.detach()
            
        # Reinforce loss
        reinforce_loss, top_ind, top_seq  = self.decode(log_probs, mask.squeeze(-1), step, writer)
        
        # Quantized 
        z_q, one_hot = self.indexing(top_ind, e) 
        
        # Calculate codebook usage 
        e_mean_np = self.codebook_usage(one_hot, mask.view(-1, 1)) # (V,) probabilities of each codebook token
        # Quantized 
        commitment_loss, smoothness_loss = self.loss(z, z_q.view(b,t,c), mask) # losses
        # Angle between the z and z_q
        if step % self.config['logging']['step'] == 0:    
            theta = torch.sum(z_flat * z_q, dim=1, keepdim=True) * mask.view(-1,1) # (batch*time, 1)
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
        
        z_q = z_flat + (z_q - z_flat).detach() # b*t,c  
        z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        z_q = z_q * mask
       
        return log_probs, z_q, smoothness_loss, commitment_loss, reinforce_loss, top_seq, self.vocab, e_mean_np 
    
    def sim(self, z_flat: torch.Tensor, e: torch.Tensor):
        # cosine similarity between z and codebooks e_j
        cos_sim = torch.matmul(z_flat, e.t()) # (b*t, vocab_size)
        # converting distance to probs
        logprobs = F.log_softmax(cos_sim, dim=1)  # (b * t, vocab_size)
        return logprobs
    
    def indexing(self, top_ind, e):
        top_ind = pad_sequence(top_ind, batch_first=True)  # shape: [batch_size, max_t]
        top_ind = top_ind.view(-1)
        one_hot = F.one_hot(top_ind, num_classes=e.shape[0]).float() # (b * t, vocab_size)
        # Quantized latents via direct indexing
        z_q = torch.matmul(one_hot, e)# (b * t, embed_dim)
        return z_q, one_hot
        
    def loss(self, z: torch.Tensor, z_q: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # commitment loss;  MSE loss between z and z_q ignoring padding positions
        commitment_loss = F.mse_loss(z, z_q.detach(), reduction='none') * mask # btc
        valid_count = mask.sum() * z.shape[-1] # Total number of valid (non-masked) elements
        commitment_loss = commitment_loss.sum() / valid_count 
        
        # 9. Smoothness loss
        smoothness_loss = F.mse_loss(z[:, :-1, :], z[:, 1:, :], reduction='none') * mask[:, 1:, :] 
        smoothness_loss = smoothness_loss.sum() / valid_count 
    
        return commitment_loss, smoothness_loss
    
    def codebook_usage(self, one_hot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        # prob for each characters
        one_hot *= mask  # (B * T, V), zero out masked positions
        prob = one_hot.sum(dim=0) / mask.sum()  # (V,) probabilities of each codebook tokenr
        return prob