import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch.nn.utils.rnn import pad_sequence

from utils import *
import random

import re
import torch

# Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
class Tokenizer(nn.Module):
    def __init__(self, config, codebook, groups=2, temp=1,epsilon=0.5):
        super(Tokenizer, self).__init__()
        # temperature > 1.0: makes the distribution softer (more uniform).
        # temperature < 1.0: makes the distribution sharper (more confident).
        # temperature == 1.0: no change.
        
        self.codebook = codebook
        
        self.epsilon = epsilon
        self.temp = temp
        self.beam_size = groups
        
        self.vocab = codebook.vocab[1:] # remove the padding token
        self.idx2char = {i:c for i,c in enumerate(self.vocab)}
        
        self.save_dir = config['logging']['dir']
        os.makedirs(f"{self.save_dir}/plots/", exist_ok=True)
        
        self.old_logits = None
        
        self.scorer = Scorer()
    
    def decode(self, log_probs, mask, iter=1):
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

        # Precompile regex to remove blanks and collapse repeats
        blank_char = re.escape('?')  # adjust if blank differs
        remove_blanks = re.compile(blank_char)
        collapse_repeats = re.compile(r'(.)\1+')

        def merge_string(s: str) -> str:
            # remove blank symbols then collapse repeats
            s = remove_blanks.sub('', s)
            return collapse_repeats.sub(r'\1', s)

        lengths = mask.sum(dim=1).long()  # [B]
        loss = 0.0
        top_sent = []

        for b in range(B):
            valid_len = lengths[b].item()
            g_log_prob = log_probs[b, :valid_len, :].unsqueeze(0)  # [1, T', V]

            # Beam search
            sequences, _ = beam_search(g_log_prob, self.beam_size)  # [1, beams, T']
            beams = sequences.size(1)

            # Sample indices and select sequences
            sample_idxs = torch.randint(0, beams, (32,), device=device)
            sampled_seqs = sequences[0, sample_idxs]  # [32, T']

            # Convert token sequences to strings using regex merge
            sentences = []
            for seq in sampled_seqs:
                chars = [self.idx2char[i] for i in seq.tolist()]
                raw = ''.join(chars)
                sentences.append(merge_string(raw))
            top_sent.append(sentences[0])

            # Compute advantages
            advantages = self.scorer.step(sentences).to(device)  # [32]

            # Gather per-token log-probs
            seq_idx = sampled_seqs.unsqueeze(-1)  # [32, T', 1]
            per_token_logps = torch.gather(
                g_log_prob.expand(32, -1, -1), 2, seq_idx
            ).squeeze(-1)  # [32, T']

            old_logps = per_token_logps.detach()
            coef = torch.exp(per_token_logps - old_logps)

            per_token_loss = -coef * advantages.unsqueeze(1)  # [32, T']
            loss += per_token_loss.mean()

        loss /= B
        return loss, top_sent

    # def decode(self, log_probs, mask, iter=1):
    #     """
    #     Args:
    #         log_probs: Tensor of shape [B, T, V]
    #         mask: Tensor of shape [B, T]
    #     Returns:
    #         loss: scalar tensor
    #         top_sent: list of best sentence strings per batch
    #     """
    #     B, T, V = log_probs.shape
    #     device = log_probs.device

    #     lengths = mask.sum(dim=1).long()  # [B]
    #     loss = 0.0
    #     top_sent = []

    #     for b in range(B):
    #         valid_len = lengths[b].item()   # now a Python int, safe for slicing
    #         g_log_prob = log_probs[b, :valid_len, :].unsqueeze(0)  # [1, T', V]

    #         # Beam search: [1, beams, T']
    #         sequences, _ = beam_search(g_log_prob, self.beam_size)
            
    #         # Sample 32 hypotheses
    #         sample_idxs = torch.randint(0, sequences.size(1), (32,))
    #         sampled_seqs = sequences[0, sample_idxs]  # [32, T']

    #         # Convert token sequences to strings
    #         sentences = [ctc_merge_string(''.join(self.idx2char[i] for i in row)) for row in sampled_seqs.tolist()]
    #         top_sent.append(sentences[0])

    #         # Score each sentence: [32]
    #         advantages = self.scorer.step(sentences).to(device)  # [32]
            
    #         # Recompute per-token log probs for sampled sequences
    #         # Prepare for gather: [1, T', V] and [32, T']
    #         seq_idx = sampled_seqs.unsqueeze(-1)  # [32, T', 1]
    #         per_token_logps = torch.gather(g_log_prob.expand(32, -1, -1), 2, seq_idx).squeeze(-1)  # [32, T']

    #         # Baseline logprobs (detached)
    #         old_logps = per_token_logps.detach()
    #         coef = torch.exp(per_token_logps - old_logps)

    #         # Loss: negative reward times the coefficient
    #         per_token_loss = -coef * advantages.unsqueeze(1)  # [32, T']
    #         loss += per_token_loss.mean()

    #     loss /= B
    #     return loss, top_sent

    
    # def decode(self, log_probs, mask, iter=1):
    #     """
    #     # Args:
    #         # log probabilities. [b,t,v]
    #         # mask: [b,t]
    #     # Returns:
    #         # List of beams of type OutputBeam with various meta information

    #     """
    #     B,T,V = log_probs.shape
    #     device = log_probs.device
        
    #     #### NEED TO MAKE SURE THAT BEAM SEARCH IS NOT BEING APPLIED ON MASKED INDICES
    #     ############ ONE WAY IS DO EVERYTHONG BATCHWISE.ALSO EASY TO GET HE GROUPING AND WCEYRTOGN  AND TO OGNORE MASKING
        
    #     g_log_probs= log_probs.clone().detach() # b,t,v # group logprob
    #     lengths = mask.sum(dim=1).to(dtype=torch.int32)  # b,
    #     g_log_probs_list = [g_log_probs[b,:lengths[b],:].unsqueeze(0) for b in range(B)] # bsz=1
        
    #     loss = 0.0
    #     top_sent = []
    #     for g_log_prob in g_log_probs_list:    
            
    #         # Perform beam search
    #         sequences, _ = beam_search(g_log_prob, self.beam_size) #  1,beamsize,seq_len,  # 1, beamsize,
    #         # To get different sequences. better and worse. 
    #         samples = random.choices(range(sequences.shape[1]), k=32)
    #         sequences = sequences[:,samples,:]  # 1, samples, seq_len

    #         # Possible sentences to score.
    #         rows = sequences[0].cpu().tolist() # beam,T
    #         sentences = [ctc_merge_string( ''.join(self.idx2char[i] for i in row) ) for row in rows] # 1 * beam ,
    #         top_sent.append(sentences[0])
    #         # print(sentences)
            
    #         # Scorer
    #         advantages = self.scorer.step(sentences).unsqueeze(-1).to(device) # samples,1
            
    #         per_token_logps = torch.gather(log_probs, 2, sequences.transpose(1,2)).transpose(1,2).squeeze(0) # samples, T
    #         old_per_token_logps = per_token_logps.detach()
        
    #         coef_1 = torch.exp(per_token_logps - old_per_token_logps) # samples, T

    #         per_token_loss = -coef_1 * advantages # samples, T
            
    #         loss += per_token_loss.mean() # ()
            
    #     loss /= B # mean by all the abtch and atheir groups
        
    
    #     return loss, top_sent
        
    def forward(self, z, mask, writer=None, step=1, iter=1):
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
        reinforce_loss, top = self.decode(log_probs, mask.squeeze(-1), iter)

        # Quantized
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # (b * t, 1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)  # (b * t, vocab_size)
        min_encodings.scatter_(1, min_encoding_indices, 1)  # (b * t, vocab_size)
        # 5. Quantized latents via direct indexing
        z_q = torch.matmul(min_encodings, e).view(b,t,c) * mask # (batch, time, channels) 
        # Losses
        commitment_loss, smoothness_loss = self.loss(z, z_q, mask)
        # Angle between the z and z_q
        if writer and step % 1000 == 0:
            theta = torch.sum(z_flat * z_q.contiguous().view(-1, c), dim=1, keepdim=True) * mask.view(-1,1) # (batch*time, 1)
            
            writer.add_scalar('tokenizer/theta_mean', theta.mean().item(), step)
            writer.add_scalar('tokenizer/theta_std', theta.std().item(), step)
            writer.add_scalar('tokenizer/theta_max', theta.max().item(), step)
            writer.add_scalar('tokenizer/theta_min', theta.min().item(), step)
            
        self.codebook_usage(min_encodings, mask.contiguous().view(-1, 1), step)
        
        return smoothness_loss, commitment_loss, reinforce_loss, top # smoothness_loss, commitment_loss, reinforce_loss
    
    
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
   
    