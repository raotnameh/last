import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch.nn.utils.rnn import pad_sequence

import numpy as np
from pyctcdecode import build_ctcdecoder
import multiprocessing

# Tokenizer module that tokenizes the speech encoder output by finding the closest codebook
class Tokenizer(nn.Module):
    def __init__(self, config, codebook, groups=2, temp=1,epsilon=0.5):
        super(Tokenizer, self).__init__()
        # temperature > 1.0: makes the distribution softer (more uniform).
        # temperature < 1.0: makes the distribution sharper (more confident).
        # temperature == 1.0: no change.
        
        self.epsilon = epsilon
        self.temp = temp
        self.codebook = codebook
        torch.compile(self.codebook.model)
        self.beam_size = groups
        self.vocab = codebook.vocab[1:] # remove the padding token
        self.save_dir = config['logging']['dir']
        os.makedirs(f"{self.save_dir}/plots/", exist_ok=True)
        self.decoder = decoder = build_ctcdecoder(["_" if x == "?" else x for x in self.vocab])
        
        self.old_logits = None
    

    def decode(self, logits, mask, iter=1):
        """
        # Args:
            # logits: logit matrix of token log probabilities. [b,t,v]
            # mask: [b,t]
            # beam_width: maximum number of beams at each step in decoding
            # beam_prune_logp: beams that are much worse than best beam will be pruned
            # token_min_logp: tokens below this logp are skipped unless they are argmax of frame

        # Returns:
            # List of beams of type OutputBeam with various meta information

        """

        device = logits.device
        # Constructing groups for GRPO
        B,T,V = logits.shape
        glogits = logits.detach().cpu().numpy() # b,t,v # group logits
        lengths = mask.sum(dim=1).to(dtype=torch.int32).cpu()  # b,
        logits_list = [glogits[b, :lengths[b], :] for b in range(B)]
        with multiprocessing.get_context("fork").Pool(processes=128) as pool: out_list = self.decoder.decode_batch(pool, logits_list, beam_width=self.beam_size)    

        ######## Advantage for each path decoded text. ########
        top, sentences, groups = [], [], [] #  b*groups , different no of groups for each seq in the batch.
        for t in out_list: 
            groups.append(len(t))
            top.append(t[0].text)
            for i in t: sentences.append(i.text) 
        # to get the start and end indices for each group
        start, indices = 0, []
        for g in groups:
            end = start + g
            indices.append((start, end))
            start = end
        
        # Get the rewards using a LLM as Judge. 
        with torch.no_grad(): 
            inputs = self.codebook.tokenizer(sentences, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device) # 0 for padding
            
            outputs = self.codebook.model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = F.log_softmax(outputs.logits, dim=-1)  # (B, T, V)
   
            target_ids = input_ids[:, 1:]
            log_probs = log_probs[:, :-1, :]
            token_log_probs = log_probs.gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
            target_mask = attention_mask[:, 1:]
            token_log_probs = token_log_probs * target_mask
            
        rewards = token_log_probs.sum(dim=1) # per-sentence sum of log-probs shape: (B,)
        
        ######## Predicted indices for each path. ########
        pred_ind = [torch.tensor(i.full_path, dtype=torch.long) for t in out_list for i in t] # list of tensors of shape T,
        pred_ind_padded = pad_sequence(pred_ind, batch_first=True, padding_value=0).unsqueeze(-1).to(device) # B*self.beamsize,T,1
        
        lengths = [len(seq) for seq in pred_ind]
        max_len = pred_ind_padded.size(1)
        pred_ind_mask = torch.zeros(pred_ind_padded.size(0), max_len, dtype=torch.int, device=device) # B*self.beamsize,T
        for i, length in enumerate(lengths): 
            pred_ind_mask[i, :length] = 1  # mark valid tokens with 1

        
        # policy probailites for each path
        per_token_logps, old_per_token_logps, advantages = [], [], []
        if iter == 1: self.old_logits = logits.detach()
        for b,(s,e) in enumerate(indices): 
            # normalize to zero mean, unit std the grouped rewards
            group_rewards = rewards[s:e]
            group_advantage = (group_rewards - group_rewards.mean()) / ( group_rewards.std(unbiased=False) + 1e-8 ) # self.beamsize,
            advantages.append(group_advantage)
            
            indices = pred_ind_padded[s:e,:,:]
            policy_log_prob = torch.gather( 
                                    logits[b,:,:].unsqueeze(0).expand(indices.shape[0],-1,-1),
                                    dim=2,
                                    index=indices)  # self.beamsize,T,1
            per_token_logps.append(policy_log_prob)
            
            if iter > 1: 
                old_policy_log_prob = torch.gather( 
                                        self.old_logits[b,:,:].unsqueeze(0).expand(indices.shape[0],-1,-1),
                                        dim=2,
                                        index=indices)  # self.beamsize,T,1
                old_per_token_logps.append(old_policy_log_prob)
        
        advantages = torch.cat(advantages, dim=0)
        per_token_logps = torch.cat(per_token_logps, dim=0)
        if iter <= 1: old_per_token_logps = per_token_logps.detach()
        else: old_per_token_logps = torch.cat(old_per_token_logps, dim=0)
        
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)

        ######## Reinforced policies ########
        per_token_loss1 = coef_1.squeeze(-1) * advantages.unsqueeze(1) # self.beamsize,T,1
        per_token_loss2 = coef_2.squeeze(-1) * advantages.unsqueeze(1)
        
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        loss = ((per_token_loss * pred_ind_mask).sum(-1) / pred_ind_mask.sum(-1).clamp(min=1.0)).mean()
        # loss = ((per_token_loss * pred_ind_mask).sum(-1)).mean()
        # loss = (per_token_loss * pred_ind_mask).sum()

        return loss, top
        

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
        log_probs = torch.nn.functional.log_softmax( -d.view(b, t, -1) / (self.temp/step), dim=-1) # shape: (b, t, vocab_size)
        log_probs = log_probs * mask + (1 - mask) * (-1e9)
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
   
    @staticmethod
    def get_very_efficient_rotation( u, q, x):
        w = F.normalize(u + q, dim=1).detach()
        return x - 2*torch.bmm(torch.bmm(x, w.unsqueeze(-1)), w.unsqueeze(1)) + 2*torch.bmm( torch.bmm(x, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
    
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          