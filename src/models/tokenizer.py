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
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.scorer = Scorer()
        self.ctcloss = nn.CTCLoss(blank=self.blank_index, zero_infinity=False, reduction='mean')


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
        mean, std = [], []
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
                rewards_dict, advantages, mean, std = self.scorer.step(sentences) # {func:rewards}, [sentences,]
                advantages = advantages.to(device)  # Move to the same device as log_probs
                
            
            # Gather per-token log-probs
            per_token_logps = torch.gather(
                log_probs[b, :lengths[b], :].unsqueeze(0).expand(self.beam_size, -1, -1),
                2,
                sequences.unsqueeze(-1)
            ).squeeze(-1)  # [self.beam_size, T']
            
            old_logps = per_token_logps.detach()
            coef = torch.exp(per_token_logps - old_logps)

            per_token_loss = -coef * advantages.unsqueeze(1)  # [self.beam_size, T'] 
            loss = loss + per_token_loss.mean()
            
        loss = loss / B
        
        if step % self.config['logging']['step']== 0:
            for k,v in rewards_dict.items(): 
                writer.add_scalar(f'perfunc-reward/mean/{k}', v.mean(), step)
                writer.add_scalar(f'perfunc-reward/std/{k}', v.std(), step)
            writer.add_scalar(f'total-reward/mean', mean, step)
            writer.add_scalar(f'total-reward/std', std, step)
        
        return loss, top_ind, top_seq
    
    def ctc_loss(self, pred_ind, log_probs, target, input_lengths, target_lengths):
        """
        Args:
            log_probs: Tensor of shape [B, T, V] - log probabilities
            target: Tensor of shape [B, L] - target sequences
            input_lengths: Tensor of shape [B] - lengths of input sequences
            target_lengths: Tensor of shape [B] - lengths of target sequences
        Returns:
            loss: scalar tensor - CTC loss
        """
        
        # Convert predicted indices to text
        pred_txts = []
        for i in range(pred_ind.size(0)):
            decoded = []
            prev = -1  # initialize with something not in vocab
            seq = pred_ind[i][:input_lengths[i]].tolist()  # Get the sequence for the i-th batch item
            for idx in seq:
                if idx != prev: decoded.append(idx)
                prev = idx
            # Step 2: Remove blanks
            decoded = [self.idx_to_char[idx] for idx in decoded if idx != self.blank_index]

            pred_txts.append(''.join(decoded))
        # print(f"Predicted texts: {pred_txts}")
        
        loss = self.ctcloss(
            log_probs, 
            target, 
            input_lengths=input_lengths, 
            target_lengths=target_lengths
        )
        
        # print(f"CTC Loss: log_probs shape: {log_probs.shape}, target shape: {target}, input_lengths shape: {input_lengths}, target_lengths shape: {target_lengths}, loss: {loss.item()}")
    
        return loss, pred_txts
    
    def forward(self, z, mask, writer=None, step=1, teacher=False, ctc=False, txts=None):
        """
        z (torch.Tensor): b,t,c
        codebook (nn.Module): A module with a weight attribute of shape (vocab_size, c).
        mask (torch.Tensor): Mask of shape (batch, time, 1) with 1s for valid positions and 0s for padding.
        """
        
        e = self.codebook.embedding.weight.clone().detach() # (vocab_size+1, c) 
        e = e[1:,:] # remove the padding idx (vocab_size, c)       
        
        b, t, c = z.shape
        z_flat = z.contiguous().view(-1, c) # (b * t, c)
        
        # For each time step gets prob corresponding to each codebook token
        log_probs = self.sim(z_flat, e).view(b, t, -1) # (b, t, vocab_size) 
        if teacher: 
            return log_probs
        
        if ctc: 
            top_ind = torch.argmax(log_probs, dim=2) # [B, T]
            input_lengths = mask.sum(dim=1).long().squeeze() # lengths of input sequences
            # convert txts to indices 
            target, target_lengths = [], torch.zeros(len(txts), dtype=torch.long) # lengths of target sequences
            for ii, txt in enumerate(txts):
                target_lengths[ii] = len(txt)
                target.append(torch.tensor([self.char_to_idx[char] for char in txt]))
            target = pad_sequence(target, batch_first=True, padding_value=self.blank_index)  # [B, L]
            ctcloss, pred_txts = self.ctc_loss(top_ind, log_probs.permute(1, 0, 2), target, input_lengths, target_lengths)
        else:
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
            writer.add_scalar('tokenizer/temp', self.temp, step)
            
        
        if ctc: 
            return ctcloss, commitment_loss, smoothness_loss, pred_txts, self.vocab, e_mean_np 
        
        z_q = z_flat + (z_q - z_flat).detach() # b*t,c  
        z_q = z_q.contiguous().view(b, t, c) # (batch, time, channels)
        z_q = z_q * mask
       
        return log_probs, z_q, smoothness_loss, commitment_loss, reinforce_loss, top_seq, self.vocab, e_mean_np 
    
    def sim(self, z_flat, e):
        # cosine similarity between z and codebooks e_j
        cos_sim = torch.matmul(z_flat, e.t()) # (b*t, vocab_size)
        # converting distance to probs    
        logits = cos_sim*self.temp
        logprobs = F.log_softmax(logits, dim=1)  # (b * t, vocab_size)
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