import torch
import numpy as np
from collections import Counter
from math import log
from tqdm.auto import tqdm
import logging

@torch.jit.script
def beam_search(log_probs: torch.Tensor, beam_size: int):
    """
    Performs beam search on a tensor of log probabilities.

    Args:
        log_probs (torch.Tensor): Tensor of shape (b, t, v) containing log probabilities.
        beam_size (int): Number of beams to keep at each time step.

    Returns:
        sequences (torch.Tensor): Tensor of shape (b, beam_size, t) containing the top sequences.
        scores (torch.Tensor): Tensor of shape (b, beam_size) containing the scores of the top sequences.
    """
    
    b, t, v = log_probs.size()
    
    initial_beam_size = min(beam_size, v) # At the very first step (time step 0), we can't have more beams than the vocabulary size. This line ensures that the initial number of beams considered doesn't exceed the number of possible first tokens.

    topk_scores, topk_indices = torch.topk(log_probs[:, 0, :], initial_beam_size, dim=-1) # Returns the k largest elements of the given input tensor along a given dimension
    sequences = topk_indices.unsqueeze(-1)  # (b, initial_beam_size, 1)
    scores = topk_scores  # (b, initial_beam_size)

    for step in range(1, t):
        # Expand the current sequences with all possible next tokens
        current_log_probs = log_probs[:, step, :].unsqueeze(1)  # (b, 1, v)
        expanded_scores = scores.unsqueeze(-1) + current_log_probs  # (b, beam_size, v)
        flat_scores = expanded_scores.view(b, -1)  # (b, beam_size * v)

        # Select the top-k scores and their corresponding indices
        topk_flat_scores, topk_indices = flat_scores.topk(beam_size, dim=-1)  # (b, beam_size)
        beam_indices = topk_indices // v  # Indices of sequences to expand
        token_indices = topk_indices % v  # New tokens to append

        # Gather the sequences to expand and append the new tokens
        sequences = torch.gather(sequences, 1, beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
        sequences = torch.cat([sequences, token_indices.unsqueeze(-1)], dim=-1)  # (b, beam_size, step+1)

        # Update the scores
        scores = topk_flat_scores

    return sequences, scores.unsqueeze(-1)


class Scorer:
    
    def __init__(self) -> None:
        with open("/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train.wrd", "r") as f:
            out = f.readlines()

            # uniq word list
            words_list = [word for x in out for word in x.strip().split(" ")]
            self.unique_words = set(words_list)
            
            # uniq char count
            all_text = ''.join(x.strip() for x in out) # Flatten to one long string of all characters
            char_counts = Counter(all_text) # Count characters
            self.char_probs = {char: count / sum(char_counts.values()) for char, count in char_counts.items()} # Probs
            
            # ratio of character to words for a sentence
            self.avg_char = sum( [ len(sent)/len(sent.split(" ")) for sent in out] )
            self.avg_char /= len(out)
            
            # ratio of words to char 
            self.avg_wrd = sum( [ len(sent.split(" "))/len(sent) for sent in out] )
            self.avg_char /= len(out)
            
            logging.info(f"avg ration of charcters to words in a seq: {self.avg_char}")
            
            
    @torch._dynamo.disable
    def step(self, sentences):
        self.sentences = sentences
        
        
        return [
            self.seen(),
            self.unigram_char(),
            self.char_to_word_ratio(),
            self.word_to_char_ratio(),
            
        ]
        
    def char_to_word_ratio(self,):
        reward = np.array([0.0]*len(self.sentences))
        
        for i, s in enumerate(self.sentences):
            cur = len(s) / (len(s.split(" "))+1)
            reward[i] = -abs( self.avg_char - cur )
        
        reward = torch.tensor(reward)
        mean, std = reward.mean(), reward.std(unbiased=False)
        if std == 0: return torch.zeros_like(reward)  # or leave as-is
        else: return (reward - mean) / std
        
    def word_to_char_ratio(self,):
        reward = np.array([0.0]*len(self.sentences))
        
        for i, s in enumerate(self.sentences):
            cur = len(s.split(" ")) / (len(s)+1) 
            reward[i] = -abs( self.avg_char - cur )
        
        reward = torch.tensor(reward)
        mean, std = reward.mean(), reward.std(unbiased=False)
        if std == 0: return torch.zeros_like(reward)  # or leave as-is
        else: return (reward - mean) / std
        
    def seen(self,):
        reward = np.array([0.0]*len(self.sentences))
        
        for i, s in enumerate(self.sentences):
            words = s.split()
            # Count how many words are not in unique_words set
            reward[i] = -sum(1 for w in words if w not in self.unique_words) / (len(words)+1)
  
        reward = torch.tensor(reward)
        mean, std = reward.mean(), reward.std(unbiased=False)
        if std == 0: return torch.zeros_like(reward)  # or leave as-is
        else: return (reward - mean) / std
    
    def unigram_char(self,):
        reward = np.array([0.0]*len(self.sentences))
        
        for i, s in enumerate(self.sentences):
            char_counts = Counter( ''.join(s) )
            char_probs = {char: count / sum(char_counts.values()) for char, count in char_counts.items()} # Probs
            
            su = 0
            for char in char_probs.keys():
                su += -abs(char_probs[char] - self.char_probs[char])            
            reward[i] = su
            
        reward = torch.tensor(reward)
        mean, std = reward.mean(), reward.std(unbiased=False)
        if std == 0: return torch.zeros_like(reward)  # or leave as-is
        else: return (reward - mean) / std
        
            
            
            
            
        
        
        
        
        
    