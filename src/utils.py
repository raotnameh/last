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
            self.avg_wrd /= len(out)
            
            logging.info(f"avg ration of charcters to words in a seq: {self.avg_char}")
            logging.info(f"avg ration of charcters to words in a seq: {self.avg_wrd}")
            
            
    def step(self, sentences):
        self.sentences = sentences
        rewards = torch.stack([
                    self.seen(),                  # shape: [num_sentences]
                    self.unigram_char(),
                    self.char_to_word_ratio(),
                    self.word_to_char_ratio(),
                ], dim=1)  # stack along new dimension → shape becomes [num_sentences, 4]
        
        return rewards

    def char_to_word_ratio(self):
        char_lens = np.array([len(s) for s in self.sentences])
        word_counts = np.array([len(s.split(" ")) + 1 for s in self.sentences])
        cur = char_lens / word_counts
        reward = -np.abs(self.avg_char - cur)

        reward = torch.tensor(reward)
        std = reward.std(unbiased=False)
        return torch.zeros_like(reward) if std == 0 else (reward - reward.mean()) / std
        
    def word_to_char_ratio(self):
        word_counts = np.array([len(s.split(" ")) for s in self.sentences])
        char_lens = np.array([len(s) + 1 for s in self.sentences])
        cur = word_counts / char_lens
        reward = -np.abs(self.avg_wrd - cur)

        reward = torch.tensor(reward)
        std = reward.std(unbiased=False)
        return torch.zeros_like(reward) if std == 0 else (reward - reward.mean()) / std
        
    def seen(self):
        reward = np.array([
            sum(w in self.unique_words for w in s.split()) / (len(s.split()) + 1)
            for s in self.sentences
        ])
        
        reward = torch.tensor(reward)
        std = reward.std(unbiased=False)
        return torch.zeros_like(reward) if std == 0 else (reward - reward.mean()) / std

    
    def unigram_char(self):
        reward = np.zeros(len(self.sentences))
        
        for i, s in enumerate(self.sentences):
            chars = ''.join(s)
            char_counts = Counter(chars)
            total_chars = sum(char_counts.values())
            reward[i] = sum(-abs(char_counts[c] / total_chars - self.char_probs[c]) for c in char_counts) / total_chars

        reward = torch.tensor(reward)
        std = reward.std(unbiased=False)
        return torch.zeros_like(reward) if std == 0 else (reward - reward.mean()) / std