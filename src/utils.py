import torch
import numpy as np
from collections import Counter
from math import log
from tqdm.auto import tqdm
import logging
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import kenlm


class Scorer:

    def __init__(self) -> None:
        '''
        The goal of the scorer is: 
            1. Teach syntax (valid characters, word lengths).
            2. Build vocabulary (valid words, diversity).
            3. Develop semantics (fluency, alignment with speech).
        Skipping phases would be like expecting a child to write essays before learning the alphabet.

        Practical Example
        Imagine training a model to transcribe speech:
            Phase 1: Reward outputs like th3 qu1ck br0wn (valid characters, plausible lengths).
            Phase 2: Reward the quck brown fox (real words, no repeats).
            Phase 3: Reward the quick brown fox (fluent, matches audio content).
        Without this progression, the model might never escape the "gibberish basin."
        
        

        '''

        with open("/raid/home/rajivratn/hemant_rajivratn/last/data/txt/train.wrd", "r") as f:
            sentences = f.readlines()
        sentences = [s for s in sentences if len(s) > 0]
            
        # uniq word list
        self.vocab = set(
            [word for sentence in sentences for word in sentence.split()]
        )
        
        # unigram character count to compute probabilities
        self.char_counter = Counter(
            [char for sentence in sentences for char in sentence]
        )
        
        self.unigram_char_prob = {
            char: count / sum(self.char_counter.values())
            for char, count in self.char_counter.items()
        }

        logging.info(f"----------Unigram character probabilities: {self.unigram_char_prob}----------")
        
        # lm fluency llm
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
        for param in self.lm.parameters():
            param.requires_grad = False
        self.lm.eval()
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        
        # Load the model (binary format loads faster)
        self.charlm = kenlm.Model('/raid/home/rajivratn/hemant_rajivratn/grpo/charlm_5gram.arpa')
    
    def step(self, sentences):
        self.sentences = sentences
        funcs = [
            
                # character level rewards
                self.unigram_character_reward,
                # self.charngram,
                
                # word level rewards
                self.length_reward,
                self.seen,
                          
                # Sentence level rewards
                # self.lm_fluency_reward,
                
                ]
        
        rewards_dict = {func.__name__: func() for func in funcs}

        return rewards_dict
    
    def charngram(self,):
        '''
        For each sentence:
        - Replace spaces with "|" and Separate each character by spaces (for char-level scoring) and Compute log10 score.
        '''
        rewards = []
        for sentence in self.sentences:
            processed = " ".join( sentence.replace(" ", "|") )
            
            score = self.charlm.score(processed) # Get log10 probability score from the model
            # score /= len(processed)
            # rewards.append(score)
            
            perplexity = 10 ** (-score / len(processed)) # Compute perplexity: 10^(-score / length_in_chars)
            rewards.append(-perplexity)
        return self._std_norm(rewards) 
    
    def unigram_character_reward(self):
        """
        Reward sentences based on the KL divergence from a unigram distribution.
        For example, a sentence that matches the unigram distribution of the training set will get a higher score.
        lower kl is better. OR. higher -kl is better.
        """
        eps = 1e-6
        rewards = []
        for sentence in self.sentences:
            C = Counter(sentence)
            L = max(1, len(sentence))
            # Q: empirical distribution from sentence
            P_emp = {ch: C[ch] / L for ch in C}
            # P: ground truth unigram distribution
            kl = 0.0
            
            for ch, p in self.unigram_char_prob.items():
                q = P_emp.get(ch, eps)  # prediction
                kl += p * math.log((p + eps) / (q + eps))

            rewards.append(-kl)  # higher = better (closer to true unigram)

        return self._std_norm(rewards)
    
    def length_reward(self):
        '''
        Penalize sentences containing words that are too short (<2 chars) or too long (>8 chars).
        '''    
        rewards = []
        for sentence in self.sentences:
            words = sentence.split()
            penalty = sum(1 if 2 <= len(word) <= 8 else -1 for word in words )
            rewards.append(penalty)
            
        return self._std_norm(rewards)

    def seen(self):
        '''
        Reward words in the vocab with +1, penalize OOV words with -1.
        '''
        reward = [
            sum(1 if (w in self.vocab and len(w)>=2) else -1 for w in sentence.split())
            for sentence in self.sentences
        ]
        return self._std_norm(reward)
    
    def _std_norm(self, arr):
        t = torch.tensor(arr, dtype=torch.float32)
        std = t.std(unbiased=False)
        if std == 0:
            return torch.zeros_like(t)
        return (t - t.mean()) / std
    
    def lm_fluency_reward(self):
        with torch.no_grad():
            inputs = self.tokenizer(self.sentences, padding=True, truncation=True, return_tensors="pt").to(self.lm.device)
            input_ids = inputs["input_ids"]
        
            outputs = self.lm(input_ids=input_ids)
            logits = outputs.logits
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
            shift_labels = input_ids[:, 1:].contiguous()   # (batch_size, seq_len-1)

            losses = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )  # (batch_size * (seq_len - 1))

            # Reshape to (batch_size, seq_len - 1)
            losses = losses.view(shift_labels.size())
            # Average loss per sentence (mean over tokens)
            sentence_losses = losses.mean(dim=1)

            # Return negative loss per sentence (reward style)
            return self._std_norm( [-loss for loss in sentence_losses] )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

    return sequences
    # return sequences, scores.unsqueeze(-1)

