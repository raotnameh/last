import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np 
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

import logging  
        
class Codebook(nn.Module):
        
    def __init__(self, vocab, model_name="meta-llama/Llama-3.2-1B-Instruct"):

        super(Codebook, self).__init__()
        
        self.vocab = vocab
        self.model_name = model_name
        
        # Initialize the model and tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Freeze LLM parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # # Set EOS token as the padding token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create the embedding matrix
        embed_tokens = self.model.get_input_embeddings()
        embedding = nn.Embedding(len(vocab), embed_tokens.weight.shape[1], padding_idx=0)
        # Initialize the embedding matrix with the pretrained model's embedding matrix with detaching the gradients
        for i, char in enumerate(vocab):
            tok = self.tokenizer(char, add_special_tokens=False)["input_ids"][0] # returns the list of token ids, 
            if char != "p": # not padding token 
                embedding.weight.data[i] = embed_tokens(torch.tensor(tok)).detach().clone()
            else:
                embedding.weight.data[i] *= 0.0 # padding token embedding is zero

        # Normalize the embedding matrix
        embedding.weight.data = F.normalize(embedding.weight.data, dim=-1)
        
        self.embedding = embedding
        self.embedding.weight.requires_grad = False # freeze the embedding matrix
        
        # print the mean, std, min, max of the embedding matrix beautifully for each character
        logging.info("Embedding matrix statistics:")
        logging.info("char\tmean\tstd\tmin\tmax")
        for i, char in enumerate(vocab):
            mean = embedding.weight.data[i].mean().item()
            std = embedding.weight.data[i].std().item()
            min_val = embedding.weight.data[i].min().item()
            max_val = embedding.weight.data[i].max().item()
            logging.info(f"{char}\t{mean:.4f}\t{std:.4f}\t{min_val:.4f}\t{max_val:.4f}")
        

        # Remove the model and tokenizer
        # del model
        # del tokenizer
        
    def forward(self, x): # x: (b,t) tensor
        return self.embedding(x) # (b,t,c)
    
    def lmscoring(self, target, inputs_embeds, attention_mask):
        """
        Args:
            target: [batch, seq_len] tensor of token ids
            inputs_embeds: [batch, seq_len, dim] tensor of input embeddings
            attention_mask: [batch, seq_len] binary mask (1 for real tokens, 0 for padding tokens)
        Returns:
            loss: cross-entropy loss per token (lower is better)
            perplexity: scalar reward. lower perplexity means higher fluency. Perplexity amplifies small differences in likelihood — sharper gradient between “okay” and “bad” generations. Perplexity will sharply punish the incoherence — because the cumulative mismatch to a fluent sequence blows up exponentially. Perplexity harshly penalizes the outputs because it reflects how unlikely they are under a real language model.
        """
        
        
        valid_mask = target != 29 # 29 is the ? token id
        
        inputs_embeds = [inputs_embeds[i][valid_mask[i]] for i in range(inputs_embeds.size(0))]
        target = [target[i][valid_mask[i]] for i in range(target.size(0))]
        attention_mask = [attention_mask[i][valid_mask[i]] for i in range(attention_mask.size(0))]
        # Pad the sequences to the same length
        inputs_embeds = pad_sequence(inputs_embeds, batch_first=True, padding_value=0)
        target = pad_sequence(target, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        

    
        attention_mask = attention_mask.float()
        outputs = self.model(inputs_embeds=inputs_embeds, labels=target, attention_mask=attention_mask)

        logits = outputs.logits[:, :-1, :]       # [B, T-1, V]
        target = target[:, 1:]             # [B, T-1]
        mask = attention_mask[:, 1:].float()  # [B, T-1]

        
        # Decode predictions to text
        # decoded_preds = "".join([
        #     self.vocab[pred_ids.item()] for pred_ids in target[0] if pred_ids.item() != 0 # 0 is the padding token id
        # ])
        # print(f"decoded_preds: {decoded_preds}")
        # exit()
        # Cross-entropy loss per token
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            target.reshape(-1), 
            reduction='none'
        ).reshape(target.shape)  # [B, T-1]

        # Mask pad tokens and compute mean loss
        masked_loss = loss * mask  # [B, T-1]
        token_count = mask.sum(dim=1)  # [B]
        loss = masked_loss.sum(dim=1) / token_count  
        perplexity = torch.exp(loss)    
        # print(f"loss: {loss.mean()}")
        # print(f"perplexity: {perplexity.mean()}")
        
        return loss.mean(), perplexity.mean()
    
        


if __name__ == "__main__":
    # Test codebook
    vocab = ['p', ' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '?']
    print(f"vocab: {vocab}")
    print(f"vocab size: {len(vocab)}")
    

    codebook = Codebook(vocab)
    print(codebook)
    x = torch.randint(0, len(vocab), (2, 3))
    print(codebook(x).shape)