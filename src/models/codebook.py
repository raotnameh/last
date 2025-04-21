import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import numpy as np 
import torch.nn.functional as F

import logging  
        
class Codebook(nn.Module):
        
    def __init__(self, vocab, model_name="meta-llama/Llama-3.2-1B-Instruct"):

        super(Codebook, self).__init__()
        
        self.vocab = vocab
        self.model_name = model_name
        
        # Initialize the model and tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Create the embedding matrix
        embedding = nn.Embedding(len(vocab), model.embed_tokens.weight.shape[1], padding_idx=0)
        # Initialize the embedding matrix with the pretrained model's embedding matrix with detaching the gradients
        for i, char in enumerate(vocab):
            tok = tokenizer(char, add_special_tokens=False)["input_ids"][0] # returns the list of token ids, 
            if char != "p": # not padding token 
                embedding.weight.data[i] = model.embed_tokens(torch.tensor(tok)).detach().clone()
            else:
                embedding.weight.data[i] *= 0.0 # padding token embedding is zero
        
        # Normalize the embedding matrix
        embedding.weight.data = F.normalize(embedding.weight.data, dim=-1)
        
        self.embedding = embedding
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
        del model
        del tokenizer
        
    @torch.no_grad()
    def forward(self, x): # x: (b,t) tensor
        return self.embedding(x) # (b,t,c)


if __name__ == "__main__":
    # Test codebook
    vocab = ['p', ' ', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '?']
    print(f"vocab: {vocab}")
    print(f"vocab size: {len(vocab)}")
    

    codebook = Codebook(vocab)
    print(codebook)
    x = torch.randint(0, len(vocab), (2, 3))
    print(codebook(x).shape)