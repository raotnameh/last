import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(Codebook, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x): # x: (b,t)
        return self.embedding(x) # (b,t,c)
    
    @torch.no_grad()
    def get_embedding(self):
        return self.embedding.weight.detach().clone()


if __name__ == "__main__":
    # Test codebook
    vocab_size = 10
    emb_dim = 4
    codebook = Codebook(vocab_size, emb_dim)
    x = torch.randint(0, vocab_size, (2, 3))
    print(codebook(x))