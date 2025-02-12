import torch
import torch.nn.functional as F


@torch.no_grad()
class FrozenVocabulary:
    def __init__(self, path):
        super().__init__()
        
        checkpoint = torch.load(path, map_location="cpu")
        self.embeddings = checkpoint["embedding"]['weight'][1:30,:]
        self.char_to_idx = checkpoint["char_to_idx"]
        self.idx_to_char = checkpoint["idx_to_char"]
        


def get_closest_vocab(z, e):
    """
    Inputs the output of the encoder network z and maps it to a discrete 
    one-hot vector that is the index of the closest embedding vector.
    """
    z = z.transpose(1, 2).contiguous() # (batch, seq, embed_dim)
    z_flattened = z.view(-1, e.shape[1]) # (batch * seq, embed_dim)
    
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = (torch.sum(z_flattened**2, dim=1, keepdim=True) \
        + torch.sum(e**2, dim=1) \
        - 2 * torch.matmul(z_flattened, e.t()))
    
    # find closest encodings
    min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
    min_encodings = torch.zeros(min_encoding_indices.shape[0], e.shape[0], device=z.device)
    min_encodings.scatter_(1, min_encoding_indices, 1)    
    
    # get quantized latent vectors
    z_q = torch.matmul(min_encodings, e).view(z.shape)
    
    commitment_loss = F.mse_loss(z, z_q.detach())
    z_q = z + (z_q - z).detach()
    
    return commitment_loss, z_q.transpose(1, 2), min_encoding_indices.view(z.shape[0], z.shape[1])


def merge_similar_indices(indices):
    batch_size, seq = indices.shape
    merged_indices = []

    for b in range(batch_size):
        unique_indices = []
        prev_idx = None

        for t in range(seq):
            current_idx = indices[b, t].item()

            if prev_idx is None or current_idx != prev_idx:
                unique_indices.append(current_idx)

            prev_idx = current_idx

        merged_indices.append(unique_indices)

    return merged_indices

# ind = get_closest_vocab(enc_out, embedding, return_indices=True).detach().cpu().numpy()
# "".join([idx_to_char[i] for i in merge_similar_indices(ind)[0]]).replace("<sil>", " ")