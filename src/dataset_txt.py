import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

class Dataset_txt(Dataset):
    def __init__(self, data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"):
        super(Dataset_txt, self).__init__()

        try: 
            # reading dataset. 
            with open(data, "r") as f:
                out = f.readlines()
            texts = [x.split("\t")[1].strip() for x in out]
            texts = [x for x in texts if len(x) > 10 and len(x) < 500] # filtering out short texts that 2 second.
        except:
            with open(data, "r") as f:
                out = f.readlines()
            texts = [x.strip() for x in out if len(x) > 10 and len(x) < 500] # filtering out short texts that 2 second.
            
        self.texts = sorted(texts, key=len)
        self.add_question_marks()
        
        # creating the vocab.
        self.vocab = self.build_vocab(texts)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)} # char to index mapping
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)} # index to char mapping
        print(F"Vocab: {self.vocab}")
        print(F"-p- is for padding and -?- is for silence")
        print(F"Vocab Size: {len(self.vocab)}")
    
    def add_question_marks(self):
        print(self.texts[:6])
        modified_texts = []
        for sentence in self.texts:
            sentence = f"?{sentence}?" # Add question marks at start and end
            
            # Randomly insert question marks with 0.25 probability
            modified_sentence = []
            for char in sentence:
                modified_sentence.append(char)
                if random.random() < 0.25:
                    modified_sentence.append("?")

            modified_texts.append("".join(modified_sentence))

        self.texts = modified_texts
        print(self.texts[:6])
        exit()

    def build_vocab(self, texts):
        """
        Creates a sorted list of unique characters with special tokens.
        special_tokens = ["p", "?"]  # "p" = PAD, "?" = silence
        """
        unique_chars = sorted(set("".join(texts)))
        return ["p"] + unique_chars + ["?"]
    
    def encode(self, text):
        """Encodes text into a list of indices."""
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        """Decodes indices back into text, removing all special tokens."""
        return "".join(self.idx_to_char[idx] for idx in indices if self.idx_to_char[idx] not in {"p", "?"})

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.encode(text)
        return input_ids  # Input sequence
        
    def collate_fn(self, batch):
        inp = [item for item in batch]
        pad_token_id = self.char_to_idx['p']
        max_length = max(len(seq) for seq in inp)

        # Pad sequences
        def pad_sequence(seq, max_length):
            return seq + [pad_token_id] * (max_length - len(seq))

        inp = torch.tensor([pad_sequence(seq, max_length) for seq in inp], dtype=torch.long)
        mask = torch.tensor([[False] * len(seq) + [True] * (max_length - len(seq)) for seq in batch], dtype=torch.bool)
    
        return inp, mask
        
        
if __name__ == "__main__":
    # create the dataset and dataloader
    data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"
    
    dataset = Dataset_txt(data=data)
    print(F"Vocab: {dataset.vocab}")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        print(batch.shape)
        break
    
    
    # Initialize embedding
    from models.codebook import Codebook
    
    vocab = dataset.vocab
    codebook = Codebook(vocab)
    print(f"Size of codebook: {codebook.embedding.weight.shape[0]} x {codebook.embedding.weight.shape[1]}")
    
    # Get a batch
    batch = next(iter(dataloader))
    inp = batch  # (batch_size, seq_len)

    # Convert input indices to embeddings
    embeddings = codebook(inp)  # (batch_size, seq_len, emb_dim)

    print(embeddings.shape)  # Expected: (batch_size, seq_len, emb_dim)