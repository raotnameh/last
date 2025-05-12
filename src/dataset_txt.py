import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from tqdm.auto import tqdm  
import logging
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class Dataset_txt(Dataset):
    def __init__(self, data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"):
        super(Dataset_txt, self).__init__()

        
        with open(data, "r") as f:
            out = f.readlines()
        texts = [x.strip() for x in tqdm(out) if len(x) > 10] # filtering out short texts that 2 second.
    
        # creating the vocab.
        self.vocab = self.build_vocab(texts)
        logging.info(f"Done building VOCAB")
        self.save_histogram(texts)
        
        self.texts = texts
        self.texts = sorted(self.texts, key=len)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)} # char to index mapping
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)} # index to char mapping
        
        logging.info(f"Vocab Size: {len(self.vocab)}")
        logging.info(f"Vocab: {self.vocab}")
        logging.info(f"-p- is for padding and -?- is for silence")
        
    def save_histogram(self, texts):
        import os
        if os.path.exists('REAL_codebook_usage_distribution.png'): 
            logging.warning("Histogram already exists. Skipping save.")
            return 
        
        texts = self.add_question_marks(texts)

        logging.info(f"Saving histogram of the REAL text data.")
        char_counts = Counter("".join(texts))  # Example output: [('a', 2), ('d', 1)]
        char_counts = dict(char_counts)
        print(f"char_counts: {char_counts}")
        c = [char_counts[v] for v in self.vocab if v not in ["p"]]  # Exclude padding and silence tokens
        c = np.array(c, dtype=np.float32)
        c /= c.sum()  # Normalize the counts to get probabilities
        
        self.prior = c # save the counts as prior for kl loss.
        
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(self.vocab[1:], c, color='blue', alpha=0.7)
        plt.xlabel('Codebook Entry (Char)')
        plt.ylabel('Probability')
        plt.title('Codebook Usage Distribution')
        plt.grid(axis='y')
        plt.savefig('REAL_codebook_usage_distribution.png', bbox_inches='tight')
        
        
    def add_question_marks(self, texts=[]):
        logging.info(f"Preprocessing the text data by adding silence tokens.")
        
        modified_texts = []
        for sentence in tqdm(texts):
            modified_sentence = ['?']# Add question marks at start 
            previous_char = None
            for char in sentence:
                # if  char == previous_char insert a question mark
                if previous_char == char:
                    modified_sentence.append("?")
                
                modified_sentence.append(char)
    
                # Randomly insert question marks with 0.25 probability
                if random.random() < 0.25 and modified_sentence[-1] != '?':
                    modified_sentence.append("?")
                
                previous_char = char
                    
            if modified_sentence[-1] != '?': 
                modified_sentence.append("?")  # Add a question mark at the end
            modified_texts.append("".join(modified_sentence))
        logging.info(f"Preprocessing done.")
        logging.info(f"Modified text sample")
        logging.info(f"{random.choice(modified_texts)}")
        logging.info(f"{random.choice(modified_texts)}")
        
        return  modified_texts

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

    def decode(self, indices, keep_special_tokens=False):
        """Decodes indices back into text, removing all special tokens."""
        if keep_special_tokens:
            return "".join(self.idx_to_char[idx] for idx in indices)
        return "".join(self.idx_to_char[idx] for idx in indices if self.idx_to_char[idx] not in {"p", "?"})

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # input_ids = self.encode(text)
        # return input_ids
        
        result = ["?"]
        prev_char = ""

        for char in text:
            if char == prev_char:
                result.append("?")
            result.append(char)

            # Slightly more efficient: only check random if not already '?'
            if result[-1] != "?" and random.random() < 0.25:
                result.append("?")

            prev_char = char

        # Ensure it ends with a question mark
        if result[-1] != "?":
            result.append("?")

        modified_text = "".join(result)
        input_ids = self.encode(modified_text)
        return input_ids
        
    def collate_fn(self, batch):
        inp = [item for item in batch]
        pad_token_id = self.char_to_idx['p']
        max_length = max(len(seq) for seq in inp)

        # Pad sequences
        def pad_sequence(seq, max_length):
            return seq + [pad_token_id] * (max_length - len(seq))

        inp = torch.tensor([pad_sequence(seq, max_length) for seq in inp], dtype=torch.long)
        mask = torch.tensor([[False] * len(seq) + [True] * (max_length - len(seq)) for seq in batch], dtype=torch.bool)
    
        return inp, mask.unsqueeze(-1)
        
        
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