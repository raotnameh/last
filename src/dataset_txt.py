import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Dataset_txt(Dataset):
    def __init__(self, data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"):
        super(Dataset_txt, self).__init__()

        # reading dataset. 
        with open(data, "r") as f:
            out = f.readlines()
        texts = [x.split("\t")[1].strip() for x in out]
        texts = sorted(texts, key=len)
        self.texts = [x for x in texts if len(x) > 10] # filtering out short texts that 2 second.
        
        # creating the vocab.
        self.vocab = self.build_vocab(texts)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)} # char to index mapping
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)} # index to char mapping
        
    def build_vocab(self, texts):
        """
        Creates a sorted list of unique characters with special tokens.
        self.special_tokens = ["p", "b", "e", "?"]  # "p" = PAD, "b" = BOS, "e" = EOS, "?" = silence
        """
        unique_chars = sorted(set("".join(texts)))
        return ["p", "?"] + unique_chars + ["b", "e"]
    
    def encode(self, text):
        """Encodes text into a list of indices with BOS and EOS tokens."""
        return [self.char_to_idx["b"]] + [self.char_to_idx[char] for char in text] + [self.char_to_idx["e"]]

    def decode(self, indices):
        """Decodes indices back into text, removing all special tokens."""
        return "".join(self.idx_to_char[idx] for idx in indices if self.idx_to_char[idx] not in {"p", "b", "e", "?"})

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.encode(text)
        return {
            "inp": input_ids[:-1],  # Input sequence
            "out": input_ids[1:]       # Target sequence (next char prediction)
        }
        
    def collate_fn(self, batch):
        inp = [item["inp"] for item in batch]
        out = [item["out"] for item in batch]

        pad_token_id = self.char_to_idx['p']
        # Find max length in the batch
        max_length = max(len(seq) for seq in inp)

        # Pad sequences
        def pad_sequence(seq, max_length):
            return seq + [pad_token_id] * (max_length - len(seq))

        inp = torch.tensor([pad_sequence(seq, max_length) for seq in inp], dtype=torch.long)
        out = torch.tensor([pad_sequence(seq, max_length) for seq in out], dtype=torch.long)

        return {"inp": inp, "out": out}
        
        
if __name__ == "__main__":
    # create the dataset and dataloader
    data="/raid/home/rajivratn/hemant_rajivratn/last/data/transcription.txt"
    
    dataset = Dataset_txt(data=data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        print(batch)
        break