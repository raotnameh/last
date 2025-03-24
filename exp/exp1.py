
# Dataloader: Done
'''
Data used: train-clean 100 hour from LibriSpeech.

1. The dataloader returns the unormalized waveform and the speaker id.
2. The waveform is of shape (B, T) and the speaker id is of shape (B,1).
'''


# Model
'''
1. Encoder: Done
(B, T) -> (B, D, T).


2. Downsampler: Done
(B, D, T) -> (B, D, T/2).

3. Codebook: 
(B, D, T/2) -> (B, D, T/2).

4. Discriminator: 
(B, D, T/2) -> (B, 1).
50% prob to repeat the real characters or not. This is done to match the distribution of the generated characters with the real characters.

5. Upsampler: 
(B, D, T/2) -> (B, D, T).

6. Decoder: 
(B, D, T) -> (B, D, T).
'''
