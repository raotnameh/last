'''
Finetuune the encoder, decoder, discriminator, and a interpretable vocab.

The losses are:
1. Reconstruction loss. This is the loss from the decoder.
2. Discriminator loss. This is the loss from the discriminator.
3. Commitment loss from VQ-VAE. It is the distance between the encoder output and the quantized output.
4. Vocab diversity loss. This is the loss from the interpretable vocab.

The training is done in the following way:
1. The encoder is trained using the reconstruction loss and the commitment loss.
2. The discriminator is trained using the discriminator loss.
3. The interpretable vocab is trained using the vocab diversity loss and discriminator loss.




Dataset-used is train-clean 100 hour from LibriSpeech.
'''