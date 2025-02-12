import torch
import torch.nn as nn


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d( in_channels, out_channels, kernel_size,),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=768, 
                 hidden_dim=256, 
                 kernel_size=7, 
                 depth=4):
        super(Discriminator, self).__init__()
        
        # First convolutional layer: from input channels to discriminator_dim filters.
        self.conv1 = Conv1d(
            in_channels=in_channels, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size
        )
        
        layers = []
        for i in range(depth-2):
            layers.append(Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
            ))
        self.network = nn.Sequential(*layers)
            
        # Second convolutional layer: from discriminator_dim filters to discriminator_dim filters.
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=1, 
            kernel_size=kernel_size
        )
        
        # mean pooling
        
        
            
        
        
    def forward(self, x):
        # Apply first conv layer and activation.
        x = self.leaky_relu(self.conv1(x))
        # Apply second conv layer and activation.
        x = self.leaky_relu(self.conv2(x))
        return x

# Example usage:
if __name__ == "__main__":
    # Create a dummy input tensor with batch size 4 and 3 color channels (e.g., 64x64 image)
    dummy_input = torch.randn(4, 3, 64, 64)
    model = TwoLayerDiscriminator(in_channels=3, discriminator_dim=384, discriminator_kernel=8, discriminator_depth=2)
    output = model(dummy_input)
    print("Output shape:", output.shape)
