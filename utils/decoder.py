import torch.nn as nn
from torch.nn.utils import weight_norm

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = weight_norm( 
                                nn.Conv1d( in_channels, out_channels, kernel_size, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2 )
                            )  

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self,channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            Conv1d(channels, channels, kernel_size=kernel_size, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.1),
               
            Conv1d(channels, channels, kernel_size=kernel_size, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
        )

    def forward(self, x):
        return x + self.net(x)
    
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_blocks, kernel_size):
        super().__init__()
        
        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock(
                hidden_dim,
                kernel_size,
            ))
        self.network = nn.Sequential(*layers)
        
        self.output = Conv1d(hidden_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.network(x)
        x = self.output(x)
        return x




class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, 
            padding=((kernel_size - 1) * dilation) // 2, output_padding=stride - 1
        )

    def forward(self, x):
        return self.conv(x)
    

class Upsampling(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super().__init__()
        self.upsampling = nn.Sequential(
            ConvTranspose1d(inp_dim, hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        x = self.upsampling(x)
        return x



# class Upsampling(nn.Module):
#     def __init__(self, inp_dim, hidden_dim):
#         super().__init__()
#         self.upsampling = Conv1d(inp_dim, hidden_dim, kernel_size=1)
        
#     def forward(self, x):
#         x = self.upsampling(x)
#         return x





# Function to calculate total parameters
def calculate_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6
    