import torch
import torch.nn as nn



class Loss:
    def __init__(self, config):
        self.config = config
        
        self.mse_loss = nn.MSELoss() # commitment loss, smoothness loss
        self.mae_loss = nn.L1Loss() # decoder loss
    
    def step(self, output):
        print("Loss step")
        print(output.keys())
        
    
    