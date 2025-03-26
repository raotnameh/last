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
        
        # reconstrunction loss :- decoder 
        rec_loss = self.mae_loss(output["dec_out"], output["gt"])
        rec_loss += self.mae_loss(output["dec_out2"], output["gt"])
        
        # quantization entropy loss
        ent_loss = 
        
        # commitment loss
        commit_loss = output["commitment_loss"]
        
        # smoothness loss :- down_out shifted by 1
        smooth_loss = self.mse_loss(output["down_out"][:,:-1,:], output["down_out"][:,1:,:])
        
        # 
        
    
        
    
    