import torch
import torch.nn as nn

class Loss:
    def __init__(self, config):
        self.config = config["loss"]
        
        self.mse_loss = nn.MSELoss() # commitment loss, smoothness loss
        self.mae_loss = nn.L1Loss() # decoder loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Discriminator loss with masking
        
    
    def step(self, output, disc=False, step=0, total_steps=None):
        
        # reconstrunction loss :- decoder 
        rec_loss = self.mae_loss(output["dec_out"], output["gt"])
        rec_loss += self.mae_loss(output["dec_out2"], output["gt"])
        rec_loss *= self.config["recon_loss_weight"]
        
        # commitment loss
        commit_loss = output["commitment_loss"] * self.config["commit_loss_weight"] 
        
        # smoothness loss :- down_out shifted by 1
        smooth_loss = self.mse_loss(output["down_out"][:,:-1,:], output["down_out"][:,1:,:])
        smooth_loss *= self.config["smooth_loss_weight"]
        
        # if disc: 
        #     real_loss = self.bce_loss(output['pred_real'], torch.ones_like(output['pred_real']))
        #     fake_loss = self.bce_loss(output['pred_fake'], torch.zeros_like(output['pred_fake']))

        #     # Apply masking if provided
        #     if mask is not None:
        #         real_loss = (real_loss * mask).sum() / mask.sum()
        #     else:
        #         real_loss = real_loss.mean()
            
        #     if dpadding_masks is not None:
        #         fake_loss = (fake_loss * (~dpadding_masks)).sum() / (~dpadding_masks).sum()
        #     else:
        #         fake_loss = fake_loss.mean()

        #     loss_D = real_loss + fake_loss

        print(f"step/total: {step}/{total_steps} rec_loss: {rec_loss}, commit_loss: {commit_loss}, smooth_loss: {smooth_loss}")
        total_loss = rec_loss + commit_loss + smooth_loss 
        
        return  total_loss
        
    
        
    
    