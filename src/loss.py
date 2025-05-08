import torch.nn as nn


class Loss:
    def __init__(self, config):

        self.config = config["loss"]

        self.mae_loss = nn.L1Loss(reduction='none') # reconstruction loss
    
    def step_disc(self, output):
        loss_fake = output["disc_fake"].mean()
        loss_real = output["disc_real"].mean()
        total_loss = -loss_real + loss_fake # total loss is sum of fake and real losses
        
        return {"total_loss": total_loss, "loss_fake": loss_fake, "loss_real": loss_real}        

    def step_gen(self, output):
        # reconstrunction loss :- decoder 
        valid_count = output["mask"].sum() * output["dec_out"].shape[-1]        
        rec_loss = self.mae_loss(output["dec_out"], output["gt"], ) * output["mask"]
        rec_loss = rec_loss.sum() / valid_count

        # generator loss
        if output["disc_fake"] is not None:
            gen_loss = -output["disc_fake"].mean() #+ output["perplexity"]
        else:
            gen_loss = 0.0
        
        loss_components = {
            "rec_loss": rec_loss * self.config["recon_loss_weight"],
            "commit_loss": output["commitment_loss"] * self.config["commit_loss_weight"],
            "smooth_loss": output["smoothness_loss"] * self.config["smooth_loss_weight"],
            "gen_loss": gen_loss * self.config["gen_loss_weight"],
        }     

        return  loss_components
    
    