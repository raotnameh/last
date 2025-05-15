import torch
import torch.nn as nn
import torch.autograd as autograd



class Loss:
    def __init__(self, config):

        self.config = config["loss"]

        self.mae_loss = nn.L1Loss(reduction='none') # reconstruction loss
        self.gp_weight = config["loss"]["gp_weight"]
    
   
    def step_gen(self, output):
        # reconstrunction loss :- decoder 
        valid_count = output["mask"].sum() * output["dec_out"].shape[-1]        
        rec_loss = self.mae_loss(output["dec_out"], output["gt"], ) * output["mask"]
        rec_loss = rec_loss.sum() / valid_count

        # generator loss
        gen_loss = 0.0
        if output['disc_fake'] is not None:
            gen_loss = -output["disc_fake"].mean() 
            gen_loss = gen_loss + output['entropy_loss']
        
        loss_components = {
            "rec_loss": rec_loss * self.config["recon_loss_weight"],
            "commit_loss": output["commitment_loss"] * self.config["commit_loss_weight"],
            "smooth_loss": output["smoothness_loss"] * self.config["smooth_loss_weight"],
            "gen_loss": gen_loss * self.config["gen_loss_weight"],
        }     

        return  loss_components
    
    def step_disc(self, output):
        loss_fake = output["disc_fake"].mean()
        loss_real = -output["disc_real"].mean()
        total_loss = loss_real + loss_fake # total loss is sum of fake and real losses
        
        # grad_pen = self.calc_gradient_penalty(output["real_x"], output["fake_x"], output["real_pad_mask"], output["fake_pad_mask"]) * self.config["gp_weight"]
        # total_loss += grad_pen
        grad_pen = 0.0
        
        return {"total_loss": total_loss, "loss_fake": loss_fake, "loss_real": loss_real, "grad_pen": grad_pen}

    def calc_gradient_penalty(self, real_data, fake_data, real_pad_mask, fake_pad_mask):
        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        real_data = real_data[:b_size, :t_size]
        fake_data = fake_data[:b_size, :t_size]
        real_pad_mask = real_pad_mask[:b_size, :t_size]
        fake_pad_mask = fake_pad_mask[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1, device=real_data.device)
        alpha = alpha.expand(real_data.size())
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).detach()
        interpolates.requires_grad_(True)
        
        interp_pad_mask = real_pad_mask | fake_pad_mask  # (B, T, 1)
  
        disc_interpolates = self.discriminator(interpolates, interp_pad_mask)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty.mean()
    