import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import logging




class GANLoss(nn.Module):
    """
    Computes adversarial loss and optional gradient penalty for GAN training.

    Args:
        gp_weight (float): Weight for the gradient penalty term.

    Inputs:
        fake (Tensor): Discriminator output for fake samples. Shape: [B, 1]
        real (Tensor): Discriminator output for real samples. Shape: [B, 1]
        fake_x (Tensor): Fake samples from generator. Shape: [B, ...]
        real_x (Tensor): Real samples from dataset. Shape: [B, ...]
        fake_smooth (float, optional): Label smoothing for fake labels. Default: 0.0
        real_smooth (float, optional): Label smoothing for real labels. Default: 0.0

    Returns:
        dict: Contains loss for fake (`loss_fake`), real (`loss_real`), and gradient penalty (`grad_pen`).
    """
    def __init__(self, gp_weight=1.0):
        super().__init__()
        self.gp_weight = gp_weight

    def forward(self, fake, real, fake_x, real_x, fake_smooth=0.0, real_smooth=0.0):
        # using zero class for real and one class for fake
        """Computes adversarial loss and gradient penalty."""
        
        # with 5 percent probability, switch fake and real data
        if np.random.rand() < 0.05:
            fake, real = real, fake
            fake_x, real_x = real_x, fake_x
            
        loss_fake = F.binary_cross_entropy_with_logits(
            fake, torch.ones_like(fake) - fake_smooth, reduction="sum"
        )
        loss_real = F.binary_cross_entropy_with_logits(
            real, torch.zeros_like(real) + real_smooth, reduction="sum"
        )
        total_loss = loss_fake + loss_real # total loss is sum of fake and real losses
        grad_pen = None
        if self.training and self.gp_weight > 0:
            grad_pen = self.calc_gradient_penalty(real_x, fake_x) * self.gp_weight
            total_loss += grad_pen
        
        return {"total_loss": total_loss, "loss_fake": loss_fake, "loss_real": loss_real, "grad_pen": grad_pen}

    def calc_gradient_penalty(self, real_data, fake_data):
        """
        Calculates the gradient penalty.
        
        Slices the real and fake data to have the same batch and temporal sizes,
        interpolates between them, and computes the gradient penalty.
        
        Inputs:
          - real_data: Real samples. Shape: [B, T, ...]
          - fake_data: Fake samples. Shape: [B, T, ...]
          
        Output:
          - Scalar gradient penalty (mean over batch).
        """
        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        real_data = real_data[:b_size, :t_size]
        fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1, device=real_data.device)
        alpha = alpha.expand(real_data.size())
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).detach()
        interpolates.requires_grad_(True)
  
        disc_interpolates = self.discriminator(interpolates, torch.zeros_like(interpolates).bool()[:,:,:1])

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
    
    
class Loss:
    def __init__(self, config):

        self.config = config["loss"]
        
        self.mse_loss = nn.MSELoss(reduction='mean') # smoothness loss
        self.mae_loss = nn.L1Loss(reduction='none') # reconstruction loss
        
        
        self.gan_loss = GANLoss(gp_weight=self.config["gp_weight"])
    
    def step_disc(self, output):
        # fake, real, fake_x, real_x, fake_smooth=0.0, real_smooth=0.0):
    
        loss = self.gan_loss(output["disc_fake"], output["disc_real"], output["disc_fake_x"], output["disc_real_x"])        
        return loss
        

    def step_gen(self, output, step=1, total_steps=1):
        
        # reconstrunction loss :- decoder 
        valid_count = output["dec_mask"].sum() * output["dec_out"].shape[-1]        
        rec_loss1 = self.mae_loss(output["dec_out"], output["gt"], ) * output["dec_mask"]
        rec_loss1 = rec_loss1.sum() / valid_count
        rec_loss2 = self.mae_loss(output["dec_out2"], output["gt"]) * output["dec_mask"]
        rec_loss2 = rec_loss2.sum() / valid_count
        rec_loss = rec_loss1 + rec_loss2
        rec_loss *= self.config["recon_loss_weight"]
        

        # generator loss
        gen_loss = F.binary_cross_entropy_with_logits(output["disc_fake"], torch.zeros_like(output["disc_fake"]))
        
        loss_components = {
            "rec_loss": rec_loss * self.config["recon_loss_weight"],
            "commit_loss": output["commitment_loss"] * self.config["commit_loss_weight"],
            "smooth_loss": output["smoothness_loss"] * self.config["smooth_loss_weight"],
            "gen_loss": gen_loss * self.config["gen_loss_weight"],
            "diversity_loss": output["diversity_loss"] * self.config["diversity_loss_weight"],
        }     
        
        return  loss_components
    
    