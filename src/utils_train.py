import torch
import logging
import os
from jiwer import wer, cer

import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from tqdm import tqdm

class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_grad(tensor, scale):
    return GradScale.apply(tensor, scale)

# norm 
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def compute_wer(real_transcripts, pred_transcripts):
    wer_pred = wer(real_transcripts, pred_transcripts)
    cer_pred = cer(real_transcripts, pred_transcripts)
    return round(cer_pred, 2), round(wer_pred, 2)


def train(
    models, 
    models_teacher,
    optimizer, 
    scheduler, 
    optimizer_decoder, 
    scheduler_decoder, 
    speech_loader, 
    config, 
    device, 
    writer, 
    save_dir,
    ):

    logging.info("Training")
    os.makedirs(f"{save_dir}/temp/", exist_ok=True) # temp folder for saving temp files
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True) # checkpoint folder for saving checkpoints
    
    # Initialize dataloader iterators
    train_speech_loader = speech_loader[0]
    val_speech_loader = speech_loader[1]
    
    accumulation_steps = config['train']['accumulation_steps']  # Get accumulation steps, default to 1 if not provided.
    steps = config['train']["steps"] * accumulation_steps  # Total steps to train, multiplied by accumulation steps.
    config['logging']['step'] *= accumulation_steps  # Adjust logging step to account for accumulation steps.
    config['checkpoint']['step'] *= accumulation_steps  # Adjust checkpoint step to account for accumulation steps
    
    decoder = config['train']['decoder']
    teacher = config['train']['teacher']
    ctc = config['train']['ctc']
    if not teacher: del models_teacher  # If not using teacher model, delete it to save memory.
    if not decoder: del models['decoder'] # If not using decoder, delete it to save memory.
    
    os.makedirs(f"{save_dir}/plots/", exist_ok=True)
        
    
    logging.info(f"Using gradient accumulation with {accumulation_steps} steps.")

    models["encoder"].eval()
    optimizer.zero_grad()  
        
    step = 1
    while step <= steps:
        if step >= steps: break
    
        for batch in train_speech_loader:
            if step >= steps: break
            # ===== Speech Data =====
            waveforms, padding_masks, dur, paths, txt, spec = batch 
            waveforms = waveforms.to(device) # [B, T]
            padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
            spec = spec.to(device)
            
            # ===== Encoder =====
            with torch.no_grad():
                enc_out, padding_mask  = models['encoder'](waveforms, padding_masks)  # [B, T//320, C], [B, T // 320, C] 
            mask = ~padding_mask # 0 for masked positions.
            mask = mask.float().unsqueeze(-1) # [B, T//320, 1]            
            # ===== Downsample =====
            down_out, dmask, temp = models['downsample'](enc_out, mask) # [B, T, codebook_dim]
            
            ctc_loss = 0.0
            dec_loss = 0.0
            per_token_kl = 0.0
            reinforce_loss = 0.0
            commitment_loss = 0.0
            smoothness_loss = 0.0
        
            if ctc: # ===== ctc =====
                # ===== Tokenizer =====
                ctc_loss, commitment_loss, smoothness_loss, top, vocab, e_mean_np = models['tokenizer'](down_out, dmask, writer, step, ctc=ctc, txts=txt, temp=temp)                
                total_loss = ctc_loss # Loss
                
            else: 
                # ===== Tokenizer =====
                per_token_logps, z_q, smoothness_loss, commitment_loss, reinforce_loss, top, vocab, e_mean_np = models['tokenizer'](down_out, dmask, writer, step, temp=temp)
                
                total_loss = reinforce_loss # Loss
                
                # ===== Decoder =====
                if decoder:
                    z_q = scale_grad(z_q, config['train']['decoder_grad_scale'])
                    dec_output = models['decoder'](z_q, padding_mask, spec) # btc
                    spec = spec[:,:dec_output.shape[1],:] # ground truth    
                    dec_loss = F.l1_loss(dec_output, spec, reduction='none') * mask # btc
                    valid_count = mask.sum() * spec.shape[-1] # Total number of valid (non-masked) elements
                    dec_loss = dec_loss.sum() / valid_count 
                        
                    total_loss = total_loss + dec_loss # Loss
                
                # ===== Teacher =====
                per_token_kl = 0
                if teacher: 
                    with torch.no_grad():
                        # ===== Encoder =====
                        enc_out_teacher, padding_mask_teacher  = models_teacher['encoder'](waveforms, padding_masks)  # [B, T//320, C], [B, T // 320, C] 
                        mask_teacher = ~padding_mask_teacher # 0 for masked positions.
                        mask_teacher = mask_teacher.float().unsqueeze(-1) # [B, T//320, 1]
                        # ===== Downsample =====
                        down_out_teacher, dmask_teacher, temp = models_teacher['downsample'](enc_out_teacher, mask_teacher) # [B, T, codebook_dim]
                        # ===== Tokenizer =====
                        ref_per_token_logps = models_teacher['tokenizer'](down_out_teacher,  dmask_teacher, writer, step, teacher, temp=temp).detach()

                    per_token_kl = ( torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1 ) * dmask_teacher
            
                    valid_count = mask.sum() * per_token_kl.shape[-1] # Total number of valid (non-masked) elements
                    per_token_kl = per_token_kl.sum() / valid_count
                    per_token_kl *= config['train']['beta']
                    
                    total_loss = total_loss + per_token_kl # Loss
            
            # print("Time taken for forward and backward pass: ", time.time() - start)
            # ===== Loss Backward ====
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            
            # ===== Logging =====
            if step % config['logging']['step'] == 0:  
                logging.info(f"TRAINING ----- step: {step} ----- ctc_loss/Reinforce_loss/dec_loss/per_token_kl: {ctc_loss}/{reinforce_loss}/{dec_loss}/{per_token_kl}")
                logging.info(f"Predicted text: {top[0]}")
                logging.info(f"Real text: {txt[0]}")
                logging.info(f"Real text path: {paths[0]}")
                
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.bar(vocab, e_mean_np.cpu().numpy(), color='blue', alpha=0.7)
                plt.xlabel('Codebook Entry (Char)')
                plt.ylabel('Probability')
                plt.title('Codebook Usage Distribution')
                plt.grid(axis='y')
                
                plt.savefig(os.path.join(f'{save_dir}/plots', f'codebook_usage_distribution_{step}.png'), bbox_inches='tight')
                plt.close()
                
                # logging losses
                writer.add_scalar('loss/loss', total_loss, step)
                writer.add_scalar('loss/ctc_loss', ctc_loss, step)
                writer.add_scalar('loss/reinforce_loss', reinforce_loss, step)
                writer.add_scalar('loss/dec_loss', dec_loss, step)
                writer.add_scalar('loss/per_token_kl', per_token_kl, step)
                writer.add_scalar('loss/commit_loss', commitment_loss, step)
                writer.add_scalar('loss/smooth_loss', smoothness_loss, step)
                # logging lr 
                writer.add_scalar('learning_rate/encoder', scheduler.get_last_lr()[0], step)
                if decoder: writer.add_scalar('learning_rate/decoder', scheduler_decoder.get_last_lr()[0], step)
                # loggin norm
                writer.add_scalar('grad_norm/encoder', get_grad_norm(models['encoder']), step)
                writer.add_scalar('grad_norm/downsample', get_grad_norm(models['downsample']), step)
                if decoder: writer.add_scalar('grad_norm/decoder', get_grad_norm(models['decoder']), step)
                
            
            if (step % accumulation_steps == 0): # Only do the following every accumulation_steps                                
                # Gradient clipping
                max_grad_norm = config['train']['grad_clip']
                torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(models['downsample'].parameters(), max_grad_norm)

                # Optimizer and Scheduler step
                optimizer.step()
                scheduler.step()
                optimizer_decoder.step()
                scheduler_decoder.step()
                
                # Zero gradients after the step
                optimizer.zero_grad()
                optimizer_decoder.zero_grad()
        
            step += 1
            
            # eval
            if step % config['checkpoint']['step'] == 0: 
                logging.info(f"Starting validation")
                with torch.no_grad():
                    models["downsample"].eval()
                    
                    pred, real = [], []
                    for batch in tqdm(val_speech_loader):
                        # ===== Speech Data =====
                        waveforms, padding_masks, dur, paths, txt, _ = batch
                        waveforms = waveforms.to(device) # [B, T]
                        padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
                        # ===== Encoder =====
                        enc_out, padding_mask  = models['encoder'](waveforms, padding_masks)  # [B, T//320, C], [B, T // 320, C]
                        mask = ~padding_mask # 0 for masked positions.
                        mask = mask.float().unsqueeze(-1)
                        # ===== Downsample =====
                        down_out, dmask, temp = models['downsample'](enc_out, mask) # [B, T, codebook_dim]
                        
                        ctc_loss = 0.0
                        reinforce_loss = 0.0
                        commitment_loss = 0.0
                        smoothness_loss = 0.0
                        
                        if ctc: # ===== ctc =====
                            # ===== Tokenizer =====
                            ctc_loss, commitment_loss, smoothness_loss, top, vocab, e_mean_np = models['tokenizer'](down_out, dmask, writer, step, ctc=ctc, txts=txt, temp=temp)                
                        else:
                            # ===== Tokenizer =====
                            _, z_q, smoothness_loss, commitment_loss, reinforce_loss, top, vocab, e_mean_np = models['tokenizer']( down_out, dmask, writer, step, temp=temp)
                    
                        real.extend(txt)
                        pred.extend(top)

                    cer_pred, wer_pred = compute_wer(real, pred)
                
                    # logging error rates
                    logging.info(f"VALIDATION ----- step: {step} ----- ctc_loss/Reinforce_loss/commitment_loss: {ctc_loss}/{reinforce_loss}/{commitment_loss} ----- CER/WER: {cer_pred}/{wer_pred}")                
                    writer.add_scalar('error/cer', cer_pred, step)
                    writer.add_scalar('error/wer', wer_pred, step)
                    writer.add_scalar('val_loss/ctc_loss', ctc_loss, step)
                    writer.add_scalar('val_loss/reinforce_loss', reinforce_loss, step)
                    writer.add_scalar('val_loss/commit_loss', commitment_loss, step)
                    writer.add_scalar('val_loss/smooth_loss', smoothness_loss, step)
                
                checkpoint_path = f"{save_dir}/checkpoints/step_{step:06d}.pt"
                torch.save({
                    'step': step,
                    'models': {k: v.state_dict() for k, v in models.items()},
                    'optimizers': optimizer.state_dict(), # Save optimizer state dict
                    'schedulers': scheduler.state_dict(), # Save scheduler state dict
                    'config': config
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
                
                models["downsample"].train()
                models['decoder'].train() if decoder else None
