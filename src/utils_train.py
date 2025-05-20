import torch
import logging
import os
from jiwer import wer, cer

import matplotlib.pyplot as plt


# norm 
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train(
    models, 
    optimizer, 
    scheduler, 
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
    
    freeze_epochs = config['train']["freeze_epochs"]
    epochs = config['train']["epochs"]
    
    os.makedirs(f"{save_dir}/plots/", exist_ok=True)
        
    
    accumulation_steps = config['train']['accumulation_steps']  # Get accumulation steps, default to 1 if not provided.
    logging.info(f"Using gradient accumulation with {accumulation_steps} steps.")

    # # torch compile
    # for k in list(models): models[k] = torch.compile(models[k])

    torch.backends.cudnn.enabled = True 
    # torch.backends.cudnn.benchmark = True
    
    optimizer.zero_grad()  
    step = 1
    for epoch in range(1, epochs+1):
        models["encoder"].eval()
        models["downsample"].train()
    
        for batch in train_speech_loader:
            # ===== Speech Data =====
            waveforms, padding_masks, dur, paths, txt = batch
            waveforms = waveforms.to(device) # [B, T]
            padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
        
            # ===== Encoder =====
            with torch.no_grad():
                enc_out, padding_mask  = models['encoder'](waveforms, padding_masks)  # [B, T//320, C], [B, T // 320, C] 
                mask = ~padding_mask # 0 for masked positions.
                mask = mask.float().unsqueeze(-1)
                
            # ===== Downsample =====
            down_out = models['downsample'](enc_out, mask) # [B, T, codebook_dim]
            
            # ===== Tokenizer =====
            smoothness_loss, commitment_loss, reinforce_loss, top, vocab, e_mean_np = models['tokenizer'](
                down_out, 
                mask,
                writer,
                step,
            )
            
            # Loss calculation      
            total_loss = reinforce_loss
        
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.warning(f"Skipping step {step} due to NaN/Inf in total_loss")
                optimizer.zero_grad()
                scheduler.step() 
                continue

            # Scale the loss, and accumulate over accumulation steps
            total_loss = total_loss / accumulation_steps # Divide loss by accumulation steps.
            total_loss.backward()
                
            if step % config['logging']['step'] == 0:  
                cer_pred, wer_pred = compute_wer(txt, top)
                
                logging.info(f"Training loss ---- epoch - step - loss - CER - WER: {epoch} - {step} - {total_loss} - {cer_pred} - {wer_pred}")                
                logging.info(F"True txt: {txt[0]}")
                logging.info(F"Pred txt: {top[0]}")
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.bar(vocab, e_mean_np, color='blue', alpha=0.7)
                plt.xlabel('Codebook Entry (Char)')
                plt.ylabel('Probability')
                plt.title('Codebook Usage Distribution')
                plt.grid(axis='y')
                
                plt.savefig(os.path.join(f'{save_dir}/plots', f'codebook_usage_distribution_{step}.png'), bbox_inches='tight')
                plt.close()
    
                # logging error rates
                writer.add_scalar('error/cer', cer_pred, step)
                writer.add_scalar('error/wer', wer_pred, step)
                # logging losses
                writer.add_scalar('loss/loss', total_loss, step)
                writer.add_scalar('loss/commit_loss', commitment_loss, step)
                writer.add_scalar('loss/smooth_loss', smoothness_loss, step)
                # logging lr 
                writer.add_scalar('learning_rate/encoder', scheduler.get_last_lr()[0], step)
                # loggin norm
                writer.add_scalar('grad_norm/encoder', get_grad_norm(models['encoder']), step)
                writer.add_scalar('grad_norm/downsample', get_grad_norm(models['downsample']), step)
            
            if (step % accumulation_steps == 0): # Only do the following every accumulation_steps                                
                # Gradient clipping
                max_grad_norm = config['train']['grad_clip']
                torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(models['downsample'].parameters(), max_grad_norm)

                # Optimizer and Scheduler step
                optimizer.step()
                scheduler.step()
                # Zero gradients after the step
                optimizer.zero_grad()
                
            step += 1
        

    checkpoint_path = f"{save_dir}/checkpoints/step_{step:06d}.pt"
    torch.save({
        'step': step,
        'epoch': epoch,
        'models': {k: v.state_dict() for k, v in models.items()},
        'optimizers': optimizer.state_dict(), # Save optimizer state dict
        'schedulers': scheduler.state_dict(), # Save scheduler state dict
        'config': config
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def compute_wer(real_transcripts, pred_transcripts):
    wer_pred = wer(real_transcripts, pred_transcripts)
    cer_pred = cer(real_transcripts, pred_transcripts)
    return cer_pred, wer_pred