import torch
import logging
import os
from torch.nn.utils.rnn import pad_sequence

from tqdm.auto import tqdm

from jiwer import wer, cer
import torch.cuda.amp

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
    
    
    accumulation_steps = config['train']['accumulation_steps']  # Get accumulation steps, default to 1 if not provided.
    logging.info(f"Using gradient accumulation with {accumulation_steps} steps.")

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()


    # torch compile
    for m in models: torch.compile(models[m])
        
    for epoch in range(1, epochs):
        models["encoder"].eval()
        models["downsample"].train()
        
        for step, batch in enumerate(train_speech_loader, start=1):
            
            # ===== Speech Data =====
            waveforms, padding_masks, dur, paths, txt = batch
            waveforms = waveforms.to(device) # [B, T]
            padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
            
            iter = 0
            while iter < config['train']['iters']:
                iter +=1
                with torch.cuda.amp.autocast():
                    # ===== Encoder =====
                    with torch.no_grad():
                        enc_out, padding_mask  = models['encoder'](waveforms, padding_masks)  # [B, T//320, C], [B, T // 320, C] 
                        mask = ~padding_mask # 0 for masked positions.
                        mask = mask.float().unsqueeze(-1)
                        
                    # ===== Downsample =====
                    down_out = models['downsample'](enc_out, mask) # [B, T, codebook_dim]
                    
                    # ===== Tokenizer =====
                    smoothness_loss, commitment_loss, reinforce_loss, top = models['tokenizer'](
                        down_out, 
                        mask,
                        writer,
                        step,
                        iter,
                    )
                    
                    # Loss calculation      
                    total_loss = reinforce_loss
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logging.warning(f"Skipping step {step} due to NaN/Inf in total_loss")
                    optimizer.zero_grad()
                    scheduler.step() 
                    continue
                
                else: 
                    # Scale the loss, and accumulate over accumulation steps
                    total_loss = total_loss / accumulation_steps # Divide loss by accumulation steps.
                    scaler.scale(total_loss).backward() # Change here    
                    
                if step % config['logging']['step'] == 0:  
                    cer_pred, wer_pred = compute_wer(txt, top)
                    
                    logging.info(f"Training loss ---- epoch - step - loss - CER - WER: {epoch} - {step} - {total_loss} - {cer_pred} - {wer_pred}")                
                    logging.info(F"True txt: {txt[0]}")
                    logging.info(F"Pred txt: {top[0]}")
        
                    # logging losses
                    writer.add_scalar('loss/loss', total_loss, step)
                    writer.add_scalar('generator_loss/commit_loss', commitment_loss, step)
                    writer.add_scalar('generator_loss/smooth_loss', smoothness_loss, step)
                    # logging lr 
                    writer.add_scalar('learning_rate/encoder', scheduler.get_last_lr()[0], step)
                    # loggin norm
                    writer.add_scalar('grad_norm/encoder', get_grad_norm(models['encoder']), step)
                    writer.add_scalar('grad_norm/downsample', get_grad_norm(models['downsample']), step)
                
                if (step % accumulation_steps == 0): # Only do the following every accumulation_steps                                
                    if scaler:
                        scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    # Gradient clipping
                    max_grad_norm = config['train']['grad_clip']
                    torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(models['downsample'].parameters(), max_grad_norm)

                    # Optimizer and 
                    scaler.step(optimizer) # Change here
                    scaler.update()
                    # Scheduler step
                    scheduler.step()
                    # Zero gradients after the step
                    optimizer.zero_grad()
            

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