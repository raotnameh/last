import torch
import logging
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import os
import contextlib
from torch.nn.utils.rnn import pad_sequence

from models.asr import WhisperWERCalculator, compute_pesq, compute_stoi
from tqdm.auto import tqdm

import torch.nn.functional as F

import json

def train(
    models, 
    optimizers, 
    schedulers, 
    speech_loader, 
    text_dataset, 
    text_loader, 
    loss_module, 
    config, 
    device, 
    writer, 
    start_step, 
    save_dir
    ):

    logging.info("Training")
    os.makedirs(f"{save_dir}/temp/", exist_ok=True) # temp folder for saving temp files
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True) # checkpoint folder for saving checkpoints
    
    num_steps = config['train']['num_steps']
    freeze_steps = config['train']['freeze_steps']
    stoi = 0.0

    # Initialize dataloader iterators
    train_speech_loader = speech_loader[0]
    val_speech_loader = speech_loader[1]
    test_speech_loader = speech_loader[2]
    
    train_speech_iter = iter(train_speech_loader)
    text_iter = iter(text_loader)
    
    loss_module.discriminator = models['discriminator']
    
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    
    for m in models: 
        if m != 'gtruth': models[m].train()

    for step in range(start_step, num_steps + 1):

        output = {}
        # ===== Speech Data =====
        try:
            waveforms, padding_masks, dur, paths, txt = next(train_speech_iter)
        except:
            train_speech_iter = iter(train_speech_loader)
            waveforms, padding_masks, dur, paths, txt = next(train_speech_iter)
        waveforms = waveforms.to(device) # [B, T]
        padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]

        # ===== Encoder or Generator =====
        with torch.no_grad() if step < freeze_steps else contextlib.ExitStack():
            enc_out = models['encoder'](waveforms, padding_masks)  # [B, T, C] # step 1
            output["cnn_out"] = enc_out['cnn_out'] # [B, T // 320, C] 
            output['encoder_out'] = enc_out['encoder_out'] # [B, T // 320, C] 
        
            mask = ~enc_out['padding_mask'] # B,T//320 # 0 for masked positions.
            mask = mask.unsqueeze(-1).float() # [B, T // 320, 1]
            output['mask'] = mask
        
        # ===== Ground Truth =====
        with torch.no_grad():
            def rescale_waveforms(waveforms, eps=1e-9):# waveforms: [B, T]
                max_vals = waveforms.abs().amax(dim=1, keepdim=True)  # [B, 1]
                return waveforms / (max_vals + eps)
            gt = models['gtruth'].encode(rescale_waveforms(waveforms).unsqueeze(1)) # [B, T//320, 1024] 
            assert enc_out['encoder_out'].shape[1]-10 <= gt.shape[1] <= enc_out['encoder_out'].shape[1]+10, f"GT shape: {gt.shape}, Encoder out shape: {enc_out['encoder_out'].shape}"
            gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
            # output['gt'] = F.normalize(gt, dim=-1) # [B, T, 1024]
            output['gt'] = gt
        
        # ===== Downsample =====
        down_out = models['downsample'](enc_out['encoder_out'], mask) # [B, T // 2, C], [B, T // 2, vocab_size]
        dmask = mask[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride'], 1]
        output['down_out'] = down_out
        output['dmask'] = dmask
        
        # ===== Tokenizer =====
        n_student_probs, smoothness_loss, commitment_loss, z_q, z_q_disc, z_q_disc_mask, selected_encodings_list, selected_encodings_repeated = models['tokenizer'](
            down_out, 
            models['codebook'], 
            dmask,
            writer,
            step,
            text_dataset.skip_non_speech
        )
        z_q_disc_mask = ~z_q_disc_mask.bool() # [B, T // 2, 1]
        output['smoothness_loss'] = smoothness_loss
        output['commitment_loss'] = commitment_loss
        output['z_q'] = z_q # already masked
        output['z_q_disc'] = z_q_disc # already masked
        
        # ===== UpSample =====
        log_student_probs, up_out = models['upsample'](z_q, n_student_probs)
        up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]
        output['up_out'] = up_out
        
        dec_out = models['decoder'](
            x=up_out,
            mask=enc_out['padding_mask'],
            s=enc_out['cnn_out'],
        )
        # output['dec_out'] = F.normalize(dec_out, dim=-1) # [B, T, 1024]
        output['dec_out'] = dec_out
            
        # ===== Discriminator Generator Update =====
        tensor_seqs = [torch.tensor(seq, dtype=torch.long) for seq in selected_encodings_list]
        padded_batch = pad_sequence(tensor_seqs, batch_first=True, padding_value=0).to(z_q_disc.device)
        disc_fake, lm_loss = models['discriminator'](z_q_disc, z_q_disc_mask, labels=padded_batch.detach())
        # lm_loss, perplexity = models['codebook'].lmscoring(
        #     target=padded_batch,
        #     inputs_embeds=z_q_disc,
        #     attention_mask=~z_q_disc_mask.squeeze(-1),
        # )
        output['disc_fake'] = disc_fake
        output["entropy_loss"] = lm_loss

        # Loss calculation
        gen_loss_components = loss_module.step_gen(output)        
        total_lossg = gen_loss_components['rec_loss']
        
        if step % config['logging']['step'] == 0:
            enc_list = [i.item() for i in selected_encodings_list[0]]
            logging.info(f"Generator encoded text path: --{paths[0]}-- of length {dur[0]} seconds--")
            logging.info( f"Generator decoded text without special tokens: --{text_dataset.decode(enc_list)}--" )
            
            if step % 1000 == 0:
                with torch.no_grad():
                    pr = models['gtruth'].decode(output['dec_out'][0].unsqueeze(0)) # [1, T]
                    pr = pr / torch.max(torch.abs(pr))
                    gt = waveforms[0].unsqueeze(0) # [1, T]
                    gt = gt / torch.max(torch.abs(gt)) 
                    gap = torch.zeros_like(gt)[:,:16000] # [1, 16000]
                    total = torch.cat([pr, gap, gt], dim=1) # [1, 2T+16000]
                    torchaudio.save(f"{save_dir}/temp/{step}.wav", total.clone().detach().cpu(), sample_rate=16000)
                
                    spec_tokens_repeated = text_dataset.decode([i.item() for i in selected_encodings_repeated[0]],keep_special_tokens=True)
                    spec_tokens = text_dataset.decode(enc_list,keep_special_tokens=True)
                    tokens = text_dataset.decode(enc_list)
                    
                    output_data = {
                        "decoded_text": tokens,
                        "decoded_text_with_special_tokens": spec_tokens,
                        "decoded_raw_text_repeated": spec_tokens_repeated,
                        "total_time_seconds": dur[0],
                        "path": paths[0]
                    }
                    # Save as pretty JSON
                    with open(f"{save_dir}/temp/{step}.json", "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=4, ensure_ascii=False)
                        
                    # save a mel spectorgram with non repeated tokens as labels
                    # === Generate Spectrogram (20ms resolution) ===
                    sr = 16000
                    frame_length = int(0.02 * sr)  # 20 ms -> 320 samples
                    hop_length = frame_length      # no overlap

                    mel_spectrogram = T.MelSpectrogram(
                        sample_rate=sr,
                        n_fft=frame_length,
                        hop_length=hop_length,
                        n_mels=80
                    )(gt.detach().cpu())[0] # 2d > 1d
                    
                    l = [t for t in spec_tokens_repeated]
                    mel_spectrogram = mel_spectrogram[:,:len(l)]
                    
                    plt.figure(figsize=(20, 6))
                    plt.imshow(mel_spectrogram.log2().numpy(), aspect="auto", origin="lower", cmap="magma")

                    # Add token labels to x-axis
                    plt.xticks(
                        ticks=range(len(l)),
                        labels=l,
                        rotation=45
                    )

                    plt.xlabel("Special Tokens (20ms per token)")
                    plt.ylabel("Mel Frequency Channels")
                    plt.title("Mel Spectrogram with Special Tokens on X-axis")
                    plt.tight_layout()
                    plt.savefig(f"{save_dir}/temp/{step}_mel_spectrogram.png", dpi=600)
                    
                    
                
            logging.info(
            f"GEN-LOSS---step/total: {step}/{num_steps} "
            f"rec_loss: {gen_loss_components['rec_loss']:.4f}, "
            f"commit_loss: {gen_loss_components['commit_loss']:.4f}, "
            f"smooth_loss: {gen_loss_components['smooth_loss']:.4f}, "
            f"gen_loss: {gen_loss_components['gen_loss']:.4f}, "
            f"Generator decoded text entropy_loss: {lm_loss:.4f}, "
                )                
 
            writer.add_scalar('generator_loss/rec_loss', gen_loss_components['rec_loss'], step)
            writer.add_scalar('generator_loss/commit_loss', gen_loss_components['commit_loss'], step)
            writer.add_scalar('generator_loss/smooth_loss', gen_loss_components['smooth_loss'], step)
            writer.add_scalar('generator_loss/gen_loss', gen_loss_components['gen_loss'], step)
            writer.add_scalar('generator_loss/entropy_loss', lm_loss, step)

            # logging lr 
            writer.add_scalar('learning_rate/encoder', schedulers['enc'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/downsample', schedulers['down'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/decoder', schedulers['dec'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/discriminator', schedulers['disc'].get_last_lr()[0], step)
        
        
        # ===== Discriminator Forward Pass =====
        if step % config['train']['discriminator_freq'] != 0:
            doutput = {}
            disc_fake = models['discriminator'](z_q_disc.clone().detach(), z_q_disc_mask)
            doutput['disc_fake'] = disc_fake
            doutput['fake_x'] = z_q_disc.clone().detach()
            doutput['fake_pad_mask'] = z_q_disc_mask
            
            try:
                text, tmask = next(text_iter)
            except StopIteration:
                text_iter = iter(text_loader)
                text, tmask = next(text_iter)
            
            text = text.to(device)
            tmask = tmask.to(device)
                 
            text_emb = models['codebook'](text)
            disc_real, dlm_loss = models['discriminator'](text_emb, tmask, labels=text)
            doutput['disc_real'] = disc_real
            doutput['real_x'] = text_emb
            doutput['real_pad_mask'] = tmask
    
            disc_loss_components = loss_module.step_disc(doutput)
            total_lossd = disc_loss_components['total_loss'] + dlm_loss
            
            # update the total loss
            total_lossg = total_lossg + total_lossd
        else: 
            # if discriminator_freq = 5, then train the generator at 0, 5, 10, 15, ... steps.
            total_lossg = total_lossg + gen_loss_components['gen_loss']

        if step % config['logging']['step'] == 0:  
                logging.info(
                f"DISC-LOSS---step/total: {step}/{num_steps} "
                f"real_loss: {disc_loss_components['loss_real']:.4f}, "
                f"fake_loss: {disc_loss_components['loss_fake']:.4f}, "
                f"total_loss: {total_lossd:.4f}, "
                f"grad_pen: {disc_loss_components['grad_pen']:.4f}, "
                f"lm_loss: {dlm_loss:.4f}, "
                )                    
        
                writer.add_scalar('Discriminator_loss/discriminator_real_loss', disc_loss_components['loss_real'], step)
                writer.add_scalar('Discriminator_loss/discriminator_fake_loss', disc_loss_components['loss_fake'], step)
                writer.add_scalar('Discriminator_loss/discriminator_total_loss', total_lossd, step)
                writer.add_scalar('Discriminator_loss/discriminator_grad_pen', disc_loss_components['grad_pen'], step)
                writer.add_scalar('Discriminator_loss/discriminator_lm_loss', dlm_loss, step)

        # Backpropagation
        if torch.isnan(total_lossg) or torch.isinf(total_lossg):
            logging.warning(f"Skipping step {step} due to NaN/Inf in total_lossg")
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            for scheduler in schedulers.values():
                scheduler.step() 
            continue
        else: 
            total_lossg.backward()
        
        # Gradient clipping        
        max_grad_norm = config['train']['grad_clip']
        torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(models['downsample'].parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(models['upsample'].parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(models['decoder'].parameters(), max_grad_norm)
 
        # norm 
        def get_grad_norm(model):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        
        if step % config['logging']['step'] == 0:  
            writer.add_scalar('grad_norm/discriminator', get_grad_norm(models['discriminator']), step)
            writer.add_scalar('grad_norm/encoder', get_grad_norm(models['encoder']), step)
            writer.add_scalar('grad_norm/downsample', get_grad_norm(models['downsample']), step)
            writer.add_scalar('grad_norm/decoder', get_grad_norm(models['decoder']), step)
            writer.add_scalar('grad_norm/upsample', get_grad_norm(models['upsample']), step) 
        
        # Optimizer step
        if step >= freeze_steps:
            optimizers['enc'].step()
        optimizers['down'].step()
        optimizers['dec'].step()

        if step % config['train']['discriminator_freq'] != 0:
            optimizers['disc'].step()
        
        # scheduler step
        for scheduler in schedulers.values():
            scheduler.step()    
        # Zero gradients after the step
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # Checkpoint
        if step % config['checkpoint']['step'] == 0:
            
            # Evaluation function
            current_stoi = eval(models, val_speech_loader, loss_module, config, device, writer=writer, step=step)
            
            # if current_stoi > stoi:
            checkpoint_path = f"{save_dir}/checkpoints/step_{step:06d}.pt"
            torch.save({
                'step': step,
                'num_steps': num_steps,
                'models': {k: v.state_dict() for k, v in models.items()},
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                'schedulers': {k: v.state_dict() for k, v in schedulers.items()},
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            stoi = max( current_stoi, stoi )
            for m in models:
                if m != 'gtruth': models[m].train()
                    
                    
                    
                    






























@torch.no_grad()
def eval(models, speech_loader, loss_module, config, device, writer=None, step=0):
    
    for m in models: models[m].eval()
            
    with torch.no_grad():
        logging.info("Evaluating")
        
        WERCalculator = WhisperWERCalculator()
        
        all_pr = []
        all_gt = []
        all_txt = []
        total_rec_loss = 0.0
        total_commit_loss = 0.0
        total_smooth_loss = 0.0
        
        for waveforms, padding_masks, dur, paths, txt in tqdm(speech_loader):
            waveforms = waveforms.to(device) 
            padding_masks = padding_masks.to(device) 
        
            output = {}
            # ===== Encoder =====
            enc_out = models['encoder'](waveforms, padding_masks) 
            
            mask = ~enc_out['padding_mask'] 
            mask = mask.unsqueeze(-1).float()
            output['mask'] = mask
            
            # ===== GT =====
            gt = models['gtruth'].encode(waveforms.unsqueeze(1)) # [B, T//320, 1024] 
            assert enc_out['encoder_out'].shape[1]-10 <= gt.shape[1] <= enc_out['encoder_out'].shape[1]+10, f"GT shape: {gt.shape}, Encoder out shape: {enc_out['encoder_out'].shape}"
            gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
            output['gt'] = gt 
                
            # ===== Downsample =====
            down_out = models['downsample'](enc_out['encoder_out'], mask) 
            dmask = mask[:, ::config["upsample"]['stride']]
            
            # ===== Tokenizer =====
            n_student_probs, smoothness_loss, commitment_loss, z_q, z_q_disc, z_q_disc_mask, selected_encodings_list, selected_encodings_repeated = models['tokenizer'](
                down_out, 
                models['codebook'], 
                dmask,
            )
            
            output['smoothness_loss'] = smoothness_loss
            output['commitment_loss'] = commitment_loss
            output['z_q'] = z_q # already masked
            
            # ===== UpSample =====
            log_student_probs, up_out = models['upsample'](z_q, n_student_probs)
            up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]
            
            dec_out = models['decoder'](
                x=up_out,
                mask=enc_out['padding_mask'],
                s=enc_out['cnn_out'],
            )
            output['dec_out'] = dec_out
                
            output['disc_fake'] = None
            
            # Loss calculation
            gen_loss_components = loss_module.step_gen(output)        
            
            total_rec_loss += gen_loss_components['rec_loss']
            total_commit_loss += gen_loss_components['commit_loss']
            total_smooth_loss += gen_loss_components['smooth_loss']        
            
            dur = [int(dur[i] * 16000) for i in range(len(dur))] # dur in samples
            
            gt = [waveforms[i].unsqueeze(0)[:,:dur[i]] for i in range(len(waveforms))] # [ [1,T], [1,T], ...]
            gt = [x / torch.max(torch.abs(x)) for x in gt]   
            
            pr = [ models['gtruth'].decode(x.unsqueeze(0))[:,:dur[e]] for e,x in enumerate(output['dec_out']) ] # [ [1,T], [1,T], ...]
            pr = [x / torch.max(torch.abs(x)) for x in pr]
            
            all_gt.extend(gt)
            all_pr.extend(pr)
            all_txt.extend(txt)
            
        # Get the ASR loos for pr and gt
        cer_pred, cer_real, wer_pred, wer_real, pred_hyps, real_hyps = WERCalculator.compute_wer(all_pr, all_gt, all_txt)
        
        # Get PESQ
        # pesq = compute_pesq(all_gt, all_pr)
        
        # Get STOI
        stoi = compute_stoi(all_gt, all_pr)
        
        total_rec_loss /= len(speech_loader)
        total_commit_loss /= len(speech_loader)
        total_smooth_loss /= len(speech_loader)
        
        logging.info(
            f"Predicted CER: {cer_pred:.4f}, "
            f"Real CER: {cer_real:.4f}, "
            f"Predicted WER: {wer_pred:.4f}, "
            f"Real WER: {wer_real:.4f}, "
            # f"PESQ: {pesq:.4f}, "
            f"STOI: {stoi:.4f}, "
            f"rec_loss: {total_rec_loss:.4f}, "
            f"commit_loss: {total_commit_loss:.4f}, "
            f"smooth_loss: {total_smooth_loss:.4f}"
        )
        
        if writer is not None:
            writer.add_scalar('val_generator_loss/cer_pred', cer_pred, step)
            writer.add_scalar('val_generator_loss/cer_real', cer_real, step)
            writer.add_scalar('val_generator_loss/wer_pred', wer_pred, step)
            writer.add_scalar('val_generator_loss/wer_real', wer_real, step)
            
            # writer.add_scalar('val_generator_loss/pesq', pesq, step)
            writer.add_scalar('val_generator_loss/stoi', stoi, step)
            
            writer.add_scalar('val_generator_loss/rec_loss', total_rec_loss, step)
            writer.add_scalar('val_generator_loss/commit_loss', total_commit_loss, step)
            writer.add_scalar('val_generator_loss/smooth_loss', total_smooth_loss, step)


        return stoi
