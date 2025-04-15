import torch
import logging
import torchaudio


def train_vqvae(models, optimizers, schedulers, speech_loader, text_dataset, text_loader, loss_module, config, device, writer, start_step):
    
    logging.info("Training VQVAE")
    
    num_steps = config['train']['num_steps']
    freeze_steps = config['train']['freeze_steps']

    
    # Initialize iterators
    speech_iter = iter(speech_loader)

    for optimizer in optimizers.values():
        optimizer.zero_grad()

    for step in range(start_step, num_steps + 1):
        output = {}
        
        # ===== Data Preparation =====
        try:
            waveforms, padding_masks, paths, dur = next(speech_iter)
        except StopIteration:
            speech_iter = iter(speech_loader)
            waveforms, padding_masks, paths, dur = next(speech_iter)
        waveforms = waveforms.to(device) # [B, T]
        padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
        
    
        # ===== Generator Forward Pass =====
        # ===== Encoder =====
        if step >= freeze_steps:
            models['encoder'].train() 
        enc_out = models['encoder'](waveforms, padding_masks)  # [B, T, C] # step 1
        output["cnn_out"] = enc_out['cnn_out'] # [B, T // 320, C] 
        output['encoder_out'] = enc_out['encoder_out'] # [B, T // 320, C] 
        
        mask = ~enc_out['padding_mask'] # B,T//320 # 0 for masked positions.
        mask = mask.unsqueeze(-1).float() # [B, T // 320, 1]
        output['mask'] = mask
        
        # ===== Ground Truth =====
        with torch.no_grad():
            gt = models['gtruth'].encode(waveforms.unsqueeze(1)) # [B, T//320, 1024] 
            assert enc_out['encoder_out'].shape[1]-10 <= gt.shape[1] <= enc_out['encoder_out'].shape[1]+10, f"GT shape: {gt.shape}, Encoder out shape: {enc_out['encoder_out'].shape}"
            gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
            output['gt'] = gt 
            
        # ===== Downsample =====
        down_out = models['downsample'](enc_out['encoder_out'], mask) # [B, T // 2, C], [B, T // 2, vocab_size]
        dmask = mask[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride'], 1]
        output['down_out'] = down_out
        output['dmask'] = dmask
        
        # ===== Tokenizer =====
        smoothness_loss, commitment_loss, z_q, z_q_disc, z_q_disc_mask, selected_encodings_list = models['tokenizer'](
            down_out, 
            models['codebook'], 
            dmask,
        )
        z_q_disc_mask = ~z_q_disc_mask.bool() # [B, T // 2, 1]
        
        output['smoothness_loss'] = smoothness_loss
        output['commitment_loss'] = commitment_loss
        output['z_q'] = z_q # already masked
        output['z_q_disc'] = z_q_disc # already masked
        
        # ===== UpSample =====
        up_out = models['upsample'](z_q)
        up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]
        output['up_out'] = up_out
        
        dec_out = models['decoder'](
            x=up_out,
            mask=enc_out['padding_mask'],
            s=enc_out['cnn_out'],
        )
        output['dec_out'] = dec_out
        
            
        output['disc_fake'] = None
        # Loss calculation
        gen_loss_components = loss_module.step_gen(output)        
        total_lossg = gen_loss_components['rec_loss']
        # total_lossg = total_lossg + gen_loss_components['commit_loss'] 
        total_lossg = total_lossg + gen_loss_components['smooth_loss']
            
            
        if step % config['logging']['step'] == 0:
            
            logging.info(f"Generator encoded text path: --{paths[0]}-- of length {dur[0]} seconds--")
            logging.info( f"Generator decoded text with special tokens: --{text_dataset.decode(selected_encodings_list[0],keep_special_tokens=True)}--" )
            logging.info( f"Generator decoded text without special tokens: --{text_dataset.decode(selected_encodings_list[0])}--" )
        
            with torch.no_grad():
                pred = models['gtruth'].decode(output['dec_out'][0].unsqueeze(0)).clone().detach().cpu()
                torchaudio.save(f"temp/{step}_{text_dataset.decode(selected_encodings_list[0])}_.wav", pred, sample_rate=16000)
            
            
            
            logging.info(
            f"VQ-VAE-LOSS---step/total: {step}/{num_steps} "
            f"rec_loss: {gen_loss_components['rec_loss']:.4f}, "
            f"commit_loss: {gen_loss_components['commit_loss']:.4f}, "
            f"smooth_loss: {gen_loss_components['smooth_loss']:.4f}, "
            f"total_loss: {total_lossg:.4f}"
                    )                
            writer.add_scalar('generator_loss/rec_loss', gen_loss_components['rec_loss'], step)
            writer.add_scalar('generator_loss/commit_loss', gen_loss_components['commit_loss'], step)
            writer.add_scalar('generator_loss/smooth_loss', gen_loss_components['smooth_loss'], step)
            writer.add_scalar('generator_loss/total_loss_gen', total_lossg, step)
    

            # logging lr 
            writer.add_scalar('learning_rate/encoder', schedulers['enc'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/downsample', schedulers['down'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/decoder', schedulers['dec'].get_last_lr()[0], step)
            
            
        # Backpropagation    
        total_lossg /= config['train']['gradient_accumulation_steps']
        total_lossg.backward()


        if step % config['train']['gradient_accumulation_steps'] == 0:
            # Gradient clipping        
            max_grad_norm = config['train']['grad_clip']
            if step >= freeze_steps:
                torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                list(models['downsample'].parameters()) + 
                list(models['upsample'].parameters()) + 
                list(models['decoder'].parameters()),
                max_grad_norm
            )
                
            # Optimizer step
            if step >= freeze_steps:
                optimizers['enc'].step()
            optimizers['down'].step()
            optimizers['dec'].step()
        
        
            # scheduler step
            for scheduler in schedulers.values():
                scheduler.step()    
           
            # Zero gradients after the step
            for optimizer in optimizers.values():
                optimizer.zero_grad()
        
            
            
        # Checkpoint
        if step % config['checkpoint']['step'] == 0:
            checkpoint_path = f"{config['checkpoint']['dir']}/step_{step:06d}.pt"
            torch.save({
                'step': step,
                'num_steps': num_steps,
                'models': {k: v.state_dict() for k, v in models.items()},
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                'schedulers': {k: v.state_dict() for k, v in schedulers.items()},
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            


def train(models, optimizers, schedulers, speech_loader, text_dataset, text_loader, loss_module, config, device, writer, start_step):
    
    num_steps = config['train']['num_steps']
    freeze_steps = config['train']['freeze_steps']

    
    # Initialize iterators
    speech_iter = iter(speech_loader)
    text_iter = iter(text_loader)
    
    loss_module.gan_loss.training = True

    for optimizer in optimizers.values():
        optimizer.zero_grad()
    

    for step in range(start_step, num_steps + 1):
        output = {}
        
        # ===== Data Preparation =====
        try:
            waveforms, padding_masks, paths, dur = next(speech_iter)
        except StopIteration:
            speech_iter = iter(speech_loader)
            waveforms, padding_masks, paths, dur = next(speech_iter)
        waveforms = waveforms.to(device) # [B, T]
        padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
        
    
        # ===== Generator Forward Pass =====
        
        # ===== Encoder =====
        enc_out = models['encoder'](waveforms, padding_masks)  # [B, T, C] # step 1
        output["cnn_out"] = enc_out['cnn_out'] # [B, T // 320, C] 
        output['encoder_out'] = enc_out['encoder_out'] # [B, T // 320, C] 
        
        mask = ~enc_out['padding_mask'] # B,T//320 # 0 for masked positions.
        mask = mask.unsqueeze(-1).float() # [B, T // 320, 1]
        output['mask'] = mask
        
        # ===== Ground Truth =====
        with torch.no_grad():
            gt = models['gtruth'].encode(waveforms.unsqueeze(1)) # [B, T//320, 1024] 
            assert enc_out['encoder_out'].shape[1]-10 <= gt.shape[1] <= enc_out['encoder_out'].shape[1]+10, f"GT shape: {gt.shape}, Encoder out shape: {enc_out['encoder_out'].shape}"
            gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
            output['gt'] = gt 
        
        # ===== Downsample =====
        down_out = models['downsample'](enc_out['encoder_out'], mask) # [B, T // 2, C], [B, T // 2, vocab_size]
        dmask = mask[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride'], 1]
        down_out = down_out[:,:dmask.shape[1],:] * dmask # [B, T // 2, C]
        output['down_out'] = down_out
        output['dmask'] = dmask
        
        # ===== Tokenizer =====
        smoothness_loss, commitment_loss, z_q, z_q_disc, z_q_disc_mask, selected_encodings_list = models['tokenizer'](
            down_out, 
            models['codebook'], 
            dmask,
        )
        z_q_disc_mask = ~z_q_disc_mask.bool() # [B, T // 2, 1]
        
        output['smoothness_loss'] = smoothness_loss
        output['commitment_loss'] = commitment_loss
        output['z_q'] = z_q # already masked
        output['z_q_disc'] = z_q_disc # already masked
        
        # ===== UpSample =====
        up_out = models['upsample'](z_q)
        up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]       
        output['up_out'] = up_out
        
        dec_out, dec_out2, dec_mask = models['decoder'](
            x=up_out,
            mask=enc_out['padding_mask'],
            s=enc_out['cnn_out'],
        )
        dec_out = dec_out[:,:dec_mask.shape[1],:] * dec_mask
        dec_out2 = dec_out2[:,:dec_mask.shape[1],:] * dec_mask
        output['dec_out'] = dec_out
        output['dec_out2'] = dec_out2
        output['dec_mask'] = dec_mask
        
            
        output['disc_fake'] = None
        # ===== Discriminator Forward Pass =====
        if step % config['train']['discriminator_freq'] != 0:
            # ===== Discriminator Generator Update =====
            disc_fake = models['discriminator'](z_q_disc, z_q_disc_mask)
            output['disc_fake'] = disc_fake
        else:
            doutput = {}
            
            disc_fake = models['discriminator'](z_q_disc.detach(), z_q_disc_mask)
            doutput['disc_fake'] = disc_fake
            doutput["disc_fake_x"] = z_q_disc
        
            try:
                text, tmask = next(text_iter)
            except StopIteration:
                text_iter = iter(text_loader)
                text, tmask = next(text_iter)
            text = text.to(device)
            tmask = tmask.to(device)
            
            
            text_emb = models['codebook'](text)
            disc_real = models['discriminator'](text_emb, tmask.unsqueeze(-1))
            doutput['disc_real'] = disc_real
            doutput['disc_real_x'] = text_emb

            loss_module.gan_loss.discriminator = models['discriminator']
            disc_loss_components = loss_module.step_disc(doutput)
            total_lossd = disc_loss_components['total_loss']
            
            if step % config['logging']['step'] == 0:  
                logging.info(
                f"DISC-LOSS---step/total: {step}/{num_steps} "
                f"real_loss: {disc_loss_components['loss_real']:.4f}, "
                f"fake_loss: {disc_loss_components['loss_fake']:.4f}, "
                f"gp_loss: {disc_loss_components['grad_pen']:.4f}, "
                f"total_loss: {disc_loss_components['total_loss']:.4f}"
                )                    
                writer.add_scalar('Discriminator_loss/discriminator_total_loss', total_lossd, step)
                writer.add_scalar('Discriminator_loss/discriminator_real_loss', disc_loss_components['loss_real'], step)
                writer.add_scalar('Discriminator_loss/discriminator_fake_loss', disc_loss_components['loss_fake'], step)
                writer.add_scalar('Discriminator_loss/discriminator_gp_loss', disc_loss_components['grad_pen'], step)

        # Loss calculation
        gen_loss_components = loss_module.step_gen(output)        
        total_lossg = gen_loss_components['rec_loss'] 
        total_lossg = total_lossg + gen_loss_components['commit_loss'] 
        total_lossg = total_lossg + gen_loss_components['smooth_loss']
        if output['disc_fake'] is not None:
            total_lossg = total_lossg + gen_loss_components['gen_loss']
            
        if step % config['logging']['step'] == 0:
            
            logging.info(f"Generator encoded text path: --{paths[0]}-- of length {dur[0]} seconds--")
            logging.info( f"Generator decoded text with special tokens: --{text_dataset.decode(selected_encodings_list[0],keep_special_tokens=True)}--" )
            logging.info( f"Generator decoded text without special tokens: --{text_dataset.decode(selected_encodings_list[0])}--" )
                
            logging.info(
            f"GEN-LOSS---step/total: {step}/{num_steps} "
            f"rec_loss: {gen_loss_components['rec_loss']:.4f}, "
            f"commit_loss: {gen_loss_components['commit_loss']:.4f}, "
            f"smooth_loss: {gen_loss_components['smooth_loss']:.4f}, "
            f"gen_loss: {gen_loss_components['gen_loss']:.4f}, "
            f"total_loss: {total_lossg:.4f}"
                    )                
            writer.add_scalar('generator_loss/rec_loss', gen_loss_components['rec_loss'], step)
            writer.add_scalar('generator_loss/commit_loss', gen_loss_components['commit_loss'], step)
            writer.add_scalar('generator_loss/smooth_loss', gen_loss_components['smooth_loss'], step)
            if output['disc_fake'] is not None: 
                writer.add_scalar('generator_loss/gen_loss', gen_loss_components['gen_loss'], step)
            writer.add_scalar('generator_loss/total_loss_gen', total_lossg, step)
    

            # logging lr 
            writer.add_scalar('learning_rate/encoder', schedulers['enc'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/downsample', schedulers['down'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/decoder', schedulers['dec'].get_last_lr()[0], step)
            writer.add_scalar('learning_rate/discriminator', schedulers['disc'].get_last_lr()[0], step)
          
            
        # Backpropagation   
        if step % config['train']['discriminator_freq'] == 0:
            total_lossg += total_lossd    
        
        total_lossg /= config['train']['gradient_accumulation_steps']
        
        total_lossg.backward()


        if step % config['train']['gradient_accumulation_steps'] == 0:
            # Gradient clipping        
            max_grad_norm = config['train']['grad_clip']
            if step >= freeze_steps:
                torch.nn.utils.clip_grad_norm_(models['encoder'].parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                list(models['downsample'].parameters()) + 
                list(models['upsample'].parameters()) + 
                list(models['decoder'].parameters()),
                max_grad_norm
            )
            if step % config['train']['discriminator_freq'] == 0:
                torch.nn.utils.clip_grad_norm_(models['discriminator'].parameters(), max_grad_norm)
                
            # Optimizer step
            if step >= freeze_steps:
                optimizers['enc'].step()
            optimizers['down'].step()
            optimizers['dec'].step()
            if step % config['train']['discriminator_freq'] == 0:
                optimizers['disc'].step()
        
        
            # scheduler step
            for scheduler in schedulers.values():
                scheduler.step()    
           
            # Zero gradients after the step
            for optimizer in optimizers.values():
                optimizer.zero_grad()
        
            
            
        # Checkpoint
        if step % config['checkpoint']['step'] == 0:
            checkpoint_path = f"{config['checkpoint']['dir']}/step_{step:06d}.pt"
            torch.save({
                'step': step,
                'num_steps': num_steps,
                'models': {k: v.state_dict() for k, v in models.items()},
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                'schedulers': {k: v.state_dict() for k, v in schedulers.items()},
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
