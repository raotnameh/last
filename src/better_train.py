import argparse
import logging
import random
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import os
import warnings
import sys
import matplotlib

sys.path.append(f"{os.getcwd()}/models/decoder")
warnings.simplefilter("ignore")
matplotlib.set_loglevel("critical")
logging.getLogger('matplotlib').disabled = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# Local imports
from dataset_speech import Dataset_speech
from dataset_txt import Dataset_txt
from models.codebook import Codebook
from models.decoder.decoder import Decoder, Upsample
from models.discriminator import Discriminator
from models.encoder import Encoder, Downsample
from models.gtruth import Gtruth
from models.tokenizer import Tokenizer
from loss import Loss


random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# step :- Prepare the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("-r", "--resume_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to run on (overrides config).")
    parser.add_argument("-l", "--log_dir", type=str, default=None, help="Directory for logs (overrides config).")
    parser.add_argument("-fp16", "--fp16", action="store_true", help="Use mixed precision training.")
    
    return parser.parse_args()

# step :- Prepare the Hyperparameters
def load_config(config_path: str) -> Dict:
    """Load and log training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Configure logging
def configure_logging(dir='logs/') -> None:
    """Initialize logging configuration with file and stream handlers."""
    os.makedirs(dir, exist_ok=True)
    log_filename = datetime.now().strftime("training_%Y-%m-%d_%H-%M-%S.log")
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dir}/{log_filename}"),
            logging.StreamHandler()
        ]
    )
    
# step :- Prepare the dataset.
def initialize_datasets(config: Dict) -> Tuple[DataLoader, DataLoader, Dict]:
    """Initialize and configure speech/text datasets with samplers."""
    class ShuffledBatchSampler(BatchSampler):
        """Custom batch sampler that shuffles batch order while maintaining sequence order within batches."""
        def __init__(self, sampler, batch_size, drop_last):
            super().__init__(sampler, batch_size, drop_last)  

        def __iter__(self):
            batches = list(super().__iter__())  
            random.shuffle(batches)  # Shuffle batch order
            return iter(batches)

    # step 1 :- Prepare the speech dataset.
    speech_dataset = Dataset_speech(
        input_manifest=config['dataset_speech']['path'],
        min_duration=config['dataset_speech']['min_duration'],
        max_duration=config['dataset_speech']['max_duration'],
    )

    speech_loader = DataLoader(
        speech_dataset,
        batch_sampler=ShuffledBatchSampler(
            sampler=SequentialSampler(speech_dataset),
            batch_size=config['dataset_speech']['batch_size'],
            drop_last=False,
        ),
        collate_fn=speech_dataset.collate_fn,
        pin_memory=True,
        num_workers=6
    )
    
    # speech_loader = DataLoader(
    #     speech_dataset,
    #     batch_size=config['dataset_txt']['batch_size'],
    #     collate_fn=speech_dataset.collate_fn,
    #     pin_memory=True,
    #     shuffle=True,
    #     num_workers=6
    # )

    # step 2 :- Prepare the text dataset.
    text_dataset = Dataset_txt(data=config['dataset_txt']['path'])
    text_loader = DataLoader(
        text_dataset,
        batch_size=config['dataset_txt']['batch_size'],
        collate_fn=text_dataset.collate_fn,
        pin_memory=True,
        shuffle=True,
        num_workers=6
    )

    logging.info(f"Number of batches in speech dataset: {len(speech_loader)}")
    logging.info(f"Number of batches in text dataset: {len(text_loader)}")
    
    return speech_loader, text_dataset, text_loader, text_dataset.vocab


# step :- Prepare the codebook
def setup_models(config: Dict, vocab: nn.Module) -> Dict:
    """Initialize and configure model components."""
    models = {
        'codebook': Codebook(vocab, config['codebook']['model_name']),
        'gtruth': Gtruth(), # Always non trainable
        'encoder': Encoder(config['encoder']['ckpt_path']),
    }
    
    models["downsample"] = Downsample(
        input_dim=models['encoder'].cfg['model']['encoder_embed_dim'],
        output_dim=models['codebook'].embedding.weight.shape[1],
        kernel_size=config['downsample']['kernel_size'],
        stride=config['downsample']['stride'],
        groups=config['downsample']['groups'],
        )
    
    models['tokenizer'] = Tokenizer(vocab=models['codebook'].vocab)
    
    models['upsample'] = Upsample(
        input_dim=models['codebook'].embedding.weight.shape[1],
        output_dim=config["decoder"]["transformer"]["decoder_hidden"],
        kernel_size=config['upsample']['kernel_size'],
        stride=config['upsample']['stride'],
        groups=config['upsample']['groups'],
        )
 
    models['decoder'] = Decoder(config['decoder'])
    
    models['discriminator'] = Discriminator(
        in_channels=models['codebook'].embedding.weight.shape[1], 
        hidden_dim=config['discriminator']['hidden_dim'], 
        kernel_size=config['discriminator']['kernel_size'])
    
    
    logging.info(f"Size of codebook: {models['codebook'].embedding.weight.shape[0]} x {models['codebook'].embedding.weight.shape[1]}")

    
    # Log model parameters
    total_params = 0
    for name, model in models.items():
        cur_params = sum(p.numel() for p in model.parameters()) / 1e6
        logging.info("%s parameters: %.4fM", 
                    name.capitalize(), 
                    cur_params)
        total_params += cur_params
    logging.info("Total parameters: %.4fM", total_params)
    
    return models

# step :- Prepare the training mode
def configure_training_mode(models: Dict, config: Dict) -> None:
    """Set model training modes and parameter requirements."""
 
    # Partially freeze encoder
    for name, param in models['encoder'].named_parameters():
        for n in config['encoder']['frozen_layers']:
            if str(f"model.encoder.layers.{n}") in name :
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # Log trainable parameters
    total_params = 0
    for name, model in models.items():
        cur_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logging.info("%s trainable parameters: %.4fM",
                    name.capitalize(), 
                    cur_params)
        total_params += cur_params
    logging.info("Total trainable parameters: %.4fM", total_params)  


# step :- Prepare the optimizer
def configure_optimizers(models: Dict, config: Dict) -> Dict:
    """Initialize optimizers for different model components using AdamW."""

    optimizers = {
        'enc': optim.AdamW(
            [p for p in models['encoder'].parameters() if p.requires_grad],
            lr=config['train']['lr_enc']
        ),
        'down': optim.AdamW(
            [p for p in models['downsample'].parameters() if p.requires_grad],
            lr=config['train']['lr_down']
        ),
        'dec': optim.AdamW(
            [p for p in models['upsample'].parameters() if p.requires_grad] +
            [p for p in models['decoder'].parameters() if p.requires_grad],
            lr=config['train']['lr_dec']
        ),
        'disc': optim.AdamW(
            [p for p in models['discriminator'].parameters() if p.requires_grad],
            lr=config['train']['lr_disc']
        )
    }
    
    
    def tri_stage_scheduler(optimizer, total_steps, phase_ratio=[0.03, 0.9, 0.07]):
        """
        Tri-stage LR scheduler that applies:
        - Warmup phase: LR increases linearly from 0 to base LR.
        - Constant phase: LR stays constant.
        - Decay phase: LR decreases linearly to 0.
        
        phase_ratio: a list with ratios for [warmup, constant, decay] phases.
        """
        warmup_steps = int(phase_ratio[0] * total_steps)
        constant_steps = int(phase_ratio[1] * total_steps)
        decay_steps = int(phase_ratio[2] * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup: from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step < warmup_steps + constant_steps:
                # Constant phase: LR stays at base value (multiplier 1)
                return 1.0
            else:
                # Linear decay: from 1 down to 0 over decay_steps
                decay_step = current_step - (warmup_steps + constant_steps)
                return max(0.0, 1.0 - float(decay_step) / float(max(1, decay_steps)))
        return LambdaLR(optimizer, lr_lambda)

    
    phase_ratio = config['lr_scheduler']['phase_ratio']
    total_steps = config['train']['num_steps']
    
    schedulers = {
        'enc': tri_stage_scheduler(optimizers['enc'], total_steps, phase_ratio),
        'down': tri_stage_scheduler(optimizers['down'], total_steps, phase_ratio),
        'dec': tri_stage_scheduler(optimizers['dec'], total_steps, phase_ratio),
        'disc': tri_stage_scheduler(optimizers['disc'], total_steps, phase_ratio),
    }
    
    return optimizers, schedulers

def load_checkpoint(checkpoint_path, models, optimizers, device):
    """Load model and optimizer states from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint['models']:
            model.load_state_dict(checkpoint['models'][name])
        else:
            logging.warning(f"No weights found for {name} in checkpoint")

    # Load optimizer states
    for name, optimizer in optimizers.items():
        if name in checkpoint['optimizers']:
            optimizer.load_state_dict(checkpoint['optimizers'][name])
        else:
            logging.warning(f"No state found for {name} optimizer")

    # Return additional info
    return {
        'step': checkpoint['step'],
        'num_steps': checkpoint['num_steps'],
        'config': checkpoint['config']
    }
    

def train(models: Dict, optimizers: Dict, schedulers:Dict, speech_loader: DataLoader, text_dataset, text_loader: DataLoader, loss_module: Loss, config: Dict, device: torch.device, start_step: int):
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config['logging'].get('log_dir', './logs'))
    
    # Initialize GradScaler
    scaler = GradScaler(enabled=config['train']['mixed_precision'])
    
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
            waveforms, padding_masks, paths = next(speech_iter)
        except StopIteration:
            speech_iter = iter(speech_loader)
            waveforms, padding_masks, paths = next(speech_iter)
        waveforms = waveforms.to(device) # [B, T]
        padding_masks = padding_masks.to(device) # [B, T] true for masked, false for not masked means [False, False, ..., True, True]
        
    
        # ===== Generator Forward Pass =====
        with autocast(enabled=config['train']['mixed_precision']):
            enc_out = models['encoder'](waveforms, padding_masks)  # [B, T, C] # step 1
            output["cnn_out"] = enc_out['cnn_out'] # [B, T // 320, C] 
            output['encoder_out'] = enc_out['encoder_out'] # [B, T // 320, C] 
            
            mask = ~enc_out['padding_mask'] # B,T//320 # 0 for masked positions.
            mask = mask.unsqueeze(-1).float() # [B, T // 320, 1]
            output['mask'] = mask
            
            # ===== Ground Truth =====
            with torch.no_grad():
                gt = models['gtruth'].encode(waveforms.unsqueeze(1)) # [B, T//320, 1024] 
                gt = gt[:,:mask.shape[1],:] * mask # [B, T, 1024]
                output['gt'] = gt 
            
            # ===== Generator Forward Pass Cont. =====
            down_out = models['downsample'](enc_out['encoder_out']) # [B, T // 2, C] 
            dmask = mask[:, ::config["upsample"]['stride']] # [B, T // config["upsample"]['stride'], 1]
            down_out = down_out[:,:dmask.shape[1],:] * dmask # [B, T // 2, C]
            output['down_out'] = down_out
            output['dmask'] = dmask
            
            smoothness_loss, commitment_loss, z_q, z_q_disc, z_q_disc_mask, selected_encodings_list = models['tokenizer'](
                down_out, 
                models['codebook'], 
                dmask
            )
            z_q_disc_mask = ~z_q_disc_mask.bool() # [B, T // 2, 1]
           
            output['smoothness_loss'] = smoothness_loss
            output['commitment_loss'] = commitment_loss
            output['z_q'] = z_q # already masked
            output['z_q_disc'] = z_q_disc # already masked
            
            up_out = models['upsample'](z_q)
            up_out = up_out[:,:mask.shape[1],:] * mask # [B, T, C]       
            output['up_out'] = up_out
            
            dec_out, dec_out2, dec_mask = models['decoder'](
                x=up_out,
                mask=enc_out['padding_mask'],
                s=enc_out['cnn_out'],
                use_s=config['decoder']['speaker']['use_s']
            )
            dec_out = dec_out[:,:dec_mask.shape[1],:] * dec_mask
            dec_out2 = dec_out2[:,:dec_mask.shape[1],:] * dec_mask
            output['dec_out'] = dec_out
            output['dec_out2'] = dec_out2
            output['dec_mask'] = dec_mask
            
            # Generator discriminator prediction
            disc_fake = models['discriminator'](z_q_disc, z_q_disc_mask)
            output['disc_fake'] = disc_fake
            
            # Loss calculation
            gen_loss_components = loss_module.step_gen(output, step=step, total_steps=num_steps)
            # total_lossg = sum(gen_loss_components.values())
            total_lossg = gen_loss_components['rec_loss'] 
            total_lossg = total_lossg + gen_loss_components['commit_loss'] 
            total_lossg = total_lossg + gen_loss_components['gen_loss']
            total_lossg = total_lossg + gen_loss_components['smooth_loss']
            
            if step % config['logging']['step'] == 0:
                
                logging.info(f"Generator encoded text path: --{paths[0]}--")
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
                writer.add_scalar('generator_loss/gen_loss', gen_loss_components['gen_loss'], step)
                writer.add_scalar('generator_loss/total_loss_gen', total_lossg, step)
            
            
            
            # ===== Discriminator Forward Pass =====
            # Get text batch (every 2 steps) with auto-reset
            if step % config['train']['discriminator_freq'] == 0:
                
                disc_fake = models['discriminator'](z_q_disc.clone().detach(), z_q_disc_mask)
                output['disc_fake'] = disc_fake
                output["disc_fake_x"] = z_q_disc
            
                try:
                    text, tmask = next(text_iter)
                except StopIteration:
                    text_iter = iter(text_loader)
                    text, tmask = next(text_iter)
                text = text.to(device)
                tmask = tmask.to(device)
                text_emb = models['codebook'](text)
                disc_real = models['discriminator'](text_emb, tmask.unsqueeze(-1))
                output['disc_real'] = disc_real
                output['disc_real_x'] = text_emb

                loss_module.gan_loss.discriminator = models['discriminator']
                
                disc_loss_components = loss_module.step_disc(output)
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
            
            
        # Backpropagation   
        if step % config['train']['discriminator_freq'] == 0:
            total_lossg += total_lossd
        total_lossg /= config['train']['gradient_accumulation_steps']
        scaler.scale(total_lossg).backward() if config['train']['mixed_precision'] else total_lossg.backward()


        if step % config['train']['gradient_accumulation_steps'] == 0:
            
            # Gradient clipping
            if config['train']['mixed_precision']:
                for optimizer in optimizers.values():
                    scaler.unscale_(optimizer)
                
            max_grad_norm = config['train']['grad_clip']
            torch.nn.utils.clip_grad_norm_(
                list(models['encoder'].parameters()) +
                list(models['downsample'].parameters()) + 
                list(models['upsample'].parameters()) + 
                list(models['decoder'].parameters()), 
                max_grad_norm
            )
            if step % config['train']['discriminator_freq'] == 0:
                torch.nn.utils.clip_grad_norm_(models['discriminator'].parameters(), max_grad_norm)
                
            # Optimizer step
            if config['train']['mixed_precision']:
                if step >= freeze_steps:
                    scaler.step(optimizers['enc'])
                scaler.step(optimizers['down'])
                scaler.step(optimizers['dec'])
                if step % config['train']['discriminator_freq'] == 0:
                    scaler.step(optimizers['disc'])
                   
                scaler.update()
            else:
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
                'config': config
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()



def main():
    args = parse_args()
    config = load_config(args.config) 
    configure_logging(config['logging']['dir'])
    # Override config if command-line args are provided
    if args.resume_checkpoint:
        config['train']['resume_checkpoint'] = True
        config['train']['checkpoint_path'] = args.resume_checkpoint
    if args.device:
        config['device'] = args.device
    if args.log_dir:
        config['logging']['dir'] = args.log_dir
    logging.info(f"Loaded config from {args.config}")
    logging.info(f"Command-line args: {args}")   
    logging.info(f"Config after command-line overrides: {config}")

    config['train']['num_steps'] *= config['train']['gradient_accumulation_steps']
    config['train']['freeze_steps'] *= config['train']['gradient_accumulation_steps']
    config['checkpoint']['step'] *= config['train']['gradient_accumulation_steps']
    
    
    # Set random seeds
    random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    
    # Initialize datasets and models
    speech_loader, text_dataset, text_loader, vocab = initialize_datasets(config)
    models = setup_models(config, vocab)
    # Training setup
    models['encoder'].train()
    models['downsample'].train()
    models['upsample'].train()
    models['decoder'].train()
    models['discriminator'].train()
    configure_training_mode(models, config)
    
    # Move models to device
    device = torch.device(config.get("device"))
    for model in models.values():
        model.to(device)
        
    # Initialize optimizers and loss
    optimizers, schedulers = configure_optimizers(models, config)
    loss_module = Loss(config)
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    
    
    # Resume training if checkpoint specified
    start_step = 1
    if config['train']['resume_checkpoint']:
        checkpoint_info = load_checkpoint(
            config['train']['checkpoint_path'],
            models,
            optimizers,
            device
        )
        start_step = checkpoint_info['step'] + 1
        logging.info(f"Resuming training from step {start_step}")

    
    # Main training loop
    try:
        train(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            speech_loader=speech_loader,
            text_loader=text_loader,
            text_dataset=text_dataset,
            loss_module=loss_module,
            config=config,
            device=device,
            start_step=start_step  # Add this
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.exception("Unexpected error occurred during training")
        

if __name__ == "__main__":
    main()