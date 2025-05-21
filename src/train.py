import argparse
import logging
import random
import numpy as np
import os
import warnings
import sys
import matplotlib
from datetime import datetime
import yaml

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

sys.path.append(f"{os.getcwd()}/models/decoder")
warnings.simplefilter("ignore")
matplotlib.set_loglevel("critical")
logging.getLogger('matplotlib').disabled = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local imports
from dataset_speech import Dataset_speech
from dataset_txt import Dataset_txt
from models.codebook import Codebook
from models.encoder import Encoder, Downsample
from models.tokenizer import Tokenizer

from utils_train import * 


# step :- Prepare the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("-r", "--resume_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to run on (overrides config).")
    parser.add_argument("-l", "--log_dir", type=str, default=None, help="Directory for logs (overrides config).")
    parser.add_argument("-fp16", "--fp16", action="store_true", help="Use mixed precision training.")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate the model instead of training.")
    
    return parser.parse_args()

# step :- Prepare the Hyperparameters
def load_config(config_path):
    """Load and log training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Configure logging
def configure_logging(dir='logs/'):
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
def initialize_datasets(config, split='train', shuffle=True, bsz=1):
    """Initialize and configure speech/text datasets with samplers."""
       
    # step 1 :- Prepare the speech dataset.
    speech_dataset = Dataset_speech(
        input_manifest=config['dataset_speech'][f'{split}_path'],
        min_duration=config['dataset_speech']['min_duration'],
        max_duration=config['dataset_speech']['max_duration'],
        split=split,
    )    
    speech_loader = DataLoader(
        speech_dataset,
        batch_size=bsz,
        shuffle=shuffle,
        collate_fn=speech_dataset.collate_fn,
        num_workers=4
    )
    
    logging.info(f"Number of batches in {split} speech dataset: {len(speech_loader)}")
    
    if split == 'train':           
        # step 2 :- Prepare the text dataset.
        text_dataset = Dataset_txt(data=config['dataset_txt']['path'], skip_non_speech=config['dataset_txt']['skip_non_speech'])
        text_loader = DataLoader(
            text_dataset,
            batch_size=config['dataset_txt']['batch_size'],
            shuffle=True,
            collate_fn=text_dataset.collate_fn,
            num_workers=4
        )

        logging.info(f"Number of batches in text dataset: {len(text_loader)}")

        return speech_loader, text_dataset, text_loader, text_dataset.vocab
    else:
        return speech_loader

# step :- Prepare the codebook
def setup_models(config, vocab):
    """Initialize and configure model components."""
    models = {
        'codebook': Codebook(vocab, config['codebook']['model_name']),
        'encoder': Encoder(config['encoder']['ckpt_path']),
    }
    
    models["downsample"] = Downsample(
        input_dim=models['encoder'].cfg['model']['encoder_embed_dim'],
        output_dim=models['codebook'].embedding.weight.shape[1],
        kernel_size=config['downsample']['kernel_size'],
        stride=config['downsample']['stride'],
        groups=config['downsample']['groups'], 
        )
    
    models['tokenizer'] = Tokenizer(config, models['codebook'], config["train"]["groups"], config["train"]["temp"])
    
    logging.info(f"Size of codebook: {models['codebook'].embedding.weight.shape[0]} x {models['codebook'].embedding.weight.shape[1]}")
    logging.info(models['encoder'])
    logging.info(models['downsample'])
    logging.info(models['codebook'])
    logging.info(models['tokenizer'])
    
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
def configure_training_mode(models, config):
    """Set model training modes and parameter requirements."""
    
    # set encoder layers to train false for config layer number
    logging.info("Layer freezing for encoder")
    count = 0
    for name, param in models['encoder'].named_parameters():
        count += 1
        if count < 177: # 193,177,161,145,129,113,97,81,65,49,33,17
            param.requires_grad = False
        elif 'model.layer_norm' in name:
            param.requires_grad = False
        elif 'model.encoder.layer_norm' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    for name, param in models['encoder'].named_parameters():
        param.requires_grad = False
        if param.requires_grad: logging.info(f"Trainable parameter: {name} - {param.requires_grad}")
            
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
def configure_optimizers(models, config, dataloader):
    """Initialize optimizers for different model components using AdamW."""

    optimizer = optim.AdamW(
            [p for p in models['encoder'].parameters() if p.requires_grad] + [p for p in models['downsample'].parameters() if p.requires_grad],
            lr=config['train']['lr'],
            betas=(0.9, 0.99),
            weight_decay=0.1, 
    )   
        

    def tri_stage_scheduler(optimizer, total_steps, phase_ratio=[0.03, 0.9, 0.07], low=1e-2):
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
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step < warmup_steps + constant_steps:
                # Constant phase: LR stays at base value (multiplier 1)
                return 1.0
            else:
                # Linear decay: from 1 down to low over decay_steps
                decay_step = current_step - (warmup_steps + constant_steps)
                return max( low, (1.0 - float(decay_step) / float(max(1, decay_steps))) )
        return LambdaLR(optimizer, lr_lambda)

    
    phase_ratio = config['lr_scheduler']['phase_ratio']
    total_steps = config['train']['steps']
    logging.info(f"Total number of steps: {total_steps}")
    
    scheduler = tri_stage_scheduler(optimizer, total_steps, phase_ratio)
    
    return optimizer, scheduler
    

def load_checkpoint(checkpoint_path, models, optimizers, schedulers):
    """Load model and optimizer states from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint['models']: 
            model.load_state_dict(checkpoint['models'][name])  # fallback for single GPU or non-wrapped
        else:
            logging.warning(f"No weights found for {name} in checkpoint")

    # Load optimizer states
    for name, optimizer in optimizers.items():
        if name in checkpoint['optimizers']:
            optimizer.load_state_dict(checkpoint['optimizers'][name])
        else:
            logging.warning(f"No state found for {name} optimizer")
    
    # Load schedulers
    for name, scheduler in schedulers.items():
        if name in checkpoint['schedulers']:
            scheduler.load_state_dict(checkpoint['schedulers'][name])


def main():
    args = parse_args()
    if args.resume_checkpoint: 
        config = torch.load(args.resume_checkpoint)["config"]
    else:
        config = load_config(args.config) 
        
    
    # Set random seeds
    random.seed(config['train']['seed'])
    torch.manual_seed(config['train']['seed'])
    np.random.seed(config['train']['seed'])
    
    # Override config if command-line args are provided
    if args.resume_checkpoint:
        config['train']['resume_checkpoint'] = True
        config['train']['checkpoint_path'] = args.resume_checkpoint
    if args.device:
        config['device'] = args.device
    if args.log_dir:
        config['logging']['dir'] = args.log_dir
    if args.fp16:
        config['train']['mixed_precision'] = True
    if args.eval:
        config['eval']['eval'] = True
        
    # Configure logging
    configure_logging(config['logging']['dir'])
    # save the config file to the log directory
    with open(f"{config['logging']['dir']}/config.yaml", "w") as f:
        yaml.dump(config, f)
    logging.info("Logging configuration:") 
    
    logging.info(f"Loaded config from {args.config}")
    logging.info(f"Command-line args: {args}")   
    logging.info(f"Config after command-line overrides: {config}")
    
    # Initialize datasets and models
    train_speech_loader, _, _, vocab = initialize_datasets(config, split='train', shuffle=True, bsz = config['dataset_speech']['batch_size'])
    val_speech_loader = initialize_datasets(config, split='val', shuffle=False, bsz = config['dataset_speech']['batch_size'])
    test_speech_loader = initialize_datasets(config, split='test', shuffle=False, bsz = config['dataset_speech']['batch_size'])
    
    speech_loader = [train_speech_loader, val_speech_loader, test_speech_loader]
    
    # Initialize models    
    models = setup_models(config, vocab)
    # Training setup
    configure_training_mode(models, config)
    
    # Determine the device (GPU or CPU)
    device = torch.device(config.get("device"))
    for name in models:
        models[name].to(device)

    # Initialize optimizers
    optimizer, scheduler = configure_optimizers(models, config, train_speech_loader)
    
    # root dir for saving 
    save_dir = config['logging']['dir']
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config['logging']['dir'])
    
    # Resume training if checkpoint specified
    if config['train']['resume_checkpoint']:
        load_checkpoint(
            config['train']['checkpoint_path'],
            models,
            optimizer,
            scheduler,
        )
    
    train(
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        speech_loader=speech_loader,
        config=config,
        device=device,
        writer=writer,
        save_dir=save_dir,
    )

    writer.close()



if __name__ == "__main__":
    main()