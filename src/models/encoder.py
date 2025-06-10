import torch
import torch.nn as nn
import fairseq
from fairseq import checkpoint_utils
import torch.nn.functional as F

arg_overrides = {
    "apply_mask": True,

    "mask_selection": "static",
    "mask_length": 10,
    "mask_other": 0,
    "mask_prob": 0.75,

    "mask_channel_selection": "static",
    "mask_channel_length": 64,
    "mask_channel_other": 0,
    "mask_channel_prob": 0.5,

    "encoder_layerdrop": 0.0,
    "dropout": 0.0,
    "activation_dropout": 0.1,
    "attention_dropout": 0.0,

    "feature_grad_mult": 0.0, # always keeps the feature grad mult to 0.0
    
    "fp16": False,
}


class Encoder(torch.nn.Module):
    def __init__(self, ckpt_path="/raid/home/rajivratn/hemant_rajivratn/last/weights/convert_iter3.pt"):
        super().__init__()

        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path, arg_overrides)
    
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], state=state)
    
        model[0].remove_pretraining_modules()
        
        self.model = model[0]
    
        self.cfg = cfg
      
    def forward(self, source, padding_mask): 
        w2v_args = {
            "source": source, # source: (B, T)
            "padding_mask": padding_mask, # padding_mask: (B, T), 
            "mask": True and self.training,
            # "mask": False,
            "ret_conv": False,
        }
                      
        _, x, padding_mask = self.model.extract_features(**w2v_args)
        
        return x, padding_mask  # B x T x C ,  B x T
        
        
class Downsample(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=256, kernel_size=9, stride=2, groups=1):
        super().__init__()
    
        self.norm = torch.nn.LayerNorm(input_dim)
        # self.lin_q = torch.nn.Linear(input_dim, output_dim)
        self.lin_q1 = torch.nn.Linear(input_dim, 32)
        self.relu = torch.nn.ReLU()
        self.lin_q2 = torch.nn.Linear(32, output_dim)
        
        # scalar temperature for cosine similarity
        self.temp = torch.nn.Parameter(torch.tensor(10.0), requires_grad=True)
        
    def forward(self, x, mask): # B x T x C
        
        x = self.norm(x)
        x = x * mask
        # x = self.lin_q(x) # B x T x C
        x = self.lin_q1(x)
        x = self.relu(x)
        x = self.lin_q2(x)
        
        x = F.normalize(x, dim=-1) # normlaize
        
        return x, mask, self.temp # B x T x C 
    
if __name__ == "__main__":
    # Test encoder
    model = Encoder()
    print(model)
    
    print("Layer freezing for hubert")
    count = 0
    for name, param in model.named_parameters():
        count += 1
        
        if count < 177:
            param.requires_grad = False
            
        elif 'model.layer_norm' in name:
            param.requires_grad = False
            
        elif 'model.final_proj' in name:
            param.requires_grad = False
        
        print(name, param.requires_grad)