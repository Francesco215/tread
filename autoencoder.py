import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

class SDVAE_EMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.bfloat16)
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.processor = VaeImageProcessor()
        
    def encode(self, x):
        return self.model.encode(x).latents * self.model.config.scaling_factor
    
    def decode(self, x):
        with torch.no_grad():
            # Note: The user example does not divide by scaling factor, so we follow that.
            x = self.model.decode(x / self.model.config.scaling_factor).sample
        
        # Use VaeImageProcessor for postprocessing
        # do_denormalize=[True, True] is taken from the user's example usage
        return self.processor.postprocess(image=x.detach(), do_denormalize=[True, True])
