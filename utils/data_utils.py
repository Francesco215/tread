import torch
import numpy as np
from typing import Any
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from streaming.base.format.mds.encodings import Encoding, _encodings



class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0

_encodings["uint8"] = uint8

def custom_collate_fn(batch):
    latents = []
    labels = []
    for item in batch:
        latent_np = item['vae_output']
        latent = torch.from_numpy(latent_np).reshape(4, 32, 32)
        latents.append(latent)
        
        labels.append(int(item['label']))
    
    return torch.stack(latents), torch.tensor(labels)

def make_loader(root, batch_size=32, num_workers=4):
    
    # StreamingDataset configuration
    dataset = StreamingDataset(
        local=root,
        remote=None, # We are using local files
        split=None, 
        shuffle=True,
        batch_size=batch_size,
        num_canonical_nodes=1, 
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
            
    return loader
