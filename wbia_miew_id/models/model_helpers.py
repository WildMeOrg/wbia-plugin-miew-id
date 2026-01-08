import torch
import sys
from .model import MiewIdNet
from wbia_miew_id.helpers import load_model_weights

def get_model(cfg, checkpoint_path=None, use_gpu=True):

    model = MiewIdNet(**dict(cfg.model_params))

    if use_gpu:
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")

    if checkpoint_path:
        load_model_weights(checkpoint_path, model, device=device)

    return model