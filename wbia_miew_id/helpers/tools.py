# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import json
import errno
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch


__all__ = [
    'mkdir_if_missing',
    'check_isfile',
    'read_json',
    'write_json',
    'set_random_seed',
    'collect_env_info',
    'save_checkpoint',
    'load_checkpoint',
    'load_model_weights',
]


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, criterion, epoch, best_score, best_cmc, config=None, swa_model=None, swa_scheduler=None):
    """Save a full checkpoint for training resumption.
    
    Args:
        checkpoint_dir: Directory to save the checkpoint
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        criterion: The loss criterion state to save
        epoch: Current epoch number
        best_score: Best validation score so far
        best_cmc: Best CMC metrics so far
        config: Optional config dict to save
        swa_model: Optional SWA model to save
        swa_scheduler: Optional SWA scheduler to save
    
    Returns:
        Path to the saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'best_score': best_score,
        'best_cmc': best_cmc,
    }
    if config is not None:
        checkpoint['config'] = dict(config) if hasattr(config, '__iter__') else config
    if swa_model is not None:
        checkpoint['swa_model_state_dict'] = swa_model.state_dict()
    if swa_scheduler is not None:
        checkpoint['swa_scheduler_state_dict'] = swa_scheduler.state_dict()
    
    checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch} to {checkpoint_path}')
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, criterion=None, device=None, swa_model=None, swa_scheduler=None):
    """Load a checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        criterion: Optional criterion to restore state
        device: Device to load the checkpoint to
        swa_model: Optional SWA model to restore
        swa_scheduler: Optional SWA scheduler to restore
    
    Returns:
        tuple: (start_epoch, best_score, best_cmc) if full checkpoint,
               or (0, 0, None) if legacy model-only checkpoint
    """
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a full checkpoint or a legacy model-only checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if criterion is not None and 'criterion_state_dict' in checkpoint:
            try:
                criterion.load_state_dict(checkpoint['criterion_state_dict'])
            except RuntimeError as e:
                print(f'WARNING: Could not load criterion state_dict (likely n_classes mismatch): {e}')
                print('Continuing with freshly initialized criterion weights...')
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_score = checkpoint.get('best_score', 0)
        best_cmc = checkpoint.get('best_cmc', None)
        
        if swa_model is not None and 'swa_model_state_dict' in checkpoint:
            swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        if swa_scheduler is not None and 'swa_scheduler_state_dict' in checkpoint:
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler_state_dict'])
        
        print(f'Loaded full checkpoint from epoch {checkpoint.get("epoch", "unknown")}, best_score: {best_score}')
        return start_epoch, best_score, best_cmc
    else:
        # Legacy model-only checkpoint (just state_dict)
        model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded legacy model-only checkpoint from {checkpoint_path}')
        return 0, 0, None


def load_model_weights(checkpoint_path, model, device=None):
    """Load model weights from either a full checkpoint or legacy model-only checkpoint.
    
    This is a convenience function for evaluation that only needs model weights.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        device: Device to load the checkpoint to
    
    Returns:
        The model with loaded weights
    """
    print(f'Loading model weights from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a full checkpoint or a legacy model-only checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model weights from full checkpoint')
    else:
        # Legacy model-only checkpoint (just state_dict)
        model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded model weights from legacy checkpoint')
    
    return model