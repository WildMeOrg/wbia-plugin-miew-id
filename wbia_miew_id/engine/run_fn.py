import torch
import torch.distributed as dist
from .train_fn import train_fn
from .eval_fn import eval_fn
from .group_eval import group_eval_fn
from helpers.swatools import update_bn
import wandb
import os

def run_fn(rank, config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
    best_score = 0.0

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.engine.epochs):
        # Training phase
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler, epoch, use_wandb=(use_wandb and rank == 0), swa_model=swa_model, swa_start=swa_start, swa_scheduler=swa_scheduler)

        # Ensure only the master process saves the checkpoint
        if rank == 0:
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch}.bin'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
            }, checkpoint_path)

        # Validation phase
        print("\nGetting metrics on validation set...")
        eval_groups = config.data.test.eval_groups

        if eval_groups:
            if rank == 0:
                valid_score, valid_cmc = group_eval_fn(config, eval_groups, model, use_wandb=use_wandb)
            else:
                valid_score, valid_cmc = group_eval_fn(config, eval_groups, model, use_wandb=False)
        else:
            valid_score, valid_cmc = eval_fn(valid_loader, model, device, use_wandb=use_wandb, return_outputs=False)

        valid_score_tensor = torch.tensor([valid_score], device=device)
        
        # Now valid_score_tensor can be safely used with dist.all_reduce
        if dist.is_initialized():
            dist.all_reduce(valid_score_tensor, op=dist.ReduceOp.SUM)
            valid_score_tensor /= dist.get_world_size()
            
        # Convert it back to a Python scalar for any further non-tensor operations if necessary
        valid_score = valid_score_tensor.item()

        # Conditional operations based on aggregated metrics
        if rank == 0:
            print(f'Aggregated Validation Score: {valid_score}')

            if use_wandb:
                wandb.log({'Validation Score': valid_score, 'epoch': epoch})

            if valid_score > best_score:
                best_score = valid_score
                best_model_path = f'{checkpoint_dir}/model_best.bin'
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_model_path)
                print(f'New best model saved at {best_model_path} (score: {best_score})')

    # SWA model update and final save, executed only by the master process
    if swa_model and rank == 0:
        print("Updating SWA model batchnorm statistics...")
        update_bn(train_loader, swa_model, device=device)
        swa_model_path = f'{checkpoint_dir}/swa_model_final.bin'
        torch.save(swa_model.state_dict(), swa_model_path)
        print(f'SWA model saved at {swa_model_path}')

    return best_score
