import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from wbia_miew_id.metrics import AverageMeter, compute_calibration


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None):
    model.train()
    loss_score = AverageMeter()
    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for bi, d in tk0:
        images = d['image']
        targets = d['label']
        batch_size = images.shape[0]

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(images, targets)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

    if swa_model and epoch > swa_start:
        print("Updating swa model...")
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    if use_wandb:
        wandb.log({
            "train loss": loss_score.avg,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })

    return loss_score
