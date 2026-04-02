import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from wbia_miew_id.metrics import AverageMeter, compute_calibration


def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch, use_wandb=True, swa_model=None, swa_start=None, swa_scheduler=None, pgr_bank=None, pgr_weight=0.1):
    model.train()
    loss_score = AverageMeter()
    pgr_score = AverageMeter()
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

        if pgr_bank is not None:
            indices = d['index'].to(device)
            pgr_loss = pgr_bank(output, targets, indices)
            loss = loss + pgr_weight * pgr_loss
            pgr_score.update(pgr_loss.detach().item(), batch_size)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        postfix = dict(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
        if pgr_bank is not None:
            postfix['PGR'] = pgr_score.avg
        tk0.set_postfix(**postfix)

    if swa_model and epoch > swa_start:
        print("Updating swa model...")
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

    if use_wandb:
        log_dict = {
            "train loss": loss_score.avg,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        }
        if pgr_bank is not None:
            log_dict["pgr loss"] = pgr_score.avg
        wandb.log(log_dict)

    return loss_score
