import torch
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import wandb

from wbia_miew_id.metrics import AverageMeter, compute_distance_matrix, compute_calibration, eval_onevsall, topk_average_precision, precision_at_k, get_accuracy
from torch.cuda.amp import autocast  

def extract_embeddings(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                batch_embeddings = model.extract_feat(batch["image"].to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels

def extract_logits(data_loader, model, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    logits = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            batch_logits = model.extract_logits(batch["image"].to(device), batch["label"].to(device)).detach().cpu()

            image_idx = batch["image_idx"].tolist()
            batch_logits_df = pd.DataFrame(batch_logits.numpy(), index=image_idx)
            logits.append(batch_logits_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
            
    logits = pd.concat(logits)
    logits = logits.values

    assert not np.isnan(logits).sum(), "NaNs found in extracted logits"

    return logits, labels

def calculate_matches(embeddings, labels, embeddings_db=None, labels_db=None, dist_metric='cosine', ranks=list(range(1, 21)), mask_matrix=None):

    q_pids = np.array(labels)
    
    qf = torch.Tensor(embeddings)
    if embeddings_db is not None:
        print('embeddings_db not note')
        dbf = torch.Tensor(embeddings_db)
        labels_db = np.array(labels_db)
    else:
        dbf = qf
        labels_db = np.array(labels)
        mask_matrix_diagonal = np.full((embeddings.shape[0], embeddings.shape[0]), False)
        np.fill_diagonal(mask_matrix_diagonal, True)
        if mask_matrix is not None:
            mask_matrix = np.logical_or(mask_matrix_diagonal, mask_matrix)
        else:
            mask_matrix = mask_matrix_diagonal

    distmat = compute_distance_matrix(qf, dbf, dist_metric)

    distmat = distmat.numpy()

    if mask_matrix is not None:
        assert mask_matrix.shape == distmat.shape, "Mask matrix must have same shape as distance matrix"
        distmat[mask_matrix] = np.inf

    print("Computing CMC and mAP ...")

    mAP = topk_average_precision(q_pids, distmat, names_db=labels_db, k=None)
    cmc, match_mat, topk_idx, topk_names = precision_at_k(q_pids, distmat, names_db=labels_db, ranks=ranks, return_matches=True)
    print(f"Computed rank metrics on {match_mat.shape[0]} examples")

    return mAP, cmc, (embeddings, q_pids, distmat)

def calculate_calibration(logits, labels, logits_db=None, labels_db=None):

    q_pids = np.array(labels)
    confidences = torch.softmax(torch.Tensor(logits), dim=1)
    top_confidences, pred_labels = confidences.max(dim=1)
    pred_labels = pred_labels.numpy()
    top_confidences = top_confidences.numpy()

    print("Computing ECE ...")
    results = compute_calibration(q_pids, pred_labels, top_confidences, num_bins=10)
    ece = results['expected_calibration_error']
    print(f"Computed ECE on {pred_labels.shape[0]} examples")

    return ece, (logits, q_pids, top_confidences, pred_labels)

def log_results(mAP, cmc, tag='Avg', use_wandb=True):
    ranks=[1, 5, 10, 20]
    print(f"** {tag} Results **")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
        if use_wandb: wandb.log({f"{tag} - Rank-{r:<3}": cmc[r - 1]})
    
    if use_wandb: wandb.log({f"{tag} - mAP": mAP})

def eval_fn(data_loader, model, device, use_wandb=True, return_outputs=False):

    embeddings, labels = extract_embeddings(data_loader, model, device)
    mAP, cmc, (embeddings, q_pids, distmat) = calculate_matches(embeddings, labels)


    log_results(mAP, cmc, use_wandb=use_wandb)

    if return_outputs:
        return mAP, cmc, (embeddings, q_pids, distmat)
    else:
        return mAP, cmc