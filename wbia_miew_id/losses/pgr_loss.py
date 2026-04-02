import torch
import torch.nn as nn
import torch.nn.functional as F


class PGRMemoryBank(nn.Module):
    """Proximity-Guided Regularization memory bank.

    Maintains a momentum-updated memory bank of L2-normalized embeddings
    for all training samples. Mines hard positives/negatives beyond the
    mini-batch to compute a proximity-weighted contrastive loss.

    Reference: Sun et al., "Part-Aware Cross-Integration Transformer with
    Proximity Guided Regularization for Domain Generalizable Animal
    Re-Identification", Expert Systems with Applications, 2026.
    """

    def __init__(self, num_samples, embedding_dim, num_classes, momentum=0.2, temperature=0.07):
        super().__init__()
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = temperature

        # Memory bank: L2-normalized embeddings for every training sample
        self.register_buffer('memory', torch.randn(num_samples, embedding_dim))
        self.memory = F.normalize(self.memory, dim=1)

        # Label lookup: class label for every training sample
        self.register_buffer('labels', torch.zeros(num_samples, dtype=torch.long))

        self._initialized = False

    def init_labels(self, all_labels):
        """Set labels for all training samples (call once before training)."""
        self.labels.copy_(torch.as_tensor(all_labels, dtype=torch.long))
        self._initialized = True

    @torch.no_grad()
    def update(self, indices, embeddings):
        """Momentum-update memory bank entries for the given batch indices."""
        embeddings = F.normalize(embeddings.detach(), dim=1)
        self.memory[indices] = (
            self.momentum * self.memory[indices]
            + (1.0 - self.momentum) * embeddings
        )
        self.memory[indices] = F.normalize(self.memory[indices], dim=1)

    def forward(self, embeddings, labels, indices):
        """Compute proximity-guided contrastive loss.

        For each sample in the batch, use the full memory bank to find
        positives (same identity) and negatives (different identity),
        then compute an InfoNCE-style loss weighted by proximity.

        Args:
            embeddings: [B, D] batch embeddings (pre-normalization ok)
            labels: [B] integer class labels
            indices: [B] dataset indices for memory bank update

        Returns:
            Scalar loss tensor.
        """
        embeddings = F.normalize(embeddings, dim=1)
        batch_size = embeddings.size(0)

        # Snapshot memory so in-place update doesn't break autograd
        memory = self.memory.detach().clone()

        # Similarity of batch embeddings against entire memory bank: [B, N]
        sim = torch.mm(embeddings, memory.t()) / self.temperature

        # Mask: which memory entries share the same label as each batch sample
        # labels: [B], self.labels: [N] -> mask: [B, N]
        pos_mask = labels.unsqueeze(1) == self.labels.unsqueeze(0)  # [B, N]

        # Exclude self from positives (the sample's own memory slot)
        self_mask = torch.zeros_like(pos_mask)
        self_mask[torch.arange(batch_size, device=indices.device), indices] = True
        pos_mask = pos_mask & ~self_mask

        # Check each sample has at least one positive; skip those that don't
        has_pos = pos_mask.any(dim=1)  # [B]
        if not has_pos.any():
            self.update(indices, embeddings)
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Proximity-weighted positive selection:
        # Weight positives by softmax of similarity (harder positives get more weight)
        # This is the "proximity-guided" aspect - distance-aware weighting
        neg_mask = ~pos_mask & ~self_mask  # [B, N]

        # For numerical stability, subtract max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Compute log-sum-exp over negatives for each sample
        neg_sim = sim.clone()
        neg_sim[~neg_mask] = float('-inf')
        log_sum_exp_neg = torch.logsumexp(neg_sim, dim=1)  # [B]

        # Compute mean of log(softmax) over positives: -log(exp(s_p) / (exp(s_p) + sum_neg))
        # = -(s_p - log(exp(s_p) + sum_neg))
        pos_sim = sim.clone()
        pos_sim[~pos_mask] = float('-inf')

        # For each sample, average over its positives
        # Use logsumexp trick: log(mean(exp(pos))) = logsumexp(pos) - log(num_pos)
        num_pos = pos_mask.float().sum(dim=1).clamp(min=1)  # [B]
        log_sum_exp_pos = torch.logsumexp(pos_sim, dim=1)  # [B]

        # Denominator: log(sum_all_non_self) = log(exp(lse_pos) + exp(lse_neg))
        log_denom = torch.logsumexp(
            torch.stack([log_sum_exp_pos, log_sum_exp_neg], dim=1), dim=1
        )  # [B]

        # Loss per sample: -1/|P| * sum_{p in P} log(exp(s_p) / denom)
        # = -1/|P| * sum_{p in P} (s_p - log_denom)
        # Approximate using: -(logsumexp(pos) - log(|P|) - log_denom)
        # = -(log_mean_exp_pos - log_denom)
        loss_per_sample = -(log_sum_exp_pos - torch.log(num_pos) - log_denom)

        # Only count samples that have positives
        loss = loss_per_sample[has_pos].mean()

        # Update memory bank
        self.update(indices, embeddings)

        return loss
