# PairX Re-ranking for MiewID Plugin

## Overview

Add optional PairX-based re-ranking to the MiewID plugin, triggered by a JSON
flag from Wildbook. When enabled, the top-k MiewID candidates are re-scored
using PairX's intermediate-representation matching metrics, which provide
spatial correspondence signals orthogonal to cosine similarity.

**Backward compatible**: default behavior is unchanged. Re-ranking only engages
when `pairx_rerank: true` is sent in the request config.

## Background

The PairX paper proposes two discriminative metrics computed from intermediate
feature maps that separate correct from incorrect matches even when cosine
similarity is tied:

1. **Inverted Residual Mean (IRM)**: Fit a homography between matched keypoint
   coordinates; measure mean reprojection error. Low residual = geometrically
   consistent = likely correct match.
2. **Match Coverage (MC)**: Aggregate relevance at matched locations
   (`ir_0[j0][i0] * ir_1[j1][i1]`). High coverage = model's attention regions
   align = likely correct match.

These are complementary to cosine similarity (which operates on a single
embedding vector) because they exploit spatial structure in intermediate layers.

## Architecture

Follows the same pattern as the Hybrid plugin (MiewID → LightGlue → fusion),
but operates *within* the MiewID plugin itself:

```
Stage 1: MiewID cosine ranking (existing, unchanged)
         ↓ top-k candidates
Stage 2: PairX re-ranking (new, optional)
         - Forward pass through model with intermediate hooks
         - BFMatcher cross-checked feature matching
         - Compute IRM and MC scores
         ↓
Stage 3: Score fusion (new, optional)
         - fused = alpha * miewid_score + (1 - alpha) * pairx_normalized
         - Non-shortlisted candidates keep miewid_score * alpha
```

## Implementation Plan

### Phase 1: Model cache

**New module-level cache in `_plugin.py`:**

`read_config_and_load_model(species)` returns a 3-tuple
`(model, config, (model_url, config_url))` and has no built-in caching — every
call loads weights from disk. Add a module-level cache (same pattern as the
Hybrid plugin's `_MIEWID_MODEL_CACHE`):

```python
_PAIRX_MODEL_CACHE = {}  # keyed by species

def _get_pairx_model(species):
    """Get or load model for PairX re-ranking. Cached per species."""
    if species not in _PAIRX_MODEL_CACHE:
        model, mconfig, (model_url, config_url) = read_config_and_load_model(species)
        device = next(model.parameters()).device
        model.eval()
        model.device = device  # MiewIdNet has no .device property; set it manually
        _PAIRX_MODEL_CACHE[species] = (model, mconfig, device)
    return _PAIRX_MODEL_CACHE[species]
```

Note: `model.device` does not exist on `MiewIdNet` (it's a plain `nn.Module`).
The vendored `pairx_draw.py:76` works around this with `model.device = device`.
We do the same here. Device is derived from `next(model.parameters()).device`.

### Phase 2: Core PairX scoring function

**New file: `wbia_miew_id/visualization/pairx/scoring.py`**

Co-located with the existing PairX code (not a separate `rerank/` package) to
keep PairX logic together and share imports.

Import helpers from the vendored PairX code:
```python
from .core import (
    get_intermediate_feature_maps_and_embedding,
    get_feature_matches,
    get_intermediate_relevances,
)
```

1. **`pairx_rerank_score(device, img_0, img_1, model, layer_key)`**
   - Zero model gradients before each call: `model.zero_grad()`
   - Set `img_0.requires_grad_(True)` and `img_1.requires_grad_(True)`
   - Call `get_intermediate_feature_maps_and_embedding()` for both images
   - Compute cosine similarity and backprop for intermediate relevances
     (same as vendored `pairx()`, but per-pair — relevances are pair-specific
     because they come from backpropagating *this pair's* cosine similarity)
   - Run `get_feature_matches()` (BFMatcher cross-checked)
   - Compute match relevances: `ir_0[j0][i0] * ir_1[j1][i1]`
   - Compute IRM: fit homography with `cv2.findHomography(RANSAC)`,
     compute reprojection residuals, return `1.0 / (1.0 + mean_residual)`.
     Keypoint coordinates come from `get_keypoints()` which maps feature map
     grid cells to image-space via `step_w*(i+0.5)` — sufficient for
     homography estimation. If fewer than 4 matches, fall back to MC-only.
   - Compute MC: `sum(top_k_relevances) / k` (top-k by relevance)
   - Return composite score: `w_irm * irm + w_mc * mc` (default 0.5/0.5)
   - **Skip** pixel-level backpropagation (`get_pixel_relevances`) — expensive,
     only needed for visualization. Saves ~40-60% of compute.
   - Wrap in `try/except` — on any failure (CUDA OOM, too few matches, etc.),
     return `None` so the caller falls back to MiewID-only score.

2. **`normalize_pairx_score(raw_score, k, x0)`**
   - Sigmoid normalization (same pattern as Hybrid's LightGlue normalization)

**Important**: No "shared query forward pass" optimization. The intermediate
relevances are pair-specific (computed by backpropagating each pair's cosine
similarity gradient). Both images must go through the full forward + backward
pass together for each candidate pair.

### Phase 3: Thread safety

The plugin runs under Gunicorn with 16 threads. PairX is not thread-safe:
- `register_forward_hook` / `register_full_backward_hook` on a shared model
- `zennit.composites.EpsilonPlus.context(model)` modifies model layers in-place
- `backward()` writes to `.grad` attributes

**Solution: `threading.Lock` per cached model.**

```python
import threading

_PAIRX_MODEL_CACHE = {}    # keyed by species
_PAIRX_MODEL_LOCKS = {}    # keyed by species
_PAIRX_CACHE_LOCK = threading.Lock()  # protects dict mutation

def _get_pairx_model(species):
    with _PAIRX_CACHE_LOCK:
        if species not in _PAIRX_MODEL_CACHE:
            model, mconfig, _ = read_config_and_load_model(species)
            device = next(model.parameters()).device
            model.eval()
            model.device = device
            _PAIRX_MODEL_CACHE[species] = (model, mconfig, device)
            _PAIRX_MODEL_LOCKS[species] = threading.Lock()
    return _PAIRX_MODEL_CACHE[species], _PAIRX_MODEL_LOCKS[species]
```

The re-ranking function acquires the lock for the entire shortlist:

```python
(model, mconfig, device), model_lock = _get_pairx_model(species)
with model_lock:
    for daid in shortlist:
        ...  # all PairX scoring happens under the lock
```

This serializes PairX re-ranking for the same species but allows concurrent
MiewID embedding computation (which uses `torch.no_grad()` and doesn't touch
hooks or gradients). Different species use different model instances and
different locks, so they can run concurrently.

### Phase 4: Config parameters

**Modify: `wbia_miew_id/_plugin.py` — `MiewIdConfig`**

Add optional parameters (all default to values that preserve existing behavior):

```python
class MiewIdConfig(dt.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('config_path', None),
            ut.ParamInfo('use_knn', True, hideif=True),
            # PairX re-ranking (all defaults = disabled)
            ut.ParamInfo('pairx_rerank', False),        # master switch
            ut.ParamInfo('pairx_shortlist_k', 20),      # top-k to re-rank
            ut.ParamInfo('pairx_alpha', 0.8),            # fusion weight (1=pure MiewID)
            ut.ParamInfo('pairx_layer_key', 'backbone.blocks.3'),
            ut.ParamInfo('pairx_sigmoid_k', 0.1),        # sigmoid steepness
            ut.ParamInfo('pairx_sigmoid_x0', 0.5),       # sigmoid midpoint
        ]
```

These can be passed from Wildbook in the `query_config_dict`:
```json
{
  "MiewId": {
    "pairx_rerank": true,
    "pairx_shortlist_k": 20,
    "pairx_alpha": 0.7
  }
}
```

### Phase 5: Integration into scoring pipeline

**Modify: `wbia_miew_id/_plugin.py` — `wbia_plugin_miew_id()`**

After computing MiewID scores (line ~386), add conditional re-ranking:

```python
qaid_score_dict[qaid] = aid_score_dict

# Optional PairX re-ranking
if config.get('pairx_rerank', False):
    qaid_score_dict[qaid] = _pairx_rerank(
        ibs, qaid, daids, qaid_score_dict[qaid], config
    )
```

**New function: `_pairx_rerank(ibs, qaid, daids, score_dict, config)`**

```python
def _pairx_rerank(ibs, qaid, daids, score_dict, config):
    shortlist_k = config.get('pairx_shortlist_k', 20)
    alpha = config.get('pairx_alpha', 0.8)
    layer_key = config.get('pairx_layer_key', 'backbone.blocks.3')
    sigmoid_k = config.get('pairx_sigmoid_k', 0.1)
    sigmoid_x0 = config.get('pairx_sigmoid_x0', 0.5)

    # Sort by MiewID score, take top-k
    sorted_daids = sorted(daids, key=lambda d: score_dict.get(d, 0), reverse=True)
    shortlist = sorted_daids[:shortlist_k]

    # Load model from cache (with thread-safety lock)
    species = ibs.get_annot_species_texts(qaid)
    (model, mconfig, device), model_lock = _get_pairx_model(species)

    from wbia_miew_id.visualization.pairx.scoring import (
        pairx_rerank_score, normalize_pairx_score,
    )

    with model_lock:
        # Prepare query image once (reuse across candidates)
        query_img = _prepare_image(ibs, qaid, mconfig, device)

        for daid in shortlist:
            db_img = _prepare_image(ibs, daid, mconfig, device)

            # Zero gradients before each pair to prevent accumulation
            model.zero_grad()
            if query_img.grad is not None:
                query_img.grad = None
            if db_img.grad is not None:
                db_img.grad = None

            raw_pairx = pairx_rerank_score(
                device, query_img, db_img, model, layer_key
            )

            if raw_pairx is None:
                # PairX failed for this pair — keep MiewID-only score (scaled)
                score_dict[daid] = alpha * score_dict[daid]
                continue

            pairx_norm = normalize_pairx_score(raw_pairx, sigmoid_k, sigmoid_x0)
            miewid_score = score_dict[daid]
            score_dict[daid] = alpha * miewid_score + (1 - alpha) * pairx_norm

    # Scale non-shortlisted scores (same pattern as Hybrid)
    for daid in sorted_daids[shortlist_k:]:
        score_dict[daid] = alpha * score_dict.get(daid, 0)

    return score_dict
```

### Phase 6: Image preparation helper

**New function: `_prepare_image(ibs, aid, config, device)`**

Loads and transforms an annotation image into a tensor suitable for PairX.
Reuse `_load_data` with a single-element list (same function used by
`miew_id_compute_embedding`) to ensure identical preprocessing:

```python
def _prepare_image(ibs, aid, config, device):
    """Load and transform a single annotation for PairX forward pass."""
    # _load_data returns a DataLoader; extract the single image tensor
    test_loader = _load_data(ibs, [aid], config, multithread=False)
    batch = next(iter(test_loader))
    img_tensor = batch[0]  # (1, C, H, W)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device).float()
    img_tensor.requires_grad_(True)  # needed for LRP backpropagation
    return img_tensor
```

Key considerations:
- Must use the same transforms as embedding computation (handled by `_load_data`)
- Must set `requires_grad_(True)` (PairX needs gradients; embedding uses
  `torch.no_grad()`)
- The `crop_bbox` behavior is handled by `_load_data` / `PluginDataset`
  automatically based on annotation bounding boxes

### Phase 7: Backbone compatibility guard

`choose_canonizer()` in the vendored PairX code only supports ResNet and
EfficientNet. MiewID also supports ViT and Swin backbones, which will crash
with `Exception("Model type not recognized for canonizer selection")`.

Add a guard in `_pairx_rerank` before entering the scoring loop:

```python
from wbia_miew_id.visualization.pairx.core import choose_canonizer

try:
    choose_canonizer(model)
except Exception:
    print(
        'PairX re-ranking skipped: unsupported backbone type %s'
        % type(model).__name__
    )
    # Return scores unchanged (no scaling, no re-ranking)
    return score_dict
```

This fails fast before doing any work, and returns the original MiewID scores
unmodified (no alpha scaling). The check runs once per query, not per candidate.

### Phase 8: Testing

**New file: `tests/test_pairx_rerank.py`**

1. Unit test `pairx_rerank_score()` with synthetic feature maps
2. Unit test IRM computation (known homography → expected residual)
3. Unit test IRM fallback when fewer than 4 matches (MC-only)
4. Unit test score fusion (alpha=1 → pure MiewID, alpha=0 → pure PairX)
5. Unit test `pairx_rerank_score` returns `None` on exception (no crash)
6. Integration test: run MiewID with `pairx_rerank=False` vs `True`, verify
   scores differ for shortlisted candidates and are scaled for non-shortlisted
7. Backward compatibility: verify default config produces identical results to
   current behavior
8. Thread safety: two concurrent PairX re-rankings on same species don't crash

## Files Changed

| File | Change |
|------|--------|
| `wbia_miew_id/_plugin.py` | Add config params, model cache, `_pairx_rerank`, `_prepare_image` |
| `wbia_miew_id/visualization/pairx/scoring.py` | New: PairX scoring (IRM, MC, composite, normalization) |
| `tests/test_pairx_rerank.py` | New: unit + integration tests |

## Risks

1. **GPU memory**: PairX requires gradient computation (no `torch.no_grad()`),
   which increases memory usage. Mitigated by small `shortlist_k` (default 20),
   processing candidates one-at-a-time, and zeroing gradients between pairs.

2. **Model compatibility**: PairX uses `zennit` for LRP canonization. Currently
   supports ResNet and EfficientNet backbones (see `choose_canonizer()`).
   ViT/Swin backbones are guarded against — PairX re-ranking is silently
   skipped with a log message, returning unmodified MiewID scores.

3. **Vendored PairX divergence**: The MiewID repo has a vendored copy of PairX
   at `visualization/pairx/` that differs from `/mnt/c/pairx/`. The vendored
   copy handles single images; the upstream handles batches. We use the vendored
   copy for consistency. Batch support can be ported later as an optimization.

4. **Sigmoid parameters**: The default `pairx_sigmoid_k` and `pairx_sigmoid_x0`
   values are placeholders and will need tuning based on the actual distribution
   of PairX scores observed in practice.

5. **Thread serialization**: The per-species model lock serializes PairX
   re-ranking, meaning only one PairX query per species runs at a time. This is
   acceptable because PairX is already expensive (~1-2s per query) and
   contention is expected to be low. MiewID embedding computation (the common
   case) is unaffected.

6. **Gradient accumulation**: `pairx()` calls `cosine_sim.backward()` which
   writes to model parameter `.grad` attributes. Without zeroing between pairs,
   gradients accumulate and consume increasing memory. Addressed by calling
   `model.zero_grad()` and clearing input `.grad` before each pair.

## Open Questions

1. **Relative weighting of IRM vs MC**: The composite score
   `w_irm * irm + w_mc * mc` needs weights. The paper's evaluation may suggest
   which metric is more discriminative. Start with equal weights (0.5/0.5) and
   tune empirically.

2. **Homography fitting**: `cv2.findHomography` with RANSAC needs a minimum of
   4 point correspondences. If PairX produces fewer matches (rare but possible
   with very dissimilar images), fall back to MC-only scoring.

3. **Layer selection**: `backbone.blocks.3` is the default in the demo, but
   different layers may be more discriminative for different species. Making
   this configurable via `pairx_layer_key` handles this.

4. **`zennit` dependency**: Verify `zennit` is already installed in the MiewID
   plugin's environment (it's used by the vendored PairX visualization code, so
   it should be). If not, add it to `requirements.txt`.
