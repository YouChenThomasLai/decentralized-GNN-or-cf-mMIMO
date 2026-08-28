"""Causal modular association controllers for Stage 5."""

import numpy as np

from association import (
    ASSOCIATION_PERIOD_FRAMES,
    HYSTERESIS_MARGIN_DB,
    hysteresis_top_l_mask,
    top_l_mask,
)


FIXED_TOP2 = "fixed_lsf_top2_t0"
H3 = "hysteresis_lsf_top2_h3_db"
STAGE4_BOUNDARY = "stage4_fixed_threshold_t0"
TRACE_POLICIES = ("sac_current", "sac_history")
ASSOCIATION_POLICIES = (FIXED_TOP2, H3, STAGE4_BOUNDARY) + TRACE_POLICIES


def build_modular_trace(
    path_loss_factors,
    policy,
    *,
    stage4_mask=None,
    external_trace=None,
    association_period_frames=ASSOCIATION_PERIOD_FRAMES,
):
    """Return decision indices, AP-local bids, and UE-arbitrated masks."""
    path_loss = np.asarray(path_loss_factors, dtype=np.float64)
    if path_loss.ndim != 4:
        raise ValueError("Path loss must have shape [trajectory,time,AP,user]")
    if association_period_frames <= 0:
        raise ValueError("Association period must be positive")
    decision_indices = np.arange(0, path_loss.shape[1], association_period_frames)
    local_bids = np.square(path_loss[:, decision_indices]).transpose(0, 1, 3, 2)

    if policy == STAGE4_BOUNDARY:
        initial = np.asarray(stage4_mask, dtype=bool)
        if initial.shape != (path_loss.shape[0], path_loss.shape[3], path_loss.shape[2]):
            raise ValueError("Stage 4 mask must have shape [trajectory,user,AP]")
        masks = np.broadcast_to(initial[:, None], local_bids.shape).copy()
    elif policy == FIXED_TOP2:
        initial = top_l_mask(local_bids[:, 0])
        masks = np.broadcast_to(initial[:, None], local_bids.shape).copy()
    elif policy == H3:
        masks = np.empty(local_bids.shape, dtype=bool)
        masks[:, 0] = top_l_mask(local_bids[:, 0])
        for epoch in range(1, len(decision_indices)):
            masks[:, epoch] = hysteresis_top_l_mask(
                local_bids[:, epoch],
                masks[:, epoch - 1],
                HYSTERESIS_MARGIN_DB,
            )
    elif policy in TRACE_POLICIES:
        masks = np.asarray(external_trace, dtype=bool)
        if masks.shape != local_bids.shape:
            raise ValueError("External association trace shape mismatch")
    else:
        raise ValueError(f"Unknown association policy: {policy}")

    return decision_indices, local_bids, masks


def expand_trace(mask_trace, num_frames, association_period_frames=ASSOCIATION_PERIOD_FRAMES):
    """Expand decision-epoch masks to one byte-identical mask per 1 ms frame."""
    epochs = np.arange(num_frames) // association_period_frames
    if epochs[-1] >= mask_trace.shape[1]:
        raise ValueError("Association trace does not cover all frames")
    return np.asarray(mask_trace, dtype=bool)[:, epochs]
