"""Frozen top-2 and H3 association rules used by Stage 5."""

import numpy as np


ASSOCIATION_PERIOD_FRAMES = 50
TOP_L = 2
HYSTERESIS_MARGIN_DB = 3.0


def _validate_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or scores.shape[-1] < TOP_L:
        raise ValueError("Scores must end in [user, AP] with at least two APs")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Association scores must be finite and nonnegative")
    return scores


def top_l_mask(scores, top_l=TOP_L):
    """Select strongest APs, resolving ties by the lowest AP index."""
    scores = _validate_scores(scores)
    if not 0 < top_l <= scores.shape[-1]:
        raise ValueError("top_l must be between one and the AP count")
    strongest = np.argsort(-scores, axis=-1, kind="stable")[..., :top_l]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, strongest, True, axis=-1)
    return mask


def hysteresis_top_l_mask(scores, previous_mask, margin_db, top_l=TOP_L):
    """Replace the weakest incumbent only when an outsider clears the margin."""
    scores = _validate_scores(scores)
    previous_mask = np.asarray(previous_mask, dtype=bool)
    if previous_mask.shape != scores.shape:
        raise ValueError("Previous mask and scores must have identical shapes")
    if np.any(previous_mask.sum(axis=-1) != top_l):
        raise ValueError("Previous mask violates fixed-cardinality contract")
    if margin_db < 0:
        raise ValueError("Hysteresis margin cannot be negative")
    if margin_db == 0:
        return top_l_mask(scores, top_l)

    result = previous_mask.copy()
    ratio = 10 ** (margin_db / 10)
    for index in np.ndindex(scores.shape[:-1]):
        values = scores[index]
        selected = result[index]
        outsiders = np.flatnonzero(~selected)
        outsiders = outsiders[np.argsort(-values[outsiders], kind="stable")]
        for outsider in outsiders:
            incumbents = np.flatnonzero(selected)
            weakest = incumbents[
                np.lexsort((incumbents, values[incumbents]))[0]
            ]
            if values[outsider] > values[weakest] * ratio:
                selected[weakest] = False
                selected[outsider] = True
            else:
                break
    return result


def switching_metrics(mask_trace, *, num_frames, frame_period_s):
    mask_trace = np.asarray(mask_trace, dtype=bool)
    if mask_trace.ndim != 4:
        raise ValueError("Mask trace must have shape [trajectory,epoch,user,AP]")
    toggles = np.abs(np.diff(mask_trace.astype(np.int8), axis=1)).sum(axis=(2, 3))
    user_changes = np.any(
        mask_trace[:, 1:] != mask_trace[:, :-1], axis=3
    ).sum(axis=2)
    total_toggles = toggles.sum(axis=1)
    total_user_changes = user_changes.sum(axis=1)
    duration_s = num_frames * frame_period_s
    if duration_s <= 0:
        raise ValueError("Trajectory duration must be positive")
    ap_load = mask_trace.sum(axis=2)
    num_users = mask_trace.shape[2]
    return {
        "link_toggles_by_epoch": toggles,
        "serving_set_replacements_by_epoch": toggles / 2,
        "users_changed_by_epoch": user_changes,
        "link_toggles_per_trajectory": total_toggles,
        "serving_set_replacements_per_trajectory": total_toggles / 2,
        "users_changed_per_trajectory": total_user_changes,
        "link_toggles_per_ue_s": total_toggles / (num_users * duration_s),
        "serving_set_changes_per_ue_s": total_user_changes / (
            num_users * duration_s
        ),
        "ap_load_mean_per_trajectory": ap_load.mean(axis=(1, 2)),
        "ap_load_max_per_trajectory": ap_load.max(axis=(1, 2)),
        "ap_load_std_per_trajectory": ap_load.std(axis=(1, 2)),
        "zero_load_ap_fraction_per_trajectory": (ap_load == 0).mean(axis=(1, 2)),
    }
