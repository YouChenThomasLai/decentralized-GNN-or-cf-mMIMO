import numpy as np


ASSOCIATION_PERIOD_FRAMES = 50
TOP_L = 2
THRESHOLD_RATIO = 0.1
HYSTERESIS_MARGINS_DB = (0.0, 3.0, 6.0)
POLICIES = (
    "stage2_fixed_instantaneous_rssi_threshold_t0",
    "fixed_lsf_threshold_t0",
    "current_lsf_threshold",
    "fixed_lsf_top2_t0",
    "current_lsf_top2",
    "hysteresis_lsf_top2_h0_db",
    "hysteresis_lsf_top2_h3_db",
    "hysteresis_lsf_top2_h6_db",
    "current_instantaneous_rssi_top2",
    "random_top2",
)
PRIMARY_TOP2_POLICIES = (
    "fixed_lsf_top2_t0",
    "current_lsf_top2",
    "hysteresis_lsf_top2_h0_db",
    "hysteresis_lsf_top2_h3_db",
    "hysteresis_lsf_top2_h6_db",
    "current_instantaneous_rssi_top2",
    "random_top2",
)


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


def threshold_mask(scores, ratio=THRESHOLD_RATIO):
    scores = _validate_scores(scores)
    if not 0 <= ratio <= 1:
        raise ValueError("Threshold ratio must be in [0, 1]")
    return scores >= scores.max(axis=-1, keepdims=True) * ratio


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
        outsiders = outsiders[
            np.argsort(-values[outsiders], kind="stable")
        ]
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


def build_association_traces(
    path_loss_factors,
    true_channels,
    *,
    seed=0,
    association_period_frames=ASSOCIATION_PERIOD_FRAMES,
):
    """Build causal decision-epoch masks for every Stage 3A policy."""
    path_loss_factors = np.asarray(path_loss_factors, dtype=np.float64)
    true_channels = np.asarray(true_channels)
    if path_loss_factors.ndim != 4:
        raise ValueError("Path loss must have shape [trajectory,time,AP,user]")
    if true_channels.ndim != 5 or true_channels.shape[:-1] != path_loss_factors.shape:
        raise ValueError("Channels must have shape [trajectory,time,AP,user,antenna]")
    if association_period_frames <= 0:
        raise ValueError("Association period must be positive")

    decision_indices = np.arange(
        0, path_loss_factors.shape[1], association_period_frames
    )
    lsf = np.square(path_loss_factors[:, decision_indices]).transpose(0, 1, 3, 2)
    rssi = np.sum(
        np.abs(true_channels[:, decision_indices]) ** 2, axis=-1
    ).transpose(0, 1, 3, 2)
    initial_top2 = top_l_mask(lsf[:, 0])

    def fixed(mask):
        return np.broadcast_to(mask[:, None], lsf.shape).copy()

    traces = {
        "stage2_fixed_instantaneous_rssi_threshold_t0": fixed(
            threshold_mask(rssi[:, 0])
        ),
        "fixed_lsf_threshold_t0": fixed(threshold_mask(lsf[:, 0])),
        "current_lsf_threshold": threshold_mask(lsf),
        "fixed_lsf_top2_t0": fixed(initial_top2),
        "current_lsf_top2": top_l_mask(lsf),
    }
    for margin_db in HYSTERESIS_MARGINS_DB:
        name = f"hysteresis_lsf_top2_h{margin_db:g}_db"
        trace = np.empty(lsf.shape, dtype=bool)
        trace[:, 0] = initial_top2
        for epoch in range(1, len(decision_indices)):
            trace[:, epoch] = hysteresis_top_l_mask(
                lsf[:, epoch], trace[:, epoch - 1], margin_db
            )
        traces[name] = trace

    rssi_trace = top_l_mask(rssi)
    rssi_trace[:, 0] = initial_top2
    traces["current_instantaneous_rssi_top2"] = rssi_trace

    random_rng = np.random.default_rng(np.random.SeedSequence((seed, 303)))
    random_trace = top_l_mask(random_rng.random(lsf.shape))
    random_trace[:, 0] = initial_top2
    traces["random_top2"] = random_trace

    if tuple(traces) != POLICIES:
        raise RuntimeError("Stage 3A policy order changed")
    return decision_indices, traces


def switching_metrics(mask_trace, *, num_frames, frame_period_s):
    mask_trace = np.asarray(mask_trace, dtype=bool)
    if mask_trace.ndim != 4:
        raise ValueError("Mask trace must have shape [trajectory,epoch,user,AP]")
    toggles = np.abs(np.diff(mask_trace.astype(np.int8), axis=1)).sum(axis=(2, 3))
    user_changes = np.any(mask_trace[:, 1:] != mask_trace[:, :-1], axis=3).sum(axis=2)
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
