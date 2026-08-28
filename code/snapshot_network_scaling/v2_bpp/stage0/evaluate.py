import time

import numpy as np
import torch

from utils_return_indivial_rates import discrete_mapping


METHODS = (
    "centralized",
    "decentralized",
    "centralized_discrete",
    "decentralized_discrete",
    "centralized_random_phase",
    "decentralized_random_phase",
    "centralized_random_phase_discrete",
    "decentralized_random_phase_discrete",
)


def validate_sample_count(name, sample_count, batch_size):
    if sample_count <= 0 or sample_count % batch_size != 0:
        raise ValueError(
            f"{name} must be a positive multiple of batch_size={batch_size}"
        )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _random_phase_like(theta, num_bits=None):
    if num_bits is None:
        phase = 2 * torch.pi * torch.rand_like(theta[..., 0])
    else:
        level = 2**num_bits
        phase = torch.randint(level, theta.shape[:-1], device=theta.device)
        phase = phase.to(theta.dtype) * (2 * torch.pi / level)
    return torch.stack((phase.cos(), phase.sin()), dim=-1)


def _check_beamformer(beamformer, association_mask, pmax_w):
    if not torch.isfinite(beamformer).all():
        raise RuntimeError("Non-finite beamformer")
    batch_size, num_users, num_ap = association_mask.shape
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        batch_size, num_ap * num_users
    )
    mask = torch.as_tensor(
        flat_mask, dtype=torch.bool, device=beamformer.device
    ).unsqueeze(1)
    if torch.count_nonzero(beamformer.masked_select(~mask)):
        raise RuntimeError("Beamformer violates the association mask")
    powers = []
    for ap in range(num_ap):
        block = beamformer[:, :, ap * num_users : (ap + 1) * num_users]
        powers.append(block.square().sum(dim=(1, 2)))
    powers = torch.stack(powers, dim=1)
    if torch.any(powers > pmax_w + 1e-6):
        raise RuntimeError("Beamformer violates the per-AP power limit")
    return powers.detach().cpu().numpy()


def _summarize(rate_arrays, details, area, elapsed, sample_count):
    metrics = {}
    for method, batches in rate_arrays.items():
        rates = np.concatenate(batches, axis=0)
        sum_rates = rates.sum(axis=1)
        metrics[method] = float(sum_rates.mean())
        details[f"{method}_per_ue_rates"] = rates
        details[f"{method}_per_sample_sum_rates"] = sum_rates
        if method in ("centralized", "decentralized"):
            metrics[f"{method}_mean_per_ue_rate"] = float(rates.mean())
            metrics[f"{method}_fifth_percentile_ue_rate"] = float(
                np.percentile(rates, 5)
            )
            metrics[f"{method}_sum_rate_per_area"] = float(
                sum_rates.mean() / area
            )
            metrics[f"{method}_seconds_per_sample"] = (
                elapsed[method] / sample_count
            )

    absolute_gap = metrics["centralized"] - metrics["decentralized"]
    metrics["absolute_gap"] = absolute_gap
    metrics["relative_gap_percent"] = (
        100 * absolute_gap / metrics["centralized"]
        if metrics["centralized"]
        else float("nan")
    )
    summary_names = (
        "serving_ap_count",
        "associated_ue_count",
        "visible_links_per_ap",
        "local_to_global_visibility_ratio",
        "nearest_ap_distance",
        "nearest_ris_distance",
        "strongest_received_power",
        "ap_pair_shared_ue",
        "connected_component_count",
        "largest_component_fraction",
        "bipartite_fully_connected",
        "visible_ap_ue_ris_feature_blocks",
        "global_ris_proposal_count_centralized",
        "global_ris_proposal_count_decentralized",
        "centralized_ap_power",
        "decentralized_ap_power",
        "ap_ris_degree",
    )
    for name in summary_names:
        values = details[name]
        if values.size:
            metrics[f"{name}_mean"] = float(values.mean())
            metrics[f"{name}_median"] = float(np.median(values))

    metrics["paired_snapshot_count"] = int(
        details["cd_pairing_verified"].sum()
    )
    metrics["learned_vs_random_centralized_positive"] = bool(
        metrics["centralized"] > metrics["centralized_random_phase"]
    )
    metrics["learned_vs_random_decentralized_positive"] = bool(
        metrics["decentralized"] > metrics["decentralized_random_phase"]
    )
    return metrics


def evaluate_snapshot(
    model,
    dataloader,
    test_sample,
    *,
    K,
    batch_size,
    associate_threshold,
    pmax_w,
    device,
    return_details=False,
):
    validate_sample_count("evaluation sample count", test_sample, batch_size)
    model.eval()
    rate_arrays = {method: [] for method in METHODS}
    elapsed = {"centralized": 0.0, "decentralized": 0.0}
    detail_names = (
        "user_locations",
        "association_mask",
        "ap_user_distances",
        "ris_user_distances",
        "serving_ap_count",
        "associated_ue_count",
        "visible_links_per_ap",
        "local_to_global_visibility_ratio",
        "nearest_ap_distance",
        "nearest_ris_distance",
        "strongest_received_power",
        "ap_pair_shared_ue",
        "connected_component_count",
        "largest_component_fraction",
        "bipartite_fully_connected",
        "visible_ap_ue_ris_feature_blocks",
        "global_ris_proposal_count_centralized",
        "global_ris_proposal_count_decentralized",
    )
    detail_batches = {name: [] for name in detail_names}
    detail_batches.update(
        {
            "centralized_ap_power": [],
            "decentralized_ap_power": [],
            "cd_pairing_verified": [],
        }
    )

    with torch.no_grad():
        for _ in range(test_sample // batch_size):
            feature, e, index, e_dir, _ = dataloader.gen_training_data(
                K, associate_threshold, associate_threshold
            )
            feature = feature.to(device)
            e = e.to(device)
            e_dir = e_dir.to(device)
            snapshot_id = dataloader.snapshot_id
            association_mask = dataloader.get_association_mask().copy()
            channel_batch = dataloader.get_channel_batch()
            topology = dataloader.get_topology_batch()

            _sync(device)
            started = time.perf_counter()
            centralized_w, centralized_theta = model(
                feature, e, index, e_dir, training=True, duplicate=False
            )
            _sync(device)
            elapsed["centralized"] += time.perf_counter() - started

            local_feature, local_e, local_index, local_e_dir = (
                dataloader.gen_testing_data(
                    K,
                    associate_threshold,
                    associate_threshold,
                    regenerate_channels=False,
                )
            )
            local_feature = [value.to(device) for value in local_feature]
            local_e = [value.to(device) for value in local_e]
            local_e_dir = [value.to(device) for value in local_e_dir]
            _sync(device)
            started = time.perf_counter()
            decentralized_w, decentralized_theta = model(
                local_feature,
                local_e,
                local_index,
                local_e_dir,
                training=False,
            )
            _sync(device)
            elapsed["decentralized"] += time.perf_counter() - started

            paired = (
                snapshot_id == dataloader.snapshot_id
                and np.array_equal(
                    association_mask, dataloader.get_association_mask()
                )
                and all(
                    np.array_equal(before, after)
                    for before, after in zip(
                        channel_batch, dataloader.get_channel_batch()
                    )
                )
            )
            if not paired:
                raise RuntimeError("C/D evaluation inputs are not paired")
            detail_batches["cd_pairing_verified"].append(
                np.ones(batch_size, dtype=bool)
            )
            detail_batches["centralized_ap_power"].append(
                _check_beamformer(centralized_w, association_mask, pmax_w)
            )
            detail_batches["decentralized_ap_power"].append(
                _check_beamformer(decentralized_w, association_mask, pmax_w)
            )

            phase_methods = {
                "centralized": (centralized_w, centralized_theta),
                "decentralized": (decentralized_w, decentralized_theta),
                "centralized_discrete": (
                    centralized_w,
                    discrete_mapping(centralized_theta, 2),
                ),
                "decentralized_discrete": (
                    decentralized_w,
                    discrete_mapping(decentralized_theta, 2),
                ),
                "centralized_random_phase": (
                    centralized_w,
                    _random_phase_like(centralized_theta),
                ),
                "decentralized_random_phase": (
                    decentralized_w,
                    _random_phase_like(decentralized_theta),
                ),
                "centralized_random_phase_discrete": (
                    centralized_w,
                    _random_phase_like(centralized_theta, 2),
                ),
                "decentralized_random_phase_discrete": (
                    decentralized_w,
                    _random_phase_like(decentralized_theta, 2),
                ),
            }
            for method, (beamformer, theta) in phase_methods.items():
                if not torch.isfinite(theta).all():
                    raise RuntimeError(f"Non-finite {method} RIS phases")
                rates = dataloader.compute_rates(
                    beamformer, theta, pmax_w, device
                )
                if not torch.isfinite(rates).all():
                    raise RuntimeError(f"Non-finite {method} rates")
                rate_arrays[method].append(rates.cpu().numpy())

            for name in detail_names:
                detail_batches[name].append(topology[name])

    details = {
        name: np.concatenate(values, axis=0)
        for name, values in detail_batches.items()
    }
    details["ap_ris_distances"] = topology["ap_ris_distances"]
    details["ap_locations"] = dataloader.BS_Loc_array.copy()
    details["ris_locations"] = dataloader.RIS_Loc_array.copy()
    details["ap_ris_degree"] = np.full(
        len(dataloader.BS_array), len(dataloader.RIS_array), dtype=int
    )
    metrics = _summarize(
        rate_arrays,
        details,
        dataloader.square_side**2,
        elapsed,
        test_sample,
    )
    if metrics["ap_ris_degree_mean"] != len(dataloader.RIS_array):
        raise RuntimeError("All-AP-all-RIS association was not preserved")
    return (metrics, details) if return_details else metrics
