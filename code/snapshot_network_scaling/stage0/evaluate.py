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
    batch_size = association_mask.shape[0]
    num_users = association_mask.shape[1]
    num_ap = association_mask.shape[2]
    flat_mask = association_mask.transpose(0, 2, 1).reshape(
        batch_size, num_ap * num_users
    )
    mask = torch.as_tensor(
        flat_mask, dtype=torch.bool, device=beamformer.device
    ).unsqueeze(1)
    if torch.count_nonzero(beamformer.masked_select(~mask)):
        raise RuntimeError("Beamformer violates the association mask")
    for ap in range(num_ap):
        block = beamformer[:, :, ap * num_users : (ap + 1) * num_users]
        power = block.square().sum(dim=(1, 2))
        if torch.any(power > pmax_w + 1e-6):
            raise RuntimeError("Beamformer violates the per-AP power limit")


def _visibility_ratio(association_mask):
    serving_ap_count = association_mask.sum(axis=2)
    visible_links = (
        association_mask.transpose(0, 2, 1) * serving_ap_count[:, None, :]
    ).sum(axis=2)
    global_links = association_mask.sum(axis=(1, 2))
    return visible_links / np.maximum(global_links[:, None], 1)


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
    for name in (
        "serving_ap_count",
        "associated_ue_count",
        "local_to_global_visibility_ratio",
        "nearest_ap_distance",
        "strongest_received_power",
        "ap_ris_degree",
    ):
        values = details[name]
        metrics[f"{name}_mean"] = float(values.mean())
        metrics[f"{name}_median"] = float(np.median(values))
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
    detail_batches = {
        "user_locations": [],
        "association_mask": [],
        "serving_ap_count": [],
        "associated_ue_count": [],
        "local_to_global_visibility_ratio": [],
        "nearest_ap_distance": [],
        "strongest_received_power": [],
    }

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

            if snapshot_id != dataloader.snapshot_id:
                raise RuntimeError("C/D evaluation channels are not paired")
            if not np.array_equal(
                association_mask, dataloader.get_association_mask()
            ):
                raise RuntimeError("C/D evaluation masks are not paired")
            _check_beamformer(centralized_w, association_mask, pmax_w)
            _check_beamformer(decentralized_w, association_mask, pmax_w)

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

            detail_batches["user_locations"].append(
                topology["user_locations"]
            )
            detail_batches["association_mask"].append(association_mask)
            detail_batches["serving_ap_count"].append(
                association_mask.sum(axis=2)
            )
            detail_batches["associated_ue_count"].append(
                association_mask.sum(axis=1)
            )
            detail_batches["local_to_global_visibility_ratio"].append(
                _visibility_ratio(association_mask)
            )
            detail_batches["nearest_ap_distance"].append(
                topology["nearest_ap_distance"]
            )
            detail_batches["strongest_received_power"].append(
                topology["strongest_received_power"]
            )

    details = {
        name: np.concatenate(values, axis=0)
        for name, values in detail_batches.items()
    }
    details["ap_ris_degree"] = np.full(
        len(dataloader.BS_array), len(dataloader.RIS_array), dtype=int
    )
    metrics = _summarize(
        rate_arrays,
        details,
        np.pi * dataloader.length**2,
        elapsed,
        test_sample,
    )
    if metrics["ap_ris_degree_mean"] != len(dataloader.RIS_array):
        raise RuntimeError("All-AP-all-RIS association was not preserved")
    return (metrics, details) if return_details else metrics
