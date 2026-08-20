import time

import numpy as np
import torch

from utils_return_indivial_rates import mrt_beamforming, rzf_beamforming


METHODS = ("centralized_gnn", "decentralized_gnn", "mrt", "rzf")


def validate_sample_count(name, sample_count, batch_size):
    if sample_count <= 0 or sample_count % batch_size != 0:
        raise ValueError(
            f"{name} must be a positive multiple of batch_size={batch_size}"
        )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_beamformer(beamformer, association_mask, pmax_w):
    if not torch.isfinite(beamformer).all():
        raise RuntimeError("Non-finite beamformer")
    batch_size, _, virtual_users = beamformer.shape
    num_ap = association_mask.shape[2]
    num_users = association_mask.shape[1]
    expected_shape = (batch_size, num_ap * num_users)
    if virtual_users != expected_shape[1]:
        raise RuntimeError("Beamformer and association dimensions do not match")
    flat_mask = association_mask.transpose(0, 2, 1).reshape(expected_shape)
    mask = torch.as_tensor(
        flat_mask, dtype=torch.bool, device=beamformer.device
    ).unsqueeze(1)
    if torch.count_nonzero(beamformer.masked_select(~mask)):
        raise RuntimeError("Beamformer violates the association mask")
    for ap in range(num_ap):
        start = ap * num_users
        stop = (ap + 1) * num_users
        power = beamformer[:, :, start:stop].square().sum(dim=(1, 2))
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
        if method in ("centralized_gnn", "decentralized_gnn"):
            metrics[f"{method}_mean_per_ue_rate"] = float(rates.mean())
            metrics[f"{method}_fifth_percentile_ue_rate"] = float(
                np.percentile(rates, 5)
            )
            metrics[f"{method}_sum_rate_per_area"] = float(
                sum_rates.mean() / area
            )
        details[f"{method}_per_ue_rates"] = rates
        details[f"{method}_per_sample_sum_rates"] = sum_rates
        metrics[f"{method}_seconds_per_sample"] = elapsed[method] / sample_count

    absolute_gap = metrics["centralized_gnn"] - metrics["decentralized_gnn"]
    metrics["absolute_gap"] = absolute_gap
    metrics["relative_gap_percent"] = (
        100 * absolute_gap / metrics["centralized_gnn"]
        if metrics["centralized_gnn"]
        else float("nan")
    )
    for name in (
        "serving_ap_count",
        "associated_ue_count",
        "local_to_global_visibility_ratio",
        "nearest_ap_distance",
        "strongest_received_power",
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
    num_of_AP,
    device,
    noise_power,
    return_details=False,
):
    validate_sample_count("evaluation sample count", test_sample, batch_size)
    if num_of_AP != len(dataloader.BS_array):
        raise ValueError("num_of_AP does not match the dataloader topology")
    model.eval()
    rate_arrays = {method: [] for method in METHODS}
    elapsed = {method: 0.0 for method in METHODS}
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
            centralized_feature, centralized_index = (
                dataloader.gen_training_data(K, associate_threshold)
            )
            centralized_feature = centralized_feature.to(device)
            channels = dataloader.get_stacked_channels().copy()
            association_mask = dataloader.get_association_mask().copy()
            topology = dataloader.get_topology_batch()

            _sync(device)
            started = time.perf_counter()
            centralized_w = model(
                centralized_feature,
                centralized_index,
                training=True,
                duplicate=False,
            )
            _sync(device)
            elapsed["centralized_gnn"] += time.perf_counter() - started

            decentralized_feature, decentralized_index = (
                dataloader.gen_testing_data(
                    K,
                    associate_threshold,
                    regenerate_channels=False,
                )
            )
            decentralized_feature = [
                feature.to(device) for feature in decentralized_feature
            ]
            _sync(device)
            started = time.perf_counter()
            decentralized_w = model(
                decentralized_feature,
                decentralized_index,
                training=False,
                duplicate=False,
            )
            _sync(device)
            elapsed["decentralized_gnn"] += time.perf_counter() - started

            if not np.array_equal(channels, dataloader.get_stacked_channels()):
                raise RuntimeError("C/D evaluation channels are not paired")
            if not np.array_equal(
                association_mask, dataloader.get_association_mask()
            ):
                raise RuntimeError("C/D evaluation masks are not paired")

            started = time.perf_counter()
            mrt_w = mrt_beamforming(
                channels, association_mask, pmax_w, device
            )
            _sync(device)
            elapsed["mrt"] += time.perf_counter() - started

            started = time.perf_counter()
            rzf_w = rzf_beamforming(
                channels,
                association_mask,
                pmax_w,
                device,
                noise_power,
            )
            _sync(device)
            elapsed["rzf"] += time.perf_counter() - started

            beamformers = {
                "centralized_gnn": centralized_w,
                "decentralized_gnn": decentralized_w,
                "mrt": mrt_w,
                "rzf": rzf_w,
            }
            for method, beamformer in beamformers.items():
                _check_beamformer(beamformer, association_mask, pmax_w)
                rates = dataloader.compute_rates(
                    beamformer, device, noise_power
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
    metrics = _summarize(
        rate_arrays,
        details,
        dataloader.square_side**2,
        elapsed,
        test_sample,
    )
    return (metrics, details) if return_details else metrics
