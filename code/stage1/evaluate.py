import numpy as np
import torch

from utils_return_indivial_rates import mrt_beamforming, rzf_beamforming


METHODS = ("centralized_gnn", "decentralized_gnn", "mrt", "rzf")


def validate_sample_count(name, sample_count, batch_size):
    if sample_count <= 0 or sample_count % batch_size != 0:
        raise ValueError(
            f"{name} must be a positive multiple of batch_size={batch_size}"
        )


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
):
    validate_sample_count(
        "evaluation sample count", test_sample, batch_size
    )
    model.eval()
    sum_rates = {method: [] for method in METHODS}

    with torch.no_grad():
        for _ in range(test_sample // batch_size):
            centralized_feature, centralized_index = (
                dataloader.gen_training_data(
                    K,
                    associate_threshold,
                    duplicate=False,
                )
            )
            centralized_feature = centralized_feature.to(device)
            centralized_w = model(
                centralized_feature,
                centralized_index,
                training=True,
                duplicate=False,
            )
            _, centralized_rate, _ = dataloader.compute_loss(
                centralized_w, device, noise_power
            )
            sum_rates["centralized_gnn"].append(centralized_rate.item())

            decentralized_feature, decentralized_index = (
                dataloader.gen_testing_data(
                    K,
                    associate_threshold,
                    duplicate=False,
                    regenerate_channels=False,
                )
            )
            decentralized_feature = [
                feature.to(device) for feature in decentralized_feature
            ]
            decentralized_w = model(
                decentralized_feature,
                decentralized_index,
                training=False,
                duplicate=False,
            )
            _, decentralized_rate, _ = dataloader.compute_loss(
                decentralized_w, device, noise_power
            )
            sum_rates["decentralized_gnn"].append(decentralized_rate.item())

            channels = dataloader.get_stacked_channels()
            association_mask = dataloader.get_association_mask()
            mrt_w = mrt_beamforming(
                channels,
                association_mask,
                pmax_w,
                device,
            )
            _, mrt_rate, _ = dataloader.compute_loss(
                mrt_w, device, noise_power
            )
            sum_rates["mrt"].append(mrt_rate.item())

            rzf_w = rzf_beamforming(
                channels,
                association_mask,
                pmax_w,
                device,
                noise_power,
            )
            _, rzf_rate, _ = dataloader.compute_loss(
                rzf_w, device, noise_power
            )
            sum_rates["rzf"].append(rzf_rate.item())

    return {
        method: float(np.mean(values))
        for method, values in sum_rates.items()
    }
