import numpy as np
import torch

from utils_return_indivial_rates import (
    calculate_rates,
    mrt_beamforming,
    rzf_beamforming,
)


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


def load_frozen_model(model, checkpoint_path, device):
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    return model


class TrajectoryEvaluator:
    """Evaluate the four Stage 1 beamformers on identical current-CSI frames."""

    def __init__(
        self,
        model,
        *,
        K,
        pmax_w,
        num_ap,
        device,
        noise_power,
        frame_batch_size=32,
        time_stride=10,
    ):
        if frame_batch_size <= 0 or time_stride <= 0:
            raise ValueError("Frame batch size and time stride must be positive")
        self.model = model
        self.K = K
        self.pmax_w = pmax_w
        self.num_ap = num_ap
        self.device = device
        self.noise_power = noise_power
        self.frame_batch_size = frame_batch_size
        self.time_stride = time_stride

    def _beamformers(self, loader, trajectory_indices, time_indices):
        central, central_mask, local, local_masks = loader.get_frames(
            trajectory_indices, time_indices
        )
        channels = loader.get_stacked_channels(
            trajectory_indices, time_indices
        )
        association_mask = loader.association_mask[trajectory_indices]
        central = central.to(self.device)
        local = [feature.to(self.device) for feature in local]
        return channels, association_mask, {
            "centralized_gnn": self.model(
                central, central_mask, training=True, duplicate=False
            ),
            "decentralized_gnn": self.model(
                local, local_masks, training=False, duplicate=False
            ),
            "mrt": mrt_beamforming(
                channels, association_mask, self.pmax_w, self.device
            ),
            "rzf": rzf_beamforming(
                channels,
                association_mask,
                self.pmax_w,
                self.device,
                self.noise_power,
            ),
        }

    def _constraint_checks(self, weights, association_mask):
        finite = bool(torch.isfinite(weights).all())
        flat_mask = association_mask.transpose(0, 2, 1).reshape(
            len(association_mask), self.num_ap * self.K
        )
        mask = torch.as_tensor(
            flat_mask, dtype=torch.bool, device=weights.device
        ).unsqueeze(1).expand_as(weights)
        unassociated = weights[~mask]
        max_unassociated = (
            float(unassociated.abs().max().item())
            if unassociated.numel()
            else 0.0
        )
        max_power = 0.0
        for ap in range(self.num_ap):
            start = ap * self.K
            stop = (ap + 1) * self.K
            power = weights[:, :, start:stop].square().sum(dim=(1, 2))
            max_power = max(max_power, float(power.max().item()))
        return finite, max_unassociated, max_power

    def evaluate(self, loader):
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Stage 2 requires a fully frozen model")
        self.model.eval()
        time_indices = np.arange(0, loader.episode_steps, self.time_stride)
        trajectory_indices = np.repeat(
            np.arange(loader.batch_size), len(time_indices)
        )
        frame_indices = np.tile(time_indices, loader.batch_size)
        user_rate_sums = {
            method: np.zeros((loader.batch_size, self.K))
            for method in METHODS
        }
        frame_counts = np.zeros(loader.batch_size, dtype=np.int64)
        checks = {
            method: {
                "finite": True,
                "max_unassociated_abs": 0.0,
                "max_ap_power_w": 0.0,
            }
            for method in METHODS
        }

        with torch.inference_mode():
            for start in range(0, len(trajectory_indices), self.frame_batch_size):
                stop = min(start + self.frame_batch_size, len(trajectory_indices))
                trajectory_batch = trajectory_indices[start:stop]
                time_batch = frame_indices[start:stop]
                channels, association_mask, beamformers = self._beamformers(
                    loader, trajectory_batch, time_batch
                )
                np.add.at(frame_counts, trajectory_batch, 1)
                for method, weights in beamformers.items():
                    finite, max_unassociated, max_power = self._constraint_checks(
                        weights, association_mask
                    )
                    checks[method]["finite"] &= finite
                    checks[method]["max_unassociated_abs"] = max(
                        checks[method]["max_unassociated_abs"],
                        max_unassociated,
                    )
                    checks[method]["max_ap_power_w"] = max(
                        checks[method]["max_ap_power_w"], max_power
                    )
                    rates = calculate_rates(
                        weights,
                        channels,
                        self.num_ap,
                        self.device,
                        self.noise_power,
                    ).cpu().numpy()
                    if not np.isfinite(rates).all():
                        checks[method]["finite"] = False
                    np.add.at(user_rate_sums[method], trajectory_batch, rates)

        summary = {}
        raw = {
            "evaluation_time_indices": time_indices,
            "evaluation_trajectory_indices": np.arange(loader.batch_size),
        }
        for method in METHODS:
            user_means = user_rate_sums[method] / frame_counts[:, None]
            trajectory_sum_rates = user_means.sum(axis=1)
            trajectory_p05 = np.percentile(user_means, 5, axis=1)
            summary[method] = float(trajectory_sum_rates.mean())
            summary[f"{method}_p05_user_rate"] = float(
                trajectory_p05.mean()
            )
            raw[f"{method}_per_trajectory_sum_rate"] = trajectory_sum_rates
            raw[f"{method}_per_trajectory_user_time_average_rates"] = user_means
            raw[f"{method}_per_trajectory_p05_user_rate"] = trajectory_p05
            raw[f"{method}_finite"] = np.asarray(checks[method]["finite"])
            raw[f"{method}_max_unassociated_abs"] = np.asarray(
                checks[method]["max_unassociated_abs"]
            )
            raw[f"{method}_max_ap_power_w"] = np.asarray(
                checks[method]["max_ap_power_w"]
            )
            raw[f"{method}_constraints_passed"] = np.asarray(
                checks[method]["finite"]
                and checks[method]["max_unassociated_abs"] == 0
                and checks[method]["max_ap_power_w"] <= self.pmax_w + 1e-6
            )
        raw["shared_current_channel"] = np.asarray(True)
        raw["shared_fixed_association_mask"] = np.asarray(True)
        raw["shared_noise_power"] = np.asarray(self.noise_power)
        raw["shared_pmax_w"] = np.asarray(self.pmax_w)
        return summary, raw
