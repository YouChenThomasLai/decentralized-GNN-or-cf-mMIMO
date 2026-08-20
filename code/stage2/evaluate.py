import json
import os

import numpy as np
import torch

from data import MOBILITY_HOTSPOT, MOBILITY_PHASE_LABELS
from utils_return_indivial_rates import (
    calculate_rates,
    mrt_beamforming,
    rzf_beamforming,
)


METHODS = ("centralized_gnn", "decentralized_gnn", "mrt", "rzf")
SUMMARY_METRICS = tuple(
    metric
    for method in METHODS
    for metric in (method, f"{method}_p05_user_rate")
)


class TrajectoryEvaluator:
    def __init__(
        self,
        model,
        K,
        pmax_w,
        num_of_AP,
        device,
        noise_power,
        episode_steps,
        eval_frame_batch_size,
        eval_time_stride,
        associate_threshold,
        bootstrap_samples,
    ):
        self.model = model
        self.K = K
        self.pmax_w = pmax_w
        self.num_of_AP = num_of_AP
        self.device = device
        self.noise_power = noise_power
        self.episode_steps = episode_steps
        self.eval_frame_batch_size = eval_frame_batch_size
        self.eval_time_stride = eval_time_stride
        self.associate_threshold = associate_threshold
        self.bootstrap_samples = bootstrap_samples

    def evaluate_checkpoint(
        self, loader, run_id, out_dir, checkpoint_path, config
    ):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"[INFO] Loaded frozen checkpoint: {checkpoint_path}")
        print("Running checkpoint-only FINAL trajectory evaluation...")
        final_metrics, raw_metrics = self.evaluate(loader)
        for metric, value in final_metrics.items():
            print(f"[Final Eval] {metric} = {value:.8g}")
        return self.save_final_evaluation(
            loader, run_id, out_dir, final_metrics, raw_metrics
        )

    def save_final_evaluation(
        self, loader, run_id, out_dir, final_metrics, raw_metrics
    ):
        final_metrics["centralized_minus_decentralized"] = (
            final_metrics["centralized_gnn"]
            - final_metrics["decentralized_gnn"]
        )
        final_dir = os.path.join(out_dir, "final_eval")
        os.makedirs(final_dir, exist_ok=True)
        np.save(
            os.path.join(final_dir, f"final_eval_run{run_id}.npy"),
            final_metrics,
        )
        with open(
            os.path.join(final_dir, f"final_eval_run{run_id}.txt"), "w"
        ) as final_file:
            for key, value in final_metrics.items():
                final_file.write(f"{key}: {value:.8g}\n")
        np.savez(
            os.path.join(final_dir, f"raw_metrics_run{run_id}.npz"),
            **raw_metrics,
        )
        self._save_environment_artifacts(loader, final_dir, run_id)
        return final_metrics

    def _beamformers(
        self,
        channels,
        association_mask,
        centralized_feature,
        centralized_index,
        decentralized_feature,
        decentralized_index,
    ):
        return {
            "centralized_gnn": self.model(
                centralized_feature,
                centralized_index,
                training=True,
                duplicate=False,
            ),
            "decentralized_gnn": self.model(
                decentralized_feature,
                decentralized_index,
                training=False,
                duplicate=False,
            ),
            "mrt": mrt_beamforming(
                channels,
                association_mask,
                self.pmax_w,
                self.device,
            ),
            "rzf": rzf_beamforming(
                channels,
                association_mask,
                self.pmax_w,
                self.device,
                self.noise_power,
            ),
        }

    def evaluate(self, loader):
        self.model.eval()
        trajectory_count = loader.batch_size
        evaluation_time_indices = np.arange(
            0, self.episode_steps, self.eval_time_stride
        )
        trajectory_indices = np.repeat(
            np.arange(trajectory_count), evaluation_time_indices.size
        )
        time_indices = np.tile(evaluation_time_indices, trajectory_count)
        frame_count = trajectory_indices.size
        period_user_rates = {
            method: np.empty((frame_count, self.K), dtype=np.float64)
            for method in METHODS
        }
        dynamic_period_user_rates = {
            method: np.empty((frame_count, self.K), dtype=np.float64)
            for method in METHODS
        }

        with torch.no_grad():
            for start in range(0, frame_count, self.eval_frame_batch_size):
                stop = min(start + self.eval_frame_batch_size, frame_count)
                trajectory_batch = trajectory_indices[start:stop]
                time_batch = time_indices[start:stop]
                (
                    centralized_feature,
                    centralized_index,
                    decentralized_feature,
                    decentralized_index,
                ) = loader.get_frames(trajectory_batch, time_batch)
                centralized_feature = centralized_feature.to(self.device)
                decentralized_feature = [
                    feature.to(self.device)
                    for feature in decentralized_feature
                ]
                channels = loader.get_stacked_channels(
                    trajectory_batch, time_batch
                )
                association_mask = loader.association_mask[trajectory_batch]

                beamformers = self._beamformers(
                    channels,
                    association_mask,
                    centralized_feature,
                    centralized_index,
                    decentralized_feature,
                    decentralized_index,
                )
                for method, weights in beamformers.items():
                    rates = calculate_rates(
                        weights,
                        channels,
                        self.num_of_AP,
                        self.device,
                        self.noise_power,
                    )
                    period_user_rates[method][start:stop] = (
                        rates.detach().cpu().numpy()
                    )

                dynamic_mask = loader.get_current_association_mask(
                    trajectory_batch,
                    time_batch,
                    self.associate_threshold,
                )
                (
                    dynamic_centralized_feature,
                    dynamic_centralized_index,
                    dynamic_decentralized_feature,
                    dynamic_decentralized_index,
                ) = loader.get_frames(
                    trajectory_batch,
                    time_batch,
                    dynamic_mask,
                )
                dynamic_beamformers = self._beamformers(
                    channels,
                    dynamic_mask,
                    dynamic_centralized_feature.to(self.device),
                    dynamic_centralized_index,
                    [
                        feature.to(self.device)
                        for feature in dynamic_decentralized_feature
                    ],
                    dynamic_decentralized_index,
                )
                for method, weights in dynamic_beamformers.items():
                    rates = calculate_rates(
                        weights,
                        channels,
                        self.num_of_AP,
                        self.device,
                        self.noise_power,
                    )
                    dynamic_period_user_rates[method][start:stop] = (
                        rates.detach().cpu().numpy()
                    )

        metrics = {}
        raw_metrics = {}
        for method, rates in period_user_rates.items():
            rates = rates.reshape(
                trajectory_count, evaluation_time_indices.size, self.K
            )
            per_trajectory_sum_rate = rates.sum(axis=2).mean(axis=1)
            per_trajectory_user_rates = rates.mean(axis=1)
            per_trajectory_p05 = np.percentile(
                per_trajectory_user_rates, 5, axis=1
            )
            metrics[method] = float(per_trajectory_sum_rate.mean())
            metrics[f"{method}_p05_user_rate"] = float(
                per_trajectory_p05.mean()
            )
            raw_metrics[f"{method}_period_user_rates"] = rates
            raw_metrics[f"{method}_per_trajectory_sum_rate"] = (
                per_trajectory_sum_rate
            )
            raw_metrics[f"{method}_per_trajectory_p05_user_rate"] = (
                per_trajectory_p05
            )
            dynamic_rates = dynamic_period_user_rates[method].reshape(
                trajectory_count,
                evaluation_time_indices.size,
                self.K,
            )
            dynamic_sum_rate = dynamic_rates.sum(axis=2).mean(axis=1)
            regret = dynamic_sum_rate - per_trajectory_sum_rate
            metrics[f"{method}_fixed_association_regret"] = float(
                regret.mean()
            )
            raw_metrics[f"{method}_dynamic_period_user_rates"] = (
                dynamic_rates
            )
            raw_metrics[
                f"{method}_per_trajectory_fixed_association_regret"
            ] = regret
        raw_metrics["evaluation_time_indices"] = evaluation_time_indices
        return metrics, raw_metrics

    def _save_environment_artifacts(self, loader, final_dir, run_id):
        rssi = np.sum(np.abs(loader.true_channels) ** 2, axis=-1)
        strongest_ap = np.argmax(rssi, axis=2)
        expanded_fixed_mask = np.broadcast_to(
            loader.association_mask[:, None],
            (*strongest_ap.shape, len(loader.BS_array)),
        )
        strongest_ap_covered = np.take_along_axis(
            expanded_fixed_mask, strongest_ap[..., None], axis=-1
        )[..., 0]
        environment = {
            "mobility_model": np.asarray(loader.mobility_model),
            "ue_positions": loader.ue_positions,
            "ue_speeds_mps": loader.ue_speeds_mps,
            "ue_directions_rad": loader.ue_directions_rad,
            "instantaneous_speeds_mps": loader.instantaneous_speeds_mps,
            "distances": loader.distances,
            "path_loss_factors": loader.path_loss_factors,
            "association_mask": loader.association_mask,
            "strongest_ap_change_count": np.sum(
                strongest_ap[:, 1:] != strongest_ap[:, :-1], axis=1
            ),
            "fixed_serving_set_coverage": strongest_ap_covered.mean(axis=1),
            "rhos": loader.rhos,
            "true_stored_max_abs_error": np.max(
                np.abs(loader.true_channels - loader.stored_channels)
            ),
        }
        if loader.mobility_model == MOBILITY_HOTSPOT:
            environment.update(
                {
                    "hotspot_centers": loader.hotspot_centers,
                    "hotspot_radius_m": loader.hotspot_radius_m,
                    "hotspot_state": loader.hotspot_state,
                    "mobility_phase": loader.mobility_phase,
                    "mobility_phase_labels": MOBILITY_PHASE_LABELS,
                    "transition_matrix": loader.transition_matrix,
                    "parent_trace_ids": loader.parent_trace_ids,
                    "clip_start_times_s": loader.clip_start_times_s,
                    "dwell_durations_s": loader.dwell_durations_s,
                    "dwell_trajectory_indices": (
                        loader.dwell_trajectory_indices
                    ),
                    "dwell_user_indices": loader.dwell_user_indices,
                    "dwell_hotspot_states": loader.dwell_hotspot_states,
                    "dwell_start_times_s": loader.dwell_start_times_s,
                    "transition_trajectory_indices": (
                        loader.transition_trajectory_indices
                    ),
                    "transition_user_indices": (
                        loader.transition_user_indices
                    ),
                    "transition_from_states": loader.transition_from_states,
                    "transition_to_states": loader.transition_to_states,
                }
            )
        np.savez(
            os.path.join(final_dir, f"environment_run{run_id}.npz"),
            **environment,
        )
        diagnostics = loader.diagnostics(
            bootstrap_samples=self.bootstrap_samples
        )
        np.savez(
            os.path.join(final_dir, f"channel_diagnostics_run{run_id}.npz"),
            **diagnostics,
        )
        if loader.mobility_model == MOBILITY_HOTSPOT:
            np.savez(
                os.path.join(
                    final_dir, f"hotspot_diagnostics_run{run_id}.npz"
                ),
                **loader.hotspot_diagnostics(),
            )
