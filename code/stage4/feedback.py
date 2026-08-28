"""Causal per-AP CSI feedback state for Stage 4."""

import numpy as np


SCHEDULERS = ("round_robin", "random", "mobility_age_priority")


class FeedbackState:
    """Track stored CSI and ages under one fixed-association policy."""

    def __init__(
        self,
        initial_true_channels,
        association_mask,
        *,
        budget,
        scheduler,
        scheduler_seed=0,
        t0_link_power=None,
        rhos=None,
    ):
        initial = np.asarray(initial_true_channels)
        association = np.asarray(association_mask, dtype=bool)
        if initial.ndim != 4:
            raise ValueError("Initial channels must have shape [B,A,K,M]")
        if association.shape != (initial.shape[0], initial.shape[2], initial.shape[1]):
            raise ValueError("Association mask must have shape [B,K,A]")
        if scheduler not in SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        if not 0 <= budget <= initial.shape[2]:
            raise ValueError("Budget must be in [0,K]")

        self.active = association.transpose(0, 2, 1)
        self.budget = int(budget)
        self.scheduler = scheduler
        self.rng = np.random.default_rng(scheduler_seed)
        self.cursors = np.zeros(self.active.shape[:2], dtype=np.int64)
        self.stored_channels = np.zeros_like(initial)
        self.stored_channels[self.active] = initial[self.active]
        self.ages = np.where(self.active, 0, -1).astype(np.int32)
        self.previous_updates = np.zeros_like(self.active)
        self.last_priority = np.zeros(self.active.shape)

        if scheduler == "mobility_age_priority":
            if t0_link_power is None or rhos is None:
                raise ValueError(
                    "mobility_age_priority requires t0_link_power and rhos"
                )
            self.t0_link_power = np.asarray(t0_link_power, dtype=np.float64)
            rho_values = np.asarray(rhos, dtype=np.float64)
            if self.t0_link_power.shape != self.active.shape:
                raise ValueError("t0_link_power must have shape [B,A,K]")
            if rho_values.shape != (initial.shape[0], initial.shape[2]):
                raise ValueError("rhos must have shape [B,K]")
            self.rhos = rho_values[:, None, :]
        else:
            self.t0_link_power = None
            self.rhos = None

    def _select_ranked(self, priority):
        updates = np.zeros_like(self.active)
        for trajectory in range(self.active.shape[0]):
            for ap in range(self.active.shape[1]):
                load = int(self.active[trajectory, ap].sum())
                count = min(self.budget, load)
                if count:
                    order = np.argsort(
                        -priority[trajectory, ap], kind="stable"
                    )
                    updates[trajectory, ap, order[:count]] = True
        return updates

    def select_updates(self):
        """Choose links using only state available before current CSI is revealed."""
        if self.scheduler == "round_robin":
            updates = np.zeros_like(self.active)
            priority = np.zeros(self.active.shape)
            for trajectory in range(self.active.shape[0]):
                for ap in range(self.active.shape[1]):
                    users = np.flatnonzero(self.active[trajectory, ap])
                    count = min(self.budget, len(users))
                    if not count:
                        continue
                    cursor = self.cursors[trajectory, ap] % len(users)
                    selected = users[(cursor + np.arange(count)) % len(users)]
                    updates[trajectory, ap, selected] = True
                    priority[trajectory, ap, selected] = np.arange(
                        count, 0, -1
                    )
                    self.cursors[trajectory, ap] = (cursor + count) % len(users)
        elif self.scheduler == "random":
            priority = self.rng.random(self.active.shape)
            priority[~self.active] = -np.inf
            updates = self._select_ranked(priority)
        else:
            priority = self.t0_link_power * (
                1 - np.abs(self.rhos) ** (2 * (self.ages + 1))
            )
            priority[~self.active] = -np.inf
            updates = self._select_ranked(priority)

        self._validate_updates(updates)
        priority[~self.active] = 0
        self.last_priority = priority
        return updates

    def _validate_updates(self, updates):
        if updates.shape != self.active.shape:
            raise ValueError("Update mask shape mismatch")
        if np.any(updates & ~self.active):
            raise RuntimeError("Scheduler selected an unassociated link")
        counts = updates.sum(axis=-1)
        loads = self.active.sum(axis=-1)
        if np.any(counts > self.budget):
            raise RuntimeError("Per-AP feedback budget exceeded")
        if np.any(counts != np.minimum(self.budget, loads)):
            raise RuntimeError("Scheduler did not fill available budget")

    def apply_updates(self, current_true_channels, updates):
        """Reveal selected current CSI, then update stored CSI and ages."""
        current = np.asarray(current_true_channels)
        updates = np.asarray(updates, dtype=bool)
        if current.shape != self.stored_channels.shape:
            raise ValueError("Current channels must match stored CSI shape")
        self._validate_updates(updates)
        self.stored_channels[updates] = current[updates]
        self.ages[self.active] += 1
        self.ages[updates] = 0
        self.ages[~self.active] = -1
        self.previous_updates = updates.copy()

    def step(self, current_true_channels):
        updates = self.select_updates()
        self.apply_updates(current_true_channels, updates)
        return updates
