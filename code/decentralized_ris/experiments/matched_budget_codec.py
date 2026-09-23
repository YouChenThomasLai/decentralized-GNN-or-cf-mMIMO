"""E07 continuation: matched-bit-budget comparison of the G2 and R0 messages.

The E07 pilot quantized only the R0 message family and stopped on a failed
identity control.  This program answers the question that pilot could not:

    at the same number of AP->CPU bits per AP-RIS pair and the same single
    fusion round, does G2's directly executable phase message reach a higher
    paper-decentralized sum rate than the R0 anchor's latent message?

Both frozen checkpoints are evaluated on *identical* channel draws, so every
contrast is paired.  Nothing is retrained: the codec sits on the wire, after an
AP has formed its message and before the CPU fuses.

    backbone   wire message                      fusion               fp32 bits
    -------------------------------------------------------------------------
    G2-150k    N phase angles + 1 energy scalar  energy circular mean       992
    R0-500k    4N latent phase feature           learned W_reduce          3840
    R0c        2N pre-projection logits          sum, project              1920

R0 is additionally given every interface G2 uses: the same phase grid, the same
pair scalar and a 2-D vector quantizer, all applied to its own logits with
codebooks fitted on a separate calibration split.  The baseline is therefore
never handicapped by being denied the compressed interface under test.

Two accountings are kept for R0.  `bits_per_pair` follows the repository's
maintained 4N count; `bits_per_pair_generous` charges only 2N(1 + 1/R) reals,
because the last 2N latent coordinates are the RIS-independent `fe_AP` block and
an AP could send them once per AP instead of once per AP-RIS pair.  Every gate
uses the generous count, so the baseline is compared at its cheapest defensible
payload.

Run from `code/decentralized_ris`:

    python -m experiments.matched_budget_codec --device cuda:0
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

from evaluate import build_model, resolve_device, seed_everything, temporary_seed
from model import load_checkpoint
from rates import RatePrecompute, quantize_phase
from simulation import ChannelSimulator
from variants import _unit_from_pairs, circular_consensus, decode_phase

from experiments.message_codec_sweep import (
    FP32_BITS,
    LOG_FLOOR,
    Codebooks,
    CodecSpec,
    LogScalarQuantizer,
    VectorQuantizer,
    aggregate,
    encode_decode,
    fit_phase_offset,
    git_output,
    lloyd_max_1d,
    project,
    sha256_file,
    t95,
)

# --------------------------------------------------------------------- decision
# Declared before the locked evaluation and recorded in
# `artifacts/decentralized_ris/e07_message_codec/prereg_matched_budget.json`.
# A failed gate is reported, never retuned.
PRIMARY_PHASE_BITS = 2
PRIMARY_SCALE_BITS = 8
PRIMARY_BUDGET_BITS = 68          # N * 2 + 8 with N = 30
PRIMARY_ACTUATION = "continuous"

DECISION = {
    "primary_budget_bits_per_pair": PRIMARY_BUDGET_BITS,
    "primary_actuation": PRIMARY_ACTUATION,
    "primary_g2_arm": f"g2_energy_bp{PRIMARY_PHASE_BITS}_bs{PRIMARY_SCALE_BITS}",
    "d1_signaling_advantage": (
        "G2 at the primary budget minus the uncompressed R0 anchor is positive "
        "and the 95% cluster interval excludes zero"
    ),
    "d2_matched_budget": (
        "G2 at the primary budget minus the best R0-family arm whose generous "
        "payload is at most the primary budget is positive and the 95% cluster "
        "interval excludes zero"
    ),
    "d3_interface_robustness": (
        "the compression penalty of G2 relative to its own fp32 message is "
        "smaller in magnitude than the penalty of the D2 baseline arm relative "
        "to the uncompressed R0 anchor, with a 95% cluster interval on the "
        "difference in differences that excludes zero"
    ),
    "d4_same_backbone_interface": (
        "descriptive only: on the R0 backbone alone, the executable-phase "
        "interface at the primary budget beats the best latent or logit "
        "interface at no more than the primary budget"
    ),
    "confirmation": (
        "every decision that passes on the discovery seed is re-evaluated on a "
        "fresh evaluation seed with the arms fixed by the discovery run; a "
        "claim stands only if the same rule passes on both seeds"
    ),
    "reference_rates_seed20260915": {
        "g2_150k_paper_decentralized_continuous": 24.535803,
        "g2_150k_paper_decentralized_2bit": 22.818733,
        "r0_500k_paper_decentralized_continuous": 22.330500,
        "r0_500k_paper_decentralized_2bit": 20.679702,
    },
    "max_reference_reproduction_error": 1e-3,
    "max_message_identity_error": 0.0,
    "max_message_relative_error": 1e-6,
    "max_conditioned_phase_error": 1e-5,
    "max_identity_rate_error": 1e-4,
    "max_unit_modulus_error": 1e-5,
    "max_latent_tail_ris_spread": 0.0,
    "min_inactive_weight_leakage": 0.0,
}

ART = "../../artifacts/decentralized_ris"
R0_RUN = f"{ART}/e01_baseline_training/iter500000/M2_N30_L4_K8_P15.0_iter350000_seed0/run0"
R0_CKPT = f"{R0_RUN}/models/model_final_run0.pt"
G2_RUN = f"{ART}/e06_graph_energy_training/g2_long_training/g2/iter150000"
G2_CKPT = f"{G2_RUN}/checkpoints/iter150000.pt"

ACTUATIONS = ("continuous", "2bit")


# ----------------------------------------------------------------- quantizers
class PerDimQuantizer:
    """Independent Lloyd-Max codebook per coordinate of a fixed-length vector.

    The R0 wire message is a 4N latent activation whose coordinates carry very
    different scales, so one shared codebook would understate what the anchor
    can do.  Fitting a separate codebook per coordinate is the strongest
    scalar quantizer available to the baseline at a given rate, and it stays
    pre-shared: nothing about it is charged to the per-message payload.
    """

    def __init__(self, bits, levels):
        self.bits = int(bits)
        self.levels = levels.double()                        # (dim, 2 ** bits)
        self._edges = None
        self._levels = None

    @classmethod
    def fit(cls, values, bits):
        flat = values.detach().double().cpu().reshape(-1, values.shape[-1])
        levels = torch.stack([lloyd_max_1d(flat[:, d], 2 ** bits)
                              for d in range(flat.shape[1])])
        return cls(bits, levels)

    def to(self, device):
        self._edges = ((self.levels[:, 1:] + self.levels[:, :-1]) / 2).to(
            device=device, dtype=torch.float32)
        self._levels = self.levels.to(device=device, dtype=torch.float32)
        return self

    def quantize(self, values):
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        if self._edges.shape[1] == 0:
            index = torch.zeros_like(flat, dtype=torch.long)
        else:
            index = torch.searchsorted(
                self._edges.expand(flat.shape[0], -1, -1).contiguous(),
                flat.unsqueeze(-1).contiguous()).squeeze(-1)
        return torch.gather(self._levels.expand(flat.shape[0], -1, -1), 2,
                            index.unsqueeze(-1)).squeeze(-1).reshape(shape)

    def describe(self):
        return {"bits": self.bits, "dim": int(self.levels.shape[0]),
                "level_min": self.levels.min(dim=1).values.tolist(),
                "level_max": self.levels.max(dim=1).values.tolist()}


def quantize_angles(angles, bits, offset):
    """Nearest point of the calibrated uniform phase grid, returned as angles."""
    step = 2 * math.pi / (2 ** bits)
    return torch.round((angles - offset) / step) * step + offset


# ----------------------------------------------------------------------- arms
class Arm:
    """One wire format: how the message is coded, and what it costs."""

    def __init__(self, name, backbone, family, bits, generous=None, detail=None):
        self.name = name
        self.backbone = backbone                 # "g2" or "r0"
        self.family = family
        self.bits = int(bits)
        self.generous = int(bits if generous is None else generous)
        self.detail = detail or {}

    def describe(self, n_ap, n_ris):
        return {
            "backbone": self.backbone,
            "family": self.family,
            "bits_per_pair": self.bits,
            "bits_per_pair_generous": self.generous,
            "bits_system": self.bits * n_ap * n_ris,
            "bits_system_generous": self.generous * n_ap * n_ris,
            "fusion_rounds": 1,
            **self.detail,
        }


def tag(bits):
    return "inf" if bits is None else str(bits)


def width(bits):
    return FP32_BITS if bits is None else int(bits)


def build_arms(n_elem, n_ris, phase_bits, scale_bits, latent_bits, vq_bits,
               mag_bits, resid_bits, extra_scale_bits):
    """Every operating point, with its payload counted from its own fields."""
    arms = []

    # --- G2: N executable phase angles, optionally one local-energy scalar ---
    for b_p in phase_bits:
        arms.append(Arm(f"g2_equal_bp{tag(b_p)}", "g2", "g2_equal",
                        n_elem * width(b_p),
                        detail={"b_phase": b_p, "b_scale": None,
                                "sends_energy": False}))
        for b_s in scale_bits:
            arms.append(Arm(f"g2_energy_bp{tag(b_p)}_bs{tag(b_s)}", "g2",
                            "g2_energy",
                            n_elem * width(b_p) + width(b_s),
                            detail={"b_phase": b_p, "b_scale": b_s,
                                    "sends_energy": True}))

    # --- R0: its own 4N latent message into the learned CPU reduction -------
    dedup = 2 * n_elem * (n_ris + 1) / n_ris                 # 2N(1 + 1/R) reals
    for b_v in latent_bits:
        arms.append(Arm(f"r0_latent_b{tag(b_v)}", "r0", "r0_latent",
                        4 * n_elem * width(b_v),
                        generous=round(dedup * width(b_v)),
                        detail={"b_latent": b_v}))

    # --- R0c: the algebraically equivalent 2N logit message -----------------
    for b_v in latent_bits:
        arms.append(Arm(f"r0c_logit_b{tag(b_v)}", "r0", "r0c_logit",
                        2 * n_elem * width(b_v), detail={"b_logit": b_v}))

    # --- the compressed interfaces G2 uses, granted to the R0 backbone ------
    for b_p in phase_bits:
        arms.append(Arm(f"r0_phase_bp{tag(b_p)}", "r0", "r0_phase",
                        n_elem * width(b_p),
                        detail={"b_phase": b_p, "b_scale": None}))
        for b_s in ((PRIMARY_SCALE_BITS,) if b_p != PRIMARY_PHASE_BITS
                    else tuple(extra_scale_bits)):
            arms.append(Arm(f"r0_pairmag_bp{tag(b_p)}_bs{tag(b_s)}", "r0",
                            "r0_pairmag",
                            n_elem * width(b_p) + width(b_s),
                            detail={"b_phase": b_p, "b_scale": b_s}))
    for b_q in vq_bits:
        arms.append(Arm(f"r0c_vq_b{tag(b_q)}", "r0", "r0c_vq",
                        n_elem * width(b_q), detail={"b_vq": b_q}))
        arms.append(Arm(f"r0c_vqgain_b{tag(b_q)}_bs{PRIMARY_SCALE_BITS}", "r0",
                        "r0c_vqgain",
                        n_elem * width(b_q) + PRIMARY_SCALE_BITS,
                        detail={"b_vq": b_q, "b_scale": PRIMARY_SCALE_BITS}))
    for b_p in phase_bits:
        for b_m in mag_bits:
            arms.append(Arm(f"r0c_polar_bp{tag(b_p)}_bm{tag(b_m)}", "r0",
                            "r0c_polar",
                            n_elem * (width(b_p) + width(b_m)),
                            detail={"b_phase": b_p, "b_mag": b_m}))
    for b_p in (2, 3):
        for b_e in resid_bits:
            arms.append(Arm(
                f"r0c_progressive_bp{b_p}_bs{PRIMARY_SCALE_BITS}_be{b_e}", "r0",
                "r0c_progressive",
                n_elem * b_p + PRIMARY_SCALE_BITS + n_elem * b_e,
                detail={"b_phase": b_p, "b_scale": PRIMARY_SCALE_BITS,
                        "b_resid": b_e}))

    # --- uncompressed references -------------------------------------------
    arms.append(Arm("g2_native_fp32", "g2", "reference",
                    n_elem * FP32_BITS + FP32_BITS,
                    detail={"note": "N fp32 angles + one fp32 energy scalar"}))
    arms.append(Arm("r0_native_fp32", "r0", "reference", 4 * n_elem * FP32_BITS,
                    generous=round(dedup * FP32_BITS),
                    detail={"note": "4N fp32 latent into the learned reduction"}))
    arms.append(Arm("r0c_native_fp32", "r0", "reference", 2 * n_elem * FP32_BITS,
                    detail={"note": "2N fp32 logits, summed then projected"}))
    return arms


def codec_spec(arm):
    """Map an R0c-side arm onto the pilot's CodecSpec so one codec path is used."""
    d = arm.detail
    if arm.family == "r0_phase":
        return CodecSpec("phase_only", b_phase=d["b_phase"])
    if arm.family == "r0_pairmag":
        return CodecSpec("pair_mag", b_phase=d["b_phase"], b_scale=d["b_scale"])
    if arm.family == "r0c_polar":
        return CodecSpec("polar", b_phase=d["b_phase"], b_mag=d["b_mag"])
    if arm.family == "r0c_progressive":
        return CodecSpec("progressive", b_phase=d["b_phase"],
                         b_scale=d["b_scale"], b_resid=d["b_resid"])
    if arm.family == "r0c_vq":
        return CodecSpec("vq", b_vq=d["b_vq"])
    if arm.family == "r0c_vqgain":
        return CodecSpec("vq_gain", b_vq=d["b_vq"], b_scale=d["b_scale"])
    return None


# ------------------------------------------------------------------ statistics
def clustered(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = t95(n - 1) * sem if n > 1 else float("nan")
    return {"mean": mean, "sem": sem, "ci95_low": mean - half,
            "ci95_high": mean + half, "clusters": n}


def paired(left, right):
    stats = clustered(np.asarray(left) - np.asarray(right))
    stats["excludes_zero"] = bool(stats["ci95_low"] > 0.0)
    return stats


def message_pass(net, features, edges, masks, direct):
    trace = {}
    beamformer, theta = net.decentralized(features, edges, masks, direct, trace)
    return beamformer, theta, trace


def draw_batch(simulator, users_per_ap, threshold, device):
    """One holdout draw, using the same RNG consumption as `evaluate_model`."""
    simulator.training_batch(users_per_ap, threshold, threshold)
    features, edges, masks, direct = simulator.decentralized_batch(
        users_per_ap, threshold, threshold, regenerate_channels=False)
    features = [t.to(device) for t in features]
    edges = [t.to(device) for t in edges]
    direct = [t.to(device) for t in direct]
    return features, edges, masks, direct




# --------------------------------------------------------- codebooks / calibration
class MatchedBooks(Codebooks):
    """The pilot's frozen codebooks plus the ones this comparison adds.

    Everything here is fitted once on a calibration split that shares no draw
    with either evaluation seed, then frozen and pre-shared, so none of it is
    charged to the per-message payload.
    """

    def __init__(self):
        super().__init__()
        self.energy = {}                 # G2 local-energy scalar, log domain
        self.latent = {}                 # R0 4N latent, per coordinate
        self.logit = {}                  # R0c 2N logits, per coordinate

    def to(self, device):
        super().to(device)
        for table in (self.energy, self.latent, self.logit):
            for quantizer in table.values():
                quantizer.to(device)
        return self

    def describe(self):
        blob = super().describe()
        blob["g2_energy_scalar"] = {str(k): q.describe()
                                    for k, q in self.energy.items()}
        blob["r0_latent_per_dim"] = {str(k): q.describe()
                                     for k, q in self.latent.items()}
        blob["r0c_logit_per_dim"] = {str(k): q.describe()
                                     for k, q in self.logit.items()}
        return blob


def calibrate(nets, simulator, config, device, samples, batch_size, seed,
              phase_bits, scale_bits, latent_bits, vq_bits, mag_bits,
              resid_bits, extra_scale_bits, clip_quantile, codebook_seed,
              n_ris, n_elem):
    """Fit every frozen codebook on draws disjoint from the holdouts."""
    threshold = config.get("assoc_threshold", 0.1)
    g2_angles, g2_energy, g2_active, r0_z, r0_latent, r0_logit = [], [], [], [], [], []
    with torch.no_grad(), temporary_seed(seed):
        for _ in range(samples // batch_size):
            features, edges, masks, direct = draw_batch(
                simulator, config["K"], threshold, device)
            _, _, trace_g2 = message_pass(nets["g2"], features, edges, masks, direct)
            _, _, trace_r0 = message_pass(nets["r0"], features, edges, masks, direct)
            g2_angles.append(trace_g2["phase_angles"].to("cpu", torch.float64))
            g2_energy.append(trace_g2["energy"].to("cpu", torch.float64))
            g2_active.append(trace_g2["active"].to("cpu", torch.float64))
            r0_z.append(trace_r0["z_pairs"].to("cpu", torch.float64))
            r0_latent.append(torch.stack(trace_r0["latents"], dim=1)
                             .to("cpu", torch.float64))
            r0_logit.append(torch.stack(
                nets["r0"].RIS_merge.local_logits(trace_r0["latents"]), dim=1)
                .to("cpu", torch.float64))

    g2_angles = torch.cat(g2_angles, dim=0)
    g2_energy = torch.cat(g2_energy, dim=0)
    g2_active = torch.cat(g2_active, dim=0)
    r0_z = torch.cat(r0_z, dim=0)
    r0_latent = torch.cat(r0_latent, dim=0)
    r0_logit = torch.cat(r0_logit, dim=0)
    r0_m = r0_z.norm(dim=-1)
    r0_s = r0_m.mean(dim=3)
    r0_angle = torch.atan2(r0_z[..., 1], r0_z[..., 0])

    # An AP that serves nobody transmits nothing, so its zero energy must not
    # enter the log-domain codebook for the APs that do transmit.
    voting = (g2_active[..., None].expand_as(g2_energy) > 0) & (g2_energy > 0)
    energy_values = g2_energy[voting]

    books = MatchedBooks()
    clip = (clip_quantile, 1.0 - clip_quantile)
    books.stats = {
        "draws": int(g2_angles.shape[0]),
        "messages": int(g2_angles.shape[0] * config["AP"] * n_ris),
        "g2_energy": {
            "voting_pairs": int(voting.sum()),
            "min": float(energy_values.min()), "max": float(energy_values.max()),
            "median": float(energy_values.median()),
            "dynamic_range_db": float(20 * math.log10(
                float(energy_values.max()) / max(float(energy_values.min()), LOG_FLOOR))),
        },
        "g2_angle_resultant_length": float(
            torch.hypot(g2_angles.cos().mean(), g2_angles.sin().mean())),
        "r0_latent": {"min": float(r0_latent.min()), "max": float(r0_latent.max()),
                      "per_dim_std_min": float(r0_latent.reshape(-1, 4 * n_elem)
                                               .std(dim=0).min()),
                      "per_dim_std_max": float(r0_latent.reshape(-1, 4 * n_elem)
                                               .std(dim=0).max())},
        "r0_logit": {"min": float(r0_logit.min()), "max": float(r0_logit.max())},
        "r0_element_magnitude": {
            "min": float(r0_m.min()), "max": float(r0_m.max()),
            "median": float(r0_m.median()),
            "dynamic_range_db": float(20 * math.log10(
                float(r0_m.max()) / max(float(r0_m.min()), LOG_FLOOR)))},
    }

    # One shared phase grid for both backbones: the G2 angles and the R0
    # projected logit angles are both very close to uniform, so the calibrated
    # rotation is fitted on the union and neither side gets a private grid.
    union = torch.cat((g2_angles.reshape(-1), r0_angle.reshape(-1)))
    for bits in phase_bits:
        if bits is None:
            continue
        offset, cost = fit_phase_offset(union, bits)
        books.phase_offset[bits] = offset
        books.phase_offset_cost[bits] = cost
        print(f"[calib] phase grid b_p={bits}: offset={offset:.6f} rad, "
              f"chord distortion={cost:.6f}")

    for bits in set(scale_bits) | set(extra_scale_bits) | {PRIMARY_SCALE_BITS}:
        if bits is not None:
            books.energy[bits] = LogScalarQuantizer.fit(energy_values, bits, clip)
    for bits in latent_bits:
        if bits is None:
            continue
        books.latent[bits] = PerDimQuantizer.fit(r0_latent, bits)
        books.logit[bits] = PerDimQuantizer.fit(r0_logit, bits)
        print(f"[calib] per-dimension Lloyd-Max b={bits} fitted for the R0 "
              f"latent ({4 * n_elem} dims) and the R0c logits ({2 * n_elem} dims)")
    for bits in mag_bits:
        if bits is not None:
            books.magnitude[bits] = LogScalarQuantizer.fit(r0_m, bits, clip)
    books.scale[PRIMARY_SCALE_BITS] = LogScalarQuantizer.fit(
        r0_s, PRIMARY_SCALE_BITS, clip)
    for bits in extra_scale_bits:
        if bits is not None and bits not in books.scale:
            books.scale[bits] = LogScalarQuantizer.fit(r0_s, bits, clip)
    books.to(device)

    r0_m_dev = r0_m.to(device=device, dtype=torch.float32)
    r0_s_dev = r0_s.to(device=device, dtype=torch.float32)
    scale_hat = books.scale[PRIMARY_SCALE_BITS].quantize(r0_s_dev)
    ratio = r0_m_dev / scale_hat[..., None].clamp_min(LOG_FLOOR)
    for bits in resid_bits:
        if bits:
            books.residual[(PRIMARY_SCALE_BITS, bits)] = LogScalarQuantizer.fit(
                ratio, bits, clip)
    for bits in vq_bits:
        if bits is None:
            continue
        books.vector[bits] = VectorQuantizer.fit(r0_z, bits, codebook_seed)
        normalized = r0_z.to(device=device, dtype=torch.float32) \
            / scale_hat[..., None, None].clamp_min(LOG_FLOOR)
        books.vector_gain[(bits, PRIMARY_SCALE_BITS)] = VectorQuantizer.fit(
            normalized, bits, codebook_seed + 1)
        print(f"[calib] fitted 2-D VQ b={bits} ({2 ** bits} codewords)")
    books.to(device)
    return books


# ------------------------------------------------------------------ evaluation
def theta_for_arm(arm, state, books, n_ris, n_elem):
    """The fused RIS phase this wire format produces on one batch."""
    if arm.family == "reference":
        return state[{"g2_native_fp32": "g2_theta",
                      "r0_native_fp32": "r0_theta",
                      "r0c_native_fp32": "r0c_theta"}[arm.name]]

    if arm.backbone == "g2":
        b_p = arm.detail["b_phase"]
        angles = state["g2_angles"]
        if b_p is not None:
            angles = quantize_angles(angles, b_p, books.phase_offset[b_p])
        weights = None
        if arm.detail["sends_energy"]:
            b_s = arm.detail["b_scale"]
            weights = state["g2_energy"]
            if b_s is not None:
                weights = books.energy[b_s].quantize(weights)
        theta, _ = circular_consensus(decode_phase(angles), state["g2_active"],
                                      None, 1.0, weights)
        return theta

    if arm.family == "r0_latent":
        b_v = arm.detail["b_latent"]
        if b_v is None:
            return state["r0_net"].RIS_merge(state["r0_latents"])
        quantized = [books.latent[b_v].quantize(latent)
                     for latent in state["r0_latents"]]
        return state["r0_net"].RIS_merge(quantized)

    if arm.family == "r0c_logit":
        b_v = arm.detail["b_logit"]
        flat = state["r0_logits"]
        if b_v is not None:
            flat = books.logit[b_v].quantize(flat)
        return aggregate(torch.stack((flat[..., :n_elem], flat[..., n_elem:]),
                                     dim=-1))

    return aggregate(encode_decode(codec_spec(arm), state["r0_z"], books))


def conditioned_error(theta, reference, proposals, weights, active):
    """max |dtheta| * rho, with rho the normalised resultant length.

    The pilot's control compared the raw post-projection phase error, which
    blows up exactly where the AP proposals nearly cancel and the projected
    direction is genuinely ill-conditioned.  Weighting by the resultant length
    removes that amplification without hiding a real reconstruction error: an
    element whose resultant is well formed still has rho near one.
    """
    delta = torch.atan2(theta[..., 1], theta[..., 0]) - torch.atan2(
        reference[..., 1], reference[..., 0])
    delta = torch.atan2(delta.sin(), delta.cos()).abs()
    w = weights * active.unsqueeze(2).expand_as(weights)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    rho = (proposals * w[..., None, None]).sum(dim=1).norm(dim=-1)
    return float((delta * rho).max())


def grid_residual(angles, bits, offset):
    """Distance from a decoded angle to its 2**bits grid point."""
    step = 2 * math.pi / (2 ** bits)
    residual = (angles - offset) / step
    return float((residual - residual.round()).abs().max())


def run_evaluation(nets, simulator, config, arms, books, device, samples,
                   batch_size, eval_seed, n_ris, n_elem):
    """Score every arm on identical channel draws and collect the controls."""
    records = {arm.name: {phase: [] for phase in ACTUATIONS} for arm in arms}
    controls = {
        "max_message_identity_error": 0.0,
        "max_message_relative_error": 0.0,
        "max_phase_grid_residual": 0.0,
        "max_conditioned_phase_error": 0.0,
        "max_identity_rate_error": 0.0,
        "max_unit_modulus_error": 0.0,
        "max_latent_tail_ris_spread": 0.0,
        "max_inactive_weight_leakage": 0.0,
        "max_fast_rate_vs_reference_error": 0.0,
        "inactive_ap_slots": 0,
        "identity_pairs": {},
    }
    arm_names = {arm.name for arm in arms}
    identity_pairs = {name: reference for name, reference in (
        ("g2_energy_bpinf_bsinf", "g2_native_fp32"),
        ("r0_latent_binf", "r0_native_fp32"),
        ("r0c_logit_binf", "r0c_native_fp32"),
        ("r0c_polar_bpinf_bminf", "r0c_native_fp32"),
    ) if name in arm_names}
    for key in identity_pairs:
        controls["identity_pairs"][key] = {"max_conditioned_phase_error": 0.0,
                                           "max_rate_error": 0.0}

    finite_phase_bits = sorted(books.phase_offset)
    threshold = config.get("assoc_threshold", 0.1)
    with torch.no_grad(), temporary_seed(eval_seed):
        for index in range(samples // batch_size):
            features, edges, masks, direct = draw_batch(
                simulator, config["K"], threshold, device)
            w_g2, theta_g2, trace_g2 = message_pass(
                nets["g2"], features, edges, masks, direct)
            w_r0, _, trace_r0 = message_pass(
                nets["r0"], features, edges, masks, direct)

            latents = trace_r0["latents"]
            state = {
                "g2_theta": theta_g2,
                "g2_angles": trace_g2["phase_angles"],
                "g2_energy": trace_g2["energy"],
                "g2_active": trace_g2["active"],
                "r0_net": nets["r0"],
                "r0_latents": latents,
                "r0_logits": torch.stack(
                    nets["r0"].RIS_merge.local_logits(latents), dim=1),
                "r0_z": trace_r0["z_pairs"],
                "r0_theta": nets["r0"].RIS_merge(latents),
                "r0c_theta": nets["r0"].RIS_merge.forward_commuted(latents),
            }

            # Message-level identity, the pilot's failed control replaced.
            # The G2 wire format is already the angle the CPU decodes, so its
            # round trip must be exact.  The R0c polar format converts between
            # Cartesian and polar coordinates, which is a float32 round trip
            # rather than an identity, so it is held to a relative bound.
            controls["max_message_identity_error"] = max(
                controls["max_message_identity_error"],
                float((decode_phase(state["g2_angles"])
                       - trace_g2["proposals"]).abs().max()))
            round_trip = encode_decode(
                CodecSpec("polar", b_phase=None, b_mag=None), state["r0_z"], books)
            controls["max_message_relative_error"] = max(
                controls["max_message_relative_error"],
                float(((round_trip - state["r0_z"]).norm(dim=-1)
                       / state["r0_z"].norm(dim=-1).clamp_min(LOG_FLOOR)).max()))
            for bits in finite_phase_bits:
                controls["max_phase_grid_residual"] = max(
                    controls["max_phase_grid_residual"],
                    grid_residual(quantize_angles(state["g2_angles"], bits,
                                                  books.phase_offset[bits]),
                                  bits, books.phase_offset[bits]))

            stacked = torch.stack(latents, dim=1)               # (B, A, R, 4N)
            tail = stacked[..., 2 * n_elem:]
            controls["max_latent_tail_ris_spread"] = max(
                controls["max_latent_tail_ris_spread"],
                float((tail - tail[:, :, :1, :]).abs().max()))
            controls["inactive_ap_slots"] += int((trace_g2["active"] <= 0).sum())
            controls["max_inactive_weight_leakage"] = max(
                controls["max_inactive_weight_leakage"],
                float((trace_g2["weights"]
                       * (trace_g2["active"] <= 0).to(theta_g2.dtype)[..., None]
                       ).abs().max()))

            pre = RatePrecompute(simulator, device)
            beamformers = {"g2": w_g2, "r0": w_r0}
            thetas, rates = {}, {}
            for arm in arms:
                theta = theta_for_arm(arm, state, books, n_ris, n_elem)
                thetas[arm.name] = theta
                controls["max_unit_modulus_error"] = max(
                    controls["max_unit_modulus_error"],
                    float((theta.norm(dim=-1) - 1.0).abs().max()))
                for phase in ACTUATIONS:
                    action = theta if phase == "continuous" else quantize_phase(theta, 2)
                    value = float(pre.sum_rate(beamformers[arm.backbone],
                                               action).mean())
                    records[arm.name][phase].append(value)
                    if phase == "continuous":
                        rates[arm.name] = value

            ones = torch.ones(state["g2_active"].shape, device=device,
                              dtype=theta_g2.dtype)
            r0_proposals = _unit_from_pairs(
                torch.cat((state["r0_z"][..., 0], state["r0_z"][..., 1]), dim=-1),
                n_ris, n_elem)
            r0_weights = state["r0_z"].norm(dim=-1).mean(dim=3)
            for name, reference in identity_pairs.items():
                entry = controls["identity_pairs"][name]
                if name.startswith("g2"):
                    proposals, weights, active = (decode_phase(state["g2_angles"]),
                                                  state["g2_energy"],
                                                  state["g2_active"])
                else:
                    proposals, weights, active = r0_proposals, r0_weights, ones
                entry["max_conditioned_phase_error"] = max(
                    entry["max_conditioned_phase_error"],
                    conditioned_error(thetas[name], thetas[reference],
                                      proposals, weights, active))
                entry["max_rate_error"] = max(
                    entry["max_rate_error"], abs(rates[name] - rates[reference]))

            _, reference_rate, _ = simulator.loss(w_r0, state["r0_theta"], device)
            controls["max_fast_rate_vs_reference_error"] = max(
                controls["max_fast_rate_vs_reference_error"],
                abs(float(reference_rate) - rates["r0_native_fp32"]))

            if (index + 1) % 10 == 0:
                print(f"[eval {eval_seed}] cluster {index + 1}/"
                      f"{samples // batch_size}  "
                      f"g2={np.mean(records['g2_native_fp32']['continuous']):.5f}  "
                      f"r0={np.mean(records['r0_native_fp32']['continuous']):.5f}")

    controls["max_conditioned_phase_error"] = max(
        (entry["max_conditioned_phase_error"]
         for entry in controls["identity_pairs"].values()), default=0.0)
    controls["max_identity_rate_error"] = max(
        (entry["max_rate_error"]
         for entry in controls["identity_pairs"].values()), default=0.0)
    return records, controls


# ------------------------------------------------------------------- decisions
EXECUTABLE_PHASE_FAMILIES = ("r0_phase", "r0_pairmag")
LATENT_FAMILIES = ("r0_latent", "r0c_logit", "r0c_vq", "r0c_vqgain")


def pareto_front(points):
    """points: (bits, rate, name) -> names on the cheap-and-high frontier."""
    front = []
    for bits, rate, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if not front or rate > front[-1][1]:
            front.append((bits, rate, name))
    return [name for _, _, name in front]


def summarise(arms, records, n_ap, n_ris):
    """Per-arm rate summary plus the contrast against its own fp32 reference."""
    arrays = {name: {phase: np.asarray(values, dtype=np.float64)
                     for phase, values in blob.items()}
              for name, blob in records.items()}
    own_reference = {"g2": "g2_native_fp32", "r0": "r0_native_fp32"}
    table = {}
    for arm in arms:
        entry = arm.describe(n_ap, n_ris)
        for phase in ACTUATIONS:
            entry[phase] = clustered(arrays[arm.name][phase])
        reference = own_reference[arm.backbone]
        entry["vs_own_fp32_continuous"] = paired(
            arrays[arm.name]["continuous"], arrays[reference]["continuous"])
        entry["vs_g2_native_continuous"] = paired(
            arrays[arm.name]["continuous"], arrays["g2_native_fp32"]["continuous"])
        table[arm.name] = entry
    for backbone in ("g2", "r0"):
        points = [(entry["bits_per_pair_generous"], entry["continuous"]["mean"], name)
                  for name, entry in table.items() if entry["backbone"] == backbone]
        front = set(pareto_front(points))
        for name in table:
            if table[name]["backbone"] == backbone:
                table[name]["on_pareto_frontier"] = name in front
    return arrays, table


def decide(arms, arrays, table, budget=PRIMARY_BUDGET_BITS, fixed=None):
    """Evaluate the pre-declared rules; `fixed` replays a discovery selection."""
    by_name = {arm.name: arm for arm in arms}
    primary = DECISION["primary_g2_arm"]
    pool = [name for name, entry in table.items()
            if entry["backbone"] == "r0" and entry["family"] != "reference"
            and entry["bits_per_pair_generous"] <= budget]
    baseline = (fixed or {}).get("d2_baseline_arm") or (
        max(pool, key=lambda n: table[n]["continuous"]["mean"]) if pool else None)

    d1 = paired(arrays[primary]["continuous"], arrays["r0_native_fp32"]["continuous"])
    result = {
        "primary_arm": primary,
        "primary_budget_bits_per_pair": budget,
        "baseline_pool_size": len(pool),
        "d1_signaling_advantage": {
            "against": "r0_native_fp32",
            "against_bits_per_pair_generous":
                table["r0_native_fp32"]["bits_per_pair_generous"],
            "payload_reduction_x": table["r0_native_fp32"][
                "bits_per_pair_generous"] / budget,
            **d1, "passed": bool(d1["ci95_low"] > 0.0),
        },
    }

    if baseline is None:
        result["d2_matched_budget"] = {"passed": False,
                                       "note": "no R0 arm fits the budget"}
        result["d3_interface_robustness"] = {"passed": False,
                                             "note": "no D2 baseline"}
    else:
        d2 = paired(arrays[primary]["continuous"], arrays[baseline]["continuous"])
        penalty_g2 = (arrays[primary]["continuous"]
                      - arrays["g2_native_fp32"]["continuous"])
        penalty_r0 = (arrays[baseline]["continuous"]
                      - arrays["r0_native_fp32"]["continuous"])
        d3 = clustered(penalty_g2 - penalty_r0)
        result["d2_matched_budget"] = {
            "baseline_arm": baseline,
            "baseline_bits_per_pair_generous":
                table[baseline]["bits_per_pair_generous"],
            **d2, "passed": bool(d2["ci95_low"] > 0.0),
        }
        result["d3_interface_robustness"] = {
            "baseline_arm": baseline,
            "g2_penalty": clustered(penalty_g2),
            "r0_penalty": clustered(penalty_r0),
            **d3, "passed": bool(d3["ci95_low"] > 0.0),
        }

    # D4: the same question inside the R0 backbone, with no G2 model involved.
    executable = [n for n, e in table.items()
                  if e["family"] in EXECUTABLE_PHASE_FAMILIES
                  and e["bits_per_pair_generous"] <= budget]
    latent = [n for n, e in table.items()
              if e["family"] in LATENT_FAMILIES
              and e["bits_per_pair_generous"] <= budget]
    if executable and latent:
        best_exec = (fixed or {}).get("d4_executable_arm") or max(
            executable, key=lambda n: table[n]["continuous"]["mean"])
        best_latent = (fixed or {}).get("d4_latent_arm") or max(
            latent, key=lambda n: table[n]["continuous"]["mean"])
        d4 = paired(arrays[best_exec]["continuous"], arrays[best_latent]["continuous"])
        result["d4_same_backbone_interface"] = {
            "executable_arm": best_exec, "latent_arm": best_latent,
            **d4, "passed": bool(d4["ci95_low"] > 0.0),
        }
    else:
        result["d4_same_backbone_interface"] = {
            "passed": False, "note": "one side of the comparison is empty"}

    # How many bits the anchor needs before it matches G2 at the primary budget.
    target = table[primary]["continuous"]["mean"]
    reached = [(table[n]["bits_per_pair_generous"], n) for n, e in table.items()
               if e["backbone"] == "r0" and e["continuous"]["mean"] >= target]
    if reached:
        bits, name = min(reached)
        result["bit_equivalence_point"] = {
            "arm": name, "bits_per_pair_generous": bits,
            "ratio_vs_primary": bits / budget,
            "rate": table[name]["continuous"]["mean"],
        }
    else:
        result["bit_equivalence_point"] = {
            "arm": None,
            "note": ("no R0-family arm reaches the primary G2 operating point "
                     "at any budget measured here, including the uncompressed "
                     "anchor"),
        }
    result["advantage_passed"] = bool(
        result["d1_signaling_advantage"]["passed"]
        and result["d2_matched_budget"]["passed"])
    result["mechanism_passed"] = bool(
        result["d3_interface_robustness"]["passed"]
        and result["d4_same_backbone_interface"]["passed"])
    result["passed"] = bool(result["advantage_passed"]
                            and result["d3_interface_robustness"]["passed"])
    return result


def matched_budget_table(table, arrays):
    """Every budget where at least one G2 and one R0 arm cost the same bits."""
    by_bits = {}
    for name, entry in table.items():
        if entry["family"] == "reference":
            continue
        by_bits.setdefault(entry["bits_per_pair_generous"], []).append(name)
    matched = {}
    for bits, names in sorted(by_bits.items()):
        g2_names = [n for n in names if table[n]["backbone"] == "g2"]
        r0_names = [n for n in names if table[n]["backbone"] == "r0"]
        if not g2_names or not r0_names:
            continue
        best_g2 = max(g2_names, key=lambda n: table[n]["continuous"]["mean"])
        best_r0 = max(r0_names, key=lambda n: table[n]["continuous"]["mean"])
        matched[str(bits)] = {
            "best_g2": best_g2, "best_r0": best_r0,
            "g2_rate": table[best_g2]["continuous"]["mean"],
            "r0_rate": table[best_r0]["continuous"]["mean"],
            "g2_minus_r0": paired(arrays[best_g2]["continuous"],
                                  arrays[best_r0]["continuous"]),
            "g2_arms": sorted(g2_names), "r0_arms": sorted(r0_names),
        }
    return matched


def check_controls(controls, table, references):
    """Pre-declared numerical gates.  A failure is reported, never retuned."""
    gates = {
        "message_identity": controls["max_message_identity_error"]
        <= DECISION["max_message_identity_error"],
        "message_round_trip": controls["max_message_relative_error"]
        <= DECISION["max_message_relative_error"],
        "phase_grid_legality": controls["max_phase_grid_residual"] <= 1e-6,
        "conditioned_phase_identity": controls["max_conditioned_phase_error"]
        <= DECISION["max_conditioned_phase_error"],
        "identity_rate": controls["max_identity_rate_error"]
        <= DECISION["max_identity_rate_error"],
        "unit_modulus": controls["max_unit_modulus_error"]
        <= DECISION["max_unit_modulus_error"],
        "latent_tail_redundancy": controls["max_latent_tail_ris_spread"]
        <= DECISION["max_latent_tail_ris_spread"],
        "inactive_ap_weight": controls["max_inactive_weight_leakage"]
        <= DECISION["min_inactive_weight_leakage"],
    }
    if references:
        errors = {key: abs(table[name][phase]["mean"] - value)
                  for key, (name, phase, value) in references.items()}
        controls["reference_reproduction_error"] = errors
        gates["reference_reproduction"] = all(
            error <= DECISION["max_reference_reproduction_error"]
            for error in errors.values())
    controls["gates"] = gates
    controls["passed"] = bool(all(gates.values()))
    return controls


# ------------------------------------------------------------------------ main
def parse_bits(text, allow_zero=False):
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "inf":
            out.append(None)
        elif token == "0" and allow_zero:
            out.append(0)
        else:
            out.append(int(token))
    return out


def load_config(run_dir):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as handle:
        return json.load(handle)["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2_run", default=G2_RUN)
    parser.add_argument("--g2_ckpt", default=G2_CKPT)
    parser.add_argument("--r0_run", default=R0_RUN)
    parser.add_argument("--r0_ckpt", default=R0_CKPT)
    parser.add_argument("--calib_samples", type=int, default=800)
    parser.add_argument("--calib_seed", type=int, default=20260916)
    parser.add_argument("--eval_samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260915,
                        help="discovery seed; also the shared E06 holdout")
    parser.add_argument("--confirm_seed", type=int, default=20260923,
                        help="fresh confirmation seed, 0 to skip")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase_bits", default="1,2,3,4,5,6,8,inf")
    parser.add_argument("--energy_bits", default="2,4,8,inf")
    parser.add_argument("--latent_bits", default="1,2,3,4,6,8,inf")
    parser.add_argument("--vq_bits", default="1,2,3,4,5,6")
    parser.add_argument("--mag_bits", default="1,2,4,8,inf")
    parser.add_argument("--resid_bits", default="1,2")
    parser.add_argument("--pair_scale_bits", default="2,4,8,inf")
    parser.add_argument("--clip_quantile", type=float, default=0.001)
    parser.add_argument("--codebook_seed", type=int, default=20260916)
    parser.add_argument("--out_dir", default=f"{ART}/e07_message_codec")
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    phase_bits = parse_bits(args.phase_bits)
    energy_bits = parse_bits(args.energy_bits)
    latent_bits = parse_bits(args.latent_bits)
    vq_bits = parse_bits(args.vq_bits)
    mag_bits = parse_bits(args.mag_bits)
    resid_bits = parse_bits(args.resid_bits)
    pair_scale_bits = parse_bits(args.pair_scale_bits)

    config_r0 = load_config(args.r0_run)
    config_g2 = load_config(args.g2_run)
    for key in ("M", "N", "L", "K", "AP", "seed", "pmax_dbm"):
        if config_r0[key] != config_g2[key]:
            raise SystemExit(f"checkpoints disagree on {key}: "
                             f"{config_r0[key]} vs {config_g2[key]}")
    if args.eval_samples % args.batch_size or args.calib_samples % args.batch_size:
        raise SystemExit("sample counts must be positive multiples of batch_size")

    device = resolve_device(args.device)
    n_ap, n_ris, n_elem = config_r0["AP"], config_r0["L"], config_r0["N"]

    # One simulator, built under the shared training seed, so both frozen
    # policies are scored on the topology they were trained on.
    seed_everything(config_r0["seed"])
    simulator = ChannelSimulator(config_r0["M"], config_r0["N"], config_r0["L"],
                                 args.batch_size, n_ap=n_ap)
    host = dict(config_r0)
    host["arch"], host["consensus"] = "r1_shared", "equal"
    host["batch_size"] = args.batch_size
    net_r0 = build_model(host, simulator, device)
    load_checkpoint(net_r0, args.r0_ckpt, device)
    net_r0.eval()
    config_eval = dict(config_g2)
    config_eval["batch_size"] = args.batch_size
    net_g2 = build_model(config_eval, simulator, device)
    load_checkpoint(net_g2, args.g2_ckpt, device)
    net_g2.eval()
    nets = {"g2": net_g2, "r0": net_r0}
    print(f"[setup] g2 arch={net_g2.arch} consensus={net_g2.consensus}; "
          f"r0 hosted as arch={net_r0.arch} consensus={net_r0.consensus}")

    print(f"[calib] {args.calib_samples} samples, seed {args.calib_seed}")
    books = calibrate(nets, simulator, config_eval, device, args.calib_samples,
                      args.batch_size, args.calib_seed, phase_bits, energy_bits,
                      latent_bits, vq_bits, mag_bits, resid_bits,
                      pair_scale_bits, args.clip_quantile, args.codebook_seed,
                      n_ris, n_elem)

    arms = build_arms(n_elem, n_ris, phase_bits, energy_bits, latent_bits,
                      vq_bits, mag_bits, resid_bits, pair_scale_bits)
    print(f"[eval] {len(arms)} arms, {args.eval_samples} samples, "
          f"discovery seed {args.eval_seed}")

    references = {
        "g2_continuous": ("g2_native_fp32", "continuous",
                          DECISION["reference_rates_seed20260915"]
                          ["g2_150k_paper_decentralized_continuous"]),
        "g2_2bit": ("g2_native_fp32", "2bit",
                    DECISION["reference_rates_seed20260915"]
                    ["g2_150k_paper_decentralized_2bit"]),
        "r0_continuous": ("r0_native_fp32", "continuous",
                          DECISION["reference_rates_seed20260915"]
                          ["r0_500k_paper_decentralized_continuous"]),
        "r0_2bit": ("r0_native_fp32", "2bit",
                    DECISION["reference_rates_seed20260915"]
                    ["r0_500k_paper_decentralized_2bit"]),
    }

    records, controls = run_evaluation(
        nets, simulator, config_eval, arms, books, device, args.eval_samples,
        args.batch_size, args.eval_seed, n_ris, n_elem)
    arrays, table = summarise(arms, records, n_ap, n_ris)
    controls = check_controls(
        controls, table,
        references if args.eval_seed == 20260915 else None)
    discovery = decide(arms, arrays, table)
    matched = matched_budget_table(table, arrays)

    confirmation = None
    if args.confirm_seed:
        if args.confirm_seed in (args.eval_seed, args.calib_seed):
            raise SystemExit("the confirmation seed must be fresh")
        print(f"[confirm] replaying the declared rules on seed {args.confirm_seed}")
        fixed = {
            "d2_baseline_arm": discovery["d2_matched_budget"].get("baseline_arm"),
            "d4_executable_arm":
                discovery["d4_same_backbone_interface"].get("executable_arm"),
            "d4_latent_arm":
                discovery["d4_same_backbone_interface"].get("latent_arm"),
        }
        records_c, controls_c = run_evaluation(
            nets, simulator, config_eval, arms, books, device, args.eval_samples,
            args.batch_size, args.confirm_seed, n_ris, n_elem)
        arrays_c, table_c = summarise(arms, records_c, n_ap, n_ris)
        controls_c = check_controls(controls_c, table_c, None)
        confirmation = {
            "seed": args.confirm_seed,
            "arms_fixed_by_discovery": fixed,
            "controls": controls_c,
            "decisions": decide(arms, arrays_c, table_c, fixed=fixed),
            "arms": table_c,
            "matched_bits": matched_budget_table(table_c, arrays_c),
        }

    source_files = ["variants.py", "model.py", "evaluate.py", "rates.py",
                    "simulation.py", "experiments/message_codec_sweep.py",
                    "experiments/matched_budget_codec.py"]
    config_blob = {
        "experiment": "e07_matched_budget_codec",
        "purpose": ("matched-bit-budget comparison of the G2 executable-phase "
                    "message against the R0 latent message, both frozen, both "
                    "scored on identical channel draws"),
        "decision": DECISION,
        "checkpoints": {
            "g2": {"run_dir": os.path.abspath(args.g2_run),
                   "checkpoint": os.path.abspath(args.g2_ckpt),
                   "sha256": sha256_file(args.g2_ckpt), "config": config_g2},
            "r0": {"run_dir": os.path.abspath(args.r0_run),
                   "checkpoint": os.path.abspath(args.r0_ckpt),
                   "sha256": sha256_file(args.r0_ckpt), "config": config_r0,
                   "hosted_as": {"arch": host["arch"],
                                 "consensus": host["consensus"],
                                 "note": ("r0's own _merge returns before the "
                                          "trace is filled, so the checkpoint "
                                          "is hosted under the shared-reduction "
                                          "arch that exposes the same latents; "
                                          "the r0 phase is recomputed with "
                                          "RIS_merge and the reference gate "
                                          "checks it against the E06 holdout")}},
        },
        "calibration": {"samples": args.calib_samples, "seed": args.calib_seed,
                        "clip_quantile": args.clip_quantile,
                        "codebook_seed": args.codebook_seed},
        "evaluation": {"samples": args.eval_samples, "batch_size": args.batch_size,
                       "discovery_seed": args.eval_seed,
                       "confirmation_seed": args.confirm_seed,
                       "cluster_note": ("one set of UE locations per batch, so "
                                        "the batch mean is the independent unit")},
        "topology": {"A": n_ap, "R": n_ris, "N": n_elem, "M": config_r0["M"],
                     "K": config_r0["K"], "ap_ris_pairs": n_ap * n_ris},
        "codec_grid": {"phase_bits": args.phase_bits,
                       "energy_bits": args.energy_bits,
                       "latent_bits": args.latent_bits, "vq_bits": args.vq_bits,
                       "mag_bits": args.mag_bits, "resid_bits": args.resid_bits,
                       "pair_scale_bits": args.pair_scale_bits},
        "payload_policy": {
            "counted_per_message": [
                "G2: N quantized phase indices plus one quantized energy index",
                "R0: 4N quantized latent indices, or the reinterpreted "
                "interface's own indices",
            ],
            "pre_shared_not_counted": [
                "every Lloyd-Max codebook and its log clipping range",
                "the calibrated uniform phase-grid rotation offsets",
                "the 2-D Cartesian VQ codebooks",
                "W_reduce and b, which both sides hold as part of the policy",
            ],
            "generous_r0_accounting": (
                "the last 2N coordinates of the R0 latent are the "
                "RIS-independent fe_AP block, so bits_per_pair_generous charges "
                "2N(1 + 1/R) reals per AP-RIS pair instead of 4N; every gate "
                "uses the generous count"),
            "fp32_accounting": "an unquantized field costs one fp32 per real",
        },
        "codebooks": books.describe(),
        "provenance": {
            "git_head": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "git_status": git_output("status", "--porcelain"),
            "source_sha256": {f: sha256_file(f) for f in source_files
                              if os.path.exists(f)},
        },
        "command": " ".join([sys.executable, "-m",
                             "experiments.matched_budget_codec", *sys.argv[1:]]),
    }
    results_blob = {
        "discovery": {"seed": args.eval_seed, "arms": table,
                      "matched_bits": matched, "controls": controls,
                      "decisions": discovery},
        "confirmation": confirmation,
        "overall_passed": bool(
            controls["passed"] and discovery["passed"]
            and (confirmation is None
                 or (confirmation["controls"]["passed"]
                     and confirmation["decisions"]["passed"]))),
    }

    with open(os.path.join(args.out_dir, "matched_budget_config.json"), "w",
              encoding="utf-8") as handle:
        json.dump(config_blob, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "matched_budget_results.json"), "w",
              encoding="utf-8") as handle:
        json.dump(results_blob, handle, indent=2, sort_keys=True)
    flat = {f"discovery__{name}__{phase}": arrays[name][phase]
            for name in arrays for phase in ACTUATIONS}
    if confirmation is not None:
        flat.update({f"confirm__{name}__{phase}": arrays_c[name][phase]
                     for name in arrays_c for phase in ACTUATIONS})
    np.savez(os.path.join(args.out_dir, "matched_budget_paired.npz"), **flat)

    rows = ["name,backbone,family,bits_per_pair,bits_per_pair_generous,"
            "continuous,continuous_sem,continuous_ci95_low,continuous_ci95_high,"
            "actuation_2bit,delta_vs_own_fp32,delta_vs_own_fp32_ci95_low,"
            "delta_vs_g2_native,delta_vs_g2_native_ci95_low,on_pareto_frontier"]
    for name in sorted(table, key=lambda n: (table[n]["backbone"],
                                             table[n]["bits_per_pair_generous"], n)):
        entry = table[name]
        rows.append(",".join(str(x) for x in (
            name, entry["backbone"], entry["family"], entry["bits_per_pair"],
            entry["bits_per_pair_generous"],
            f"{entry['continuous']['mean']:.6f}",
            f"{entry['continuous']['sem']:.6f}",
            f"{entry['continuous']['ci95_low']:.6f}",
            f"{entry['continuous']['ci95_high']:.6f}",
            f"{entry['2bit']['mean']:.6f}",
            f"{entry['vs_own_fp32_continuous']['mean']:.6f}",
            f"{entry['vs_own_fp32_continuous']['ci95_low']:.6f}",
            f"{entry['vs_g2_native_continuous']['mean']:.6f}",
            f"{entry['vs_g2_native_continuous']['ci95_low']:.6f}",
            int(entry["on_pareto_frontier"]))))
    with open(os.path.join(args.out_dir, "matched_budget_frontier.csv"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    with open(os.path.join(args.out_dir, "matched_budget_run_meta.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"wall_clock_seconds": round(time.time() - started, 2),
                   "hostname": platform.node(), "python": sys.version.split()[0],
                   "torch": torch.__version__, "numpy": np.__version__,
                   "device": str(device),
                   "cuda_device": (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None)},
                  handle, indent=2, sort_keys=True)

    # ---------------------------------------------------------------- console
    def report(label, blob, controls_blob):
        print(f"\n[{label}] controls")
        for gate, ok in controls_blob["gates"].items():
            print(f"  {gate:28s} {'pass' if ok else 'FAIL'}")
        print(f"  message identity          {controls_blob['max_message_identity_error']:.3e}")
        print(f"  message round trip (rel)  {controls_blob['max_message_relative_error']:.3e}")
        print(f"  conditioned phase error   {controls_blob['max_conditioned_phase_error']:.3e}")
        print(f"  identity rate error       {controls_blob['max_identity_rate_error']:.3e}")
        print(f"  unit modulus              {controls_blob['max_unit_modulus_error']:.3e}")
        if "reference_reproduction_error" in controls_blob:
            for key, value in controls_blob["reference_reproduction_error"].items():
                print(f"  reference {key:16s} {value:.3e}")
        print(f"[{label}] decisions")
        for key in ("d1_signaling_advantage", "d2_matched_budget",
                    "d3_interface_robustness", "d4_same_backbone_interface"):
            entry = blob[key]
            if "mean" in entry:
                print(f"  {key:28s} {entry['mean']:+8.4f} "
                      f"[{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}] "
                      f"{'pass' if entry['passed'] else 'FAIL'}"
                      + (f"  vs {entry.get('baseline_arm') or entry.get('latent_arm')}"
                         if entry.get("baseline_arm") or entry.get("latent_arm") else ""))
            else:
                print(f"  {key:28s} {entry.get('note')}")
        print(f"  bit equivalence           {blob['bit_equivalence_point']}")

    report(f"discovery seed {args.eval_seed}", discovery, controls)
    if confirmation is not None:
        report(f"confirmation seed {args.confirm_seed}",
               confirmation["decisions"], confirmation["controls"])

    print(f"\n[done] {args.out_dir}  overall_passed="
          f"{results_blob['overall_passed']}")
    if not controls["passed"] or (confirmation is not None
                                  and not confirmation["controls"]["passed"]):
        raise SystemExit("CONTROL FAILURE: see matched_budget_results.json")


if __name__ == "__main__":
    main()
