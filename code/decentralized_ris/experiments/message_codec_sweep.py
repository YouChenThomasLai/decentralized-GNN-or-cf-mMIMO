"""Consensus-before-quantization sweep of the AP->CPU RIS message interface.

The repo's existing 2-bit evaluation quantizes the *final* RIS phase, after the
CPU has already aggregated fp32 AP messages, so it says nothing about fronthaul
cost.  This module puts encoder / quantizer / decoder exactly where the wire is:
each AP l produces its pre-projection logits

    z_{l,r,n} = (W_reduce v_{l,r} + b / A)_n  in R^2,

encodes them into a finite number of bits, and the CPU decodes z_hat and only
then runs the parameter-free consensus

    theta_{r,n} = Pi( sum_l z_hat_{l,r,n} ).

Codec families (bits per AP-RIS pair, with `inf` accounted as fp32 = 32 bit):

    phase_only   N b_p                    direction only (the R1-Shared limit)
    pair_mag     N b_p + b_s              direction + one pair scale s_{l,r}
    polar        N (b_p + b_m)            direction + per-element magnitude
    progressive  N b_p + b_s + N b_eps    pair scale + per-element residual
    vq           N b_vq                   2-D Cartesian VQ, radius and angle free
    vq_gain      N b_vq + b_s             per-pair gain + 2-D VQ on z / s_hat

The two VQ families are *reconstruction-aware*: their codebooks minimise the
Euclidean reconstruction error of z on the calibration split.  They are neither
rate-aware nor task-aware - no sum-rate signal enters the codebook fit.

Every clipping range, scalar codebook, phase-grid offset and vector codebook is
estimated on the calibration split only and is then frozen and pre-shared
between the APs and the CPU, so none of them is charged to the per-message
payload; `payload_bits` counts only what an AP actually puts on the wire for one
AP-RIS pair of one channel realisation.

Run from `code/decentralized_ris`:

    python -m experiments.message_codec_sweep --run_dir <run> --ckpt <ckpt.pt>
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from evaluate import (
    build_model,
    resolve_device,
    seed_everything,
    temporary_seed,
)
from model import load_checkpoint
from rates import RatePrecompute
from simulation import ChannelSimulator
from variants import _unit_from_pairs, circular_consensus

FP32_BITS = 32
DEGENERATE = 1e-12
LOG_FLOOR = 1e-30

# Student-t 97.5% quantiles; the pilot uses 50 batch clusters -> df = 49.
_T95 = {39: 2.022690911734728, 49: 2.009575234489209, 99: 1.9842169515086827,
        399: 1.9659334533247917}


def t95(df):
    return _T95.get(df, 1.959963984540054)


# --------------------------------------------------------------------- helpers
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:                                      # not a checkout / no git
        return ""


def project(vec):
    """Unit projection with the repo's degenerate-pair fallback (phase 0)."""
    norm = vec.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(vec)
    fallback[..., 0] = 1.0
    return torch.where(norm > DEGENERATE, vec / norm.clamp(min=DEGENERATE), fallback)


def aggregate(z_hat):
    """CPU side: sum the decoded AP messages, then project. (B, A, R, N, 2)."""
    return project(z_hat.sum(dim=1))


# ------------------------------------------------------------------ quantizers
def lloyd_max_1d(x, levels, iters=400, tol=1e-13):
    """Deterministic Lloyd-Max codebook for a 1-D sample, float64 on the CPU."""
    x = x.detach().double().cpu().reshape(-1)
    if levels == 1:
        return x.mean().reshape(1)
    probs = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    centroids = torch.quantile(x, probs)
    if not bool(torch.all(centroids[1:] > centroids[:-1])):
        centroids = torch.linspace(float(x.min()), float(x.max()), levels,
                                   dtype=torch.float64)
    for _ in range(iters):
        edges = (centroids[1:] + centroids[:-1]) / 2
        index = torch.bucketize(x, edges)
        count = torch.bincount(index, minlength=levels).double()
        total = torch.zeros(levels, dtype=torch.float64).scatter_add_(0, index, x)
        updated = torch.where(count > 0, total / count.clamp(min=1.0), centroids)
        updated, _ = torch.sort(updated)
        shift = float((updated - centroids).abs().max())
        centroids = updated
        if shift < tol:
            break
    return centroids


class LogScalarQuantizer:
    """Lloyd-Max quantizer on log-magnitude, calibrated once and pre-shared.

    Magnitudes span orders of magnitude, so the codebook is fitted in the log
    domain; reconstruction is `exp(centroid)`.  The clipping range is the
    calibration `clip_q` percentile pair of the log values and only affects the
    fit - at encode time nearest-centroid assignment already maps the tails onto
    the extreme centroids.
    """

    def __init__(self, bits, log_levels, log_clip):
        self.bits = int(bits)
        self.log_levels = log_levels.double()
        self.log_clip = (float(log_clip[0]), float(log_clip[1]))
        self._edges = None
        self._levels = None

    @classmethod
    def fit(cls, values, bits, clip_q=(0.001, 0.999)):
        log_values = values.detach().double().cpu().reshape(-1).clamp_min(LOG_FLOOR).log()
        lo = float(torch.quantile(log_values, clip_q[0]))
        hi = float(torch.quantile(log_values, clip_q[1]))
        clipped = log_values.clamp(lo, hi)
        return cls(bits, lloyd_max_1d(clipped, 2 ** bits), (lo, hi))

    def to(self, device):
        self._edges = ((self.log_levels[1:] + self.log_levels[:-1]) / 2).to(
            device=device, dtype=torch.float32)
        self._levels = self.log_levels.exp().to(device=device, dtype=torch.float32)
        return self

    def quantize(self, magnitude):
        log_value = magnitude.clamp_min(LOG_FLOOR).log().clamp(*self.log_clip)
        if self._edges.numel() == 0:                       # single-level codebook
            index = torch.zeros_like(log_value, dtype=torch.long)
        else:
            index = torch.bucketize(log_value.contiguous(), self._edges)
        return self._levels[index]

    def describe(self):
        return {"bits": self.bits, "levels": self.log_levels.exp().tolist(),
                "log_clip": list(self.log_clip)}


def fit_phase_offset(angles, bits, trials=256):
    """Calibrated rotation of the uniform 2^bits phase grid (chord distortion)."""
    angles = angles.detach().double().cpu().reshape(-1)
    step = 2 * math.pi / (2 ** bits)
    offsets = torch.arange(trials, dtype=torch.float64) * step / trials
    best_offset, best_cost = 0.0, float("inf")
    for offset in offsets.tolist():
        error = angles - (torch.round((angles - offset) / step) * step + offset)
        cost = float((1.0 - torch.cos(error)).mean())
        if cost < best_cost - 1e-15:
            best_offset, best_cost = offset, cost
    return best_offset, best_cost


def phase_grid_quantize(z, bits, offset):
    """Nearest point of the calibrated uniform phase grid, as a unit 2-vector."""
    angle = torch.atan2(z[..., 1], z[..., 0])
    step = 2 * math.pi / (2 ** bits)
    quantized = torch.round((angle - offset) / step) * step + offset
    return torch.stack((quantized.cos(), quantized.sin()), dim=-1)


def kmeans_2d(x, clusters, seed, iters=400, tol=1e-13):
    """Deterministic k-means++ / Lloyd in R^2, float64 on the CPU."""
    x = x.detach().double().cpu().reshape(-1, 2)
    generator = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(x.shape[0], (1,), generator=generator))
    centers = [x[first]]
    closest = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(clusters - 1):
        weight = closest.clamp_min(0)
        if float(weight.sum()) <= 0:
            pick = int(torch.randint(x.shape[0], (1,), generator=generator))
        else:
            pick = int(torch.multinomial(weight / weight.sum(), 1, generator=generator))
        centers.append(x[pick])
        closest = torch.minimum(closest, ((x - centers[-1]) ** 2).sum(dim=1))
    book = torch.stack(centers)
    energy = (x ** 2).sum(dim=1, keepdim=True)
    for _ in range(iters):
        index = (energy - 2 * x @ book.T + (book ** 2).sum(dim=1)).argmin(dim=1)
        count = torch.bincount(index, minlength=clusters).double()
        total = torch.zeros_like(book).index_add_(0, index, x)
        updated = torch.where(count[:, None] > 0, total / count[:, None].clamp(min=1.0), book)
        shift = float((updated - book).abs().max())
        book = updated
        if shift < tol:
            break
    order = torch.argsort(torch.atan2(book[:, 1], book[:, 0]))
    return book[order]


class VectorQuantizer:
    """Reconstruction-aware 2-D Cartesian codebook; angle and radius co-vary."""

    def __init__(self, bits, book):
        self.bits = int(bits)
        self.book = book.double()
        self._book = None

    @classmethod
    def fit(cls, samples, bits, seed):
        return cls(bits, kmeans_2d(samples, 2 ** bits, seed))

    def to(self, device):
        self._book = self.book.to(device=device, dtype=torch.float32)
        return self

    def quantize(self, z):
        shape = z.shape
        flat = z.reshape(-1, 2)
        index = torch.cdist(flat, self._book).argmin(dim=1)
        return self._book[index].reshape(shape)

    def describe(self):
        return {"bits": self.bits, "codewords": self.book.tolist()}


# ----------------------------------------------------------------- codec specs
class CodecSpec:
    """One operating point: what is transmitted, and how many bits it costs."""

    def __init__(self, family, b_phase=None, b_scale=None, b_mag=None,
                 b_resid=None, b_vq=None):
        self.family = family
        self.b_phase = b_phase
        self.b_scale = b_scale
        self.b_mag = b_mag
        self.b_resid = b_resid
        self.b_vq = b_vq

    @staticmethod
    def _width(bits):
        """`None` means the field is sent unquantized; charge it as one fp32."""
        return FP32_BITS if bits is None else int(bits)

    @staticmethod
    def _tag(bits):
        return "inf" if bits is None else str(bits)

    def payload(self, n_elem):
        """Per AP-RIS pair payload, itemised from the codec configuration."""
        items = {}
        if self.family in ("phase_only", "pair_mag", "polar", "progressive"):
            items["direction"] = n_elem * self._width(self.b_phase)
        if self.family in ("pair_mag", "progressive", "vq_gain"):
            items["pair_scale"] = self._width(self.b_scale)
        if self.family == "polar":
            items["element_magnitude"] = n_elem * self._width(self.b_mag)
        if self.family == "progressive" and self.b_resid:
            items["element_residual"] = n_elem * int(self.b_resid)
        if self.family in ("vq", "vq_gain"):
            items["vq_index"] = n_elem * self._width(self.b_vq)
        return items

    def bits_per_pair(self, n_elem):
        return sum(self.payload(n_elem).values())

    @property
    def name(self):
        if self.family == "phase_only":
            return f"phase_only_bp{self._tag(self.b_phase)}"
        if self.family == "pair_mag":
            return f"pair_mag_bp{self._tag(self.b_phase)}_bs{self._tag(self.b_scale)}"
        if self.family == "polar":
            return f"polar_bp{self._tag(self.b_phase)}_bm{self._tag(self.b_mag)}"
        if self.family == "progressive":
            return (f"progressive_bp{self._tag(self.b_phase)}"
                    f"_bs{self._tag(self.b_scale)}_be{int(self.b_resid)}")
        if self.family == "vq":
            return f"vq_cart_b{self._tag(self.b_vq)}"
        return f"vq_cart_gain_b{self._tag(self.b_vq)}_bs{self._tag(self.b_scale)}"

    def describe(self, n_elem, n_ap, n_ris):
        payload = self.payload(n_elem)
        bits = sum(payload.values())
        return {
            "name": self.name,
            "family": self.family,
            "b_phase": self.b_phase, "b_scale": self.b_scale, "b_mag": self.b_mag,
            "b_resid": self.b_resid, "b_vq": self.b_vq,
            "payload_bits_per_pair": payload,
            "bits_per_pair": bits,
            "bits_system": bits * n_ap * n_ris,
            "is_fp32_reference": self._has_fp32_field(),
            "reconstruction_aware_codebook": self.family in ("vq", "vq_gain"),
            "rate_aware_codebook": False,
            "task_aware_codebook": False,
        }

    def _has_fp32_field(self):
        fields = {
            "phase_only": (self.b_phase,),
            "pair_mag": (self.b_phase, self.b_scale),
            "polar": (self.b_phase, self.b_mag),
            "progressive": (self.b_phase, self.b_scale),
            "vq": (self.b_vq,),
            "vq_gain": (self.b_vq, self.b_scale),
        }[self.family]
        return any(f is None for f in fields)


class Codebooks:
    """Everything estimated on the calibration split; frozen and pre-shared."""

    def __init__(self):
        self.phase_offset = {}
        self.phase_offset_cost = {}
        self.magnitude = {}
        self.scale = {}
        self.residual = {}
        self.vector = {}
        self.vector_gain = {}
        self.stats = {}

    def to(self, device):
        for table in (self.magnitude, self.scale, self.residual,
                      self.vector, self.vector_gain):
            for quantizer in table.values():
                quantizer.to(device)
        return self

    def describe(self):
        return {
            "phase_grid_offset_rad": {str(k): v for k, v in self.phase_offset.items()},
            "phase_grid_chord_distortion": {
                str(k): v for k, v in self.phase_offset_cost.items()},
            "element_magnitude": {str(k): q.describe() for k, q in self.magnitude.items()},
            "pair_scale": {str(k): q.describe() for k, q in self.scale.items()},
            "residual_ratio": {f"bs{k[0]}_be{k[1]}": q.describe()
                               for k, q in self.residual.items()},
            "vq_cartesian": {str(k): q.describe() for k, q in self.vector.items()},
            "vq_cartesian_gain": {f"b{k[0]}_bs{k[1]}": q.describe()
                                  for k, q in self.vector_gain.items()},
            "calibration_statistics": self.stats,
        }


def encode_decode(spec, z, books):
    """AP-side encode + CPU-side decode of one batch of messages.

    z: (B, A, R, N, 2) pre-projection logits.  Returns z_hat of the same shape.
    """
    magnitude = z.norm(dim=-1)                                       # (B, A, R, N)

    if spec.family == "vq":
        if spec.b_vq is None:
            return z
        return books.vector[spec.b_vq].quantize(z)

    if spec.family == "vq_gain":
        scale = magnitude.mean(dim=3)                                # (B, A, R)
        scale_hat = (scale if spec.b_scale is None
                     else books.scale[spec.b_scale].quantize(scale))
        normalized = z / scale_hat[..., None, None].clamp_min(LOG_FLOOR)
        if spec.b_vq is None:
            return normalized * scale_hat[..., None, None]
        book = books.vector_gain[(spec.b_vq, spec.b_scale)]
        return book.quantize(normalized) * scale_hat[..., None, None]

    direction = (project(z) if spec.b_phase is None
                 else phase_grid_quantize(z, spec.b_phase,
                                          books.phase_offset[spec.b_phase]))

    if spec.family == "phase_only":
        return direction

    if spec.family == "polar":
        magnitude_hat = (magnitude if spec.b_mag is None
                         else books.magnitude[spec.b_mag].quantize(magnitude))
        return magnitude_hat[..., None] * direction

    scale = magnitude.mean(dim=3)                                    # (B, A, R)
    scale_hat = (scale if spec.b_scale is None
                 else books.scale[spec.b_scale].quantize(scale))

    if spec.family == "pair_mag" or not spec.b_resid:
        # `progressive` with b_eps = 0 transmits no residual, so it takes this
        # exact code path and is bit-identical to the `pair_mag` endpoint.
        return scale_hat[..., None, None] * direction

    ratio = magnitude / scale_hat[..., None].clamp_min(LOG_FLOOR)
    ratio_hat = books.residual[(spec.b_scale, spec.b_resid)].quantize(ratio)
    return (scale_hat[..., None] * ratio_hat)[..., None] * direction


def build_specs(phase_bits, scale_bits, mag_bits, resid_bits, vq_bits, vq_gain_scale):
    specs = [CodecSpec("phase_only", b_phase=b) for b in phase_bits]
    specs += [CodecSpec("pair_mag", b_phase=p, b_scale=s)
              for p in phase_bits for s in scale_bits]
    specs += [CodecSpec("polar", b_phase=p, b_mag=m)
              for p in phase_bits for m in mag_bits]
    specs += [CodecSpec("progressive", b_phase=p, b_scale=s, b_resid=e)
              for p in phase_bits for s in scale_bits for e in resid_bits]
    specs += [CodecSpec("vq", b_vq=b) for b in vq_bits]
    specs += [CodecSpec("vq_gain", b_vq=b, b_scale=vq_gain_scale) for b in vq_bits]
    return specs


# --------------------------------------------------------------- forward hooks
def message_pass(net, simulator, users_per_ap, threshold, device):
    """One evaluation draw: channels, beamformer, per-AP logits and latents."""
    features, edges, masks, direct, _ = simulator.training_batch(
        users_per_ap, threshold, threshold)
    del features, edges, masks, direct
    features, edges, masks, direct = simulator.decentralized_batch(
        users_per_ap, threshold, threshold, regenerate_channels=False)
    features = [tensor.to(device) for tensor in features]
    edges = [tensor.to(device) for tensor in edges]
    direct = [tensor.to(device) for tensor in direct]
    trace = {}
    beamformer, theta_shared = net.decentralized(features, edges, masks, direct, trace)
    return beamformer, theta_shared, trace["z_pairs"], trace["latents"]


def native_interfaces(net, latents, z, n_ris, n_elem):
    """The four unquantized reference interfaces, through the library code."""
    flat = torch.cat((z[..., 0], z[..., 1]), dim=-1)                 # (B, A, R, 2N)
    proposals = _unit_from_pairs(flat, n_ris, n_elem)
    active = torch.ones(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
    theta_shared, _ = circular_consensus(proposals, active)
    weights = z.norm(dim=-1).mean(dim=3)
    theta_apmag, _ = circular_consensus(proposals, active, None, 1.0, weights)
    return {
        "r0_fp32": net.RIS_merge(latents),
        "r0c_fp32": net.RIS_merge.forward_commuted(latents),
        "r1_shared_fp32": theta_shared,
        "r1_ap_ris_mag_fp32": theta_apmag,
    }


# ------------------------------------------------------------------ statistics
def paired_stats(values, reference):
    difference = values - reference
    n = len(difference)
    mean = float(difference.mean())
    sem = float(difference.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = t95(n - 1) * sem if n > 1 else float("nan")
    return {"mean": mean, "sem": sem,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "t": mean / sem if sem > 0 else float("inf"), "clusters": n}


def pareto_front(points):
    """points: list of (bits, mean, name) -> names on the upper-left frontier."""
    front = []
    for bits, mean, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if not front or mean > front[-1][1] + 0.0:
            front.append((bits, mean, name))
    return [name for _, _, name in front]


# ------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="run directory holding summary.json")
    parser.add_argument("--ckpt", required=True, help="checkpoint file to evaluate")
    parser.add_argument("--calib_samples", type=int, default=800)
    parser.add_argument("--calib_seed", type=int, default=20260919)
    parser.add_argument("--eval_samples", type=int, default=400)
    parser.add_argument("--eval_seed", type=int, default=20260920)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase_bits", default="1,2,3,inf")
    parser.add_argument("--scale_bits", default="2,4,8,inf")
    parser.add_argument("--mag_bits", default="1,2,4,8,inf")
    parser.add_argument("--resid_bits", default="0,1,2,4")
    parser.add_argument("--vq_bits", default="1,2,3,4,5,6")
    parser.add_argument("--vq_gain_scale", type=int, default=8)
    parser.add_argument("--clip_quantile", type=float, default=0.001)
    parser.add_argument("--codebook_seed", type=int, default=20260919)
    parser.add_argument("--out_dir",
                        default="../../artifacts/decentralized_ris/e07_message_codec")
    args = parser.parse_args()

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

    phase_bits = parse_bits(args.phase_bits)
    scale_bits = parse_bits(args.scale_bits)
    mag_bits = parse_bits(args.mag_bits)
    resid_bits = parse_bits(args.resid_bits, allow_zero=True)
    vq_bits = parse_bits(args.vq_bits)

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.run_dir, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"].copy()
    if config["arch"] != "r0":
        raise SystemExit(f"expected an r0 checkpoint, got arch={config['arch']}")
    if args.batch_size != config["batch_size"]:
        print(f"[warn] overriding checkpoint batch_size {config['batch_size']} "
              f"with {args.batch_size}")
    config["batch_size"] = args.batch_size
    for samples, label in ((args.calib_samples, "calib"), (args.eval_samples, "eval")):
        if samples <= 0 or samples % args.batch_size:
            raise SystemExit(f"{label}_samples must be a positive multiple of batch_size")

    device = resolve_device(args.device)
    n_ap, n_ris, n_elem = config["AP"], config["L"], config["N"]
    threshold = config.get("assoc_threshold", 0.1)

    # The action path is irrelevant to the backbone; r1_shared is the arch whose
    # trace exposes both the per-AP latents and the pre-projection logits z.
    host = config.copy()
    host["arch"], host["consensus"] = "r1_shared", "equal"

    seed_everything(config["seed"])
    simulator = ChannelSimulator(config["M"], config["N"], config["L"],
                                 config["batch_size"], n_ap=config["AP"])
    net = build_model(host, simulator, device)
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    # ------------------------------------------------------------ calibration
    print(f"[calib] {args.calib_samples} samples, seed {args.calib_seed}")
    calibration = []
    with torch.no_grad(), temporary_seed(args.calib_seed):
        for _ in range(args.calib_samples // args.batch_size):
            _, _, z, _ = message_pass(net, simulator, config["K"], threshold, device)
            calibration.append(z.detach().to("cpu", torch.float64))
    calib_z = torch.cat(calibration, dim=0)                          # (Bc, A, R, N, 2)
    calib_m = calib_z.norm(dim=-1)
    calib_s = calib_m.mean(dim=3)
    calib_angle = torch.atan2(calib_z[..., 1], calib_z[..., 0])

    books = Codebooks()
    books.stats = {
        "draws": int(calib_z.shape[0]),
        "messages": int(calib_z.shape[0] * n_ap * n_ris),
        "element_magnitude": {
            "min": float(calib_m.min()), "max": float(calib_m.max()),
            "mean": float(calib_m.mean()), "median": float(calib_m.median()),
            "q001": float(torch.quantile(calib_m.reshape(-1), 0.001)),
            "q999": float(torch.quantile(calib_m.reshape(-1), 0.999)),
            "dynamic_range_db": float(20 * math.log10(
                float(calib_m.max()) / max(float(calib_m.min()), LOG_FLOOR))),
        },
        "pair_scale": {
            "min": float(calib_s.min()), "max": float(calib_s.max()),
            "mean": float(calib_s.mean()), "median": float(calib_s.median()),
        },
        "within_pair_relative_spread": float(
            (calib_m / calib_s[..., None]).std().item()),
        "angle_resultant_length": float(
            torch.hypot(calib_angle.cos().mean(), calib_angle.sin().mean())),
    }
    print("[calib] element magnitude: "
          f"min={books.stats['element_magnitude']['min']:.4g} "
          f"median={books.stats['element_magnitude']['median']:.4g} "
          f"max={books.stats['element_magnitude']['max']:.4g} "
          f"({books.stats['element_magnitude']['dynamic_range_db']:.1f} dB)")
    print("[calib] angular mean resultant length "
          f"{books.stats['angle_resultant_length']:.5f} (0 = uniform directions)")

    clip = (args.clip_quantile, 1.0 - args.clip_quantile)
    for bits in phase_bits:
        if bits is None:
            continue
        offset, cost = fit_phase_offset(calib_angle, bits)
        books.phase_offset[bits] = offset
        books.phase_offset_cost[bits] = cost
        print(f"[calib] phase grid b_p={bits}: offset={offset:.6f} rad, "
              f"chord distortion={cost:.6f}")
    for bits in mag_bits:
        if bits is not None:
            books.magnitude[bits] = LogScalarQuantizer.fit(calib_m, bits, clip)
    for bits in scale_bits:
        if bits is not None:
            books.scale[bits] = LogScalarQuantizer.fit(calib_s, bits, clip)
    books.to(device)

    calib_m_dev = calib_m.to(device=device, dtype=torch.float32)
    calib_s_dev = calib_s.to(device=device, dtype=torch.float32)
    for b_scale in scale_bits:
        scale_hat = (calib_s_dev if b_scale is None
                     else books.scale[b_scale].quantize(calib_s_dev))
        ratio = calib_m_dev / scale_hat[..., None].clamp_min(LOG_FLOOR)
        for b_resid in resid_bits:
            if b_resid:
                books.residual[(b_scale, b_resid)] = LogScalarQuantizer.fit(
                    ratio, b_resid, clip)

    for bits in vq_bits:
        if bits is None:
            continue
        books.vector[bits] = VectorQuantizer.fit(calib_z, bits, args.codebook_seed)
        gain = (calib_s_dev if args.vq_gain_scale is None
                else books.scale[args.vq_gain_scale].quantize(calib_s_dev))
        normalized = calib_z.to(device=device, dtype=torch.float32) \
            / gain[..., None, None].clamp_min(LOG_FLOOR)
        books.vector_gain[(bits, args.vq_gain_scale)] = VectorQuantizer.fit(
            normalized, bits, args.codebook_seed + 1)
        print(f"[calib] fitted 2-D VQ b={bits} "
              f"({2 ** bits} codewords, reconstruction-aware)")
    books.to(device)

    del calibration, calib_z, calib_m, calib_s, calib_angle, calib_m_dev, calib_s_dev

    # ------------------------------------------------------------- evaluation
    specs = build_specs(phase_bits, scale_bits, mag_bits, resid_bits,
                        vq_bits, args.vq_gain_scale)
    references = ("r0_fp32", "r0c_fp32", "r1_shared_fp32", "r1_ap_ris_mag_fp32")
    arm_names = list(references) + [spec.name for spec in specs]
    records = {name: [] for name in arm_names}
    distortion = {spec.name: [0.0, 0.0] for spec in specs}
    controls = {
        "r0_vs_r0c_max_theta_error": 0.0,
        "native_shared_vs_recomputed_max_error": 0.0,
        "unit_modulus_error": 0.0,
        "fast_rate_vs_reference_max_error": 0.0,
        "inf_codec_identity": {},
        "progressive_zero_residual_bit_exact": True,
    }
    inf_pairs = {codec: reference for codec, reference in (
        ("polar_bpinf_bminf", "r0c_fp32"),
        ("phase_only_bpinf", "r1_shared_fp32"),
        ("pair_mag_bpinf_bsinf", "r1_ap_ris_mag_fp32"),
    ) if codec in arm_names}
    for key in inf_pairs:
        controls["inf_codec_identity"][key] = {"max_theta_error": 0.0,
                                               "max_rate_error": 0.0}

    print(f"[eval] {args.eval_samples} samples, seed {args.eval_seed}, "
          f"{len(arm_names)} arms")
    with torch.no_grad(), temporary_seed(args.eval_seed):
        for batch_index in range(args.eval_samples // args.batch_size):
            beamformer, theta_shared, z, latents = message_pass(
                net, simulator, config["K"], threshold, device)
            pre = RatePrecompute(simulator, device)
            natives = native_interfaces(net, latents, z, n_ris, n_elem)

            controls["r0_vs_r0c_max_theta_error"] = max(
                controls["r0_vs_r0c_max_theta_error"],
                float((natives["r0_fp32"] - natives["r0c_fp32"]).abs().max()))
            controls["native_shared_vs_recomputed_max_error"] = max(
                controls["native_shared_vs_recomputed_max_error"],
                float((natives["r1_shared_fp32"] - theta_shared).abs().max()))

            thetas = dict(natives)
            z_hats = {}
            for spec in specs:
                z_hat = encode_decode(spec, z, books)
                z_hats[spec.name] = z_hat
                thetas[spec.name] = aggregate(z_hat)
                error = (z_hat - z).pow(2).sum(dim=-1).mean()
                distortion[spec.name][0] += float(error)
                distortion[spec.name][1] += float(z.pow(2).sum(dim=-1).mean())

            rates = {}
            for name, theta in thetas.items():
                controls["unit_modulus_error"] = max(
                    controls["unit_modulus_error"],
                    float((theta.norm(dim=-1) - 1.0).abs().max()))
                value = float(pre.sum_rate(beamformer, theta).mean())
                rates[name] = value
                records[name].append(value)

            for codec_name, reference in inf_pairs.items():
                entry = controls["inf_codec_identity"][codec_name]
                entry["max_theta_error"] = max(
                    entry["max_theta_error"],
                    float((thetas[codec_name] - thetas[reference]).abs().max()))
                entry["max_rate_error"] = max(
                    entry["max_rate_error"], abs(rates[codec_name] - rates[reference]))

            for spec in specs:
                if spec.family != "progressive" or spec.b_resid:
                    continue
                twin = CodecSpec("pair_mag", b_phase=spec.b_phase,
                                 b_scale=spec.b_scale).name
                if not torch.equal(z_hats[spec.name], z_hats[twin]):
                    controls["progressive_zero_residual_bit_exact"] = False

            # Cross-check the cached fast rate against the reference implementation.
            _, reference_rate, _ = simulator.loss(
                beamformer, natives["r0_fp32"], device)
            controls["fast_rate_vs_reference_max_error"] = max(
                controls["fast_rate_vs_reference_max_error"],
                abs(float(reference_rate) - rates["r0_fp32"]))

            if (batch_index + 1) % 10 == 0:
                print(f"[eval] cluster {batch_index + 1}/"
                      f"{args.eval_samples // args.batch_size} "
                      f"r0={np.mean(records['r0_fp32']):.5f}")

    arrays = {name: np.asarray(values, dtype=np.float64)
              for name, values in records.items()}
    clusters = len(arrays["r0_fp32"])

    spec_by_name = {spec.name: spec for spec in specs}
    reference_bits = {
        "r0_fp32": 4 * n_elem * FP32_BITS,
        "r0c_fp32": 2 * n_elem * FP32_BITS,
        "r1_shared_fp32": n_elem * FP32_BITS,
        "r1_ap_ris_mag_fp32": n_elem * FP32_BITS + FP32_BITS,
    }
    reference_payload = {
        "r0_fp32": {"phase_feature": 4 * n_elem * FP32_BITS},
        "r0c_fp32": {"cartesian_logits": 2 * n_elem * FP32_BITS},
        "r1_shared_fp32": {"direction": n_elem * FP32_BITS},
        "r1_ap_ris_mag_fp32": {"direction": n_elem * FP32_BITS,
                               "pair_scale": FP32_BITS},
    }

    arms = {}
    for name in arm_names:
        values = arrays[name]
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / math.sqrt(clusters))
        half = t95(clusters - 1) * sem
        if name in references:
            entry = {"family": "reference", "bits_per_pair": reference_bits[name],
                     "payload_bits_per_pair": reference_payload[name],
                     "bits_system": reference_bits[name] * n_ap * n_ris,
                     "is_fp32_reference": True,
                     "reconstruction_aware_codebook": False,
                     "rate_aware_codebook": False, "task_aware_codebook": False}
        else:
            entry = spec_by_name[name].describe(n_elem, n_ap, n_ris)
            entry.pop("name")
            total, energy = distortion[name]
            entry["relative_logit_mse"] = total / max(energy, 1e-300)
        entry.update({
            "decentralized_sum_rate": mean,
            "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half,
            "vs_r0c_fp32": paired_stats(values, arrays["r0c_fp32"]),
            "vs_r1_shared_fp32": paired_stats(values, arrays["r1_shared_fp32"]),
        })
        arms[name] = entry

    # Matched-budget comparisons: every pair of arms sharing a bits/pair value.
    by_bits = {}
    for name, entry in arms.items():
        by_bits.setdefault(entry["bits_per_pair"], []).append(name)
    matched = {}
    for bits, names in sorted(by_bits.items()):
        if len(names) < 2:
            continue
        best = max(names, key=lambda n: arms[n]["decentralized_sum_rate"])
        matched[str(bits)] = {
            "arms": sorted(names),
            "best": best,
            "vs_best": {n: paired_stats(arrays[n], arrays[best])
                        for n in sorted(names) if n != best},
        }

    finite = [(entry["bits_per_pair"], entry["decentralized_sum_rate"], name)
              for name, entry in arms.items()]
    frontier = pareto_front(finite)
    for name in arms:
        arms[name]["on_pareto_frontier"] = name in frontier

    controls["passed"] = bool(
        controls["r0_vs_r0c_max_theta_error"] <= 1e-5
        and controls["unit_modulus_error"] < 1e-6
        and controls["progressive_zero_residual_bit_exact"]
        and all(v["max_theta_error"] <= 1e-6
                for v in controls["inf_codec_identity"].values())
    )
    controls["thresholds"] = {
        "r0_vs_r0c_max_theta_error": 1e-5,
        "unit_modulus_error": 1e-6,
        "inf_codec_identity_max_theta_error": 1e-6,
    }

    source_files = ["variants.py", "model.py", "evaluate.py", "rates.py",
                    "simulation.py", "experiments/message_codec_sweep.py"]
    provenance = {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_diff_stat": git_output("diff", "--stat"),
        "git_status": git_output("status", "--porcelain"),
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "source_sha256": {f: sha256_file(f) for f in source_files if os.path.exists(f)},
    }

    payload_policy = {
        "counted_per_message": [
            "quantized direction indices (N per AP-RIS pair)",
            "quantized pair scale index (1 per AP-RIS pair, when the codec sends one)",
            "quantized per-element magnitude or residual indices (N, when sent)",
            "2-D VQ codeword indices (N per AP-RIS pair, for the VQ families)",
        ],
        "pre_shared_not_counted": [
            "all Lloyd-Max scalar codebooks and their log clipping ranges",
            "the calibrated uniform phase-grid rotation offsets",
            "the 2-D Cartesian VQ codebooks",
            "W_reduce and b, which both sides already hold as part of the policy",
        ],
        "note": ("Every codebook is fitted once on the calibration split and then "
                 "frozen, so it is a one-off configuration exchange rather than "
                 "per-message side information. No codec here needs a per-message "
                 "scale, range or codebook identifier; if one did, its bits would "
                 "have to be added to payload_bits_per_pair."),
        "fp32_accounting": ("`inf` fields are charged as one fp32 (32 bit) real per "
                            "transmitted scalar, using the angle-aware "
                            "serialisation: a direction costs one real, not two. "
                            "This makes polar(inf, inf) = 2N fp32 = R0c and "
                            "r0_fp32 = 4N fp32, matching the repo's payload table."),
    }

    config_blob = {
        "experiment": "message_codec_pilot",
        "purpose": ("bits vs decentralized sum-rate frontier for the AP->CPU RIS "
                    "message, with encoder/quantizer/decoder placed after each AP "
                    "forms z_{l,r,n} and before CPU aggregation"),
        "checkpoint_config": config,
        "host_arch": {"arch": host["arch"], "consensus": host["consensus"],
                      "note": "backbone and latents are identical across the "
                              "shared-reduction arches; only aggregation differs"},
        "training_seed": config["seed"],
        "calibration": {"samples": args.calib_samples, "seed": args.calib_seed,
                        "clip_quantile": args.clip_quantile,
                        "codebook_seed": args.codebook_seed},
        "evaluation": {"samples": args.eval_samples, "seed": args.eval_seed,
                       "batch_size": args.batch_size,
                       "batch_clusters": clusters,
                       "cluster_note": ("ChannelSimulator draws one set of UE "
                                        "locations per batch, so the batch mean is "
                                        "the independent unit")},
        "topology": {"A": n_ap, "R": n_ris, "N": n_elem, "M": config["M"],
                     "K": config["K"], "ap_ris_pairs": n_ap * n_ris},
        "codec_grid": {"phase_bits": args.phase_bits, "scale_bits": args.scale_bits,
                       "mag_bits": args.mag_bits, "resid_bits": args.resid_bits,
                       "vq_bits": args.vq_bits, "vq_gain_scale": args.vq_gain_scale},
        "payload_policy": payload_policy,
        "codebooks": books.describe(),
        "provenance": provenance,
        "command": " ".join([sys.executable, "-m", "experiments.message_codec_sweep",
                             *sys.argv[1:]]),
    }

    results_blob = {
        "arms": arms,
        "matched_bits": matched,
        "pareto_frontier": frontier,
        "controls": controls,
        "clusters": clusters,
    }

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_blob, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results_blob, handle, indent=2, sort_keys=True)
    np.savez(os.path.join(args.out_dir, "paired.npz"), **arrays)

    rows = ["name,family,bits_per_pair,bits_system,decentralized_sum_rate,sem,"
            "ci95_low,ci95_high,delta_vs_r0c,delta_sem,delta_ci95_low,"
            "delta_ci95_high,relative_logit_mse,on_pareto_frontier"]
    for name in sorted(arms, key=lambda n: (arms[n]["bits_per_pair"], n)):
        entry = arms[name]
        delta = entry["vs_r0c_fp32"]
        rows.append(",".join(str(x) for x in (
            name, entry["family"], entry["bits_per_pair"], entry["bits_system"],
            f"{entry['decentralized_sum_rate']:.6f}", f"{entry['sem']:.6f}",
            f"{entry['ci95_low']:.6f}", f"{entry['ci95_high']:.6f}",
            f"{delta['mean']:.6f}", f"{delta['sem']:.6f}",
            f"{delta['ci95_low']:.6f}", f"{delta['ci95_high']:.6f}",
            f"{entry.get('relative_logit_mse', float('nan')):.6e}",
            int(entry["on_pareto_frontier"]))))
    with open(os.path.join(args.out_dir, "frontier.csv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")

    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "wall_clock_seconds": round(time.time() - started, 2),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
        }, handle, indent=2, sort_keys=True)

    print("\n[controls]")
    print(f"  R0 vs R0c max theta error      {controls['r0_vs_r0c_max_theta_error']:.3e} "
          f"(<= 1e-5)")
    print(f"  unit-modulus error             {controls['unit_modulus_error']:.3e} "
          f"(< 1e-6)")
    print(f"  fast rate vs reference loss    "
          f"{controls['fast_rate_vs_reference_max_error']:.3e}")
    print(f"  native shared vs recomputed    "
          f"{controls['native_shared_vs_recomputed_max_error']:.3e}")
    for codec_name, entry in controls["inf_codec_identity"].items():
        print(f"  inf codec {codec_name:24s} theta {entry['max_theta_error']:.3e} "
              f"rate {entry['max_rate_error']:.3e}")
    print(f"  progressive b_eps=0 bit-exact  "
          f"{controls['progressive_zero_residual_bit_exact']}")
    print(f"  ALL CONTROLS PASSED: {controls['passed']}")

    print("\n[frontier] Pareto-optimal operating points")
    for name in sorted(frontier, key=lambda n: arms[n]["bits_per_pair"]):
        entry = arms[name]
        print(f"  {entry['bits_per_pair']:>6d} bits/pair "
              f"({entry['bits_system']:>6d} system)  "
              f"{entry['decentralized_sum_rate']:8.5f} "
              f"+-{entry['sem']:.5f}  {name}")
    print(f"\n[done] {args.out_dir}")
    if not controls["passed"]:
        raise SystemExit("CONTROL FAILURE: see results.json controls block")


if __name__ == "__main__":
    main()
