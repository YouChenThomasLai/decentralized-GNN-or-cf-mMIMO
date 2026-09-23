"""Numerical checks for the E07 matched-bit-budget message comparison."""

import math

import torch

from experiments.matched_budget_codec import (
    PerDimQuantizer,
    build_arms,
    quantize_angles,
)


N_ELEM = 30
N_RIS = 4


def arms_by_name(**overrides):
    settings = dict(phase_bits=[1, 2, 3, None], scale_bits=[4, 8, None],
                    latent_bits=[1, 2, None], vq_bits=[2, 3], mag_bits=[1, None],
                    resid_bits=[1], extra_scale_bits=[2, 4, 8, None])
    settings.update(overrides)
    arms = build_arms(N_ELEM, N_RIS, settings["phase_bits"], settings["scale_bits"],
                      settings["latent_bits"], settings["vq_bits"],
                      settings["mag_bits"], settings["resid_bits"],
                      settings["extra_scale_bits"])
    return {arm.name: arm for arm in arms}


def test_payload_matches_the_transmitted_fields():
    arms = arms_by_name()
    # G2 sends N angles plus one scalar; R0's latent is 4N reals.
    assert arms["g2_energy_bp2_bs8"].bits == N_ELEM * 2 + 8
    assert arms["g2_equal_bp2"].bits == N_ELEM * 2
    assert arms["g2_native_fp32"].bits == N_ELEM * 32 + 32
    assert arms["r0_native_fp32"].bits == 4 * N_ELEM * 32
    assert arms["r0c_native_fp32"].bits == 2 * N_ELEM * 32
    assert arms["r0_latent_b2"].bits == 4 * N_ELEM * 2
    assert arms["r0c_logit_b2"].bits == 2 * N_ELEM * 2
    assert arms["r0c_vqgain_b3_bs8"].bits == N_ELEM * 3 + 8


def test_generous_r0_accounting_drops_the_ris_independent_block():
    """The last 2N latent coordinates are shared by every RIS of one AP."""
    arms = arms_by_name()
    expected = 2 * N_ELEM * (N_RIS + 1) / N_RIS                      # 75 reals
    for bits, name in ((1, "r0_latent_b1"), (2, "r0_latent_b2"),
                       (32, "r0_native_fp32")):
        assert arms[name].generous == round(expected * bits), name
    # Only the latent family is re-counted; every other wire format is literal.
    for name in ("g2_energy_bp2_bs8", "r0c_logit_b2", "r0_pairmag_bp2_bs8"):
        assert arms[name].generous == arms[name].bits, name


def test_matched_budgets_exist_across_both_backbones():
    """The declared grids N*b_p and N*b_p + 8 must line up arm for arm."""
    arms = arms_by_name()
    for b_p in (1, 2, 3):
        budget = N_ELEM * b_p + 8
        names = {n for n, a in arms.items() if a.bits == budget}
        assert any(a.startswith("g2_") for a in names), budget
        assert any(a.startswith("r0") for a in names), budget


def test_phase_grid_quantization_lands_on_the_grid():
    angles = torch.linspace(-math.pi, math.pi, 4096, dtype=torch.float64)
    for bits in (1, 2, 3, 4, 6, 8):
        offset = 0.31
        step = 2 * math.pi / (2 ** bits)
        quantized = quantize_angles(angles, bits, offset)
        residual = (quantized - offset) / step
        assert float((residual - residual.round()).abs().max()) < 1e-9
        # Nearest-point property: no grid point is closer than the chosen one.
        error = (angles - quantized).abs()
        assert float(error.max()) <= step / 2 + 1e-9


def test_per_dim_quantizer_is_exact_when_every_value_is_a_level():
    generator = torch.Generator().manual_seed(0)
    levels = torch.tensor([-2.0, 0.5], dtype=torch.float64)
    values = levels[torch.randint(2, (64, 5), generator=generator)]
    quantizer = PerDimQuantizer.fit(values, 1).to(torch.device("cpu"))
    torch.testing.assert_close(quantizer.quantize(values.float()), values.float())


def test_per_dim_quantizer_uses_an_independent_codebook_per_coordinate():
    """A coordinate with a large offset must not be dragged by its neighbours."""
    generator = torch.Generator().manual_seed(1)
    values = torch.randn((512, 3), generator=generator, dtype=torch.float64)
    values[:, 2] += 100.0
    quantizer = PerDimQuantizer.fit(values, 3).to(torch.device("cpu"))
    decoded = quantizer.quantize(values.float())
    error = (decoded - values.float()).abs().mean(dim=0)
    # The shifted coordinate is quantized as accurately as the centred ones.
    assert float(error[2]) < 1.25 * float(error[:2].max())
    assert float(quantizer.levels[2].min()) > 90.0
    assert float(quantizer.levels[:2].max()) < 10.0


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"[ok] {name}")
