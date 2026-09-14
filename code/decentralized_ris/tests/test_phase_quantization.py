import torch

from rates import quantize_phase


def test_phase_quantization():
    theta = torch.tensor(
        [[[[0.9, 0.1], [-0.1, 0.9], [-0.9, -0.1], [0.1, -0.9]]]]
    )
    original = theta.clone()
    expected = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]]
    )

    mapped = quantize_phase(theta, num_bits=2)

    assert torch.equal(theta, original)
    assert mapped.data_ptr() != theta.data_ptr()
    assert torch.allclose(mapped, expected, atol=1e-6)


if __name__ == "__main__":
    test_phase_quantization()
