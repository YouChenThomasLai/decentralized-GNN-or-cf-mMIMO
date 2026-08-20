import torch

from utils_return_indivial_rates import discrete_mapping


def test_discrete_mapping():
    theta = torch.tensor(
        [[[[0.9, 0.1], [-0.1, 0.9], [-0.9, -0.1], [0.1, -0.9]]]]
    )
    original = theta.clone()
    expected = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]]
    )

    mapped = discrete_mapping(theta, num_bits=2)

    assert torch.equal(theta, original)
    assert mapped.data_ptr() != theta.data_ptr()
    assert torch.allclose(mapped, expected, atol=1e-6)


if __name__ == "__main__":
    test_discrete_mapping()
