"""Small checks for the topology controls and cross-AP CSI copy count."""

import json
from pathlib import Path

import numpy as np

from experiments.topology_screen import csi_values
from experiments.summarize_rate_matched import welch
from experiments.visibility_calibration import ring_layout
from simulation import ChannelSimulator


def main():
    masks = np.array([
        [[1, 1, 0], [1, 0, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ], dtype=bool)
    copies, values = csi_values(masks, 2, 4, 30)
    assert copies.tolist() == [4, 0]
    assert values.tolist() == [4 * 484, 0]

    root = Path(__file__).resolve().parents[1] / "experiments"
    for topology in ("t1", "t2", "vlow", "vhigh"):
        layout = json.loads((root / f"topology_{topology}.json").read_text())
        simulator = ChannelSimulator(2, 30, 4, 8, n_ap=5, layout=layout)
        np.testing.assert_allclose(simulator.ap_locations, layout["ap_locations"])
        np.testing.assert_allclose(simulator.base_stations[0].ris_locations,
                                   layout["ris_locations"])
    for name, radius in (("vlow", 140), ("vhigh", 350)):
        stored = json.loads((root / f"topology_{name}.json").read_text())
        rebuilt = ring_layout(name.upper(), radius, 5, 4, 100)
        assert stored["ap_locations"] == rebuilt["ap_locations"], name
        assert stored["ris_locations"] == rebuilt["ris_locations"], name
    for name, radius in (("mlow", 280), ("mhigh", 170)):
        stored = json.loads((root / f"topology_{name}.json").read_text())
        rebuilt = ring_layout(name.upper(), radius, 5, 4, 100)
        assert stored["ap_locations"] == rebuilt["ap_locations"], name
        simulator = ChannelSimulator(2, 30, 4, 8, n_ap=5, layout=stored)
        assert simulator.radius == stored["user_radius"], name
    assert ChannelSimulator(2, 30, 4, 8, n_ap=5).radius == 100

    left, right = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
    result = welch(left, right)
    assert abs(result["difference"] - 1.0) < 1e-12
    assert abs(result["se"] - np.sqrt(1.0 / 3.0)) < 1e-12

    print("topology screen checks passed")


if __name__ == "__main__":
    main()
