import glob
import os
import re
import unittest

import numpy as np
import pandas as pd


def final_extremas(run_dir):
    """Last row of the last run's temporals/tempExtremas in run_dir"""
    path = os.path.join(run_dir, "temporals", "tempExtremas")
    with open(path) as f:
        lines = f.read().splitlines()
    # The file repeats its header for every run appended to it
    last_header = max(i for i, l in enumerate(lines) if l == lines[0])
    data = pd.read_csv(path, skiprows=last_header).apply(
        pd.to_numeric, errors="coerce"
    )
    return data.iloc[-1]


@unittest.skipUnless(os.path.exists("spray-sdc-2d.inp"), "not the spray_SDC test")
class SpraySDCTestCase(unittest.TestCase):
    """spray_SDC: second order in time for the spray coupling"""

    # Extrema of the gas state at the final time; without spray_SDC these
    # converge at first order
    fields = [
        "max_temp",
        "min_density",
        "min_x_velocity",
        "max_rho.Y(NC7H16)",
        "max_rho.Y(NC10H22)",
    ]
    min_rate = 1.75

    def test_temporal_order(self):
        """Do the extrema converge at second order in dt?"""
        runs = sorted(
            glob.glob("n[0-9]*"), key=lambda d: int(re.sub(r"\D", "", d))
        )
        self.assertGreaterEqual(len(runs), 3, "need at least three dt")
        rows = [final_extremas(d) for d in runs]
        times = [r["time"] for r in rows]
        np.testing.assert_allclose(times, times[0], rtol=1e-10)

        for field in self.fields:
            vals = np.array([r[field] for r in rows])
            self.assertFalse(np.isnan(vals).any(), field)
            # Self-convergence: differences between successive dt
            diffs = np.abs(np.diff(vals))
            rates = np.log2(diffs[:-1] / diffs[1:])
            print(f"{field}: differences {diffs}, rates {rates}")
            self.assertGreater(
                rates[-1],
                self.min_rate,
                f"{field} converges at {rates[-1]:.2f} in time",
            )


if __name__ == "__main__":
    unittest.main()
