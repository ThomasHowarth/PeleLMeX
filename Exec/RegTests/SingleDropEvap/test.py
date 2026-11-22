import glob
import os
import re
import unittest

import numpy as np


def read_plotfile(pltdir):
    """Time and {name: array} of a single-level plotfile (native doubles)"""
    with open(os.path.join(pltdir, "Header")) as f:
        f.readline()
        names = [f.readline().strip() for _ in range(int(f.readline()))]
        dim = int(f.readline())
        time = float(f.readline())
        assert int(f.readline()) == 0, "single-level plotfiles only"
        for _ in range(3):  # prob_lo, prob_hi, ref ratios
            f.readline()
        dom = [int(x) for x in re.findall(r"-?\d+", f.readline())]
    lo, hi = dom[:dim], dom[dim : 2 * dim]
    data = np.full([len(names)] + [h - l + 1 for l, h in zip(lo, hi)][::-1], np.nan)
    lev = os.path.join(pltdir, "Level_0")
    with open(os.path.join(lev, "Cell_H")) as f:
        fabs = re.findall(r"FabOnDisk: (\S+) (\d+)", f.read())
    for fname, offset in fabs:
        with open(os.path.join(lev, fname), "rb") as fb:
            fb.seek(int(offset))
            # FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((lo) (hi) (type)) ncomp
            hdr = fb.readline().decode()
            m = re.search(r"\(\(([-\d,]+)\) \(([-\d,]+)\) \([-\d,]+\)\) (\d+)", hdr)
            blo = [int(x) for x in m.group(1).split(",")]
            bhi = [int(x) for x in m.group(2).split(",")]
            shape = [int(m.group(3))] + [h - l + 1 for l, h in zip(blo, bhi)][::-1]
            arr = np.fromfile(fb, dtype="<f8", count=int(np.prod(shape)))
        box = tuple(slice(b - g, e - g + 1) for b, e, g in zip(blo, bhi, lo))
        data[(slice(None),) + box[::-1]] = arr.reshape(shape)
    return time, dict(zip(names, data))


class SpraySDCTestCase(unittest.TestCase):
    """spray_SDC: second order in time for the spray coupling"""

    # Gas fields at the final time; without spray_SDC these converge at
    # first order
    fields = ["temp", "density", "x_velocity", "y_velocity", "divu"]
    min_rate = 1.6

    def test_temporal_order(self):
        """Do the gas fields converge at second order in dt?"""
        runs = sorted(
            glob.glob("n[0-9]*"), key=lambda d: int(re.sub(r"\D", "", d))
        )
        self.assertGreaterEqual(len(runs), 3, "need at least three dt")
        # Last plotfile of each run (one run per directory)
        plts = [read_plotfile(sorted(glob.glob(f"{d}/plt*"))[-1]) for d in runs]
        times = [t for t, _ in plts]
        np.testing.assert_allclose(times, times[0], rtol=1e-10)

        for field in self.fields:
            vals = [p[field] for _, p in plts]
            self.assertFalse(any(np.isnan(v).any() for v in vals), field)
            # Self-convergence: L1 norms of the differences between
            # successive dt
            diffs = np.array([np.abs(a - b).mean() for a, b in zip(vals, vals[1:])])
            # dt halves from run to run: fit the rate over all the pairs,
            # since one pair alone can show any rate by chance at first order
            rate = -np.polyfit(np.arange(len(diffs)), np.log2(diffs), 1)[0]
            print(f"{field}: differences {diffs}, fitted rate {rate:.2f}")
            self.assertGreater(
                rate, self.min_rate, f"{field} converges at {rate:.2f} in time"
            )


if __name__ == "__main__":
    unittest.main()
