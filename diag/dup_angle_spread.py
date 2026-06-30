"""
dup_angle_spread.py — for every multi-edge face pair (>1 topo edge), how much do the
per-edge dihedral angles vary within the pair? Determines whether the loader's
angle-selection among duplicate edges can be a simple representative (they're equal)
or needs a real tiebreak (they differ).

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/dup_angle_spread.py
"""
import os
import numpy as np
import h5py

REGEN = "MFCAD++_dataset/hierarchical_graphs_regen"
SPLITS = ["training_MFCAD++.h5", "val_MFCAD++.h5", "test_MFCAD++.h5"]


def main():
    spreads = []
    n_multi = 0
    for fname in SPLITS:
        with h5py.File(os.path.join(REGEN, fname), "r") as f:
            for bk in f.keys():
                b = f[bk]
                idx = b["idx"][()]
                V1n = b["V_1"].shape[0]
                A1 = b["A_1_idx"][()]
                AV = b["A_1_values"][()]
                base = int(idx[0, 0])
                for mi in range(len(idx)):
                    s = int(idx[mi, 0]) - base
                    e = (int(idx[mi + 1, 0]) - base) if mi + 1 < len(idx) else V1n
                    m = (A1[:, 0] >= s) & (A1[:, 0] < e) & (A1[:, 1] >= s) & (A1[:, 1] < e)
                    rows = A1[m] - s
                    av = AV[m]
                    per = {}
                    for (u, v), ang in zip(rows.tolist(), av.tolist()):
                        if u == v:
                            continue
                        c = (u, v) if u < v else (v, u)
                        if (u, v) == c:
                            per.setdefault(c, []).append(float(ang))
                    for c, angs in per.items():
                        if len(angs) > 1:
                            n_multi += 1
                            spreads.append(np.degrees(max(angs) - min(angs)))
    spreads = np.array(spreads)
    print(f"multi-edge pairs: {n_multi}")
    print(f"per-pair angle spread (deg): max={spreads.max():.4f} "
          f"mean={spreads.mean():.6f} median={np.median(spreads):.6f}")
    for thr in (0.01, 0.1, 0.5, 1.0, 5.0):
        print(f"  pairs with spread > {thr:>4} deg: {(spreads > thr).sum()}")


if __name__ == "__main__":
    main()
