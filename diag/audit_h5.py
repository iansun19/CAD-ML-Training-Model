"""
audit_h5.py — find unused per-face geometry in the MFCAD++ H5.

Goes beyond `dataset.py --inspect-h5`: for the first batch of a split it prints
the real shape + dtype + value range of every dataset, and decodes the columns of
V_1 / V_2 / edge-value arrays so we can see exactly what signal the loader is
leaving on the table.

Usage:
    python diag/audit_h5.py MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5
"""

import argparse
import h5py
import numpy as np


def _summ(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return "empty"
    if np.issubdtype(a.dtype, np.number):
        return (f"min={a.min():.4g} max={a.max():.4g} "
                f"mean={a.mean():.4g} uniq={min(np.unique(a).size, 9999)}")
    return f"non-numeric sample={a.flat[0]!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path")
    args = ap.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        batch_keys = list(f.keys())
        print(f"file: {args.h5_path}")
        print(f"batches: {len(batch_keys)}  first={batch_keys[0]!r}\n")

        b = f[batch_keys[0]]
        print(f"=== datasets in {batch_keys[0]} ===")
        for k in sorted(b.keys()):
            d = b[k]
            line = f"{k:16s} shape={str(d.shape):18s} dtype={str(d.dtype):10s}"
            # only summarize reasonably small / numeric arrays fully
            try:
                if d.size <= 5_000_000:
                    line += "  " + _summ(d[()])
            except Exception as e:
                line += f"  (summary failed: {e})"
            print(line)

        # --- decode V_1 columns (B-rep face features) ---
        if "V_1" in b:
            v1 = np.asarray(b["V_1"][: min(b["V_1"].shape[0], 200000)])
            print(f"\n=== V_1 column-by-column (first {len(v1)} faces) ===")
            print("docs say: [surface area, centroid x, y, z, surface type]")
            for c in range(v1.shape[1]):
                col = v1[:, c]
                print(f"  col {c}: {_summ(col)}")

        # --- decode V_2 columns (mesh-level features) ---
        if "V_2" in b:
            v2 = np.asarray(b["V_2"][: min(b["V_2"].shape[0], 200000)])
            print(f"\n=== V_2 column-by-column (first {len(v2)} facets) ===")
            print("docs say: [normal x, y, z, d coefficient]")
            for c in range(v2.shape[1]):
                print(f"  col {c}: {_summ(v2[:, c])}")

        # --- edge value arrays: are they meaningful or all ones? ---
        for ek in ("A_1_values", "E_1_values", "E_2_values", "E_3_values"):
            if ek in b:
                ev = np.asarray(b[ek][()])
                print(f"\n=== {ek} === shape={ev.shape}  {_summ(ev)}")

        # --- labels present in this batch ---
        if "labels" in b:
            lab = np.asarray(b["labels"][()])
            uniq, cnt = np.unique(lab, return_counts=True)
            print(f"\n=== labels in {batch_keys[0]} ===")
            print(f"classes present: {uniq.tolist()}")
            print(f"counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
