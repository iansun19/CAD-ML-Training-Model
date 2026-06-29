"""
audit_meshpool.py — Step 1 audit for the facet->face mesh-pooling feature.

Verifies, from the actual data (NOT the docs), the layout of:
  - V_2   : per-facet mesh features (normals + plane offset d)
  - A_3_* : facet<->face incidence (how facets map to parent B-rep faces)

so we can write pooling code against the real layout. Prints shapes, dtypes,
first rows, and runs sanity tests (unit-vector check on candidate normal cols,
incidence orientation check against V_1/V_2 row counts).

Usage:
    python diag/audit_meshpool.py MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5
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
    ap.add_argument("--batch", type=int, default=0, help="which batch group to audit")
    args = ap.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        batch_keys = list(f.keys())
        bk = batch_keys[args.batch]
        b = f[bk]
        print(f"file : {args.h5_path}")
        print(f"batches: {len(batch_keys)}  auditing={bk!r}\n")

        # --- which datasets exist that we care about ---
        print("=== relevant datasets present ===")
        for k in sorted(b.keys()):
            if k.startswith(("V_1", "V_2", "A_3", "A_1", "idx", "labels", "facet")):
                d = b[k]
                print(f"  {k:14s} shape={str(d.shape):20s} dtype={d.dtype}")
        print()

        v1 = np.asarray(b["V_1"])
        v2 = np.asarray(b["V_2"])
        num_faces = v1.shape[0]
        num_facets = v2.shape[0]
        print(f"num B-rep faces (V_1 rows) = {num_faces}")
        print(f"num mesh facets (V_2 rows) = {num_facets}")
        print(f"facets/faces ratio         = {num_facets / max(num_faces,1):.2f}\n")

        # --- V_2 layout + first rows ---
        print("=== V_2 first 10 rows ===")
        print(f"docs claim columns = [normal_x, normal_y, normal_z, d]")
        np.set_printoptions(precision=5, suppress=True, linewidth=120)
        print(v2[:10])
        print("\n=== V_2 column summaries ===")
        for c in range(v2.shape[1]):
            print(f"  col {c}: {_summ(v2[:, c])}")

        # --- unit-vector test: which 3 columns satisfy x^2+y^2+z^2 ~= 1 ? ---
        print("\n=== unit-norm test on column triples ===")
        ncol = v2.shape[1]
        from itertools import combinations
        for combo in combinations(range(ncol), 3):
            norms = np.sqrt((v2[:, list(combo)] ** 2).sum(axis=1))
            frac_unit = np.mean(np.abs(norms - 1.0) < 1e-3)
            print(f"  cols {combo}: ||.||  mean={norms.mean():.4f} "
                  f"std={norms.std():.4f}  frac(|n-1|<1e-3)={frac_unit:.3f}")

        # --- A_3 incidence: idx / values / shape ---
        print("\n=== A_3 (facet<->face incidence) ===")
        for suf in ("idx", "values", "shape"):
            key = f"A_3_{suf}"
            if key in b:
                d = np.asarray(b[key])
                print(f"  {key:12s} shape={str(d.shape):16s} dtype={d.dtype}")
        if "A_3_shape" in b:
            print(f"  A_3_shape value = {np.asarray(b['A_3_shape']).tolist()}")
        if "A_3_idx" in b:
            a3 = np.asarray(b["A_3_idx"])
            print(f"\n  A_3_idx first 12 rows:\n{a3[:12]}")
            print(f"\n  A_3_idx col 0: {_summ(a3[:, 0])}")
            print(f"  A_3_idx col 1: {_summ(a3[:, 1])}")
            # Which column indexes faces (range ~ num_faces) vs facets (~ num_facets)?
            c0max, c1max = a3[:, 0].max(), a3[:, 1].max()
            print(f"\n  col0 max={c0max} (num_faces-1={num_faces-1}, num_facets-1={num_facets-1})")
            print(f"  col1 max={c1max} (num_faces-1={num_faces-1}, num_facets-1={num_facets-1})")
            # count distinct entries on each side
            print(f"  distinct col0 = {np.unique(a3[:,0]).size}, distinct col1 = {np.unique(a3[:,1]).size}")
            # is the mapping many-facets-to-one-face? check counts per face-side
            if "A_3_values" in b:
                print(f"  A_3_values: {_summ(np.asarray(b['A_3_values']))}")
        if "facet_labels" in b:
            fl = np.asarray(b["facet_labels"])
            print(f"\n  facet_labels shape={fl.shape} {_summ(fl)}")


if __name__ == "__main__":
    main()
