"""
dup_edge_scan.py — full-scale duplicate-edge audit of the regenerated H5.

A single B-rep face pair can meet along MORE THAN ONE topological edge. In the
regen schema each such topo edge appends its own (i,j)+(j,i) rows to A_1_idx
(with its own dihedral angle in A_1_values) and its own pair into exactly one of
E_1/E_2/E_3 (convex/concave/smooth). So one face pair can produce several A_1 rows
and can even land in more than one convexity bucket.

This script, reading ONLY the H5 (the same bytes the loader will read), reports for
every model in every split:
  * total undirected face pairs
  * "multi-edge" pairs   : a pair that meets along >1 topo edge
  * "diff-convexity" pairs: a pair whose topo edges fall in >1 convexity bucket
For each diff-convexity pair it records the bucket set and the per-edge angle list,
and dumps them to JSON for STEP ground-truth attribution (dup_edge_attribute.py).

Counting note: A_1 stores both directions, so each topo edge -> one (a,b) row and
one (b,a) row. #topo-edges(pair) = #rows equal to the directed key (a,b), a<b.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/dup_edge_scan.py
"""
import json
import os
import numpy as np
import h5py

REGEN = "MFCAD++_dataset/hierarchical_graphs_regen"
SPLITS = {"train": "training_MFCAD++.h5", "val": "val_MFCAD++.h5",
          "test": "test_MFCAD++.h5"}
OUT_JSON = "diag/dup_edge_cases.json"


def canon(u, v):
    return (u, v) if u < v else (v, u)


def bucket_set(idx, s, e):
    """Canonical (a,b), a<b, non-self-loop edges of one model from a bucket idx array."""
    if idx.size == 0:
        return set()
    m = (idx[:, 0] >= s) & (idx[:, 0] < e) & (idx[:, 1] >= s) & (idx[:, 1] < e)
    out = set()
    for u, v in (idx[m] - s):
        u, v = int(u), int(v)
        if u != v:
            out.add(canon(u, v))
    return out


def main():
    tot_models = 0
    tot_pairs = 0
    tot_multi = 0          # pairs meeting along >1 topo edge
    tot_diffcvx = 0        # pairs whose edges span >1 convexity bucket
    models_with_multi = 0
    models_with_diff = 0
    multi_same_bucket = 0  # multi-edge but all edges in the SAME bucket
    diff_cases = []        # detailed records for STEP attribution

    for split, fname in SPLITS.items():
        path = os.path.join(REGEN, fname)
        with h5py.File(path, "r") as f:
            s_models = s_multi = s_diff = 0
            for bk in f.keys():
                b = f[bk]
                idx = b["idx"][()]
                V1n = b["V_1"].shape[0]
                A1 = b["A_1_idx"][()]
                AV = b["A_1_values"][()]
                E1 = b["E_1_idx"][()]
                E2 = b["E_2_idx"][()]
                E3 = b["E_3_idx"][()]
                ids = b["CAD_model"][()]
                base = int(idx[0, 0])
                for mi in range(len(idx)):
                    s = int(idx[mi, 0]) - base
                    e = (int(idx[mi + 1, 0]) - base) if mi + 1 < len(idx) else V1n
                    pid = ids[mi]
                    pid = pid.decode() if isinstance(pid, bytes) else str(pid)
                    tot_models += 1
                    s_models += 1

                    m = (A1[:, 0] >= s) & (A1[:, 0] < e) & (A1[:, 1] >= s) & (A1[:, 1] < e)
                    rows = A1[m] - s
                    avals = AV[m]
                    e1 = bucket_set(E1, s, e)
                    e2 = bucket_set(E2, s, e)
                    e3 = bucket_set(E3, s, e)

                    # group directed rows by canonical pair, keep angles of the (a,b) dir
                    per = {}  # canon -> {"fwd_ang": [...], "n_rows": int}
                    for (u, v), ang in zip(rows.tolist(), avals.tolist()):
                        if u == v:
                            continue
                        c = canon(u, v)
                        d = per.setdefault(c, {"fwd_ang": [], "n_rows": 0})
                        d["n_rows"] += 1
                        if (u, v) == c:        # directed key (a,b), a<b: one per topo edge
                            d["fwd_ang"].append(float(ang))

                    for c, d in per.items():
                        tot_pairs += 1
                        n_edges = len(d["fwd_ang"])    # topo edges between this pair
                        buckets = []
                        if c in e1:
                            buckets.append("convex")
                        if c in e2:
                            buckets.append("concave")
                        if c in e3:
                            buckets.append("smooth")
                        if n_edges > 1:
                            tot_multi += 1
                            s_multi += 1
                            if len(buckets) <= 1:
                                multi_same_bucket += 1
                        if len(buckets) > 1:
                            tot_diffcvx += 1
                            s_diff += 1
                            diff_cases.append({
                                "split": split, "pid": pid,
                                "i": int(c[0]), "j": int(c[1]),
                                "buckets": buckets,
                                "n_edges": n_edges,
                                "angles_deg": [round(np.degrees(a), 3)
                                               for a in d["fwd_ang"]],
                            })
            models_with_multi += sum(1 for _ in range(0))  # placeholder (per-model below)
            print(f"[{split}] models={s_models} multi-edge-pairs={s_multi} "
                  f"diff-convexity-pairs={s_diff}", flush=True)

    print("\n================ FULL-SCALE DUPLICATE-EDGE SUMMARY ================")
    print(f"models scanned          : {tot_models}")
    print(f"undirected face pairs    : {tot_pairs}")
    print(f"multi-edge pairs (>1 topo edge): {tot_multi}  "
          f"({100*tot_multi/max(tot_pairs,1):.4f}% of pairs)")
    print(f"  of which all same bucket     : {multi_same_bucket}")
    print(f"  of which span >1 bucket      : {tot_diffcvx}")
    print(f"diff-convexity pairs     : {tot_diffcvx}  "
          f"({100*tot_diffcvx/max(tot_pairs,1):.5f}% of pairs)")
    pids_multi = {(c['split'], c['pid']) for c in diff_cases}
    print(f"models containing a diff-convexity pair: {len(pids_multi)}  "
          f"({100*len(pids_multi)/max(tot_models,1):.4f}% of models)")

    with open(OUT_JSON, "w") as f:
        json.dump(diff_cases, f, indent=1)
    print(f"\nwrote {len(diff_cases)} diff-convexity cases -> {OUT_JSON}")

    # quick angle-equality peek straight from H5 (per-edge angles within a pair)
    eqtol = 0.5  # deg
    n_equal = n_diff = 0
    for c in diff_cases:
        a = c["angles_deg"]
        if len(a) >= 2 and (max(a) - min(a)) > eqtol:
            n_diff += 1
        else:
            n_equal += 1
    print(f"\ndiff-convexity pairs whose per-edge angles are ~equal (<= {eqtol} deg "
          f"spread): {n_equal}; angles differ: {n_diff}")
    print("(STEP attribution needed to map each angle to its convex/concave edge.)")


if __name__ == "__main__":
    main()
