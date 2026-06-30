"""
dup_edge_attribute.py — STEP ground-truth attribution for the flagged diff-convexity
duplicate face pairs (from dup_edge_scan.py -> dup_edge_cases.json).

For each flagged (split, pid, i, j) it re-reads the STEP part EXACTLY as
regen_dataset.read_model does (same face order via TopologyExplorer, same per-edge
midpoint normals, same arccos(n0.n1) angle, same sign-of-triple-product convexity
bucket), then lists every topo edge between faces i and j with its convexity bucket
AND its angle, side by side. This resolves the angle<->bucket attribution that the
H5 alone cannot (E_1/E_2/E_3 carry no per-edge angle).

Finally it answers the decision question: across all flagged pairs, does the
"keep concave" choice ever pick a DIFFERENT edge than "keep smaller-angle"?

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/dup_edge_attribute.py
"""
import json
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FORWARD
from OCC.Extend.TopologyUtils import TopologyExplorer

import sys
sys.path.insert(0, "diag")
from regen_dihedral_check import edge_midpnt_tangent, normal_on_face_at_point

BUCKET = {1: "convex", -1: "concave", 0: "smooth"}


def read_edges(path):
    """Replicate regen face order + per-edge (facepair, bucket, angle_deg)."""
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    fidx = {f: i for i, f in enumerate(faces)}
    edges = []   # (i, j, bucket_str, angle_deg)
    for edge in topo.edges():
        ef = list(topo.faces_from_edge(edge))
        if len(ef) != 2:
            continue
        i, j = fidx[ef[0]], fidx[ef[1]]
        if i == j:
            continue
        mid, tan = edge_midpnt_tangent(edge)
        ang = np.pi
        sgn = 0
        if mid is not None:
            n0 = normal_on_face_at_point(mid, ef[0])
            n1 = normal_on_face_at_point(mid, ef[1])
            if n0 is not None and n1 is not None:
                cos = np.clip(n0 @ n1 / (np.linalg.norm(n0) * np.linalg.norm(n1)), -1, 1)
                ang = float(np.arccos(cos))
                r2 = (np.dot(np.cross(n0, n1), tan) if edge.Orientation() == TopAbs_FORWARD
                      else np.dot(np.cross(n1, n0), tan))
                sgn = int(np.sign(r2))
        a, b = (i, j) if i < j else (j, i)
        edges.append((a, b, BUCKET[sgn], float(np.degrees(ang))))
    return edges


def main():
    with open("diag/dup_edge_cases.json") as f:
        cases = json.load(f)

    disagree = 0           # concave-choice edge != smaller-angle-choice edge
    concave_missing = 0    # flagged pair has NO concave edge in STEP (shouldn't happen)
    print(f"flagged diff-convexity pairs: {len(cases)}\n")
    print(f"{'split':5} {'pid':>7} {'pair':>9}   per-edge (bucket@angle_deg)")
    print("-" * 78)
    for c in cases:
        path = os.path.join("MFCAD++_dataset", "step", c["split"], f"{c['pid']}.step")
        if not os.path.isfile(path):
            print(f"{c['split']:5} {c['pid']:>7}  MISSING STEP {path}")
            continue
        try:
            edges = read_edges(path)
        except Exception as ex:
            print(f"{c['split']:5} {c['pid']:>7}  READ FAIL {ex}")
            continue
        a, b = c["i"], c["j"]
        mine = [(bk, ang) for (i, j, bk, ang) in edges if i == a and j == b]
        desc = "  ".join(f"{bk}@{ang:.1f}" for bk, ang in mine)
        print(f"{c['split']:5} {c['pid']:>7} {f'({a},{b})':>9}   {desc}")

        concave = [ang for bk, ang in mine if bk == "concave"]
        if not concave:
            concave_missing += 1
            continue
        # "keep concave": angle we'd assign = the concave edge's angle
        concave_ang = min(concave)   # if several concave, they're equal anyway
        # "keep smaller angle": angle of the globally smallest-angle edge
        smallest_ang = min(ang for _, ang in mine)
        smallest_bk = min(mine, key=lambda t: t[1])[0]
        # disagreement: the smaller-angle rule would select a NON-concave edge
        # whose angle is strictly smaller than the concave edge's angle
        if smallest_bk != "concave" and smallest_ang < concave_ang - 0.5:
            disagree += 1
            print(f"        ^ DISAGREE: smaller-angle picks {smallest_bk}@{smallest_ang:.1f} "
                  f"but concave is @{concave_ang:.1f}")

    print("\n================ DECISION ================")
    print(f"pairs where 'keep concave' and 'keep smaller-angle' pick DIFFERENT edges: "
          f"{disagree}")
    print(f"flagged pairs with no concave edge in STEP (unexpected): {concave_missing}")
    if disagree == 0:
        print("=> On this dataset the two rules never select a different edge "
              "(angles tie within each duplicate pair).")
        print("=> Implement the unambiguous rule anyway: ALWAYS keep the concave edge,")
        print("   regardless of angle magnitude (concave-first bucket precedence).")
    else:
        print("=> The rules DISAGREE somewhere. The rule MUST be: always keep the "
              "concave edge, never an angle-magnitude tiebreak.")


if __name__ == "__main__":
    main()
