"""Sanity-check surface-type decode and dump raw facet normals for sample faces."""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
np.set_printoptions(precision=4, suppress=True, linewidth=120)

with h5py.File(path, "r") as f:
    b = f["0"]
    v1 = np.asarray(b["V_1"])
    v2 = np.asarray(b["V_2"])
    a3 = np.asarray(b["A_3_idx"])

face_col = a3[:, 1]; facet_col = a3[:, 0]
normals = v2[:, :3]

print("raw V_1 col4 (surface type code) unique values:")
u = np.unique(v1[:, 4])
print(f"  {u}")
print(f"  *11 = {np.round(u*11,3)}   *11-1 = {np.round(u*11-1,3)}\n")

stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, 5)
counts = np.bincount(face_col, minlength=v1.shape[0])

# pick the cylinder face with the most facets, and a plane with many facets
TYPE = {0:"plane",1:"cyl",2:"cone",3:"sphere",4:"torus",5:"other"}
for want_type, label in [(1, "CYLINDER"), (0, "PLANE")]:
    cand = [fc for fc in range(v1.shape[0]) if stype[fc] == want_type]
    cand.sort(key=lambda fc: counts[fc], reverse=True)
    if not cand:
        print(f"no {label} faces"); continue
    fc = cand[0]
    m = facet_col[face_col == fc]
    fn = normals[face_col == fc]
    print(f"=== {label} face {fc}: {len(fn)} facets, type_code={v1[fc,4]:.4f} ===")
    print(f"  first 8 facet normals (raw V_2 cols0-2):")
    print(fn[:8])
    print(f"  per-column std over its facets = {fn.std(axis=0)}")
    print(f"  mean normal = {fn.mean(axis=0)}  ||mean||={np.linalg.norm(fn.mean(axis=0)):.4f}")
    print(f"  decoded(2v-1) first 4:\n{2*fn[:4]-1}\n")
