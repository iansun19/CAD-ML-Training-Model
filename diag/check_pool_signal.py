"""Decisive check: does per-face facet-normal variance separate planar vs curved faces?

Groups facets to faces via A_3_idx, computes per-face normal std, and buckets by
V_1 surface type. If planar faces have ~0 std and curved faces have higher std,
the pooled variance signal is meaningful regardless of the exact normal encoding.
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
np.set_printoptions(precision=4, suppress=True, linewidth=120)

NUM_SURFACE_TYPES = 6
TYPE_NAME = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "other"}

with h5py.File(path, "r") as f:
    b = f["0"]
    v1 = np.asarray(b["V_1"])          # [F,5]
    v2 = np.asarray(b["V_2"])          # [M,4]
    a3 = np.asarray(b["A_3_idx"])      # [M,2] -> (facet_idx, face_idx)

num_faces = v1.shape[0]
facet_col, face_col = a3[:, 0], a3[:, 1]

# surface type per face (same decode as dataset.py)
stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, NUM_SURFACE_TYPES - 1)

# verify A_3 is a clean facet->face map: each facet appears once
print(f"A_3 rows={len(a3)}  unique facets={np.unique(facet_col).size} "
      f"(num_facets={v2.shape[0]})  unique faces={np.unique(face_col).size} "
      f"(num_faces={num_faces})")
print("=> one row per facet, col0=facet id, col1=parent face id\n")

# facets per face distribution
counts = np.bincount(face_col, minlength=num_faces)
print(f"facets/face: min={counts.min()} max={counts.max()} "
      f"mean={counts.mean():.1f}  faces with 0 facets={np.sum(counts == 0)}\n")

# per-face normal mean-length + component std, bucketed by surface type
normals = v2[:, :3]
per_type = {t: [] for t in range(NUM_SURFACE_TYPES)}
meanlen_type = {t: [] for t in range(NUM_SURFACE_TYPES)}
for face in range(num_faces):
    m = face_col == face
    if not m.any():
        continue
    fn = normals[m]
    # component-wise std averaged over xyz = scalar curvature signal
    comp_std = fn.std(axis=0).mean()
    per_type[stype[face]].append(comp_std)
    meanlen_type[stype[face]].append(np.linalg.norm(fn.mean(axis=0)))

print("per-surface-type facet-normal spread (mean over faces of that type):")
print(f"  {'type':10s} {'nfaces':>7s} {'meanNormalStd':>14s} {'meanVecLen':>11s}")
for t in range(NUM_SURFACE_TYPES):
    if per_type[t]:
        print(f"  {TYPE_NAME[t]:10s} {len(per_type[t]):7d} "
              f"{np.mean(per_type[t]):14.4f} {np.mean(meanlen_type[t]):11.4f}")
