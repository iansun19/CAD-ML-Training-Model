"""Verify per-model facet/face ranges and A_3 face-index basis within a batch."""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
np.set_printoptions(precision=2, suppress=True, linewidth=140)

with h5py.File(path, "r") as f:
    b = f["0"]
    idx = np.asarray(b["idx"])           # [num_models, 2]
    v1n = b["V_1"].shape[0]
    v2n = b["V_2"].shape[0]
    a3 = np.asarray(b["A_3_idx"])        # [num_facets, 2] (facet, face)

print(f"V_1 rows (faces)={v1n}  V_2 rows (facets)={v2n}  models={len(idx)}")
print(f"idx array (col0=face start, col1=mesh/facet bound):\n{idx}\n")

base_face = int(idx[0, 0])
base_facet = int(idx[0, 1])
print(f"base_face={base_face} base_facet={base_facet}")

# For first 3 models, derive face & facet ranges, then check A_3 consistency
for mi in range(min(3, len(idx))):
    fstart = int(idx[mi, 0]) - base_face
    fend = int(idx[mi + 1, 0]) - base_face if mi + 1 < len(idx) else v1n
    cstart = int(idx[mi, 1]) - base_facet
    cend = int(idx[mi + 1, 1]) - base_facet if mi + 1 < len(idx) else v2n
    # A_3 rows whose face is in [fstart,fend)
    mface = (a3[:, 1] >= fstart) & (a3[:, 1] < fend)
    facets_for_faces = a3[mface, 0]
    print(f"\nmodel {mi}: faces[{fstart}:{fend}] ({fend-fstart})  "
          f"facets[{cstart}:{cend}] ({cend-cstart})")
    print(f"  A_3 rows w/ face in range: {mface.sum()}  "
          f"facet idx min={facets_for_faces.min()} max={facets_for_faces.max()}")
    print(f"  => facets-by-face range matches facet slice? "
          f"{facets_for_faces.min()==cstart and facets_for_faces.max()==cend-1}")
