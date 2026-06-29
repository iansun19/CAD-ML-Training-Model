"""Does per-face facet-normal variance separate FLAT vs CURVED faces?

Uses DECODED normals (2v-1 then per-facet L2-normalize), groups facets to faces
via A_3, and buckets faces by their geometric surface type from V_1 (ground-truth
flatness: plane=flat, cylinder/cone/sphere/torus=curved). Reports an angular
spread metric and component std so we can tell whether the earlier "inverted"
reading was a metric/sign artifact or genuine scrambling.
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
np.set_printoptions(precision=4, suppress=True, linewidth=120)

NAME = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus", 5: "other"}


def decode_unit(n_raw):
    """[0,1]->[-1,1] then L2-normalize per facet (guard near-zero)."""
    n = 2.0 * n_raw - 1.0
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(norm, 1e-6, None)


ang_by_type = {t: [] for t in range(6)}
std_by_type = {t: [] for t in range(6)}

with h5py.File(path, "r") as f:
    keys = list(f.keys())[:N_BATCHES]
    for k in keys:
        b = f[k]
        v1 = np.asarray(b["V_1"])
        v2 = np.asarray(b["V_2"])
        a3 = np.asarray(b["A_3_idx"])
        face_col = a3[:, 1]
        stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, 5)
        units = decode_unit(v2[:, :3])
        nf = v1.shape[0]
        for face in range(nf):
            m = face_col == face
            if not m.any():
                continue
            fn = units[m]
            mean = fn.mean(axis=0)
            mn = np.linalg.norm(mean)
            mean_unit = mean / max(mn, 1e-6)
            # angular spread: mean angle (deg) between each facet normal & face mean
            cos = np.clip(fn @ mean_unit, -1, 1)
            ang = np.degrees(np.arccos(cos)).mean()
            ang_by_type[stype[face]].append(ang)
            std_by_type[stype[face]].append(fn.std(axis=0).mean())

print(f"batches scanned: {len(keys)}\n")
print(f"{'type':10s} {'nfaces':>7s} {'angSpread_deg(median)':>22s} "
      f"{'angSpread_deg(mean)':>20s} {'compStd(mean)':>14s}")
flat_ang, curved_ang = [], []
for t in range(6):
    if ang_by_type[t]:
        a = np.array(ang_by_type[t]); s = np.array(std_by_type[t])
        print(f"{NAME[t]:10s} {len(a):7d} {np.median(a):22.3f} "
              f"{a.mean():20.3f} {s.mean():14.4f}")
        if t == 0:
            flat_ang.append(a)
        else:
            curved_ang.append(a)

flat = np.concatenate(flat_ang) if flat_ang else np.array([])
curved = np.concatenate(curved_ang) if curved_ang else np.array([])
print("\n--- FLAT (plane) vs CURVED (cyl/cone/sphere/torus) angular spread ---")
if flat.size:
    print(f"  flat  : n={flat.size:6d} median={np.median(flat):.3f} "
          f"mean={flat.mean():.3f} p90={np.percentile(flat,90):.3f}")
if curved.size:
    print(f"  curved: n={curved.size:6d} median={np.median(curved):.3f} "
          f"mean={curved.mean():.3f} p90={np.percentile(curved,90):.3f}")
if flat.size and curved.size:
    print(f"  => curved/flat median ratio = {np.median(curved)/max(np.median(flat),1e-9):.2f}x")
