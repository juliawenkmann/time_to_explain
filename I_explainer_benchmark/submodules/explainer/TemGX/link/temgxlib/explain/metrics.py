import numpy as np

def fidelity_minus(z_full, z_removed):
    return np.abs(z_full - z_removed)

def fidelity_plus(z_full, z_expl):
    return 1.0 - np.abs(z_expl - z_full)

def sparsity(num_edges_expl, num_edges_Lhop):
    return num_edges_expl / max(1, num_edges_Lhop)

def aufsc(points):
    pts = sorted(points, key=lambda x: x[0])
    area = 0.0
    for (s0, f0), (s1, f1) in zip(pts[:-1], pts[1:]):
        area += 0.5 * (f0 + f1) * (s1 - s0)
    return area