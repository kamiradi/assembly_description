#!/usr/bin/env python3
"""Experiment with CoACD convex decomposition on hole meshes."""

import sys
from pathlib import Path

import coacd
import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {path}")
    return mesh


def decompose(mesh: trimesh.Trimesh, threshold: float = 0.05,
              preprocess_resolution: int = 50, preprocess_mode: str = "auto") -> list[trimesh.Trimesh]:
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold,
                            preprocess_mode=preprocess_mode,
                            preprocess_resolution=preprocess_resolution)
    return [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]


def batch() -> None:
    """Non-interactive decomposition for scripted parameter sweeps.

    Usage:
        coacd_collision.py batch <mesh.obj> <out_dir> <prefix> \
            <threshold> <preprocess_resolution> [preprocess_mode]

    Writes <out_dir>/<prefix>_convex_coll_NNN.obj (clearing any existing ones).
    """
    mesh_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    prefix = sys.argv[4]
    threshold = float(sys.argv[5])
    pre_res = int(sys.argv[6])
    pre_mode = sys.argv[7] if len(sys.argv) > 7 else "auto"

    try:
        coacd.set_log_level("error")
    except Exception:
        pass

    mesh = load_mesh(mesh_path)
    parts = decompose(mesh, threshold=threshold,
                      preprocess_resolution=pre_res, preprocess_mode=pre_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_convex_coll_*.obj"):
        old.unlink()
    for i, part in enumerate(parts):
        part.export(str(out_dir / f"{prefix}_convex_coll_{i:03d}.obj"))
    print(f"[batch] {mesh_path.name}: threshold={threshold} preprocess_resolution={pre_res} "
          f"mode={pre_mode} -> {len(parts)} parts written to {out_dir}")


def random_colors(n: int) -> list[list[int]]:
    rng = np.random.default_rng(42)
    colors = (rng.random((n, 3)) * 200 + 55).astype(int)
    return [[r, g, b, 180] for r, g, b in colors]


def save_decomposition(parts: list[trimesh.Trimesh], output_dir: Path, geom_name: str) -> list[Path]:
    """Save each convex part as an OBJ under output_dir/geom_name/, all in the original mesh frame."""
    geom_dir = output_dir / geom_name
    geom_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, part in enumerate(parts):
        out = geom_dir / f"{geom_name}_convex_coll_{i:03d}.obj"
        part.export(str(out))
        paths.append(out)
    print(f"  Saved {len(parts)} parts to {geom_dir}")
    return paths


def visualise(mesh_path: Path, threshold: float = 0.05) -> list[trimesh.Trimesh]:
    print(f"Loading {mesh_path} ...")
    original = load_mesh(mesh_path)
    print(f"  Vertices: {len(original.vertices)}, Faces: {len(original.faces)}")

    print(f"\nRunning CoACD decomposition (threshold={threshold}) ...")
    parts = decompose(original, threshold=threshold)
    print(f"  Decomposed into {len(parts)} convex parts")

    # --- Scene 1: original mesh ---
    original.visual.face_colors = [180, 180, 220, 200]
    scene_original = trimesh.Scene([original])

    # --- Scene 2: convex decomposition ---
    colors = random_colors(len(parts))
    for i, part in enumerate(parts):
        part.visual.face_colors = colors[i]
    scene_decomp = trimesh.Scene(parts)

    print("\nShowing original mesh (close window to continue) ...")
    scene_original.show(caption=f"Original: {mesh_path.name}")

    print("Showing convex decomposition ...")
    scene_decomp.show(caption=f"CoACD decomposition ({len(parts)} parts): {mesh_path.name}")

    return parts


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch()
        return

    default_mesh = (
        Path(__file__).parent.parent / "urdf" / "meshes" / "arch_hole.obj"
    )

    if len(sys.argv) > 1:
        mesh_path = Path(sys.argv[1])
    else:
        mesh_path = default_mesh

    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    threshold = 0.05
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])

    parts = visualise(mesh_path, threshold=threshold)

    answer = input(f"\nSave {len(parts)} convex parts? [y/N] ").strip().lower()
    if answer == "y":
        output_dir = mesh_path.parent
        geom_name = mesh_path.stem
        save_decomposition(parts, output_dir, geom_name)


if __name__ == "__main__":
    main()
