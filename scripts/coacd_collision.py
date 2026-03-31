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


def decompose(mesh: trimesh.Trimesh, threshold: float = 0.05) -> list[trimesh.Trimesh]:
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    return [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]


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
