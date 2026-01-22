#!/usr/bin/env python3

import argparse
from pathlib import Path

import pyvista
import tetgen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tetrahedral mesh from a non-convex surface mesh."
    )
    parser.add_argument(
        "input",
        help="Input surface mesh (e.g. OBJ).",
    )
    parser.add_argument(
        "--ply-output",
        dest="ply_output",
        help="Output PLY file path.",
    )
    parser.add_argument(
        "--vtk-output",
        dest="vtk_output",
        help="Output VTK file path.",
    )
    parser.add_argument(
        "--max-volume",
        dest="max_volume",
        type=float,
        help="Maximum tetrahedron volume to enforce.",
    )
    parser.add_argument(
        "--max-volume-factor",
        dest="max_volume_factor",
        type=float,
        default=0.01,
        help="Fallback max volume as fraction of mesh volume.",
    )
    return parser.parse_args()


def estimate_maxvolume(mesh: pyvista.PolyData, factor: float) -> float:
    volume = abs(mesh.volume)
    if volume <= 0:
        bounds = mesh.bounds
        volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
    return max(volume * factor, 0.0)


def tetrahedralize(tgen: tetgen.TetGen, max_volume: float | None) -> pyvista.UnstructuredGrid:
    if max_volume is None or max_volume <= 0:
        max_volume = -1.0
    tgen.tetrahedralize(
        quality=True,
        nobisect=True,
        nomergefacet=True,
        nomergevertex=True,
        vtksurfview=True,
        vtkview=True,
        verbose=True,
        maxvolume=max_volume,
    )
    return tgen.grid


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.ply_output:
        ply_output = Path(args.ply_output)
    else:
        ply_output = input_path.with_suffix(".ply")

    if args.vtk_output:
        vtk_output = Path(args.vtk_output)
    else:
        vtk_output = input_path.with_suffix(".vtk")

    reader = pyvista.get_reader(str(input_path))
    mesh = reader.read()
    mesh = mesh.extract_surface().triangulate()
    mesh.save(str(ply_output), binary=False)

    tgen = tetgen.TetGen(str(ply_output))
    grid = tetrahedralize(tgen, args.max_volume)

    surface_points = grid.extract_surface().n_points
    if surface_points == grid.n_points:
        fallback_max_volume = args.max_volume
        if fallback_max_volume is None:
            fallback_max_volume = estimate_maxvolume(mesh, args.max_volume_factor)
        if fallback_max_volume and fallback_max_volume > 0:
            grid = tetrahedralize(tgen, fallback_max_volume)

    grid.save(str(vtk_output), binary=False)


if __name__ == "__main__":
    main()
