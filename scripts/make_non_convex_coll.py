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
    return parser.parse_args()


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
    tgen.tetrahedralize(
        quality=True,
        nobisect=True,
        nomergefacet=True,
        nomergevertex=True,
        vtksurfview=True,
        vtkview=True,
        verbose=True,
    )
    tgen.grid.save(str(vtk_output), binary=False)


if __name__ == "__main__":
    main()
