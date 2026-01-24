import argparse
from pathlib import Path

from pydrake.geometry import MeshSource
from pydrake.geometry import RefineVolumeMeshIntoVtkFileContents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a VTK volume mesh using Drake."
    )
    parser.add_argument("input", help="Input VTK volume mesh file path.")
    parser.add_argument(
        "--output",
        help="Output refined VTK file path. Defaults to <input>_refined.vtk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_refined")

    vtk_string = RefineVolumeMeshIntoVtkFileContents(
        mesh_source=MeshSource(str(input_path))
    )

    with open(output_path, "w") as file:
        file.write(vtk_string)

    print(f"Refined mesh written to: {output_path}")


if __name__ == "__main__":
    main()
