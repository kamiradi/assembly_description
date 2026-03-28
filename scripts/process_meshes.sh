#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESHES_DIR="$SCRIPT_DIR/../urdf/meshes"
TEMP_DIR="$SCRIPT_DIR/../output_temp"
FTETWILD="$SCRIPT_DIR/../../fTetWild/build/FloatTetwild_bin"

OBJECTS=(
    arch_hole
    arch_peg
    rectangle_peg_teeth
    rectangle_hole_teeth
    ellipse_hole
    ellipse_peg
    ellipse_hole_teeth
    ellipse_peg_teeth
)

mkdir -p "$TEMP_DIR"

for name in "${OBJECTS[@]}"; do
    obj="$MESHES_DIR/${name}.obj"
    vtk="$MESHES_DIR/${name}.vtk"

    echo "=== Processing: $name ==="

    echo "  [1/3] Tetrahedralizing $obj"
    "$FTETWILD" -i "$obj" -o "$TEMP_DIR/output_tetmesh.msh"

    echo "  [2/3] Converting .msh -> .vtk"
    meshio convert "$TEMP_DIR/output_tetmesh.msh" "$TEMP_DIR/output_tetmesh.vtk"

    echo "  [3/3] Refining -> $vtk"
    python3 "$SCRIPT_DIR/refine_mesh.py" "$TEMP_DIR/output_tetmesh.vtk" --output "$vtk"

    echo "  Cleaning output_temp/"
    rm -f "$TEMP_DIR"/*

    echo "  Done: $vtk"
done

echo "All meshes processed."
