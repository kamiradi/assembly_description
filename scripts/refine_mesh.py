from pydrake.geometry import MeshSource
from pydrake.geometry import RefineVolumeMeshIntoVtkFileContents

input_file = "/home/aditya/Documents/workspace/assembly_description/urdf/meshes/rectangular_hole.vtk"
vtk_string = RefineVolumeMeshIntoVtkFileContents(
             mesh_source=MeshSource(input_file))

output_file = "/home/damrongguoy/output_mesh.vtk"
output_file = "/home/aditya/Documents/workspace/assembly_description/urdf/meshes/rectangular_hole_refined.vtk"
with open(output_file, "w") as file:
    file.write(vtk_string)
