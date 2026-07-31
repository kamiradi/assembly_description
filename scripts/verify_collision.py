#!/usr/bin/env python3
"""Verify CoACD convex decompositions against their source meshes.

Two checks:

A) DEVIATION -- how far does the convex decomposition bulge *outside* the
   original mesh? Convex parts can only ever be equal-to or larger-than the
   original at concave features, so we measure that outward growth via the
   axis-aligned bounding box and via sampled signed-distance.

B) TOLERANCE -- does the peg still fit into the hole after decomposition?
   We slice both parts perpendicular to the insertion axis and measure the
   radial gap between the peg cross-section and the hole's cavity. We do this
   for the ORIGINAL meshes and for the DECOMPOSED collision geometry so the
   clearance lost to decomposition is explicit.

Only depends on trimesh / numpy / shapely (run it in your coacd venv).

Usage:
    # full arch peg+hole report using repo-default paths
    python verify_collision.py

    # deviation of one decomposition
    python verify_collision.py deviation <original.obj> <decomp_dir>

    # insertion tolerance for an explicit peg/hole pair
    python verify_collision.py tolerance \
        <peg.obj> <peg_decomp_dir> <hole.obj> <hole_decomp_dir>

    # 3D view of the mated pose (transparent geoms, interference in red)
    python verify_collision.py visualise [decomposed|original]

Assumptions (tweak the constants below if they don't hold):
    * Insertion axis is +Z.
    * Peg and hole are authored in a common frame and assembled concentrically
      (i.e. the nominal inserted pose has no relative XY translation). The
      tolerance check also reports a "best-centred" number that removes any
      residual centroid offset, which isolates pure geometric clearance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union

INSERTION_AXIS = 1  # 0=x, 1=y, 2=z -- bore/length axis of the peg in its frame
                    # (peg extends 180mm along Y; hole depth aligns to Y after
                    #  the mating transform)
N_SLICES = 25       # cross-sections sampled across the engagement region
N_SURFACE_SAMPLES = 40000  # points sampled for the deviation signed-distance


# --------------------------------------------------------------------------- #
# Mating transform
#
# The peg/hole meshes live in *different* model frames. The assembled (mated)
# pose comes from config/assembly_arch.dmd.yaml: the peg mates peg::peg_origin
# (== peg::base_link, identity) onto hole::hole_frame, which is defined relative
# to hole::base_link as translation [0,0,0.02] then Rpy(deg)=[-90,0,90].
# To express the hole mesh in the peg frame we apply X_holeframe_holebase =
# inv(translation . rotation), so the cavity axis lines up with the peg's +Z.
# --------------------------------------------------------------------------- #
def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Drake RollPitchYaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in rad."""
    return _rot("z", yaw) @ _rot("y", pitch) @ _rot("x", roll)


def hole_to_peg_frame(translation=(0.0, 0.0, 0.02), rpy_deg=(-90.0, 0.0, 90.0)) -> np.ndarray:
    """4x4 mapping hole base_link coords -> peg (mating) frame."""
    R = rpy_matrix(*np.radians(rpy_deg))
    t = np.asarray(translation, dtype=float)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def peg_flip_transform(axial_offset: float = 0.05) -> np.ndarray:
    """180 deg flip of the peg about its bore (insertion) axis, so its arch
    profile is in phase with the hole cavity, plus an axial offset along the
    bore (default +5 cm along the insertion axis)."""
    angles = [0.0, 0.0, 0.0]
    angles[INSERTION_AXIS] = np.pi
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*angles)
    T[INSERTION_AXIS, 3] = axial_offset
    return T


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single mesh from {path}")
    return mesh


def load_decomposition(decomp_dir: Path) -> list[trimesh.Trimesh]:
    parts = []
    for obj in sorted(decomp_dir.glob("*.obj")):
        m = trimesh.load(str(obj), force="mesh")
        if isinstance(m, trimesh.Trimesh) and len(m.faces):
            parts.append(m)
    if not parts:
        raise ValueError(f"No .obj parts found in {decomp_dir}")
    return parts


def combine(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(parts)


# --------------------------------------------------------------------------- #
# A) Deviation
# --------------------------------------------------------------------------- #
def check_deviation(original: trimesh.Trimesh, parts: list[trimesh.Trimesh], label: str) -> None:
    decomp = combine(parts)

    print(f"\n=== DEVIATION: {label} ===")
    print(f"  original : {len(original.vertices):>6} verts, {len(original.faces):>6} faces")
    print(f"  decomp   : {len(parts)} parts, {len(decomp.vertices):>6} verts")

    # --- AABB growth (mm), per axis ---
    o_min, o_max = original.bounds
    d_min, d_max = decomp.bounds
    grow_lo = (o_min - d_min) * 1000.0   # positive => decomp extends below original
    grow_hi = (d_max - o_max) * 1000.0   # positive => decomp extends above original
    print("  AABB growth (mm)   [+ = decomposition is larger]")
    for ax, name in enumerate("xyz"):
        print(f"    {name}: low {grow_lo[ax]:+.4f}   high {grow_hi[ax]:+.4f}   "
              f"total {grow_lo[ax] + grow_hi[ax]:+.4f}")

    # --- outward surface deviation via signed distance ---
    # trimesh signed_distance: + inside the surface, - outside.
    pts, _ = trimesh.sample.sample_surface(decomp, N_SURFACE_SAMPLES)
    sd = trimesh.proximity.signed_distance(original, pts)
    outward = -sd  # + => decomposition surface lies outside the original
    out = outward[outward > 0]
    frac_outside = len(out) / len(outward)
    print("  outward surface deviation (decomp beyond original)")
    print(f"    fraction of decomp surface outside original : {frac_outside * 100:.1f}%")
    if len(out):
        print(f"    max  outward bulge : {out.max() * 1000:.4f} mm")
        print(f"    mean outward bulge : {out.mean() * 1000:.4f} mm "
              f"(over the outside portion)")
    else:
        print("    decomposition lies entirely within the original (no bulge)")


# --------------------------------------------------------------------------- #
# B) Tolerance
# --------------------------------------------------------------------------- #
def _plane_segments_2d(mesh: trimesh.Trimesh, height: float) -> np.ndarray | None:
    """Raw plane/mesh intersection segments, projected to the 2D slice plane.

    Uses trimesh.intersections.mesh_plane (no networkx / no path machinery)."""
    normal = np.zeros(3)
    normal[INSERTION_AXIS] = 1.0
    origin = np.zeros(3)
    origin[INSERTION_AXIS] = height
    lines = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    if lines is None or len(lines) == 0:
        return None
    keep = [i for i in range(3) if i != INSERTION_AXIS]
    return np.asarray(lines)[:, :, keep]  # (N, 2, 2)


def section_convex_part(part: trimesh.Trimesh, height: float) -> Polygon | None:
    """Cross-section of a *convex* mesh = convex hull of its plane-intersection
    points. Exact for convex parts, and immune to open/non-manifold loops."""
    segs = _plane_segments_2d(part, height)
    if segs is None:
        return None
    pts = segs.reshape(-1, 2)
    if len(pts) < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    return hull if hull.geom_type == "Polygon" and hull.area > 0 else None


def section_mesh(mesh: trimesh.Trimesh, height: float):
    """Solid cross-section of an arbitrary (non-convex) mesh as a shapely polygon
    with holes. Builds polygons from segments via shapely.polygonize, then keeps
    faces at odd containment depth (even-odd fill) so cavities become holes."""
    segs = _plane_segments_2d(mesh, height)
    if segs is None:
        return None
    lines = [seg for seg in segs if not np.allclose(seg[0], seg[1])]
    if not lines:
        return None
    noded = unary_union([LineString(s) for s in lines])
    faces = [f for f in polygonize(noded) if f.is_valid and f.area > 0]
    if not faces:
        return None
    filled = [Polygon(f.exterior) for f in faces]
    solid = []
    for f in faces:
        pt = f.representative_point()
        depth = sum(1 for g in filled if g.contains(pt))
        if depth % 2 == 1:
            solid.append(f)
    if not solid:
        return None
    return unary_union(solid)


def peg_outline(solid) -> Polygon:
    """The solid peg cross-section (largest exterior, holes ignored)."""
    geoms = solid.geoms if solid.geom_type == "MultiPolygon" else [solid]
    g = max(geoms, key=lambda p: p.area)
    return Polygon(g.exterior)


def hole_cavity(solid) -> Polygon | None:
    """The empty cavity = largest interior ring across the solid cross-section."""
    geoms = solid.geoms if solid.geom_type == "MultiPolygon" else [solid]
    rings = [Polygon(r) for g in geoms for r in g.interiors]
    if not rings:
        return None
    return max(rings, key=lambda p: p.area)


def clearance_at(peg: Polygon, cavity: Polygon, recentre: bool) -> dict | None:
    """Radial clearance of `peg` inside `cavity`.

    Returns dict with keys: fits (bool), gap (m, min boundary-to-boundary gap
    when fitting), intrusion (m, how far peg pokes through the wall when not).
    """
    if recentre:
        dx = cavity.centroid.x - peg.centroid.x
        dy = cavity.centroid.y - peg.centroid.y
        peg = Polygon(np.asarray(peg.exterior.coords) + (dx, dy))

    areas = {"peg_area": peg.area, "cav_area": cavity.area}
    outside = peg.difference(cavity)
    if outside.is_empty or outside.area <= 1e-12:
        return {"fits": True,
                "gap": peg.exterior.distance(cavity.exterior),
                "intrusion": 0.0, **areas}
    # peg pokes into the wall: deepest intrusion = max distance from the
    # protruding region back to the cavity boundary.
    intr = outside.hausdorff_distance(cavity.exterior)
    return {"fits": False, "gap": 0.0, "intrusion": intr, **areas}


class MeshSource:
    """Sections an arbitrary (non-convex) original mesh via even-odd polygonize."""
    def __init__(self, mesh: trimesh.Trimesh):
        self.bounds = mesh.bounds
        self._mesh = mesh

    def section(self, height: float):
        return section_mesh(self._mesh, height)


class ConvexPartsSource:
    """Sections a set of convex collision parts; union of per-part convex hulls."""
    def __init__(self, parts: list[trimesh.Trimesh]):
        self.bounds = combine(parts).bounds
        self._parts = parts

    def section(self, height: float):
        polys = [section_convex_part(p, height) for p in self._parts]
        polys = [p for p in polys if p is not None and p.area > 0]
        if not polys:
            return None
        return unary_union(polys)


def sweep(peg_src, hole_src, recentre: bool) -> dict | None:
    """Worst-case clearance across the engaged span of the insertion axis."""
    ax = INSERTION_AXIS
    lo = max(peg_src.bounds[0][ax], hole_src.bounds[0][ax])
    hi = min(peg_src.bounds[1][ax], hole_src.bounds[1][ax])
    if hi <= lo:
        return None
    span = hi - lo
    heights = np.linspace(lo + 0.02 * span, hi - 0.02 * span, N_SLICES)

    worst = None
    for h in heights:
        pp = peg_src.section(h)
        hp = hole_src.section(h)
        if pp is None or hp is None:
            continue
        cav = hole_cavity(hp)
        if cav is None:
            continue
        res = clearance_at(peg_outline(pp), cav, recentre)
        if res is None:
            continue
        # signed metric: + gap when fitting, - intrusion when not
        signed = res["gap"] if res["fits"] else -res["intrusion"]
        res["height"] = h
        res["signed"] = signed
        if worst is None or signed < worst["signed"]:
            worst = res
    return worst


def report_tolerance(peg_orig, peg_parts, hole_orig, hole_parts,
                     hole_transform=None, peg_transform=None) -> None:
    print("\n=== INSERTION TOLERANCE ===")
    if hole_transform is not None:
        hole_orig = hole_orig.copy()
        hole_orig.apply_transform(hole_transform)
        hole_parts = _transform_copies(hole_parts, hole_transform)
    if peg_transform is not None:
        peg_orig = peg_orig.copy()
        peg_orig.apply_transform(peg_transform)
        peg_parts = _transform_copies(peg_parts, peg_transform)
    peg_orig_src = MeshSource(peg_orig)
    hole_orig_src = MeshSource(hole_orig)
    peg_decomp_src = ConvexPartsSource(peg_parts)
    hole_decomp_src = ConvexPartsSource(hole_parts)

    for label, recentre in (("nominal (concentric, as authored)", False),
                            ("best-centred (residual offset removed)", True)):
        print(f"\n  -- {label} --")
        for tag, pm, hm in (("ORIGINAL  meshes", peg_orig_src, hole_orig_src),
                            ("DECOMPOSED collision", peg_decomp_src, hole_decomp_src)):
            w = sweep(pm, hm, recentre)
            if w is None:
                print(f"    {tag}: no overlapping cross-section found")
                continue
            areas = (f"[peg {w['peg_area'] * 1e6:.1f} mm^2 vs cavity "
                     f"{w['cav_area'] * 1e6:.1f} mm^2]")
            if w["fits"]:
                print(f"    {tag}: FITS  -- min radial gap {w['gap'] * 1000:.4f} mm "
                      f"(worst slice @ {INSERTION_AXIS_NAME}={w['height'] * 1000:.2f} mm) {areas}")
            else:
                print(f"    {tag}: INTERFERENCE -- peg intrudes {w['intrusion'] * 1000:.4f} mm "
                      f"into wall (worst slice @ {INSERTION_AXIS_NAME}={w['height'] * 1000:.2f} mm) {areas}")

    print("\n  NOTE: 'DECOMPOSED' gap < 'ORIGINAL' gap is the clearance eaten by\n"
          "        convex decomposition. If DECOMPOSED shows INTERFERENCE while\n"
          "        ORIGINAL fits, the collision geometry will block insertion.")


INSERTION_AXIS_NAME = "xyz"[INSERTION_AXIS]


# --------------------------------------------------------------------------- #
# 3D visualisation
# --------------------------------------------------------------------------- #
PEG_COLOR = [70, 130, 255, 90]    # transparent blue
HOLE_COLOR = [170, 170, 170, 55]  # transparent grey
INTERF_COLOR = [255, 25, 25, 255]  # solid red


def _transform_copies(meshes: list[trimesh.Trimesh], T) -> list[trimesh.Trimesh]:
    out = []
    for m in meshes:
        c = m.copy()
        if T is not None:
            c.apply_transform(T)
        out.append(c)
    return out


def _as_solid(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Best-effort single watertight solid for boolean ops (union if multiple)."""
    if len(meshes) == 1:
        return meshes[0]
    try:
        return trimesh.boolean.union(meshes)
    except Exception:
        return combine(meshes)


def interference_volume(peg_solid, hole_solid):
    """3D boolean intersection (peg embedded in hole material). None if empty
    or no boolean engine available."""
    try:
        inter = trimesh.boolean.intersection([peg_solid, hole_solid])
    except Exception as exc:
        print(f"  [interference] boolean engine unavailable ({exc}); "
              "showing transparent geometry only.")
        return None
    if inter is None or inter.is_empty or len(inter.faces) == 0 or inter.volume <= 1e-12:
        return None
    return inter


def visualise(peg_parts, hole_parts, hole_transform=None, peg_transform=None, label="") -> None:
    """Show peg (blue) inserted into hole (grey) at the mated pose, both
    transparent, with the interfering volume drawn solid red."""
    hole_parts = _transform_copies(hole_parts, hole_transform)
    peg_parts = _transform_copies(peg_parts, peg_transform)

    scene = trimesh.Scene()
    for i, m in enumerate(peg_parts):
        m.visual.face_colors = PEG_COLOR
        scene.add_geometry(m, geom_name=f"peg_{i}")
    for i, m in enumerate(hole_parts):
        m.visual.face_colors = HOLE_COLOR
        scene.add_geometry(m, geom_name=f"hole_{i}")

    print(f"\nComputing interference volume for {label} ...")
    inter = interference_volume(_as_solid(peg_parts), _as_solid(hole_parts))
    if inter is not None:
        inter.visual.face_colors = INTERF_COLOR
        scene.add_geometry(inter, geom_name="interference")
        print(f"  interference volume: {inter.volume * 1e9:.1f} mm^3  (red)")
    else:
        print("  no interference volume (peg clears the hole material) — "
              "only transparent geometry shown")

    out = Path("/tmp") / f"insertion_{label or 'view'}.glb"
    try:
        scene.export(str(out))
        print(f"  saved scene to {out}")
    except Exception:
        pass
    print("  opening viewer (close window to exit) ...")
    scene.show(caption=f"Insertion: {label}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# Named peg/hole families: (peg_orig, peg_decomp_dir, hole_orig, hole_decomp_dir).
# The mated pose (hole_to_peg_frame + peg_flip_transform) is shared because the
# rectangle assembly (config/assembly_rect.dmd.yaml) uses the same hole_frame
# offset/rotation and insertion axis as the arch assembly.
FAMILIES = {
    "arch": ("arch_peg.obj", "arch_peg", "arch_hole.obj", "arch_hole_1"),
    "rect": ("rectangle_peg_collision.obj", "rectangle_peg",
             "rectangle_hole_chamfer.obj", "rectangle_hole"),
}


def full_report(meshes: Path, family: str) -> None:
    peg_o, peg_d, hole_o, hole_d = FAMILIES[family]
    peg_o, hole_o = meshes / peg_o, meshes / hole_o
    peg_d, hole_d = meshes / peg_d, meshes / hole_d
    check_deviation(load_mesh(peg_o), load_decomposition(peg_d), f"{family}_peg")
    check_deviation(load_mesh(hole_o), load_decomposition(hole_d), f"{family}_hole")
    report_tolerance(load_mesh(peg_o), load_decomposition(peg_d),
                     load_mesh(hole_o), load_decomposition(hole_d),
                     hole_transform=hole_to_peg_frame(),
                     peg_transform=peg_flip_transform())


def main() -> None:
    meshes = Path(__file__).parent.parent / "urdf" / "meshes"
    args = sys.argv[1:]

    if not args:  # default arch peg+hole full report
        full_report(meshes, "arch")
        return

    # `rect` / `arch` -> full deviation + mated tolerance report for that family.
    if args[0] in FAMILIES and len(args) == 1:
        full_report(meshes, args[0])
        return

    cmd = args[0]
    if cmd == "deviation" and len(args) == 3:
        check_deviation(load_mesh(Path(args[1])), load_decomposition(Path(args[2])),
                        Path(args[1]).stem)
    elif cmd == "tolerance" and len(args) == 5:
        report_tolerance(load_mesh(Path(args[1])), load_decomposition(Path(args[2])),
                         load_mesh(Path(args[3])), load_decomposition(Path(args[4])))
    elif cmd == "visualise":
        # visualise [family] [decomposed|original]  -- peg+hole in mated pose
        rest = args[1:]
        family = rest[0] if rest and rest[0] in FAMILIES else "arch"
        rest = [a for a in rest if a not in FAMILIES]
        which = rest[0] if rest else "decomposed"
        peg_o, peg_d, hole_o, hole_d = FAMILIES[family]
        if which == "original":
            peg = [load_mesh(meshes / peg_o)]
            hole = [load_mesh(meshes / hole_o)]
        else:
            peg = load_decomposition(meshes / peg_d)
            hole = load_decomposition(meshes / hole_d)
        visualise(peg, hole, hole_transform=hole_to_peg_frame(),
                  peg_transform=peg_flip_transform(), label=f"{family}_{which}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
