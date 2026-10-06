#!/usr/bin/env python3
"""
01_build_2d_wires.py
====================
Project-agnostic geometry engine for the Powerline Vectors - Wires product.

Purpose
-------
Build a raw 2D circuit wire from ONLY NM geometry:
    1. 3D wires linework
    2. circuit centreline/alignment linework

No client source data, SoW KMZ, RoW or voltage is required here.
Those are deliberately handled by 02_conflate_2d_wires.py.

Geometry model
--------------
* 3D wire endpoints are clustered into support locations.
* The corresponding centreline is the geometric truth for the support location.
* Each support cluster is projected onto the centreline to obtain the local
  structure/support station.
* Local wire endpoints are projected onto the normal to the centreline.
* The two lateral extremes are retained as QA candidates.
* Horizontal phase arrangement override: when a clear left-centre-right row of
  attachment points is detected at similar Z and the centre attachment passes
  over the structure/centreline, that centre attachment is treated as the
  "most outer" point for the 2D product.
* Otherwise a continuity solver chooses one consistent outer side through each
  connected circuit component; this avoids left/right zig-zag on symmetric
  structures.
* Consecutive selected attachment points are connected by straight 2D segments.

The QA output retains left/right candidates and the detected centre candidate,
plus the rule used at every support.

Accepted input formats
----------------------
.dgn, .dxf, .shp, .gpkg, .zip

DXF levels and DGN CAD levels are used as circuit identifiers. Common suffixes
such as _Lines, _Wires, _Centreline, _Alignment are removed automatically.
Where attributes contain CIRCUIT / CIRCUIT_ID / LINE_NO, those values are
preferred.

Output
------
01_raw_2d_wires.gpkg
    wires_2d        final raw aggregate(s), one feature per connected component
    span_edges      straight support-to-support segments (preferred input to 02)
    support_points  survey-derived structure/support stations on centreline
    outer_candidates both lateral extremes and selected flag
    centrelines     centreline geometry used by the engine

raw_2d_wires.shp    convenience copy of wires_2d
01_geometry_audit.csv

Dependencies
------------
pip install geopandas shapely pyproj pyogrio pandas numpy ezdxf
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points, unary_union

try:
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def info(msg: str = "") -> None:
    print(msg, flush=True)


def txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clean_path(value: str) -> Path:
    s = txt(value)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1]
    return Path(s).expanduser()


def decode_cad_name(value: Any) -> str:
    """Decode AutoCAD-style __x0036__ escapes often written by OGR."""
    s = txt(value)
    return re.sub(
        r"__x([0-9A-Fa-f]{4})__",
        lambda m: chr(int(m.group(1), 16)),
        s,
    )


def normalise_circuit(value: Any) -> str:
    """Human-readable circuit label inferred from a CAD level/layer."""
    s = decode_cad_name(value).strip()
    # Remove common role suffixes but preserve the actual circuit text.
    suffix = re.compile(
        r"(?:[ _\-]+(?:WIRES?|CONDUCTORS?|LINES?|CENTRELINES?|CENTERLINES?|ALIGNMENTS?|ALIGNMENT|CIRCUIT))+$",
        flags=re.I,
    )
    old = None
    while old != s:
        old = s
        s = suffix.sub("", s).strip(" _-")
    return s


def circuit_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalise_circuit(value).upper())


def is_line(geom) -> bool:
    return geom is not None and not geom.is_empty and geom.geom_type in {
        "LineString", "MultiLineString"
    }


def coords_of(geom) -> list[tuple[float, ...]]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return list(geom.coords)
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
        if not parts:
            return []
        return list(max(parts, key=lambda g: g.length).coords)
    return []


def unit_name(crs) -> str:
    try:
        c = CRS.from_user_input(crs)
        if c.axis_info:
            return txt(c.axis_info[0].unit_name).lower()
    except Exception:
        pass
    return ""


def default_support_tolerance(crs) -> float:
    u = unit_name(crs)
    if "metre" in u or "meter" in u:
        return 15.0
    return 50.0  # feet / US survey feet


def safe_layer_name(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", txt(value)).strip("_")
    return s[:60] or "layer"


def prompt_path(label: str, allowed: set[str]) -> Path:
    while True:
        p = clean_path(input(label))
        if not p.exists():
            info(f"Path does not exist: {p}")
            continue
        if p.suffix.lower() not in allowed:
            info("Expected: " + ", ".join(sorted(allowed)))
            continue
        return p


def prompt_epsg() -> int:
    while True:
        raw = input("Input CRS EPSG code: ").strip()
        raw = re.sub(r"^EPSG\s*:\s*", "", raw, flags=re.I)
        if raw.isdigit():
            try:
                CRS.from_epsg(int(raw))
                return int(raw)
            except Exception:
                pass
        info("Enter a valid numeric EPSG code.")


# -----------------------------------------------------------------------------
# CAD / vector readers
# -----------------------------------------------------------------------------

CIRCUIT_FIELDS = [
    "circuit_id", "circuitid", "circuit", "ckt_id", "cktid", "ckt",
    "line_no", "lineno", "line",
]
LEVEL_FIELDS = [
    "levelname", "level_name", "level", "layername", "layer_name", "layer",
]


def find_field(columns: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def line_from_dxf_entity(entity) -> LineString | None:
    typ = entity.dxftype()
    try:
        if typ == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            return LineString([(float(a.x), float(a.y), float(a.z)),
                               (float(b.x), float(b.y), float(b.z))])
        if typ == "POLYLINE":
            pts = []
            for v in entity.vertices:
                p = v.dxf.location
                pts.append((float(p.x), float(p.y), float(p.z)))
            return LineString(pts) if len(pts) >= 2 else None
        if typ == "LWPOLYLINE":
            # LWPOLYLINE coordinates are stored in the entity OCS, not
            # necessarily directly in world XYZ. 3D wire exports often use a
            # tilted extrusion vector + elevation to encode the wire plane.
            # vertices_in_wcs() performs the required OCS -> WCS transform.
            pts = [(float(p.x), float(p.y), float(p.z))
                   for p in entity.vertices_in_wcs()]
            return LineString(pts) if len(pts) >= 2 else None
        if typ == "SPLINE":
            # Approximate only if present. Real production DGN/DXF wires are
            # normally polylines, but this keeps the reader tolerant.
            tool = entity.construction_tool()
            pts = [(float(p.x), float(p.y), float(p.z)) for p in tool.approximate(segments=80)]
            return LineString(pts) if len(pts) >= 2 else None
    except Exception:
        return None
    return None


def read_dxf(path: Path) -> gpd.GeoDataFrame:
    if ezdxf is None:
        raise RuntimeError("DXF input requires ezdxf: pip install ezdxf")
    doc = ezdxf.readfile(path)
    rows = []
    for entity in doc.modelspace():
        geom = line_from_dxf_entity(entity)
        if geom is None or geom.is_empty:
            continue
        level = decode_cad_name(entity.dxf.layer)
        rows.append({
            "SOURCE": path.name,
            "LEVEL": level,
            "CIRCUIT": normalise_circuit(level),
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def read_ogr_file(path: Path) -> gpd.GeoDataFrame:
    """Read DGN/SHP/GPKG and expose CAD level + circuit consistently."""
    all_rows: list[gpd.GeoDataFrame] = []
    try:
        layers = pyogrio.list_layers(path)
    except Exception:
        layers = np.array([[None, None]], dtype=object)

    layer_names = [txt(x[0]) for x in layers] if len(layers) else [""]
    if path.suffix.lower() == ".shp":
        layer_names = [""]

    for layer in layer_names:
        try:
            kwargs = {"engine": "pyogrio"}
            if layer:
                kwargs["layer"] = layer
            g = gpd.read_file(path, **kwargs)
        except Exception:
            continue
        if g.empty or "geometry" not in g:
            continue
        g = g[g.geometry.apply(is_line)].copy()
        if g.empty:
            continue

        level_field = find_field(g.columns, LEVEL_FIELDS)
        circuit_field = find_field(g.columns, CIRCUIT_FIELDS)

        if level_field:
            levels = g[level_field].apply(decode_cad_name)
        else:
            levels = pd.Series(layer or path.stem, index=g.index)

        if circuit_field:
            circuits = g[circuit_field].apply(txt)
            circuits = circuits.where(circuits != "", levels.apply(normalise_circuit))
        else:
            circuits = levels.apply(normalise_circuit)

        g["SOURCE"] = path.name
        g["LEVEL"] = levels
        g["CIRCUIT"] = circuits
        all_rows.append(g[["SOURCE", "LEVEL", "CIRCUIT", "geometry"]])

    if not all_rows:
        return gpd.GeoDataFrame(columns=["SOURCE", "LEVEL", "CIRCUIT", "geometry"], geometry="geometry")

    crs = next((g.crs for g in all_rows if g.crs is not None), None)
    out = pd.concat(all_rows, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def read_vector(path: Path, temp_dir: Path) -> gpd.GeoDataFrame:
    ext = path.suffix.lower()
    if ext == ".dxf":
        return read_dxf(path)
    if ext in {".dgn", ".shp", ".gpkg"}:
        return read_ogr_file(path)
    if ext == ".zip":
        dest = temp_dir / safe_layer_name(path.stem)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        candidates = []
        for pattern in ("*.dgn", "*.dxf", "*.shp", "*.gpkg"):
            candidates.extend(dest.rglob(pattern))
        frames = []
        for p in sorted(candidates):
            try:
                frames.append(read_vector(p, temp_dir))
            except Exception as exc:
                info(f"  SKIP {p.name}: {exc}")
        frames = [g for g in frames if not g.empty]
        if not frames:
            raise RuntimeError(f"No supported linework found in {path}")
        crs = next((g.crs for g in frames if g.crs is not None), None)
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    raise RuntimeError(f"Unsupported input: {path}")


def set_or_reproject(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(target_crs, allow_override=True)
    if CRS.from_user_input(gdf.crs) != CRS.from_user_input(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


# -----------------------------------------------------------------------------
# Topology helpers
# -----------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def cluster_xy(points: list[tuple[float, float]], tolerance: float) -> list[int]:
    if not points:
        return []
    uf = UnionFind(len(points))
    cell = max(float(tolerance), 1e-9)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(points):
        cx, cy = math.floor(x / cell), math.floor(y / cell)
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for j in grid.get((gx, gy), []):
                    if math.hypot(x - points[j][0], y - points[j][1]) <= tolerance:
                        uf.union(i, j)
        grid[(cx, cy)].append(i)
    roots = [uf.find(i) for i in range(len(points))]
    root_map = {r: n + 1 for n, r in enumerate(sorted(set(roots)))}
    return [root_map[r] for r in roots]


def line_parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    return []


def nearest_line_and_station(point: Point, lines: list[LineString]) -> tuple[LineString, float, Point, float]:
    if not lines:
        raise RuntimeError("No centreline geometry available")
    best_line = None
    best_d = float("inf")
    best_station = 0.0
    best_point = None
    for ln in lines:
        s = ln.project(point)
        q = ln.interpolate(s)
        d = point.distance(q)
        if d < best_d:
            best_line, best_d, best_station, best_point = ln, d, s, q
    assert best_line is not None and best_point is not None
    return best_line, best_station, best_point, best_d


def local_tangent(line: LineString, station: float) -> tuple[float, float]:
    length = max(line.length, 1e-9)
    delta = max(min(length * 0.005, 20.0), min(length * 0.0001, 0.25))
    a = line.interpolate(max(0.0, station - delta))
    b = line.interpolate(min(length, station + delta))
    dx, dy = b.x - a.x, b.y - a.y
    n = math.hypot(dx, dy)
    if n < 1e-9:
        c = list(line.coords)
        if len(c) >= 2:
            dx, dy = c[-1][0] - c[0][0], c[-1][1] - c[0][1]
            n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else (1.0, 0.0)


@dataclass
class EndpointRec:
    wire_idx: int
    end: int
    x: float
    y: float
    z: float | None
    support: int = 0


@dataclass
class SupportRec:
    support_id: int
    centre: Point
    centreline_distance: float
    tangent: tuple[float, float]
    endpoint_indices: list[int]
    candidate_a: int
    candidate_b: int
    candidate_center: int | None = None
    horizontal_center: bool = False
    horizontal_row_width: float = 0.0
    horizontal_z_span: float = 0.0
    horizontal_ratio: float = 0.0
    selected: int = 0
    component: int = 0
    ambiguous: bool = False


def _local_cross(centre: Point, tangent: tuple[float, float], e: EndpointRec) -> float:
    """Signed lateral offset from the local centreline normal."""
    tx, ty = tangent
    nx, ny = -ty, tx
    return (e.x - centre.x) * nx + (e.y - centre.y) * ny


def detect_horizontal_center_candidate(
    centre: Point,
    tangent: tuple[float, float],
    endpoint_indices: list[int],
    endpoints: list[EndpointRec],
    tolerance: float,
    max_z_ratio: float = 0.05,
) -> tuple[int | None, float, float, float]:
    """Detect a horizontal left-centre-right attachment row.

    This implements the agreed special case: if conductors are arranged
    horizontally and a middle attachment runs over the structure/centreline,
    the middle attachment becomes the selected 2D wire point.

    Detection is deliberately geometric and project-agnostic. It requires:
      * an attachment on each side of the centreline;
      * a candidate close to the centreline;
      * a broad row spanning a substantial part of the support width; and
      * small vertical spread relative to the row width.

    Returns (endpoint_index, row_width, z_span, z_span/row_width).
    """
    vals = []
    for i in endpoint_indices:
        e = endpoints[i]
        if e.z is None or not math.isfinite(e.z):
            continue
        vals.append((_local_cross(centre, tangent, e), float(e.z), i))
    if len(vals) < 3:
        return None, 0.0, 0.0, 0.0

    xs = [v[0] for v in vals]
    xmin, xmax = min(xs), max(xs)
    full_width = xmax - xmin
    if full_width <= 1e-9 or xmin >= 0.0 or xmax <= 0.0:
        return None, 0.0, 0.0, 0.0

    half_width = max(abs(xmin), abs(xmax), 1e-9)
    # Centre phase must genuinely pass over/very close to the structure.
    # tolerance-based floor makes this robust to small endpoint/CAD mismatch.
    center_limit = max(0.05 * tolerance, 0.20 * half_width)
    centres = [v for v in vals if abs(v[0]) <= center_limit]
    lefts = [v for v in vals if v[0] < -0.10 * half_width]
    rights = [v for v in vals if v[0] > 0.10 * half_width]
    if not centres or not lefts or not rights:
        return None, 0.0, 0.0, 0.0

    best = None
    for c in centres:
        for l in lefts:
            for r in rights:
                row_width = r[0] - l[0]
                if row_width < 0.45 * full_width:
                    continue
                z_span = max(l[1], c[1], r[1]) - min(l[1], c[1], r[1])
                ratio = z_span / max(row_width, 1e-9)
                # A genuinely horizontal phase row should be much wider than
                # its vertical spread. Default 0.05 is intentionally conservative
                # and can be tuned from the CLI during validation.
                if ratio > max_z_ratio:
                    continue
                center_ratio = abs(c[0]) / half_width
                width_ratio = row_width / full_width
                score = 2.0 * width_ratio - 2.5 * ratio - 1.5 * center_ratio
                item = (score, c[2], row_width, z_span, ratio)
                if best is None or item[0] > best[0]:
                    best = item

    if best is None:
        return None, 0.0, 0.0, 0.0
    _, idx, row_width, z_span, ratio = best
    return idx, float(row_width), float(z_span), float(ratio)


def endpoints_for_wires(wires: gpd.GeoDataFrame) -> list[EndpointRec]:
    out = []
    for idx, geom in wires.geometry.items():
        c = coords_of(geom)
        if len(c) < 2:
            continue
        for end, p in ((0, c[0]), (1, c[-1])):
            z = float(p[2]) if len(p) >= 3 else None
            out.append(EndpointRec(int(idx), end, float(p[0]), float(p[1]), z))
    return out


def build_supports(
    wires: gpd.GeoDataFrame,
    centreline_geoms: list[LineString],
    tolerance: float,
    use_horizontal_center: bool = True,
    horizontal_z_ratio: float = 0.05,
) -> tuple[list[EndpointRec], dict[int, SupportRec], list[tuple[int, int, int]]]:
    """Return endpoints, supports, and wire edges (wire index, support A, support B)."""
    eps = endpoints_for_wires(wires.reset_index(drop=True))
    xy = [(e.x, e.y) for e in eps]
    labels = cluster_xy(xy, tolerance)
    for e, lab in zip(eps, labels):
        e.support = lab

    by_support: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(eps):
        by_support[e.support].append(i)

    supports: dict[int, SupportRec] = {}
    for sid, eidxs in by_support.items():
        cx = float(np.mean([eps[i].x for i in eidxs]))
        cy = float(np.mean([eps[i].y for i in eidxs]))
        approx = Point(cx, cy)
        ln, st, q, d = nearest_line_and_station(approx, centreline_geoms)
        tx, ty = local_tangent(ln, st)
        # Normal sign itself is arbitrary; we retain both extremes. Selection is
        # done later using actual offset vectors and continuity.
        nx, ny = -ty, tx
        projections = []
        for i in eidxs:
            e = eps[i]
            proj = (e.x - q.x) * nx + (e.y - q.y) * ny
            projections.append((proj, i))
        projections.sort(key=lambda t: t[0])
        a = projections[0][1]
        b = projections[-1][1]
        if use_horizontal_center:
            c_idx, row_width, z_span, h_ratio = detect_horizontal_center_candidate(
                q, (tx, ty), eidxs, eps, tolerance, horizontal_z_ratio
            )
        else:
            c_idx, row_width, z_span, h_ratio = None, 0.0, 0.0, 0.0
        supports[sid] = SupportRec(
            support_id=sid,
            centre=q,
            centreline_distance=float(d),
            tangent=(tx, ty),
            endpoint_indices=eidxs,
            candidate_a=a,
            candidate_b=b,
            candidate_center=c_idx,
            horizontal_center=c_idx is not None,
            horizontal_row_width=row_width,
            horizontal_z_span=z_span,
            horizontal_ratio=h_ratio,
        )

    # Recover which support pair each wire connects.
    per_wire: dict[int, dict[int, int]] = defaultdict(dict)
    for e in eps:
        per_wire[e.wire_idx][e.end] = e.support
    wire_edges = []
    for wi, ends in per_wire.items():
        if 0 in ends and 1 in ends and ends[0] != ends[1]:
            wire_edges.append((wi, ends[0], ends[1]))
    return eps, supports, wire_edges


def unique_span_pairs(wire_edges: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for _, a, b in wire_edges:
        pair = tuple(sorted((a, b)))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def connected_components(nodes: Iterable[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    node_set = set(nodes)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        node_set.update([a, b])
    comps = []
    seen = set()
    for root in sorted(node_set):
        if root in seen:
            continue
        q = [root]
        seen.add(root)
        comp = []
        while q:
            u = q.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def vec_for_candidate(s: SupportRec, e: EndpointRec) -> tuple[float, float, float]:
    dx, dy = e.x - s.centre.x, e.y - s.centre.y
    r = math.hypot(dx, dy)
    return dx, dy, r


def cosine(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    if v1[2] < 1e-9 or v2[2] < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (v1[2] * v2[2])))


def selected_endpoint_index(s: SupportRec, side_choice: int) -> int:
    """Return selected endpoint, applying horizontal-centre override first."""
    if s.horizontal_center and s.candidate_center is not None:
        return s.candidate_center
    return s.candidate_a if side_choice == 0 else s.candidate_b


def selection_rule(s: SupportRec) -> str:
    return "HORIZONTAL_CENTER" if s.horizontal_center and s.candidate_center is not None else "OUTER_CONTINUITY"


def select_component_side(
    comp: list[int],
    edges: list[tuple[int, int]],
    supports: dict[int, SupportRec],
    endpoints: list[EndpointRec],
) -> tuple[dict[int, int], float, float, bool]:
    """Choose one continuous outer side for a component.

    We run the propagation twice, once from each root extreme, then score each
    solution by lateral-vector continuity plus a smaller preference for the
    physically farther candidate. This prevents alternating left/right points.
    """
    comp_set = set(comp)
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        if a in comp_set and b in comp_set:
            adj[a].append(b)
            adj[b].append(a)

    # Root with the most connections gives stable propagation through branches.
    root = max(comp, key=lambda n: (len(adj[n]), -n))

    def run(root_choice: int) -> tuple[dict[int, int], float]:
        assignment = {root: root_choice}
        q = deque([root])
        while q:
            u = q.popleft()
            su = supports[u]
            ui = su.candidate_a if assignment[u] == 0 else su.candidate_b
            vu = vec_for_candidate(su, endpoints[ui])
            for v in adj[u]:
                if v in assignment:
                    continue
                sv = supports[v]
                options = [sv.candidate_a, sv.candidate_b]
                scores = []
                radii = [vec_for_candidate(sv, endpoints[i])[2] for i in options]
                scale = max(max(radii), 1.0)
                for choice, ei in enumerate(options):
                    vv = vec_for_candidate(sv, endpoints[ei])
                    continuity = cosine(vu, vv)
                    outer = vv[2] / scale
                    scores.append((2.0 * continuity + 0.35 * outer, choice))
                assignment[v] = max(scores)[1]
                q.append(v)

        # Handle unusual graph fragments inside comp defensively.
        for n in comp:
            if n not in assignment:
                s = supports[n]
                ra = vec_for_candidate(s, endpoints[s.candidate_a])[2]
                rb = vec_for_candidate(s, endpoints[s.candidate_b])[2]
                assignment[n] = 0 if ra >= rb else 1

        continuity_total = 0.0
        n_edges = 0
        for a, b in edges:
            if a not in comp_set or b not in comp_set:
                continue
            sa, sb = supports[a], supports[b]
            ia = sa.candidate_a if assignment[a] == 0 else sa.candidate_b
            ib = sb.candidate_a if assignment[b] == 0 else sb.candidate_b
            continuity_total += cosine(
                vec_for_candidate(sa, endpoints[ia]),
                vec_for_candidate(sb, endpoints[ib]),
            )
            n_edges += 1
        outer_total = 0.0
        all_r = []
        for n in comp:
            s = supports[n]
            ra = vec_for_candidate(s, endpoints[s.candidate_a])[2]
            rb = vec_for_candidate(s, endpoints[s.candidate_b])[2]
            all_r.extend([ra, rb])
            i = s.candidate_a if assignment[n] == 0 else s.candidate_b
            outer_total += vec_for_candidate(s, endpoints[i])[2]
        scale = np.median([r for r in all_r if r > 0]) if any(r > 0 for r in all_r) else 1.0
        score = continuity_total + 0.15 * (outer_total / max(scale, 1e-9))
        if n_edges:
            score /= n_edges
        return assignment, float(score)

    sol0, score0 = run(0)
    sol1, score1 = run(1)
    chosen = sol0 if score0 >= score1 else sol1
    denom = max(abs(score0), abs(score1), 1e-9)
    ambiguous = abs(score0 - score1) / denom < 0.05
    return chosen, score0, score1, ambiguous


def build_circuit(
    circuit: str,
    wires: gpd.GeoDataFrame,
    centrelines: gpd.GeoDataFrame,
    tolerance: float,
    use_horizontal_center: bool = True,
    horizontal_z_ratio: float = 0.05,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    cl_parts = []
    for g in centrelines.geometry:
        cl_parts.extend(line_parts(g))
    if not cl_parts:
        raise RuntimeError(f"No centreline geometry for circuit {circuit}")

    wires = wires.reset_index(drop=True)
    endpoints, supports, wire_edges = build_supports(
        wires, cl_parts, tolerance, use_horizontal_center, horizontal_z_ratio
    )
    spans = unique_span_pairs(wire_edges)
    comps = connected_components(supports.keys(), spans)

    support_to_comp = {}
    support_choice: dict[int, int] = {}
    comp_meta = {}
    for comp_id, comp in enumerate(comps, 1):
        sol, s0, s1, amb = select_component_side(comp, spans, supports, endpoints)
        for sid, choice in sol.items():
            support_choice[sid] = choice
            support_to_comp[sid] = comp_id
            supports[sid].component = comp_id
            supports[sid].selected = choice
            supports[sid].ambiguous = amb
        comp_meta[comp_id] = {"score_a": s0, "score_b": s1, "ambiguous": amb}

    # QA support + candidates
    support_rows = []
    candidate_rows = []
    for sid, s in sorted(supports.items()):
        selected_idx = selected_endpoint_index(s, support_choice.get(sid, 0))
        sel = endpoints[selected_idx]
        support_rows.append({
            "CIRCUIT": circuit,
            "COMP_ID": s.component,
            "SUPPORT_ID": sid,
            "N_ENDPOINTS": len(s.endpoint_indices),
            "CL_OFFSET": s.centreline_distance,
            "AMBIGUOUS": int(s.ambiguous),
            "SEL_RULE": selection_rule(s),
            "HORIZ_CTR": int(s.horizontal_center),
            "ROW_WIDTH": s.horizontal_row_width,
            "ROW_ZSPAN": s.horizontal_z_span,
            "ROW_RATIO": s.horizontal_ratio,
            "SEL_X": sel.x,
            "SEL_Y": sel.y,
            "SEL_Z": sel.z,
            "geometry": s.centre,
        })
        cand_items = [("A", s.candidate_a), ("B", s.candidate_b)]
        if s.candidate_center is not None:
            cand_items.append(("C", s.candidate_center))
        seen_candidate_indices = set()
        for label, ei in cand_items:
            if ei in seen_candidate_indices:
                continue
            seen_candidate_indices.add(ei)
            e = endpoints[ei]
            dx, dy, radius = vec_for_candidate(s, e)
            candidate_rows.append({
                "CIRCUIT": circuit,
                "COMP_ID": s.component,
                "SUPPORT_ID": sid,
                "CANDIDATE": label,
                "SELECTED": int(ei == selected_idx),
                "RULE": selection_rule(s),
                "OFFSET": radius,
                "Z": e.z,
                "geometry": Point(e.x, e.y),
            })

    # Straight span edges between selected support endpoints.
    edge_rows = []
    for span_no, (a, b) in enumerate(spans, 1):
        if a not in supports or b not in supports:
            continue
        ca = supports[a]
        cb = supports[b]
        ia = selected_endpoint_index(ca, support_choice.get(a, 0))
        ib = selected_endpoint_index(cb, support_choice.get(b, 0))
        ea, eb = endpoints[ia], endpoints[ib]
        geom = LineString([(ea.x, ea.y), (eb.x, eb.y)])
        comp_id = support_to_comp.get(a, support_to_comp.get(b, 0))
        edge_rows.append({
            "CIRCUIT": circuit,
            "COMP_ID": comp_id,
            "SPAN_ID": f"AUTO_{span_no:06d}",
            "SUPPORT_A": a,
            "SUPPORT_B": b,
            "AMBIGUOUS": int(comp_meta.get(comp_id, {}).get("ambiguous", False)),
            "RULE_A": selection_rule(ca),
            "RULE_B": selection_rule(cb),
            "HORIZ_CTR": int(ca.horizontal_center or cb.horizontal_center),
            "LENGTH": float(geom.length),
            "geometry": geom,
        })

    crs = wires.crs
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=crs)
    support_gdf = gpd.GeoDataFrame(support_rows, geometry="geometry", crs=crs)
    cand_gdf = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=crs)

    audit = {
        "CIRCUIT": circuit,
        "N_WIRES": len(wires),
        "N_SUPPORTS": len(supports),
        "N_SPANS": len(spans),
        "N_COMPONENTS": len(comps),
        "N_AMBIG_COMPONENTS": sum(int(v["ambiguous"]) for v in comp_meta.values()),
        "N_HORIZONTAL_CENTER": sum(int(s.horizontal_center) for s in supports.values()),
        "PCT_HORIZONTAL_CENTER": (100.0 * sum(int(s.horizontal_center) for s in supports.values()) / len(supports)) if supports else 0.0,
        "MED_CL_OFFSET": float(np.median([s.centreline_distance for s in supports.values()])) if supports else None,
        "MAX_CL_OFFSET": float(max([s.centreline_distance for s in supports.values()], default=0.0)),
    }
    return edges_gdf, support_gdf, cand_gdf, audit


def aggregate_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.empty:
        return gpd.GeoDataFrame(columns=["CIRCUIT", "PART_ID", "N_SPANS", "LENGTH", "AMBIGUOUS", "geometry"], geometry="geometry", crs=edges.crs)
    rows = []
    for (circuit, comp), sg in edges.groupby(["CIRCUIT", "COMP_ID"], sort=False):
        merged_input = unary_union(list(sg.geometry))
        if merged_input.geom_type == "LineString":
            geom = merged_input
        elif merged_input.geom_type == "MultiLineString":
            geom = linemerge(merged_input)
        else:
            geom = merged_input
        rows.append({
            "CIRCUIT": txt(circuit),
            "PART_ID": int(comp),
            "N_SPANS": int(len(sg)),
            "LENGTH": float(sum(sg.geometry.length)),
            "AMBIGUOUS": int(sg["AMBIGUOUS"].max()),
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)


# -----------------------------------------------------------------------------
# Circuit matching between wires and centrelines
# -----------------------------------------------------------------------------

def circuit_groups(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    out = {}
    for circuit, sg in gdf.groupby("CIRCUIT", dropna=False, sort=False):
        label = normalise_circuit(circuit)
        if not label:
            continue
        out[circuit_key(label)] = sg.assign(CIRCUIT=label)
    return out




def is_generic_circuit_label(value: Any) -> bool:
    """Return True when a CAD layer describes wire ROLE rather than circuit identity.

    Many production DXF/DGN exports use levels such as:
      - Mainline Earth Wire
      - Transmission Phase 1_Top or Left
      - Transmission Phase 2_Middle
      - Transmission Phase 3_Bottom or Right

    Those are wire/phase roles, not circuit IDs. They must therefore be assigned
    spatially to the named circuit centreline(s), exactly like a generic WIRES layer.
    """
    s = normalise_circuit(value).strip().lower()

    if s in {
        "", "0", "default", "wire", "wires", "conductor", "conductors",
        "line", "lines", "centreline", "centrelines", "centerline",
        "centerlines", "alignment", "alignments"
    }:
        return True

    # Common role-based CAD level names from engineering wire exports.
    role_patterns = (
        r"\bphase\b",
        r"\bearth\b",
        r"\bshield\b",
        r"\bground(?:wire)?\b",
        r"\bneutral\b",
        r"\bstatic\b",
        r"\bconductor\b",
        r"\bmainline\b.*\bearth\b",
        r"\btop\s+or\s+left\b",
        r"\bbottom\s+or\s+right\b",
        r"\bmiddle\b",
    )
    return any(re.search(p, s, flags=re.I) for p in role_patterns)


def _line2d(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "LineString":
        return LineString([(c[0], c[1]) for c in geom.coords])
    if geom.geom_type == "MultiLineString":
        return MultiLineString([
            LineString([(c[0], c[1]) for c in part.coords])
            for part in geom.geoms
        ])
    return geom


def assign_generic_wires_to_centrelines(
    wires: gpd.GeoDataFrame,
    centre: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign wires from a generic CAD level (e.g. WIRES) to named circuits.

    Some production CAD exports place every 3D wire on one generic level while
    the separate centreline file carries the circuit identity on its levels.
    This is still project-agnostic: the circuit is inferred purely from NM
    geometry. We sample each wire at 25/50/75% of its plan geometry and assign
    it to the centreline with the smallest median plan distance.
    """
    if wires.empty or centre.empty:
        return wires

    centre_named = centre[~centre["CIRCUIT"].apply(is_generic_circuit_label)].copy()
    if centre_named.empty:
        return wires

    centre_groups: dict[str, tuple[str, Any]] = {}
    for circuit, sg in centre_named.groupby("CIRCUIT", dropna=False, sort=False):
        label = normalise_circuit(circuit)
        key = circuit_key(label)
        geoms = [_line2d(g) for g in sg.geometry if g is not None and not g.is_empty]
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if key and geoms:
            centre_groups[key] = (label, unary_union(geoms))
    if not centre_groups:
        return wires

    out = wires.copy()
    generic_mask = out["CIRCUIT"].apply(is_generic_circuit_label)
    if not generic_mask.any():
        return out

    info(f"Spatially assigning {int(generic_mask.sum()):,} generic wire features to {len(centre_groups):,} named centreline circuit(s)...")
    assign_dist = pd.Series(np.nan, index=out.index, dtype=float)
    assign_method = pd.Series("LEVEL", index=out.index, dtype=object)

    for idx in out.index[generic_mask]:
        geom = _line2d(out.at[idx, "geometry"])
        if geom is None or geom.is_empty or geom.length <= 0:
            continue
        samples = [geom.interpolate(f, normalized=True) for f in (0.25, 0.5, 0.75)]
        scores = []
        for key, (label, clgeom) in centre_groups.items():
            ds = [p.distance(clgeom) for p in samples]
            scores.append((float(np.median(ds)), label))
        best_d, best_label = min(scores, key=lambda t: t[0])
        out.at[idx, "CIRCUIT"] = best_label
        assign_dist.at[idx] = best_d
        assign_method.at[idx] = "SPATIAL_CENTRELINE"

    out["ASSIGN_METHOD"] = assign_method
    out["ASSIGN_DIST"] = assign_dist
    return out

def assign_anonymous_centrelines(
    wire_groups: dict[str, gpd.GeoDataFrame],
    centre: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatial fallback for centreline levels such as CAD Layer 0.

    Only features whose inferred level does not match a wire circuit are
    considered. Each is assigned to the nearest unmatched wire circuit.
    """
    if centre.empty:
        return centre
    known_keys = set(wire_groups)
    centre = centre.copy()
    ckeys = centre["CIRCUIT"].apply(circuit_key)
    unmatched_mask = ~ckeys.isin(known_keys)
    if not unmatched_mask.any():
        return centre

    wire_unions = {k: unary_union(list(g.geometry)) for k, g in wire_groups.items()}
    for idx in centre.index[unmatched_mask]:
        geom = centre.at[idx, "geometry"]
        if geom is None or geom.is_empty:
            continue
        best = min(wire_unions, key=lambda k: geom.distance(wire_unions[k]))
        # Do not relabel obvious named non-circuit layers unless they are Layer 0
        # or blank. This avoids absorbing SUBSTATIONS / SWATHE frame linework.
        level = txt(centre.at[idx, "LEVEL"]).lower()
        if level in {"", "0", "default"}:
            centre.at[idx, "CIRCUIT"] = normalise_circuit(wire_groups[best]["CIRCUIT"].iloc[0])
    return centre


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_layer(gdf: gpd.GeoDataFrame, gpkg: Path, layer: str) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio")


def safe_shapefile_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    # Shapefile cannot store arbitrary GeometryCollection. Ensure only linework.
    out = out[out.geometry.apply(is_line)].copy()
    return out


def run(
    wires_path: Path,
    centreline_path: Path,
    output: Path,
    epsg: int | None,
    support_tolerance: float | None,
    use_horizontal_center: bool = True,
    horizontal_z_ratio: float = 0.05,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wires2d_01_") as td:
        td = Path(td)
        info("Reading wires...")
        wires = read_vector(wires_path, td)
        info(f"  {len(wires):,} wire line features")
        info("Reading centrelines...")
        centre = read_vector(centreline_path, td)
        info(f"  {len(centre):,} centreline features")

        if wires.empty:
            raise RuntimeError("No line geometry found in wires input")
        if centre.empty:
            raise RuntimeError("No line geometry found in centreline input")

        target_crs = wires.crs or centre.crs
        if target_crs is None:
            if epsg is None:
                epsg = prompt_epsg()
            target_crs = CRS.from_epsg(epsg)
        wires = set_or_reproject(wires, target_crs)
        centre = set_or_reproject(centre, target_crs)

        tol = support_tolerance if support_tolerance is not None else default_support_tolerance(target_crs)
        info(f"Support clustering tolerance: {tol:g} {unit_name(target_crs) or 'CRS units'}")

        # First resolve generic wire levels such as a single CAD level named
        # WIRES against named circuit centrelines. This must happen before the
        # normal circuit-key matching.
        wires = assign_generic_wires_to_centrelines(wires, centre)
        wg = circuit_groups(wires)
        centre = assign_anonymous_centrelines(wg, centre)
        cg = circuit_groups(centre)

        matched = sorted(set(wg) & set(cg))
        unmatched_w = sorted(set(wg) - set(cg))
        if unmatched_w:
            info("WARNING: no centreline for: " + ", ".join(wg[k]["CIRCUIT"].iloc[0] for k in unmatched_w))
        if not matched:
            raise RuntimeError("No circuit IDs could be matched between wires and centrelines")

        all_edges = []
        all_supports = []
        all_candidates = []
        audits = []
        used_centres = []

        for key in matched:
            w = wg[key].copy()
            c = cg[key].copy()
            circuit = txt(w["CIRCUIT"].iloc[0])
            info(f"\n{circuit}: {len(w):,} wires, {len(c):,} centreline feature(s)")
            edges, supports, candidates, audit = build_circuit(
                circuit, w, c, tol, use_horizontal_center, horizontal_z_ratio
            )
            info(f"  supports={audit['N_SUPPORTS']:,} spans={audit['N_SPANS']:,} components={audit['N_COMPONENTS']}")
            info(f"  horizontal-centre rule={audit['N_HORIZONTAL_CENTER']:,} supports ({audit['PCT_HORIZONTAL_CENTER']:.1f}%)")
            if audit["N_AMBIG_COMPONENTS"]:
                info(f"  QA: {audit['N_AMBIG_COMPONENTS']} component(s) have near-equal left/right solutions")
            all_edges.append(edges)
            all_supports.append(supports)
            all_candidates.append(candidates)
            audits.append(audit)
            used_centres.append(c.assign(CIRCUIT=circuit))

        edges = gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True), geometry="geometry", crs=target_crs) if all_edges else gpd.GeoDataFrame(geometry=[], crs=target_crs)
        supports = gpd.GeoDataFrame(pd.concat(all_supports, ignore_index=True), geometry="geometry", crs=target_crs) if all_supports else gpd.GeoDataFrame(geometry=[], crs=target_crs)
        candidates = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), geometry="geometry", crs=target_crs) if all_candidates else gpd.GeoDataFrame(geometry=[], crs=target_crs)
        centres = gpd.GeoDataFrame(pd.concat(used_centres, ignore_index=True), geometry="geometry", crs=target_crs) if used_centres else gpd.GeoDataFrame(geometry=[], crs=target_crs)
        raw = aggregate_edges(edges)

        gpkg = output / "01_raw_2d_wires.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        write_layer(raw, gpkg, "wires_2d")
        write_layer(edges, gpkg, "span_edges")
        write_layer(supports, gpkg, "support_points")
        write_layer(candidates, gpkg, "outer_candidates")
        write_layer(centres, gpkg, "centrelines")

        shp = output / "raw_2d_wires.shp"
        for p in output.glob("raw_2d_wires.*"):
            if p.is_file():
                p.unlink()
        safe_shapefile_gdf(raw).to_file(shp, driver="ESRI Shapefile", engine="pyogrio")

        audit_df = pd.DataFrame(audits)
        audit_df.to_csv(output / "01_geometry_audit.csv", index=False)

        info("\nWritten:")
        info(f"  {gpkg}")
        info(f"  {shp}")
        info(f"  {output / '01_geometry_audit.csv'}")
        return gpkg


def main() -> int:
    parser = argparse.ArgumentParser(description="Build project-agnostic raw 2D wire aggregates from NM wires + centrelines")
    parser.add_argument("wires", nargs="?", help="3D wires: DGN/DXF/SHP/GPKG/ZIP")
    parser.add_argument("centrelines", nargs="?", help="Circuit centrelines: DGN/DXF/SHP/GPKG/ZIP")
    parser.add_argument("-o", "--output", default="01_2d_wires_output")
    parser.add_argument("--epsg", type=int, default=None, help="CRS EPSG when CAD input carries no CRS")
    parser.add_argument("--support-tolerance", type=float, default=None, help="Endpoint clustering tolerance in CRS units")
    parser.add_argument(
        "--horizontal-z-ratio",
        type=float,
        default=0.05,
        help="Maximum vertical-spread / lateral-width ratio for the horizontal-centre rule (default: 0.05)",
    )
    parser.add_argument(
        "--no-horizontal-centre", "--no-horizontal-center",
        action="store_true",
        help="Disable the horizontal middle-phase-over-structure override",
    )
    args = parser.parse_args()

    allowed = {".dgn", ".dxf", ".shp", ".gpkg", ".zip"}
    wires = clean_path(args.wires) if args.wires else prompt_path("3D wires (.dgn/.dxf/.shp/.gpkg/.zip): ", allowed)
    centre = clean_path(args.centrelines) if args.centrelines else prompt_path("Circuit centrelines (.dgn/.dxf/.shp/.gpkg/.zip): ", allowed)
    try:
        run(
            wires, centre, Path(args.output), args.epsg, args.support_tolerance,
            use_horizontal_center=not args.no_horizontal_centre,
            horizontal_z_ratio=args.horizontal_z_ratio,
        )
        return 0
    except Exception as exc:
        info(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
