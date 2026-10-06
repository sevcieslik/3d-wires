#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from sklearn.cluster import DBSCAN

DEFAULT_WIRE_CLASS = 187
DEFAULT_STRUCTURE_CLASS = 215
SOURCE_LINES_NAME = "T_OH_Lines_LiDARexportUpdate.shp"
SOURCE_STRUCTURES_NAME = "T_Structures_LiDARexport.shp"
KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

def find_lidar_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".las", ".laz"}:
        return [root]
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".las", ".laz"})
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under: {root}")
    return files

def header_crs(path: Path):
    try:
        with laspy.open(path) as reader:
            return reader.header.parse_crs()
    except Exception:
        return None

def resolve_lidar_crs(files: list[Path], epsg: int | None):
    embedded = []
    for path in files:
        crs = header_crs(path)
        if crs is not None:
            embedded.append((path, CRS.from_user_input(crs)))
    if embedded:
        base = embedded[0][1]
        for path, other in embedded[1:]:
            if not base.equals(other, ignore_axis_order=True):
                raise RuntimeError(f"Mixed LiDAR CRS detected between {embedded[0][0].name} and {path.name}")
        print(f"LiDAR CRS: {base.to_string()} [embedded]")
        return base
    if epsg is None:
        raise RuntimeError(
            "LiDAR files do not contain CRS metadata. Source GIS/KMZ is geographic, so a LiDAR CRS is required.\n"
            "Run again with e.g. --epsg <projected EPSG code>."
        )
    crs = CRS.from_epsg(epsg)
    if not crs.is_projected:
        raise RuntimeError(f"EPSG:{epsg} is not projected. A projected LiDAR CRS is required.")
    print(f"LiDAR CRS: {crs.to_string()} [--epsg]")
    return crs

def unit_to_metre(crs: CRS) -> float:
    try:
        value = float(crs.axis_info[0].unit_conversion_factor)
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return 1.0

def load_lidar_classes_one_pass(files: list[Path], wire_class: int, structure_class: int, chunk_size: int = 2_000_000):
    wires, structures = [], []
    total_wire = total_structure = 0
    print("\nReading LiDAR once for topology classes...")
    for i, path in enumerate(files, start=1):
        if i == 1 or i % 10 == 0 or i == len(files):
            print(f"  tile {i:,}/{len(files):,}: {path.name}")
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                cls = np.asarray(chunk.classification)
                mask_wire = cls == wire_class
                if np.any(mask_wire):
                    arr = np.column_stack((
                        np.asarray(chunk.x, dtype=np.float64)[mask_wire],
                        np.asarray(chunk.y, dtype=np.float64)[mask_wire],
                        np.asarray(chunk.z, dtype=np.float64)[mask_wire],
                    ))
                    wires.append(arr)
                    total_wire += len(arr)
                mask_structure = cls == structure_class
                if np.any(mask_structure):
                    arr = np.column_stack((
                        np.asarray(chunk.x, dtype=np.float64)[mask_structure],
                        np.asarray(chunk.y, dtype=np.float64)[mask_structure],
                        np.asarray(chunk.z, dtype=np.float64)[mask_structure],
                    ))
                    structures.append(arr)
                    total_structure += len(arr)
    wire_xyz = np.vstack(wires) if wires else np.empty((0, 3), dtype=np.float64)
    structure_xyz = np.vstack(structures) if structures else np.empty((0, 3), dtype=np.float64)
    print(f"  class {wire_class} wire points: {total_wire:,}")
    print(f"  class {structure_class} structure points: {total_structure:,}")
    return wire_xyz, structure_xyz


def load_or_cache_topology_classes(
    files: list[Path],
    wire_class: int,
    structure_class: int,
    cache_dir: Path,
    rebuild: bool = False,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    wire_cache = cache_dir / f"class_{wire_class}_xyz.npy"
    structure_cache = cache_dir / f"class_{structure_class}_xyz.npy"

    if not rebuild and wire_cache.exists() and structure_cache.exists():
        print("\nLoading cached topology classes...")
        wire_xyz = np.load(wire_cache, mmap_mode=None)
        structure_xyz = np.load(structure_cache, mmap_mode=None)
        print(f"  class {wire_class} wire points: {len(wire_xyz):,} [cache]")
        print(f"  class {structure_class} structure points: {len(structure_xyz):,} [cache]")
        return wire_xyz, structure_xyz

    wire_xyz, structure_xyz = load_lidar_classes_one_pass(
        files,
        wire_class,
        structure_class,
    )
    np.save(wire_cache, wire_xyz)
    np.save(structure_cache, structure_xyz)
    print(f"  cache written: {cache_dir}")
    return wire_xyz, structure_xyz

def locate_source_shapefiles(source: Path):
    temp = None
    if source.is_file() and source.suffix.lower() == ".zip":
        temp = tempfile.TemporaryDirectory(prefix="nm_source_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(temp.name)
        root = Path(temp.name)
    else:
        root = source
    lines = list(root.rglob(SOURCE_LINES_NAME))
    structures = list(root.rglob(SOURCE_STRUCTURES_NAME))
    if not lines:
        raise FileNotFoundError(f"Could not find {SOURCE_LINES_NAME} under {source}")
    if not structures:
        raise FileNotFoundError(f"Could not find {SOURCE_STRUCTURES_NAME} under {source}")
    return temp, lines[0], structures[0]

def read_source_data(source: Path, target_crs: CRS):
    temp, lines_path, structures_path = locate_source_shapefiles(source)
    print("\nReading client source:")
    print(f"  lines:      {lines_path.name}")
    print(f"  structures: {structures_path.name}")
    lines = gpd.read_file(lines_path)
    structures = gpd.read_file(structures_path)
    if lines.crs is None or structures.crs is None:
        raise RuntimeError("Client source shapefile CRS is missing.")
    print(f"  source line features: {len(lines):,}")
    print(f"  source structures:    {len(structures):,}")
    print(f"  source CRS:           {lines.crs}")
    for fld in ("LINE_NO",):
        if fld not in lines.columns:
            raise RuntimeError(f"Source lines missing required field: {fld}")
    for fld in ("LINE_NO", "GIS_NO"):
        if fld not in structures.columns:
            raise RuntimeError(f"Source structures missing required field: {fld}")
    # Remove source records that cannot participate in spatial matching.
    # The supplied NMIP26068 structure layer contains at least one NULL geometry.
    bad_lines = lines.geometry.isna() | lines.geometry.is_empty
    bad_structures = structures.geometry.isna() | structures.geometry.is_empty

    if bad_lines.any():
        print(f"  source lines with null/empty geometry skipped: {int(bad_lines.sum()):,}")
        lines = lines.loc[~bad_lines].copy()

    if bad_structures.any():
        print(f"  source structures with null/empty geometry skipped: {int(bad_structures.sum()):,}")
        structures = structures.loc[~bad_structures].copy()

    lines = lines.to_crs(target_crs)
    structures = structures.to_crs(target_crs)
    lines["LINE_NO"] = lines["LINE_NO"].astype(str).str.strip()
    structures["LINE_NO"] = structures["LINE_NO"].astype(str).str.strip()
    structures["GIS_NO"] = structures["GIS_NO"].astype(str).str.strip()
    return temp, lines, structures

def _kml_text(elem, tag):
    child = elem.find(f"kml:{tag}", KML_NS)
    return child.text.strip() if child is not None and child.text else ""

def read_kmz_lines(kmz_path: Path, target_crs: CRS) -> gpd.GeoDataFrame:
    rows = []
    with zipfile.ZipFile(kmz_path) as zf:
        names = zf.namelist()
        kml_name = "doc.kml" if "doc.kml" in names else next((n for n in names if n.lower().endswith(".kml")), None)
        if not kml_name:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        root = ET.fromstring(zf.read(kml_name))
    def walk(container, folder_path):
        for child in list(container):
            tag = child.tag.split("}")[-1]
            if tag == "Folder":
                folder_name = _kml_text(child, "name")
                walk(child, folder_path + ([folder_name] if folder_name else []))
            elif tag == "Placemark":
                name = _kml_text(child, "name")
                for ls in child.findall(".//kml:LineString", KML_NS):
                    node = ls.find("kml:coordinates", KML_NS)
                    if node is None or not node.text:
                        continue
                    coords = []
                    for token in node.text.replace("\n", " ").split():
                        parts = token.split(",")
                        if len(parts) >= 2:
                            try:
                                coords.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                pass
                    if len(coords) >= 2:
                        rows.append({"KMZ_NAME": name, "FOLDER_PATH": " / ".join(folder_path), "geometry": LineString(coords)})
    document = root.find(".//kml:Document", KML_NS)
    if document is not None:
        walk(document, [])
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if not gdf.empty:
        gdf = gdf.to_crs(target_crs)
    print(f"  KMZ line features:    {len(gdf):,}")
    return gdf

def cluster_lidar_structures(xyz, crs, unit_to_m, eps_m, min_points):
    if len(xyz) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    labels = DBSCAN(eps=eps_m / unit_to_m, min_samples=min_points, n_jobs=-1).fit_predict(xyz[:, :2])
    rows = []
    next_id = 1
    print("\nClustering LiDAR structures...")
    for label in sorted(set(int(v) for v in labels if v >= 0)):
        pts = xyz[labels == label]
        centre = np.median(pts, axis=0)
        spread = np.linalg.norm(pts[:, :2] - centre[None, :2], axis=1)
        rows.append({
            "LIDAR_STRUCTURE_ID": f"LS{next_id:05d}",
            "POINTS_215": int(len(pts)),
            "X": float(centre[0]), "Y": float(centre[1]), "Z_MED": float(centre[2]),
            "R95_M": float(np.quantile(spread, 0.95) * unit_to_m),
            "geometry": Point(float(centre[0]), float(centre[1]), float(centre[2])),
        })
        next_id += 1
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    print(f"  LiDAR structure clusters: {len(result):,}")
    return result

def parse_structure_series(value: str) -> tuple[str, int | None]:
    """
    Split a structure identifier into a branch/series and terminal ordinal.

    Examples:
      13~136  -> ("13", 136)
      13A~7   -> ("13A", 7)
      136     -> ("DEFAULT", 136)
      P12     -> ("P", 12)

    The series is used only to prevent unrelated branches within the same
    LINE_NO from being interleaved by reference-line stationing.
    """
    raw = str(value or "").strip()
    if not raw:
        return "UNKNOWN", None

    parts = [part.strip() for part in raw.split("~") if part.strip()]
    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[-1]):
        return "~".join(parts[:-1]), int(parts[-1])

    match = re.fullmatch(r"(.*?)(\d+)", raw)
    if match:
        prefix = match.group(1).rstrip("~-_ ").strip()
        return (prefix or "DEFAULT"), int(match.group(2))

    return raw, None


def match_structures(client, lidar, unit_to_m, max_distance_m):
    if client.empty or lidar.empty:
        return gpd.GeoDataFrame(geometry=[], crs=lidar.crs)
    lidar_xy = np.column_stack((lidar["X"], lidar["Y"]))
    tree = cKDTree(lidar_xy)
    radius = max_distance_m / unit_to_m
    candidates = []
    for client_idx, row in client.iterrows():
        p = np.asarray([row.geometry.x, row.geometry.y], dtype=np.float64)
        for lidar_idx in tree.query_ball_point(p, r=radius):
            candidates.append((float(np.linalg.norm(lidar_xy[lidar_idx] - p)), client_idx, int(lidar_idx)))
    candidates.sort(key=lambda x: x[0])
    used_client, used_lidar, rows = set(), set(), []
    for d_source, client_idx, lidar_idx in candidates:
        if client_idx in used_client or lidar_idx in used_lidar:
            continue
        c = client.loc[client_idx]
        l = lidar.iloc[lidar_idx]
        structure_no = str(c.get("STRUCTURE_", "")).strip()
        if not structure_no:
            gis_no = str(c.get("GIS_NO", "")).strip()
            structure_no = gis_no.split("~", 1)[1] if "~" in gis_no else gis_no
        series, series_index = parse_structure_series(structure_no)

        rows.append({
            "LINE_NO": str(c.get("LINE_NO", "")),
            "GIS_NO": str(c.get("GIS_NO", "")),
            "STRUCTURE_NO": structure_no,
            "STRUCTURE_SERIES": series,
            "SERIES_INDEX": series_index,
            "CLIENT_GLOBALID": str(c.get("GLOBALID", "")),
            "LIDAR_STRUCTURE_ID": l["LIDAR_STRUCTURE_ID"],
            "MATCH_DIST_M": d_source * unit_to_m,
            "POINTS_215": int(l["POINTS_215"]),
            "X": float(l["X"]), "Y": float(l["Y"]), "Z_MED": float(l["Z_MED"]),
            "geometry": l.geometry,
        })
        used_client.add(client_idx)
        used_lidar.add(lidar_idx)
    matched = gpd.GeoDataFrame(rows, geometry="geometry", crs=lidar.crs)
    print("\nClient/LiDAR structure matching:")
    print(f"  matched:          {len(matched):,}")
    print(f"  client unmatched: {len(client) - len(used_client):,}")
    print(f"  LiDAR unmatched:  {len(lidar) - len(used_lidar):,}")
    return matched

def merged_line_by_line_no(lines):
    result = {}

    for line_no, group in lines.groupby("LINE_NO"):
        geoms = [
            g for g in group.geometry
            if g is not None and not g.is_empty
        ]
        if not geoms:
            continue

        # Shapely 2.x linemerge expects a multi-line collection/sequence.
        # unary_union may legitimately return a single LineString.
        if len(geoms) == 1:
            merged = geoms[0]
        else:
            unioned = unary_union(geoms)

            if isinstance(unioned, LineString):
                merged = unioned

            elif isinstance(unioned, MultiLineString):
                merged = linemerge(unioned)

            else:
                try:
                    line_parts = [
                        g for g in unioned.geoms
                        if isinstance(g, LineString)
                    ]
                except Exception:
                    line_parts = []

                if not line_parts:
                    continue

                merged = (
                    line_parts[0]
                    if len(line_parts) == 1
                    else linemerge(line_parts)
                )

        result[str(line_no)] = merged

    return result

def best_linestring_for_points(reference_geom, points_xy):
    if isinstance(reference_geom, LineString):
        return reference_geom
    if isinstance(reference_geom, MultiLineString):
        parts = list(reference_geom.geoms)
    else:
        try:
            parts = [g for g in reference_geom.geoms if isinstance(g, LineString)]
        except Exception:
            return None
    if not parts:
        return None
    pts = [Point(float(x), float(y)) for x, y in points_xy]
    return min(parts, key=lambda part: sum(part.distance(p) for p in pts))

def span_frame(a_xy, b_xy):
    delta = b_xy - a_xy
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return None
    u = delta / length
    return {"origin": a_xy, "u": u, "p": np.asarray([-u[1], u[0]], dtype=np.float64), "length": length}

def local_wire_subset(wire_xyz, wire_tree, frame, half_width_source, end_margin_source):
    midpoint = frame["origin"] + 0.5 * frame["length"] * frame["u"]
    radius = math.hypot(0.5 * frame["length"] + end_margin_source, half_width_source)
    ids = wire_tree.query_ball_point(midpoint, r=radius)
    return wire_xyz[np.asarray(ids, dtype=np.int64)] if ids else np.empty((0, 3), dtype=np.float64)

def wire_evidence(local, frame, unit_to_m, half_width_m, end_margin_m, bins):
    empty = {"WIRE_POINTS": 0, "WIRE_COVERAGE": 0.0, "LONGEST_RUN": 0.0, "START_OK": False, "END_OK": False, "MED_ABS_T_M": float("inf")}
    if len(local) == 0:
        return empty
    half, end = half_width_m / unit_to_m, end_margin_m / unit_to_m
    centred = local[:, :2] - frame["origin"][None, :]
    s, t = centred @ frame["u"], centred @ frame["p"]
    mask = (s >= -end) & (s <= frame["length"] + end) & (np.abs(t) <= half)
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return empty
    s_inside = s[ids]
    valid = (s_inside >= 0.0) & (s_inside <= frame["length"])
    occupied = np.zeros(bins, dtype=bool)
    if np.any(valid):
        idx = np.floor(np.clip(s_inside[valid] / max(frame["length"], 1e-9), 0.0, 0.999999) * bins).astype(int)
        occupied[np.unique(idx)] = True
    longest = current = 0
    for value in occupied:
        current = current + 1 if value else 0
        longest = max(longest, current)
    edge_bins = max(1, min(3, bins // 5))
    return {
        "WIRE_POINTS": int(len(ids)),
        "WIRE_COVERAGE": float(occupied.mean()),
        "LONGEST_RUN": float(longest / bins),
        "START_OK": bool(np.any(occupied[:edge_bins])),
        "END_OK": bool(np.any(occupied[-edge_bins:])),
        "MED_ABS_T_M": float(np.median(np.abs(t[ids])) * unit_to_m),
    }

def build_reference_ordered_spans(
    matched,
    source_lines,
    wire_xyz,
    unit_to_m,
    corridor_halfwidth_m,
    end_margin_m,
    evidence_bins,
    min_wire_points,
    min_wire_coverage,
    min_longest_run,
    min_span_m,
    max_span_m,
):
    line_map = merged_line_by_line_no(source_lines)
    wire_tree = cKDTree(wire_xyz[:, :2])
    accepted, rejected, reference_rows = [], [], []

    line_numbers = sorted(set(matched["LINE_NO"]) & set(line_map))
    print("\nBuilding ordered topology:")
    print(
        f"  LINE_NO values with matched structures + source line: "
        f"{len(line_numbers):,}"
    )

    branch_count = 0

    for line_i, line_no in enumerate(line_numbers, start=1):
        line_group = matched[matched["LINE_NO"] == line_no].copy()
        if len(line_group) < 2:
            continue

        all_points_xy = np.column_stack((line_group["X"], line_group["Y"]))
        ref_line = best_linestring_for_points(
            line_map[line_no],
            all_points_xy,
        )
        if ref_line is None:
            continue

        reference_rows.append({
            "LINE_NO": line_no,
            "geometry": ref_line,
        })

        line_group["_station"] = [
            float(ref_line.project(Point(float(x), float(y))))
            for x, y in all_points_xy
        ]

        # Critical topology rule:
        # LINE_NO can contain parallel/branch structure sequences such as
        # 13~136...140 and 13A~1...10. Do not interleave them by station.
        series_groups = []
        for series, branch in line_group.groupby(
            "STRUCTURE_SERIES",
            dropna=False,
        ):
            branch = branch.copy()
            if len(branch) < 2:
                continue

            # Station is the primary order because client numbering can have gaps.
            # SERIES_INDEX is a stable tie-breaker only.
            branch["_series_sort"] = pd.to_numeric(
                branch["SERIES_INDEX"],
                errors="coerce",
            )
            branch = branch.sort_values(
                ["_station", "_series_sort"],
                na_position="last",
            ).reset_index(drop=True)

            series_groups.append((str(series), branch))

        branch_count += len(series_groups)

        if line_i == 1 or line_i % 25 == 0 or line_i == len(line_numbers):
            branch_summary = ", ".join(
                f"{series}:{len(branch)}"
                for series, branch in series_groups
            )
            print(
                f"  line {line_i:,}/{len(line_numbers):,}: "
                f"{line_no} | {len(line_group):,} matched structures | "
                f"series [{branch_summary}]"
            )

        for series, group in series_groups:
            for i in range(len(group) - 1):
                a = group.iloc[i]
                b = group.iloc[i + 1]

                a_xy = np.asarray([a["X"], a["Y"]], dtype=np.float64)
                b_xy = np.asarray([b["X"], b["Y"]], dtype=np.float64)

                frame = span_frame(a_xy, b_xy)
                if frame is None:
                    continue

                length_m = frame["length"] * unit_to_m

                base = {
                    "LINE_NO": line_no,
                    "STRUCTURE_SERIES": series,
                    "SERIES_INDEX_A": a.get("SERIES_INDEX"),
                    "SERIES_INDEX_B": b.get("SERIES_INDEX"),
                    "STRUCT_A": a["GIS_NO"],
                    "STRUCT_B": b["GIS_NO"],
                    "LIDAR_A": a["LIDAR_STRUCTURE_ID"],
                    "LIDAR_B": b["LIDAR_STRUCTURE_ID"],
                    "LENGTH_M": float(length_m),
                    "REF_STATION_A": float(a["_station"]),
                    "REF_STATION_B": float(b["_station"]),
                    "MATCH_A_M": float(a["MATCH_DIST_M"]),
                    "MATCH_B_M": float(b["MATCH_DIST_M"]),
                    "geometry": LineString([
                        (float(a_xy[0]), float(a_xy[1])),
                        (float(b_xy[0]), float(b_xy[1])),
                    ]),
                }

                if length_m < min_span_m:
                    rejected.append({
                        **base,
                        "REJECT_REASON": "TOO_SHORT",
                    })
                    continue

                if length_m > max_span_m:
                    rejected.append({
                        **base,
                        "REJECT_REASON":
                            "TOO_LONG_OR_MISSING_INTERMEDIATE_STRUCTURE",
                    })
                    continue

                local = local_wire_subset(
                    wire_xyz,
                    wire_tree,
                    frame,
                    corridor_halfwidth_m / unit_to_m,
                    end_margin_m / unit_to_m,
                )

                evidence = wire_evidence(
                    local,
                    frame,
                    unit_to_m,
                    corridor_halfwidth_m,
                    end_margin_m,
                    evidence_bins,
                )

                row = {**base, **evidence}

                if evidence["WIRE_POINTS"] < min_wire_points:
                    row["REJECT_REASON"] = "INSUFFICIENT_WIRE_POINTS"
                    rejected.append(row)
                    continue

                if evidence["WIRE_COVERAGE"] < min_wire_coverage:
                    row["REJECT_REASON"] = "LOW_WIRE_COVERAGE"
                    rejected.append(row)
                    continue

                if evidence["LONGEST_RUN"] < min_longest_run:
                    row["REJECT_REASON"] = "WIRE_NOT_CONTINUOUS"
                    rejected.append(row)
                    continue

                if not evidence["START_OK"]:
                    row["REJECT_REASON"] = "NO_WIRE_NEAR_STRUCTURE_A"
                    rejected.append(row)
                    continue

                if not evidence["END_OK"]:
                    row["REJECT_REASON"] = "NO_WIRE_NEAR_STRUCTURE_B"
                    rejected.append(row)
                    continue

                row["SPAN_ID"] = f"SP{len(accepted) + 1:05d}"
                row["TOPOLOGY_SOURCE"] = (
                    "CLIENT_LINE_SERIES_ORDER_"
                    "LIDAR_GEOMETRY_WIRE_CONFIRMED"
                )
                accepted.append(row)

    crs = matched.crs
    accepted_gdf = gpd.GeoDataFrame(
        accepted,
        geometry="geometry",
        crs=crs,
    )
    rejected_gdf = gpd.GeoDataFrame(
        rejected,
        geometry="geometry",
        crs=crs,
    )
    reference_gdf = gpd.GeoDataFrame(
        reference_rows,
        geometry="geometry",
        crs=crs,
    )

    print(f"  structure series processed: {branch_count:,}")
    print("\nTopology result:")
    print(f"  accepted spans: {len(accepted_gdf):,}")
    print(f"  rejected spans: {len(rejected_gdf):,}")

    if not rejected_gdf.empty:
        print("\nReject reasons:")
        for reason, count in rejected_gdf["REJECT_REASON"].value_counts().items():
            print(f"  {reason}: {count:,}")

    return accepted_gdf, rejected_gdf, reference_gdf

def build_parser():
    p = argparse.ArgumentParser(description="Build survey-derived topology using client source for identity/order and LiDAR 215/187 for geometry/evidence.")
    p.add_argument("lidar", help="LAS/LAZ file or folder")
    p.add_argument("--source", required=True, help="NMIP26068 source ZIP or extracted source folder")
    p.add_argument("--kmz", default=None, help="Optional SoW KMZ retained as reference evidence")
    p.add_argument("--epsg", type=int, default=None, help="LiDAR projected EPSG; required if LAS/LAZ lacks CRS metadata")
    p.add_argument("-o", "--output", default="01_reference_topology_output")
    p.add_argument("--wire-class", type=int, default=187)
    p.add_argument("--structure-class", type=int, default=215)
    p.add_argument("--structure-cluster-eps-m", type=float, default=1.5)
    p.add_argument("--structure-min-points", type=int, default=3)
    p.add_argument("--structure-match-m", type=float, default=35.0)
    p.add_argument("--span-corridor-halfwidth-m", type=float, default=18.0)
    p.add_argument("--span-end-margin-m", type=float, default=12.0)
    p.add_argument("--span-evidence-bins", type=int, default=20)
    p.add_argument("--span-min-wire-points", type=int, default=40)
    p.add_argument("--span-min-wire-coverage", type=float, default=0.70)
    p.add_argument("--span-min-longest-run", type=float, default=0.60)
    p.add_argument("--min-span-m", type=float, default=10.0)
    p.add_argument("--max-span-m", type=float, default=650.0)
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached class 187/215 arrays and reread all LAS/LAZ files.",
    )
    return p

def main():
    args = build_parser().parse_args()
    lidar_root = Path(args.lidar).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    kmz_path = Path(args.kmz).expanduser().resolve() if args.kmz else None
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("REFERENCE-ASSISTED LIDAR TOPOLOGY BUILDER")
    print("=" * 80)
    files = find_lidar_files(lidar_root)
    print(f"LiDAR tiles: {len(files):,}")
    crs = resolve_lidar_crs(files, args.epsg)
    u2m = unit_to_metre(crs)
    print(f"CRS unit to metre factor: {u2m}")
    cache_dir = output / "_cache"
    wire_xyz, structure_xyz = load_or_cache_topology_classes(
        files,
        args.wire_class,
        args.structure_class,
        cache_dir,
        rebuild=args.rebuild_cache,
    )
    if len(wire_xyz) == 0:
        raise RuntimeError(f"No points found in wire class {args.wire_class}.")
    if len(structure_xyz) == 0:
        raise RuntimeError(f"No points found in structure class {args.structure_class}.")
    temp, source_lines, source_structures = read_source_data(source_path, crs)
    kmz_lines = gpd.GeoDataFrame(geometry=[], crs=crs)
    if kmz_path is not None:
        print("\nReading optional KMZ reference:")
        kmz_lines = read_kmz_lines(kmz_path, crs)
    lidar_structures = cluster_lidar_structures(structure_xyz, crs, u2m, args.structure_cluster_eps_m, args.structure_min_points)
    matched = match_structures(source_structures, lidar_structures, u2m, args.structure_match_m)
    accepted, rejected, used_reference_lines = build_reference_ordered_spans(
        matched, source_lines, wire_xyz, u2m,
        args.span_corridor_halfwidth_m, args.span_end_margin_m,
        args.span_evidence_bins, args.span_min_wire_points,
        args.span_min_wire_coverage, args.span_min_longest_run,
        args.min_span_m, args.max_span_m,
    )
    gpkg = output / "01_reference_topology.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    lidar_structures.to_file(gpkg, layer="lidar_structures", driver="GPKG", engine="pyogrio")
    matched.to_file(gpkg, layer="matched_structures", driver="GPKG", engine="pyogrio")
    used_reference_lines.to_file(gpkg, layer="reference_lines_used", driver="GPKG", engine="pyogrio")
    if not accepted.empty:
        accepted.to_file(gpkg, layer="accepted_spans", driver="GPKG", engine="pyogrio")
    if not rejected.empty:
        rejected.to_file(gpkg, layer="rejected_spans", driver="GPKG", engine="pyogrio")
    if not kmz_lines.empty:
        kmz_lines.to_file(gpkg, layer="kmz_reference", driver="GPKG", engine="pyogrio")
    if not matched.empty:
        matched.drop(columns="geometry").to_csv(output / "matched_structures.csv", index=False)
    if not accepted.empty:
        accepted.drop(columns="geometry").to_csv(output / "accepted_spans.csv", index=False)
    if not rejected.empty:
        rejected.drop(columns="geometry").to_csv(output / "rejected_spans.csv", index=False)
    print("\nWritten:")
    print(f"  {gpkg}")
    print("\nFirst review these GPKG layers:")
    print("  1. lidar_structures")
    print("  2. matched_structures")
    print("  3. reference_lines_used")
    print("  4. accepted_spans")
    print("  5. rejected_spans")
    _ = temp
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
