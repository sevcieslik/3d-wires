from __future__ import annotations

import hashlib
import json
import math
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from sklearn.cluster import DBSCAN


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
SOURCE_LINES_NAME = "T_OH_Lines_LiDARexportUpdate.shp"
SOURCE_STRUCTURES_NAME = "T_Structures_LiDARexport.shp"


def rows_to_gdf(rows, crs):
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([], crs=crs)},
        geometry="geometry",
        crs=crs,
    )


def find_lidar_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".las", ".laz"}:
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under: {root}")
    return files


def resolve_lidar_crs(files: list[Path], epsg: int | None) -> CRS:
    embedded = []
    for path in files:
        try:
            with laspy.open(path) as reader:
                crs = reader.header.parse_crs()
        except Exception:
            crs = None
        if crs is not None:
            embedded.append((path, CRS.from_user_input(crs)))

    if embedded:
        base = embedded[0][1]
        for path, other in embedded[1:]:
            if not base.equals(other, ignore_axis_order=True):
                raise RuntimeError(
                    f"Mixed CRS metadata: {embedded[0][0].name} vs {path.name}"
                )
        return base

    if epsg is None:
        raise RuntimeError(
            "LiDAR CRS is missing. Supply --epsg with the projected LiDAR CRS."
        )

    crs = CRS.from_epsg(epsg)
    if not crs.is_projected:
        raise RuntimeError(f"EPSG:{epsg} is not projected.")
    return crs


def unit_to_metre(crs: CRS) -> float:
    try:
        value = float(crs.axis_info[0].unit_conversion_factor)
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return 1.0


def dataset_signature(files: list[Path]) -> dict:
    records = []
    for path in files:
        st = path.stat()
        records.append({
            "path": str(path.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        })
    raw = json.dumps(records, sort_keys=True).encode("utf-8")
    return {
        "files": records,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_classes_cached(
    files: list[Path],
    classes: list[int],
    cache_dir: Path,
    rebuild: bool = False,
    chunk_size: int = 2_000_000,
) -> dict[int, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "cache_manifest.json"
    current_sig = dataset_signature(files)

    class_paths = {
        code: cache_dir / f"class_{code}_xyz.npy"
        for code in classes
    }

    cache_valid = False
    if not rebuild and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_valid = (
                previous.get("dataset_sha256") == current_sig["sha256"]
                and all(path.exists() for path in class_paths.values())
            )
        except Exception:
            cache_valid = False

    if cache_valid:
        print("\nLoading cached LiDAR classes...")
        result = {}
        for code, path in class_paths.items():
            arr = np.load(path)
            result[code] = arr
            print(f"  class {code}: {len(arr):,} points [cache]")
        return result

    if manifest_path.exists() and not rebuild:
        print("\nLiDAR dataset changed since cache was built; rebuilding cache.")

    print("\nReading LiDAR once for required classes...")
    parts = {code: [] for code in classes}
    counts = {code: 0 for code in classes}

    for i, path in enumerate(files, start=1):
        if i == 1 or i % 10 == 0 or i == len(files):
            print(f"  tile {i:,}/{len(files):,}: {path.name}")

        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                cls = np.asarray(chunk.classification)
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)

                for code in classes:
                    mask = cls == code
                    if not np.any(mask):
                        continue
                    arr = np.column_stack((x[mask], y[mask], z[mask]))
                    parts[code].append(arr)
                    counts[code] += len(arr)

    result = {}
    for code in classes:
        arr = (
            np.vstack(parts[code])
            if parts[code]
            else np.empty((0, 3), dtype=np.float64)
        )
        np.save(class_paths[code], arr)
        result[code] = arr
        print(f"  class {code}: {len(arr):,} points")

    manifest = {
        "dataset_sha256": current_sig["sha256"],
        "files": current_sig["files"],
        "classes": classes,
        "counts": {str(k): int(v) for k, v in counts.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  cache written: {cache_dir}")
    return result


def cluster_lidar_structures(
    xyz: np.ndarray,
    crs,
    unit_to_m: float,
    eps_m: float = 1.5,
    min_points: int = 3,
) -> gpd.GeoDataFrame:
    if len(xyz) == 0:
        return rows_to_gdf([], crs)

    labels = DBSCAN(
        eps=eps_m / unit_to_m,
        min_samples=min_points,
        n_jobs=-1,
    ).fit_predict(xyz[:, :2])

    rows = []
    number = 1

    for label in sorted(set(int(v) for v in labels if int(v) >= 0)):
        pts = xyz[labels == label]
        centre = np.median(pts, axis=0)
        spread = np.linalg.norm(
            pts[:, :2] - centre[None, :2],
            axis=1,
        )

        rows.append({
            "LIDAR_STRUCTURE_ID": f"LS{number:05d}",
            "POINTS_215": int(len(pts)),
            "X": float(centre[0]),
            "Y": float(centre[1]),
            "Z_MED": float(centre[2]),
            "R95_M": float(np.quantile(spread, 0.95) * unit_to_m),
            "geometry": Point(
                float(centre[0]),
                float(centre[1]),
                float(centre[2]),
            ),
        })
        number += 1

    return rows_to_gdf(rows, crs)


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
        raise FileNotFoundError(
            f"Could not find {SOURCE_LINES_NAME} under {source}"
        )
    if not structures:
        raise FileNotFoundError(
            f"Could not find {SOURCE_STRUCTURES_NAME} under {source}"
        )

    return temp, lines[0], structures[0]


def read_source_data(source: Path, target_crs):
    temp, line_path, structure_path = locate_source_shapefiles(source)

    lines = gpd.read_file(line_path)
    structures = gpd.read_file(structure_path)

    if lines.crs is None or structures.crs is None:
        raise RuntimeError("Source GIS CRS is missing.")

    bad_lines = lines.geometry.isna() | lines.geometry.is_empty
    bad_structures = structures.geometry.isna() | structures.geometry.is_empty

    if bad_lines.any():
        print(
            f"  source lines with null/empty geometry skipped: "
            f"{int(bad_lines.sum()):,}"
        )
        lines = lines.loc[~bad_lines].copy()

    if bad_structures.any():
        print(
            f"  source structures with null/empty geometry skipped: "
            f"{int(bad_structures.sum()):,}"
        )
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


def read_kmz_lines(kmz_path: Path, target_crs) -> gpd.GeoDataFrame:
    rows = []

    with zipfile.ZipFile(kmz_path) as zf:
        names = zf.namelist()
        kml_name = (
            "doc.kml"
            if "doc.kml" in names
            else next(
                (n for n in names if n.lower().endswith(".kml")),
                None,
            )
        )
        if not kml_name:
            return rows_to_gdf([], target_crs)
        root = ET.fromstring(zf.read(kml_name))

    def walk(container, folder_path):
        for child in list(container):
            tag = child.tag.split("}")[-1]

            if tag == "Folder":
                folder_name = _kml_text(child, "name")
                walk(
                    child,
                    folder_path + ([folder_name] if folder_name else []),
                )

            elif tag == "Placemark":
                name = _kml_text(child, "name")

                for ls in child.findall(".//kml:LineString", KML_NS):
                    node = ls.find("kml:coordinates", KML_NS)
                    if node is None or not node.text:
                        continue

                    coords = []
                    for token in node.text.replace("\n", " ").split():
                        parts = token.split(",")
                        if len(parts) < 2:
                            continue
                        try:
                            coords.append(
                                (float(parts[0]), float(parts[1]))
                            )
                        except ValueError:
                            continue

                    if len(coords) >= 2:
                        folder = " / ".join(folder_path)
                        rows.append({
                            "KMZ_NAME": name,
                            "FOLDER_PATH": folder,
                            "IS_CIRCUIT": "circuit" in folder.lower(),
                            "geometry": LineString(coords),
                        })

    document = root.find(".//kml:Document", KML_NS)
    if document is not None:
        walk(document, [])

    if not rows:
        return rows_to_gdf([], target_crs)

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )
    return gdf.to_crs(target_crs)


def merged_line_by_line_no(lines: gpd.GeoDataFrame) -> dict[str, object]:
    result = {}

    for line_no, group in lines.groupby("LINE_NO"):
        geoms = [
            geom for geom in group.geometry
            if geom is not None and not geom.is_empty
        ]
        if not geoms:
            continue

        if len(geoms) == 1:
            merged = geoms[0]
        else:
            unioned = unary_union(geoms)

            if isinstance(unioned, LineString):
                merged = unioned
            elif isinstance(unioned, MultiLineString):
                merged = linemerge(unioned)
            else:
                parts = [
                    geom for geom in getattr(unioned, "geoms", [])
                    if isinstance(geom, LineString)
                ]
                if not parts:
                    continue
                merged = (
                    parts[0]
                    if len(parts) == 1
                    else linemerge(parts)
                )

        result[str(line_no)] = merged

    return result


def choose_reference_component(reference_geom, points_xy):
    if isinstance(reference_geom, LineString):
        return reference_geom

    if isinstance(reference_geom, MultiLineString):
        parts = list(reference_geom.geoms)
    else:
        parts = [
            geom for geom in getattr(reference_geom, "geoms", [])
            if isinstance(geom, LineString)
        ]

    if not parts:
        return None

    points = [Point(float(x), float(y)) for x, y in points_xy]
    return min(
        parts,
        key=lambda part: sum(part.distance(point) for point in points),
    )


def span_frame(a_xy: np.ndarray, b_xy: np.ndarray):
    delta = b_xy - a_xy
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return None

    u = delta / length
    p = np.asarray([-u[1], u[0]], dtype=np.float64)

    return {
        "origin": a_xy,
        "u": u,
        "p": p,
        "length": length,
    }


def project_xyz(xyz: np.ndarray, frame: dict):
    centered = xyz[:, :2] - frame["origin"][None, :]
    return centered @ frame["u"], centered @ frame["p"]


def local_wire_subset(
    wire_xyz: np.ndarray,
    wire_tree: cKDTree,
    frame: dict,
    half_width_source: float,
    end_margin_source: float,
):
    midpoint = frame["origin"] + 0.5 * frame["length"] * frame["u"]
    radius = math.hypot(
        0.5 * frame["length"] + end_margin_source,
        half_width_source,
    )

    ids = wire_tree.query_ball_point(midpoint, r=radius)
    if not ids:
        return np.empty((0, 3), dtype=np.float64)

    local = wire_xyz[np.asarray(ids, dtype=np.int64)]
    s, t = project_xyz(local, frame)

    mask = (
        (s >= -end_margin_source)
        & (s <= frame["length"] + end_margin_source)
        & (np.abs(t) <= half_width_source)
    )
    return local[mask]


def wire_evidence(
    wire_xyz_local: np.ndarray,
    frame: dict,
    unit_to_m: float,
    half_width_m: float,
    end_margin_m: float,
    bins: int = 20,
):
    empty = {
        "WIRE_POINTS": 0,
        "WIRE_COVERAGE": 0.0,
        "LONGEST_RUN": 0.0,
        "START_OK": False,
        "END_OK": False,
        "MED_ABS_T_M": float("inf"),
    }

    if len(wire_xyz_local) == 0:
        return empty

    half = half_width_m / unit_to_m
    end = end_margin_m / unit_to_m

    s, t = project_xyz(wire_xyz_local, frame)
    mask = (
        (s >= -end)
        & (s <= frame["length"] + end)
        & (np.abs(t) <= half)
    )

    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return empty

    s_inside = s[ids]
    valid = (
        (s_inside >= 0.0)
        & (s_inside <= frame["length"])
    )

    occupied = np.zeros(bins, dtype=bool)

    if np.any(valid):
        bin_index = np.floor(
            np.clip(
                s_inside[valid] / max(frame["length"], 1e-9),
                0.0,
                0.999999,
            ) * bins
        ).astype(int)
        occupied[np.unique(bin_index)] = True

    longest = 0
    current = 0
    for value in occupied:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    edge_bins = max(1, min(3, bins // 5))

    return {
        "WIRE_POINTS": int(len(ids)),
        "WIRE_COVERAGE": float(occupied.mean()),
        "LONGEST_RUN": float(longest / bins),
        "START_OK": bool(np.any(occupied[:edge_bins])),
        "END_OK": bool(np.any(occupied[-edge_bins:])),
        "MED_ABS_T_M": float(
            np.median(np.abs(t[ids])) * unit_to_m
        ),
    }


def evidence_passes(
    evidence: dict,
    min_wire_points: int,
    min_wire_coverage: float,
    min_longest_run: float,
) -> tuple[bool, str]:
    if evidence["WIRE_POINTS"] < min_wire_points:
        return False, "INSUFFICIENT_WIRE_POINTS"
    if evidence["WIRE_COVERAGE"] < min_wire_coverage:
        return False, "LOW_WIRE_COVERAGE"
    if evidence["LONGEST_RUN"] < min_longest_run:
        return False, "WIRE_NOT_CONTINUOUS"
    if not evidence["START_OK"]:
        return False, "NO_WIRE_NEAR_STRUCTURE_A"
    if not evidence["END_OK"]:
        return False, "NO_WIRE_NEAR_STRUCTURE_B"
    return True, "ACCEPTED"
