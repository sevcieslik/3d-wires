from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import json
import math
import re

import numpy as np
import pandas as pd
import laspy
import geopandas as gpd
from pyproj import CRS
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point


KNOWN_CONDUCTOR_CLASSES = {
    187: "Wire",
    24: "Low voltage wire",
    25: "High voltage wire",
    26: "Railroad wire",
}

PTC_POSITIVE = (
    "wire", "wires", "conductor", "conductors", "phase",
    "earth wire", "shield wire", "ground wire",
)
PTC_NEGATIVE = (
    "guy", "guy-wire", "guy wire", "tension", "insulator",
)


@dataclass
class SourceGroup:
    source_id: str
    las_class: int
    label: str
    method: str
    confidence: str
    point_count: int


@dataclass
class DatasetInfo:
    files: list[str]
    crs: str | None
    crs_status: str
    source_unit: str
    unit_to_metre: float
    class_counts: dict[int, int]


def lidar_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".las", ".laz"}:
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".las", ".laz"}],
        key=lambda p: str(p).lower(),
    )
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under {root}")
    return files


def _header_crs(path: Path):
    with laspy.open(path) as reader:
        try:
            return reader.header.parse_crs()
        except Exception:
            return None


def _unit_to_metre(crs) -> tuple[str, float]:
    if crs is None:
        return "unknown_assumed_metre", 1.0
    try:
        axis = crs.axis_info
        if axis:
            name = str(axis[0].unit_name or "unknown")
            factor = float(axis[0].unit_conversion_factor)
            if np.isfinite(factor) and factor > 0:
                return name, factor
    except Exception:
        pass
    return "unknown_assumed_metre", 1.0


def inspect_dataset(files: list[Path], chunk_size: int = 2_000_000) -> DatasetInfo:
    counts: dict[int, int] = defaultdict(int)
    crs_values = []

    for path in files:
        crs = _header_crs(path)
        if crs is not None:
            crs_values.append((path, crs))
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                values, n = np.unique(np.asarray(chunk.classification), return_counts=True)
                for code, count in zip(values, n):
                    counts[int(code)] += int(count)

    resolved = None
    if crs_values:
        resolved = crs_values[0][1]
        for path, other in crs_values[1:]:
            try:
                equal = resolved.equals(other, ignore_axis_order=True)
            except Exception:
                equal = resolved == other
            if not equal:
                raise RuntimeError(
                    f"Non-equivalent CRS metadata detected: {crs_values[0][0].name} and {path.name}"
                )

    unit, factor = _unit_to_metre(resolved)
    return DatasetInfo(
        files=[str(p) for p in files],
        crs=resolved.to_string() if resolved is not None else None,
        crs_status="embedded" if resolved is not None else "missing",
        source_unit=unit,
        unit_to_metre=factor,
        class_counts=dict(sorted(counts.items())),
    )


def parse_ptc(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(.+?)\s+(-?\d+(?:\.\d+)?)\s*$")
    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("*"):
            continue
        match = pattern.match(raw)
        if not match:
            continue
        code = int(match.group(1))
        label = match.group(2).strip()
        if label:
            mapping[code] = label
    return mapping


def _ptc_is_conductor(label: str) -> bool:
    value = re.sub(r"[_\-]+", " ", label.lower()).strip()
    if any(term in value for term in PTC_NEGATIVE):
        return False
    return any(term in value for term in PTC_POSITIVE)


def resolve_sources(info: DatasetInfo, ptc: Path | None = None) -> list[SourceGroup]:
    present = info.class_counts

    # Standard remapped class 187 is authoritative enough to be the preferred
    # source. Raw FLAI classes are retained as separate sources only when 187
    # is absent, avoiding duplicate vectorisation of mixed intermediary data.
    candidates: list[tuple[int, str, str, str]] = []
    if present.get(187, 0) > 0:
        candidates.append((187, KNOWN_CONDUCTOR_CLASSES[187], "KNOWN_STANDARD", "HIGH"))
    else:
        for code in (24, 25, 26):
            if present.get(code, 0) > 0:
                candidates.append((code, KNOWN_CONDUCTOR_CLASSES[code], "KNOWN_FLAI", "HIGH"))

    if not candidates and ptc is not None:
        mapping = parse_ptc(ptc)
        for code, label in sorted(mapping.items()):
            if present.get(code, 0) > 0 and _ptc_is_conductor(label):
                candidates.append((code, label, "PTC", "HIGH"))

    groups = []
    for number, (code, label, method, confidence) in enumerate(candidates, start=1):
        groups.append(
            SourceGroup(
                source_id=f"C{number:02d}",
                las_class=code,
                label=label,
                method=method,
                confidence=confidence,
                point_count=int(present.get(code, 0)),
            )
        )
    return groups


def load_class_points(
    files: list[Path],
    class_code: int,
    chunk_size: int = 2_000_000,
    max_points: int | None = None,
) -> np.ndarray:
    parts = []
    total = 0
    for path in files:
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                cls = np.asarray(chunk.classification)
                mask = cls == class_code
                if not np.any(mask):
                    continue
                xyz = np.column_stack((
                    np.asarray(chunk.x, dtype=np.float64)[mask],
                    np.asarray(chunk.y, dtype=np.float64)[mask],
                    np.asarray(chunk.z, dtype=np.float64)[mask],
                ))
                parts.append(xyz)
                total += len(xyz)
                if max_points and total >= max_points:
                    break
        if max_points and total >= max_points:
            break
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    xyz = np.vstack(parts)
    if max_points and len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
    return xyz


def _voxel_xy(xy: np.ndarray, size: float) -> np.ndarray:
    if len(xy) == 0:
        return xy
    key = np.floor(xy / max(size, 1e-9)).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return xy[np.sort(first)]


def split_networks(
    xyz: np.ndarray,
    unit_to_m: float,
    corridor_eps_m: float = 25.0,
    min_samples: int = 4,
) -> list[np.ndarray]:
    if len(xyz) == 0:
        return []
    eps_source = corridor_eps_m / max(unit_to_m, 1e-12)

    # Coarse XY occupancy is enough for network corridor discovery and prevents
    # DBSCAN from running on millions of source points.
    coarse = _voxel_xy(xyz[:, :2], max(eps_source * 0.35, 1e-6))
    labels_coarse = DBSCAN(
        eps=eps_source,
        min_samples=min_samples,
        n_jobs=1,
    ).fit_predict(coarse)

    valid = sorted(set(int(v) for v in labels_coarse if int(v) >= 0))
    if not valid:
        # Preserve the data rather than failing: a single network is safer than
        # throwing away conductors because the corridor grouping was uncertain.
        return [xyz]

    # Assign every original point to the nearest coarse sample label.
    from scipy.spatial import cKDTree
    tree = cKDTree(coarse)
    _, nearest = tree.query(xyz[:, :2], k=1)
    point_labels = labels_coarse[np.asarray(nearest, dtype=np.int64)]

    networks = []
    for label in valid:
        current = xyz[point_labels == label]
        if len(current) >= 20:
            networks.append(current)

    noise = xyz[point_labels < 0]
    if len(noise) >= 20:
        networks.append(noise)
    return networks or [xyz]


def derive_axis(xyz: np.ndarray) -> dict:
    xy = xyz[:, :2]
    origin = np.median(xy, axis=0)
    centered = xy - origin[None, :]
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    u = vectors[:, int(np.argmax(values))]
    if u[0] < 0 or (abs(u[0]) < 1e-12 and u[1] < 0):
        u = -u
    p = np.asarray([-u[1], u[0]], dtype=np.float64)
    s = centered @ u
    t = centered @ p
    s_min = float(np.quantile(s, 0.005))
    s_max = float(np.quantile(s, 0.995))
    return {
        "origin": origin,
        "u": u,
        "p": p,
        "s_min": s_min,
        "s_max": s_max,
        "length": max(0.0, s_max - s_min),
        "s": s,
        "t": t,
    }


def _section_observations(
    xyz: np.ndarray,
    axis: dict,
    section_length_source: float,
    cluster_eps_source: float,
    min_cluster_points: int,
) -> list[list[dict]]:
    s = axis["s"]
    t = axis["t"]
    z = xyz[:, 2]
    start, end = axis["s_min"], axis["s_max"]
    n_sections = max(2, int(math.ceil((end - start) / section_length_source)))
    sections: list[list[dict]] = []

    for index in range(n_sections):
        a = start + index * section_length_source
        b = min(end, a + section_length_source)
        mask = (s >= a) & (s < b if index < n_sections - 1 else s <= b)
        ids = np.flatnonzero(mask)
        if len(ids) < min_cluster_points:
            sections.append([])
            continue

        # Cross-section clustering uses t and z. Scaling both dimensions in
        # source CRS units keeps the method independent of absolute XY.
        tz = np.column_stack((t[ids], z[ids]))
        labels = DBSCAN(
            eps=cluster_eps_source,
            min_samples=min_cluster_points,
            n_jobs=1,
        ).fit_predict(tz)

        obs = []
        for lab in sorted(set(int(v) for v in labels if int(v) >= 0)):
            local = ids[labels == lab]
            if len(local) < min_cluster_points:
                continue
            obs.append({
                "section": index,
                "s": float(np.median(s[local])),
                "t": float(np.median(t[local])),
                "z": float(np.median(z[local])),
                "points": int(len(local)),
            })
        sections.append(obs)
    return sections


def _link_tracks(
    sections: list[list[dict]],
    max_t_jump_source: float,
    max_z_jump_source: float,
    max_missed: int = 2,
) -> list[dict]:
    tracks: list[dict] = []
    next_id = 1

    for observations in sections:
        active = [
            tr for tr in tracks
            if tr["missed"] <= max_missed
        ]

        if active and observations:
            matrix = np.full((len(active), len(observations)), 1e9, dtype=np.float64)
            for i, tr in enumerate(active):
                prev = tr["obs"][-1]
                for j, cur in enumerate(observations):
                    dt = abs(cur["t"] - prev["t"])
                    dz = abs(cur["z"] - prev["z"])
                    if dt <= max_t_jump_source and dz <= max_z_jump_source:
                        matrix[i, j] = math.sqrt(
                            (dt / max(max_t_jump_source, 1e-9)) ** 2
                            + (dz / max(max_z_jump_source, 1e-9)) ** 2
                        )

            rows, cols = linear_sum_assignment(matrix)
            used_tracks = set()
            used_obs = set()
            for r, c in zip(rows, cols):
                if matrix[r, c] >= 1e8:
                    continue
                active[int(r)]["obs"].append(observations[int(c)])
                active[int(r)]["missed"] = 0
                used_tracks.add(id(active[int(r)]))
                used_obs.add(int(c))

            for tr in active:
                if id(tr) not in used_tracks:
                    tr["missed"] += 1
        else:
            used_obs = set()
            for tr in active:
                tr["missed"] += 1

        for j, obs in enumerate(observations):
            if j in used_obs:
                continue
            tracks.append({"id": next_id, "obs": [obs], "missed": 0})
            next_id += 1

    return tracks


def _track_geometry(track: dict, axis: dict, min_sections: int) -> tuple[LineString | None, dict]:
    obs = sorted(track["obs"], key=lambda row: row["s"])
    if len(obs) < min_sections:
        return None, {}

    s = np.asarray([row["s"] for row in obs], dtype=np.float64)
    t = np.asarray([row["t"] for row in obs], dtype=np.float64)
    z = np.asarray([row["z"] for row in obs], dtype=np.float64)

    if np.ptp(s) <= 1e-9:
        return None, {}

    t_coef = np.polyfit(s, t, 1)
    z_degree = 2 if len(obs) >= 3 else 1
    z_coef = np.polyfit(s, z, z_degree)

    n = max(8, min(200, len(obs) * 3))
    sample_s = np.linspace(float(s.min()), float(s.max()), n)
    sample_t = np.polyval(t_coef, sample_s)
    sample_z = np.polyval(z_coef, sample_s)

    xy = (
        axis["origin"][None, :]
        + sample_s[:, None] * axis["u"][None, :]
        + sample_t[:, None] * axis["p"][None, :]
    )
    geom = LineString([
        (float(x), float(y), float(zz))
        for (x, y), zz in zip(xy, sample_z)
    ])

    t_rmse = float(np.sqrt(np.mean((t - np.polyval(t_coef, s)) ** 2)))
    z_rmse = float(np.sqrt(np.mean((z - np.polyval(z_coef, s)) ** 2)))
    section_ids = [int(row["section"]) for row in obs]
    extent = int(max(section_ids) - min(section_ids) + 1)
    coverage = len(set(section_ids)) / max(extent, 1)

    return geom, {
        "observed_sections": int(len(obs)),
        "coverage": float(coverage),
        "t_rmse": t_rmse,
        "z_rmse": z_rmse,
        "support_points": int(sum(row["points"] for row in obs)),
    }


def vectorise_source(
    xyz: np.ndarray,
    source: SourceGroup,
    crs,
    unit_to_m: float,
    corridor_eps_m: float = 25.0,
    section_length_m: float = 8.0,
    cross_section_eps_m: float = 1.2,
    max_t_jump_m: float = 2.5,
    max_z_jump_m: float = 4.0,
    min_cluster_points: int = 2,
    min_track_sections: int = 4,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[dict]]:
    networks = split_networks(
        xyz,
        unit_to_m,
        corridor_eps_m=corridor_eps_m,
    )

    wire_rows = []
    axis_rows = []
    network_stats = []
    wire_number = 1

    for network_number, network_xyz in enumerate(networks, start=1):
        network_id = f"{source.source_id}_NET_{network_number:03d}"
        axis = derive_axis(network_xyz)

        section_length_source = section_length_m / max(unit_to_m, 1e-12)
        cross_eps_source = cross_section_eps_m / max(unit_to_m, 1e-12)
        max_t_source = max_t_jump_m / max(unit_to_m, 1e-12)
        max_z_source = max_z_jump_m / max(unit_to_m, 1e-12)

        sections = _section_observations(
            network_xyz,
            axis,
            section_length_source,
            cross_eps_source,
            min_cluster_points,
        )
        tracks = _link_tracks(
            sections,
            max_t_source,
            max_z_source,
        )

        a = axis["origin"] + axis["s_min"] * axis["u"]
        b = axis["origin"] + axis["s_max"] * axis["u"]
        axis_geom = LineString([
            (float(a[0]), float(a[1])),
            (float(b[0]), float(b[1])),
        ])

        accepted = 0
        for track in tracks:
            geom, metrics = _track_geometry(track, axis, min_track_sections)
            if geom is None:
                continue

            wire_id = f"W{wire_number:06d}"
            wire_number += 1
            accepted += 1

            # QA remains deliberately transparent. Geometry is retained even if
            # the fit is weak, but weak tracks are routed to REVIEW.
            rmse_m = max(metrics["t_rmse"], metrics["z_rmse"]) * unit_to_m
            if metrics["coverage"] >= 0.55 and rmse_m <= 1.5:
                status = "FINAL"
                confidence = "HIGH" if metrics["coverage"] >= 0.75 and rmse_m <= 0.75 else "MEDIUM"
            else:
                status = "REVIEW"
                confidence = "LOW"

            wire_rows.append({
                "WIRE_ID": wire_id,
                "NETWORK_ID": network_id,
                "SOURCE_ID": source.source_id,
                "LAS_CLASS": int(source.las_class),
                "SOURCE_LABEL": source.label,
                "SOURCE_METHOD": source.method,
                "STATUS": status,
                "CONFIDENCE": confidence,
                "OBS_SECTIONS": metrics["observed_sections"],
                "COVERAGE": metrics["coverage"],
                "T_RMSE_M": metrics["t_rmse"] * unit_to_m,
                "Z_RMSE_M": metrics["z_rmse"] * unit_to_m,
                "SUPPORT_PTS": metrics["support_points"],
                "LENGTH_M": float(geom.length) * unit_to_m,
                "geometry": geom,
            })

        axis_rows.append({
            "NETWORK_ID": network_id,
            "SOURCE_ID": source.source_id,
            "LAS_CLASS": int(source.las_class),
            "POINTS": int(len(network_xyz)),
            "WIRES": int(accepted),
            "AXIS_SOURCE": "DERIVED_WIRES",
            "LENGTH_M": float(axis_geom.length) * unit_to_m,
            "geometry": axis_geom,
        })
        network_stats.append({
            "NETWORK_ID": network_id,
            "SOURCE_ID": source.source_id,
            "LAS_CLASS": int(source.las_class),
            "POINTS": int(len(network_xyz)),
            "TRACKS_RAW": int(len(tracks)),
            "WIRES_ACCEPTED": int(accepted),
        })

    wires = gpd.GeoDataFrame(wire_rows, geometry="geometry", crs=crs)
    axes = gpd.GeoDataFrame(axis_rows, geometry="geometry", crs=crs)
    return wires, axes, network_stats


def export_outputs(
    output: Path,
    info: DatasetInfo,
    sources: list[SourceGroup],
    wires: gpd.GeoDataFrame,
    axes: gpd.GeoDataFrame,
    network_stats: list[dict],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    gpkg = output / "01_conductor_geometry.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    source_df = pd.DataFrame([asdict(s) for s in sources])
    source_df.to_csv(output / "source_groups.csv", index=False)

    class_df = pd.DataFrame([
        {"LAS_CLASS": code, "POINT_COUNT": count}
        for code, count in info.class_counts.items()
    ])
    class_df.to_csv(output / "class_summary.csv", index=False)

    if not wires.empty:
        wires.to_file(gpkg, layer="wires_3d", driver="GPKG", engine="pyogrio")

        w2 = wires.copy()
        w2["geometry"] = [
            LineString([(x, y) for x, y, *_ in geom.coords])
            for geom in w2.geometry
        ]
        w2.to_file(gpkg, layer="wires_2d", driver="GPKG", engine="pyogrio")

        endpoint_rows = []
        for row in wires.itertuples():
            coords = list(row.geometry.coords)
            for end_name, coord in (("START", coords[0]), ("END", coords[-1])):
                endpoint_rows.append({
                    "WIRE_ID": row.WIRE_ID,
                    "NETWORK_ID": row.NETWORK_ID,
                    "END": end_name,
                    "geometry": Point(*coord),
                })
        endpoints = gpd.GeoDataFrame(endpoint_rows, geometry="geometry", crs=wires.crs)
        endpoints.to_file(gpkg, layer="wire_endpoints", driver="GPKG", engine="pyogrio")

        review = wires[wires["STATUS"] != "FINAL"].copy()
        if not review.empty:
            review.to_file(gpkg, layer="qa_review", driver="GPKG", engine="pyogrio")

        wires.drop(columns="geometry").to_csv(output / "wire_summary.csv", index=False)

    if not axes.empty:
        axes.to_file(gpkg, layer="network_axes", driver="GPKG", engine="pyogrio")

    # A non-spatial source_groups layer is written as a tiny point table only
    # when a GPKG exists, keeping source provenance inside the package.
    if not sources:
        conductor_status = "NO_CONDUCTOR_CANDIDATES"
    elif wires.empty:
        conductor_status = "SOURCES_FOUND_NO_WIRES"
    else:
        conductor_status = "WIRES_GENERATED"

    manifest = {
        "stage": "01_agnostic_conductor_extraction",
        "version": "0.1-dev",
        "crs": info.crs,
        "crs_status": info.crs_status,
        "source_unit": info.source_unit,
        "unit_to_metre": info.unit_to_metre,
        "tiles": len(info.files),
        "class_counts": {str(k): int(v) for k, v in info.class_counts.items()},
        "conductor_sources": [asdict(s) for s in sources],
        "networks_detected": int(len(axes)),
        "wires_total": int(len(wires)),
        "wires_final": int((wires["STATUS"] == "FINAL").sum()) if not wires.empty else 0,
        "wires_review": int((wires["STATUS"] != "FINAL").sum()) if not wires.empty else 0,
        "status": conductor_status,
        "geometry_truth": "LIDAR",
        "reference_data_role": "OPTIONAL_EVIDENCE_ONLY",
        "network_stats": network_stats,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
