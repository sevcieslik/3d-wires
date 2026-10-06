from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point

from stage1_core import (
    SourceGroup,
    parse_ptc,
    _section_observations,
    _link_tracks,
)

KNOWN_STRUCTURE_CLASS = 215
PTC_STRUCTURE_TERMS = ("structure", "structures", "tower", "pole", "support")
PTC_STRUCTURE_EXCLUDE = ("wire", "conductor", "vegetation", "ground", "road")


def resolve_structure_class(info, ptc: Path | None = None) -> tuple[int | None, str, str]:
    if info.class_counts.get(KNOWN_STRUCTURE_CLASS, 0) > 0:
        return KNOWN_STRUCTURE_CLASS, "Structure", "KNOWN_STANDARD"

    if ptc is not None:
        mapping = parse_ptc(ptc)
        for code, label in sorted(mapping.items()):
            if info.class_counts.get(code, 0) <= 0:
                continue
            value = re.sub(r"[_\-]+", " ", label.lower()).strip()
            if any(term in value for term in PTC_STRUCTURE_EXCLUDE):
                continue
            if any(term in value for term in PTC_STRUCTURE_TERMS):
                return code, label, "PTC"

    return None, "", "NONE"


def cluster_structures(
    xyz: np.ndarray,
    crs,
    unit_to_m: float,
    eps_m: float = 5.0,
    min_points: int = 3,
) -> gpd.GeoDataFrame:
    if len(xyz) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    eps_source = eps_m / max(unit_to_m, 1e-12)
    labels = DBSCAN(
        eps=eps_source,
        min_samples=min_points,
        n_jobs=-1,
    ).fit_predict(xyz[:, :2])

    rows = []
    number = 1
    for label in sorted(set(int(v) for v in labels if int(v) >= 0)):
        pts = xyz[labels == label]
        if len(pts) < min_points:
            continue
        med = np.median(pts, axis=0)
        spread = np.sqrt(np.sum((pts[:, :2] - med[:2]) ** 2, axis=1))
        rows.append({
            "STRUCTURE_ID": f"S{number:05d}",
            "POINTS": int(len(pts)),
            "X": float(med[0]),
            "Y": float(med[1]),
            "Z_MED": float(med[2]),
            "R95_M": float(np.quantile(spread, 0.95) * unit_to_m),
            "geometry": Point(float(med[0]), float(med[1]), float(med[2])),
        })
        number += 1

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _span_frame(a_xy: np.ndarray, b_xy: np.ndarray) -> dict | None:
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
        "s_min": 0.0,
        "s_max": length,
        "length": length,
    }


def _project_to_span(xyz: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    centered = xyz[:, :2] - frame["origin"][None, :]
    return centered @ frame["u"], centered @ frame["p"]


def _local_wire_subset(
    wire_xyz: np.ndarray,
    wire_tree: cKDTree,
    frame: dict,
    half_width_source: float,
    end_margin_source: float,
) -> np.ndarray:
    midpoint = frame["origin"] + 0.5 * frame["length"] * frame["u"]
    along = 0.5 * frame["length"] + end_margin_source
    radius = math.hypot(along, half_width_source)
    ids = wire_tree.query_ball_point(midpoint, r=radius)
    if not ids:
        return np.empty((0, 3), dtype=np.float64)
    return wire_xyz[np.asarray(ids, dtype=np.int64)]


def estimate_structure_directions(
    structures: gpd.GeoDataFrame,
    wire_xyz: np.ndarray,
    crs,
    unit_to_m: float,
    inner_radius_m: float = 4.0,
    outer_radius_m: float = 55.0,
    angle_bin_deg: float = 5.0,
    peak_tolerance_deg: float = 15.0,
    min_points_per_peak: int = 15,
    max_peaks: int = 6,
):
    """
    Estimate directions in which conductor points leave each structure.

    Direction is computed from local wire points only. A histogram over full
    360 degrees is used so opposite sides of a through-line remain separate.
    """
    if structures.empty or len(wire_xyz) == 0:
        return {}, gpd.GeoDataFrame(geometry=[], crs=crs)

    wire_tree = cKDTree(wire_xyz[:, :2])
    inner = inner_radius_m / max(unit_to_m, 1e-12)
    outer = outer_radius_m / max(unit_to_m, 1e-12)
    bins = max(12, int(round(360.0 / angle_bin_deg)))
    actual_bin = 360.0 / bins

    direction_map: dict[str, list[float]] = {}
    ray_rows = []

    for row in structures.itertuples():
        origin = np.asarray([row.X, row.Y], dtype=np.float64)
        ids = wire_tree.query_ball_point(origin, r=outer)
        if not ids:
            direction_map[row.STRUCTURE_ID] = []
            continue

        pts = wire_xyz[np.asarray(ids, dtype=np.int64), :2]
        delta = pts - origin[None, :]
        dist = np.linalg.norm(delta, axis=1)
        mask = (dist >= inner) & (dist <= outer)
        if not np.any(mask):
            direction_map[row.STRUCTURE_ID] = []
            continue

        delta = delta[mask]
        dist = dist[mask]
        angles = (np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) + 360.0) % 360.0

        hist, edges = np.histogram(angles, bins=bins, range=(0.0, 360.0))

        # Circular smoothing suppresses isolated noise but preserves real span rays.
        smooth = (
            np.roll(hist, -2)
            + 2 * np.roll(hist, -1)
            + 3 * hist
            + 2 * np.roll(hist, 1)
            + np.roll(hist, 2)
        )

        order = np.argsort(smooth)[::-1]
        peaks = []
        for idx in order:
            if smooth[idx] < min_points_per_peak:
                break
            angle = (idx + 0.5) * actual_bin
            if any(_angle_diff_deg(angle, other) < peak_tolerance_deg for other in peaks):
                continue
            peaks.append(float(angle))
            if len(peaks) >= max_peaks:
                break

        direction_map[row.STRUCTURE_ID] = peaks

        ray_length = outer * 0.75
        for n, angle in enumerate(peaks, start=1):
            rad = math.radians(angle)
            end = origin + ray_length * np.asarray([math.cos(rad), math.sin(rad)])
            ray_rows.append({
                "STRUCTURE_ID": row.STRUCTURE_ID,
                "DIR_ID": n,
                "ANGLE_DEG": angle,
                "geometry": LineString([
                    (float(origin[0]), float(origin[1])),
                    (float(end[0]), float(end[1])),
                ]),
            })

    return direction_map, gpd.GeoDataFrame(ray_rows, geometry="geometry", crs=crs)


def _direction_supported(
    structure_id: str,
    bearing_deg: float,
    direction_map: dict[str, list[float]],
    tolerance_deg: float,
) -> bool:
    peaks = direction_map.get(structure_id, [])
    return any(_angle_diff_deg(bearing_deg, peak) <= tolerance_deg for peak in peaks)


def _wire_evidence(
    local_wire_xyz: np.ndarray,
    frame: dict,
    unit_to_m: float,
    corridor_halfwidth_m: float,
    end_margin_m: float,
    bins: int,
) -> dict:
    if len(local_wire_xyz) == 0:
        return {
            "points": 0,
            "coverage": 0.0,
            "longest_run": 0.0,
            "median_abs_t_m": float("inf"),
            "start_ok": False,
            "end_ok": False,
        }

    half = corridor_halfwidth_m / max(unit_to_m, 1e-12)
    end = end_margin_m / max(unit_to_m, 1e-12)
    s, t = _project_to_span(local_wire_xyz, frame)
    mask = (
        (s >= -end)
        & (s <= frame["length"] + end)
        & (np.abs(t) <= half)
    )
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return {
            "points": 0,
            "coverage": 0.0,
            "longest_run": 0.0,
            "median_abs_t_m": float("inf"),
            "start_ok": False,
            "end_ok": False,
        }

    inside_s = s[ids]
    valid = (inside_s >= 0.0) & (inside_s <= frame["length"])
    occupied = np.zeros(bins, dtype=bool)

    if np.any(valid):
        bin_index = np.floor(
            np.clip(
                inside_s[valid] / max(frame["length"], 1e-9),
                0,
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
    start_ok = bool(np.any(occupied[:edge_bins]))
    end_ok = bool(np.any(occupied[-edge_bins:]))

    return {
        "points": int(len(ids)),
        "coverage": float(occupied.mean()) if bins else 0.0,
        "longest_run": float(longest / max(bins, 1)),
        "median_abs_t_m": float(np.median(np.abs(t[ids])) * unit_to_m),
        "start_ok": start_ok,
        "end_ok": end_ok,
    }


def _prune_candidates(rows: list[dict], max_degree: int = 4, angle_slot_deg: float = 18.0):
    """
    Keep only the strongest local connection in a similar direction at each node.
    This suppresses fan-out and long skip connections.
    """
    accepted = []
    node_angles: dict[str, list[float]] = {}
    node_degree: dict[str, int] = {}

    for row in sorted(
        rows,
        key=lambda r: (
            -r["EVIDENCE_SCORE"],
            r["LENGTH_M"],
        ),
    ):
        a = row["STRUCT_A"]
        b = row["STRUCT_B"]
        angle_ab = row["BEARING_A_DEG"]
        angle_ba = row["BEARING_B_DEG"]

        if node_degree.get(a, 0) >= max_degree or node_degree.get(b, 0) >= max_degree:
            continue

        a_angles = node_angles.get(a, [])
        b_angles = node_angles.get(b, [])

        if any(_angle_diff_deg(angle_ab, x) < angle_slot_deg for x in a_angles):
            continue
        if any(_angle_diff_deg(angle_ba, x) < angle_slot_deg for x in b_angles):
            continue

        accepted.append(row)
        node_angles.setdefault(a, []).append(angle_ab)
        node_angles.setdefault(b, []).append(angle_ba)
        node_degree[a] = node_degree.get(a, 0) + 1
        node_degree[b] = node_degree.get(b, 0) + 1

    accepted.sort(key=lambda r: r["SPAN_ID"])
    return accepted


def build_span_candidates(
    structures: gpd.GeoDataFrame,
    wire_xyz: np.ndarray,
    crs,
    unit_to_m: float,
    min_span_m: float = 12.0,
    max_span_m: float = 550.0,
    structure_on_line_tol_m: float = 12.0,
    corridor_halfwidth_m: float = 18.0,
    end_margin_m: float = 10.0,
    evidence_bins: int = 20,
    min_wire_points: int = 40,
    min_wire_coverage: float = 0.72,
    min_longest_run: float = 0.65,
    direction_radius_m: float = 55.0,
    direction_tolerance_deg: float = 16.0,
    max_degree: int = 4,
):
    if structures.empty or len(structures) < 2 or len(wire_xyz) == 0:
        return (
            gpd.GeoDataFrame(geometry=[], crs=crs),
            gpd.GeoDataFrame(geometry=[], crs=crs),
        )

    structure_xy = np.column_stack(
        (structures["X"].to_numpy(), structures["Y"].to_numpy())
    )
    max_source = max_span_m / max(unit_to_m, 1e-12)
    min_source = min_span_m / max(unit_to_m, 1e-12)
    half_source = corridor_halfwidth_m / max(unit_to_m, 1e-12)
    end_source = end_margin_m / max(unit_to_m, 1e-12)

    structure_tree = cKDTree(structure_xy)
    wire_tree = cKDTree(wire_xyz[:, :2])

    direction_map, direction_rays = estimate_structure_directions(
        structures,
        wire_xyz,
        crs,
        unit_to_m,
        outer_radius_m=direction_radius_m,
    )

    pairs = sorted(structure_tree.query_pairs(r=max_source))
    print(f"  candidate structure pairs by distance: {len(pairs):,}")

    rows = []
    span_number = 1

    for pair_no, (a_idx, b_idx) in enumerate(pairs, start=1):
        if pair_no % 5000 == 0:
            print(
                f"    topology: {pair_no:,}/{len(pairs):,} pairs checked, "
                f"{len(rows):,} direction+wire candidates"
            )

        a_id = str(structures.iloc[a_idx]["STRUCTURE_ID"])
        b_id = str(structures.iloc[b_idx]["STRUCTURE_ID"])
        a_xy = structure_xy[a_idx]
        b_xy = structure_xy[b_idx]

        frame = _span_frame(a_xy, b_xy)
        if frame is None or frame["length"] < min_source:
            continue

        bearing_ab = (
            math.degrees(math.atan2(b_xy[1] - a_xy[1], b_xy[0] - a_xy[0])) + 360.0
        ) % 360.0
        bearing_ba = (bearing_ab + 180.0) % 360.0

        if not _direction_supported(
            a_id,
            bearing_ab,
            direction_map,
            direction_tolerance_deg,
        ):
            continue
        if not _direction_supported(
            b_id,
            bearing_ba,
            direction_map,
            direction_tolerance_deg,
        ):
            continue

        local_wire_xyz = _local_wire_subset(
            wire_xyz,
            wire_tree,
            frame,
            half_source,
            end_source,
        )
        if len(local_wire_xyz) < min_wire_points:
            continue

        evidence = _wire_evidence(
            local_wire_xyz,
            frame,
            unit_to_m,
            corridor_halfwidth_m,
            end_margin_m,
            evidence_bins,
        )

        if evidence["points"] < min_wire_points:
            continue
        if evidence["coverage"] < min_wire_coverage:
            continue
        if evidence["longest_run"] < min_longest_run:
            continue
        if not evidence["start_ok"] or not evidence["end_ok"]:
            continue

        score = (
            0.50 * evidence["coverage"]
            + 0.30 * evidence["longest_run"]
            + 0.10 * (1.0 if evidence["start_ok"] else 0.0)
            + 0.10 * (1.0 if evidence["end_ok"] else 0.0)
        )

        rows.append({
            "SPAN_ID": f"SP{span_number:05d}",
            "STRUCT_A": a_id,
            "STRUCT_B": b_id,
            "LENGTH_M": float(frame["length"] * unit_to_m),
            "WIRE_POINTS": evidence["points"],
            "WIRE_COVERAGE": evidence["coverage"],
            "LONGEST_RUN": evidence["longest_run"],
            "MED_ABS_T_M": evidence["median_abs_t_m"],
            "START_OK": evidence["start_ok"],
            "END_OK": evidence["end_ok"],
            "BEARING_A_DEG": bearing_ab,
            "BEARING_B_DEG": bearing_ba,
            "EVIDENCE_SCORE": float(score),
            "AXIS_SOURCE": "STRUCTURE_DIRECTIONS_CONFIRMED_BY_WIRES",
            "geometry": LineString([
                (float(a_xy[0]), float(a_xy[1])),
                (float(b_xy[0]), float(b_xy[1])),
            ]),
        })
        span_number += 1

    print(f"  candidates after directional + continuity tests: {len(rows):,}")
    rows = _prune_candidates(rows, max_degree=max_degree)
    print(f"  candidates after fan-out pruning: {len(rows):,}")

    for i, row in enumerate(rows, start=1):
        row["SPAN_ID"] = f"SP{i:05d}"

    return (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=crs),
        direction_rays,
    )


def _axis_from_span_geometry(span_geom: LineString, xyz: np.ndarray) -> dict:
    coords = list(span_geom.coords)
    a = np.asarray(coords[0][:2], dtype=np.float64)
    b = np.asarray(coords[-1][:2], dtype=np.float64)
    frame = _span_frame(a, b)
    s, t = _project_to_span(xyz, frame)
    frame["s"] = s
    frame["t"] = t
    return frame


def _track_geometry_to_supports(track: dict, axis: dict, min_sections: int):
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

    n = max(12, min(240, len(obs) * 4))
    sample_s = np.linspace(0.0, float(axis["length"]), n)
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


def vectorise_source_by_spans(
    wire_xyz: np.ndarray,
    source: SourceGroup,
    spans: gpd.GeoDataFrame,
    crs,
    unit_to_m: float,
    corridor_halfwidth_m: float = 18.0,
    end_margin_m: float = 10.0,
    section_length_m: float = 8.0,
    cross_section_eps_m: float = 1.2,
    max_t_jump_m: float = 2.5,
    max_z_jump_m: float = 4.0,
    min_cluster_points: int = 2,
    min_track_sections: int = 4,
):
    if spans.empty:
        return (
            gpd.GeoDataFrame(geometry=[], crs=crs),
            gpd.GeoDataFrame(geometry=[], crs=crs),
            [],
        )

    wire_rows = []
    axis_rows = []
    stats = []
    wire_number = 1

    half = corridor_halfwidth_m / max(unit_to_m, 1e-12)
    end = end_margin_m / max(unit_to_m, 1e-12)
    section_length_source = section_length_m / max(unit_to_m, 1e-12)
    cross_eps_source = cross_section_eps_m / max(unit_to_m, 1e-12)
    max_t_source = max_t_jump_m / max(unit_to_m, 1e-12)
    max_z_source = max_z_jump_m / max(unit_to_m, 1e-12)

    wire_tree = cKDTree(wire_xyz[:, :2])
    total_spans = len(spans)

    for span_no, span in enumerate(spans.itertuples(), start=1):
        if span_no % 250 == 0:
            print(
                f"    vectorisation: {span_no:,}/{total_spans:,} spans processed, "
                f"{len(wire_rows):,} wire tracks built"
            )

        coords = list(span.geometry.coords)
        a = np.asarray(coords[0][:2], dtype=np.float64)
        b = np.asarray(coords[-1][:2], dtype=np.float64)
        base = _span_frame(a, b)
        if base is None:
            continue

        local_wire_xyz = _local_wire_subset(
            wire_xyz,
            wire_tree,
            base,
            half,
            end,
        )
        if len(local_wire_xyz) == 0:
            continue

        s, t = _project_to_span(local_wire_xyz, base)
        mask = (
            (s >= -end)
            & (s <= base["length"] + end)
            & (np.abs(t) <= half)
        )
        xyz = local_wire_xyz[mask]
        if len(xyz) == 0:
            continue

        axis = _axis_from_span_geometry(span.geometry, xyz)

        sections = _section_observations(
            xyz,
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

        accepted = 0
        for track in tracks:
            geom, metrics = _track_geometry_to_supports(
                track,
                axis,
                min_track_sections,
            )
            if geom is None:
                continue

            rmse_m = max(metrics["t_rmse"], metrics["z_rmse"]) * unit_to_m
            if metrics["coverage"] >= 0.55 and rmse_m <= 1.5:
                status = "FINAL"
                confidence = (
                    "HIGH"
                    if metrics["coverage"] >= 0.75 and rmse_m <= 0.75
                    else "MEDIUM"
                )
            else:
                status = "REVIEW"
                confidence = "LOW"

            wire_rows.append({
                "WIRE_ID": f"W{wire_number:06d}",
                "SPAN_ID": span.SPAN_ID,
                "NETWORK_ID": span.SPAN_ID,
                "STRUCT_A": span.STRUCT_A,
                "STRUCT_B": span.STRUCT_B,
                "SOURCE_ID": source.source_id,
                "LAS_CLASS": int(source.las_class),
                "SOURCE_LABEL": source.label,
                "SOURCE_METHOD": source.method,
                "AXIS_SOURCE": "STRUCTURE_DIRECTIONS_CONFIRMED_BY_WIRES",
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
            wire_number += 1
            accepted += 1

        axis_rows.append({
            "NETWORK_ID": span.SPAN_ID,
            "SPAN_ID": span.SPAN_ID,
            "STRUCT_A": span.STRUCT_A,
            "STRUCT_B": span.STRUCT_B,
            "SOURCE_ID": source.source_id,
            "LAS_CLASS": int(source.las_class),
            "POINTS": int(len(xyz)),
            "WIRES": int(accepted),
            "AXIS_SOURCE": "STRUCTURE_DIRECTIONS_CONFIRMED_BY_WIRES",
            "LENGTH_M": float(span.LENGTH_M),
            "EVIDENCE": float(span.EVIDENCE_SCORE),
            "geometry": span.geometry,
        })

        stats.append({
            "NETWORK_ID": span.SPAN_ID,
            "SPAN_ID": span.SPAN_ID,
            "SOURCE_ID": source.source_id,
            "LAS_CLASS": int(source.las_class),
            "POINTS": int(len(xyz)),
            "TRACKS_RAW": int(len(tracks)),
            "WIRES_ACCEPTED": int(accepted),
        })

    return (
        gpd.GeoDataFrame(wire_rows, geometry="geometry", crs=crs),
        gpd.GeoDataFrame(axis_rows, geometry="geometry", crs=crs),
        stats,
    )
