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
        n_jobs=1,
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


def _has_intermediate_structure(
    a_idx: int,
    b_idx: int,
    xy: np.ndarray,
    frame: dict,
    tolerance_source: float,
) -> bool:
    centered = xy - frame["origin"][None, :]
    s = centered @ frame["u"]
    t = np.abs(centered @ frame["p"])
    margin = min(frame["length"] * 0.08, tolerance_source)
    mask = (
        (np.arange(len(xy)) != a_idx)
        & (np.arange(len(xy)) != b_idx)
        & (s > margin)
        & (s < frame["length"] - margin)
        & (t <= tolerance_source)
    )
    return bool(np.any(mask))


def _wire_evidence(
    wire_xyz: np.ndarray,
    frame: dict,
    unit_to_m: float,
    corridor_halfwidth_m: float,
    end_margin_m: float,
    bins: int,
) -> dict:
    half = corridor_halfwidth_m / max(unit_to_m, 1e-12)
    end = end_margin_m / max(unit_to_m, 1e-12)
    s, t = _project_to_span(wire_xyz, frame)
    mask = (
        (s >= -end)
        & (s <= frame["length"] + end)
        & (np.abs(t) <= half)
    )
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return {
            "ids": ids,
            "points": 0,
            "coverage": 0.0,
            "longest_run": 0.0,
            "median_abs_t_m": float("inf"),
        }

    inside_s = s[ids]
    valid = (inside_s >= 0.0) & (inside_s <= frame["length"])
    if np.any(valid):
        bin_index = np.floor(
            np.clip(inside_s[valid] / max(frame["length"], 1e-9), 0, 0.999999) * bins
        ).astype(int)
        occupied = np.zeros(bins, dtype=bool)
        occupied[np.unique(bin_index)] = True
    else:
        occupied = np.zeros(bins, dtype=bool)

    longest = 0
    current = 0
    for value in occupied:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return {
        "ids": ids,
        "points": int(len(ids)),
        "coverage": float(occupied.mean()) if bins else 0.0,
        "longest_run": float(longest / max(bins, 1)),
        "median_abs_t_m": float(np.median(np.abs(t[ids])) * unit_to_m),
    }


def build_span_candidates(
    structures: gpd.GeoDataFrame,
    wire_xyz: np.ndarray,
    crs,
    unit_to_m: float,
    min_span_m: float = 12.0,
    max_span_m: float = 550.0,
    structure_on_line_tol_m: float = 12.0,
    corridor_halfwidth_m: float = 25.0,
    end_margin_m: float = 10.0,
    evidence_bins: int = 16,
    min_wire_points: int = 40,
    min_wire_coverage: float = 0.55,
    min_longest_run: float = 0.45,
) -> gpd.GeoDataFrame:
    if structures.empty or len(structures) < 2 or len(wire_xyz) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    xy = np.column_stack((structures["X"].to_numpy(), structures["Y"].to_numpy()))
    max_source = max_span_m / max(unit_to_m, 1e-12)
    min_source = min_span_m / max(unit_to_m, 1e-12)
    intermediate_tol = structure_on_line_tol_m / max(unit_to_m, 1e-12)

    tree = cKDTree(xy)
    pairs = sorted(tree.query_pairs(r=max_source))
    rows = []
    span_number = 1

    for a_idx, b_idx in pairs:
        frame = _span_frame(xy[a_idx], xy[b_idx])
        if frame is None or frame["length"] < min_source:
            continue

        # Adjacent supports only. If another clustered structure sits between
        # the pair close to the candidate line, this is a skip-span and is rejected.
        if _has_intermediate_structure(
            a_idx,
            b_idx,
            xy,
            frame,
            intermediate_tol,
        ):
            continue

        evidence = _wire_evidence(
            wire_xyz,
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

        a_id = str(structures.iloc[a_idx]["STRUCTURE_ID"])
        b_id = str(structures.iloc[b_idx]["STRUCTURE_ID"])
        score = (
            0.60 * evidence["coverage"]
            + 0.30 * evidence["longest_run"]
            + 0.10 * min(1.0, evidence["points"] / max(min_wire_points * 5, 1))
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
            "EVIDENCE_SCORE": float(score),
            "AXIS_SOURCE": "STRUCTURES_CONFIRMED_BY_WIRES",
            "geometry": LineString([
                (float(xy[a_idx, 0]), float(xy[a_idx, 1])),
                (float(xy[b_idx, 0]), float(xy[b_idx, 1])),
            ]),
        })
        span_number += 1

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


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

    # Structures define the physical span extent. The conductor fit is
    # extrapolated only to the two support axes, never beyond them.
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
    corridor_halfwidth_m: float = 25.0,
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

    for span in spans.itertuples():
        base = _axis_from_span_geometry(span.geometry, wire_xyz)
        s = base["s"]
        t = base["t"]
        mask = (
            (s >= -end)
            & (s <= base["length"] + end)
            & (np.abs(t) <= half)
        )
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            continue

        xyz = wire_xyz[ids]
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
                "AXIS_SOURCE": "STRUCTURES_CONFIRMED_BY_WIRES",
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
            "AXIS_SOURCE": "STRUCTURES_CONFIRMED_BY_WIRES",
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
