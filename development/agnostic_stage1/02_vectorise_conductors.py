#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN


def find_lidar_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".las", ".laz"}:
        return [root]
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under: {root}")
    return files


def load_class_points(files, class_code, chunk_size=2_000_000):
    parts = []
    total = 0
    print(f"\nReading LiDAR class {class_code}...")
    for i, path in enumerate(files, start=1):
        if i == 1 or i % 10 == 0 or i == len(files):
            print(f"  tile {i:,}/{len(files):,}: {path.name}")
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
    print(f"  class {class_code} points: {total:,}")
    return np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64)


def rows_to_gdf(rows, crs):
    if rows:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([], crs=crs)},
        geometry="geometry",
        crs=crs,
    )


def span_frame(geom):
    coords = list(geom.coords)
    a = np.asarray(coords[0][:2], dtype=np.float64)
    b = np.asarray(coords[-1][:2], dtype=np.float64)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return None
    u = delta / length
    return {"origin": a, "u": u, "p": np.asarray([-u[1], u[0]]), "length": length}


def project_xyz(xyz, frame):
    centered = xyz[:, :2] - frame["origin"][None, :]
    return centered @ frame["u"], centered @ frame["p"]


def local_wire_subset(wire_xyz, tree, frame, half, end):
    midpoint = frame["origin"] + 0.5 * frame["length"] * frame["u"]
    radius = math.hypot(0.5 * frame["length"] + end, half)
    ids = tree.query_ball_point(midpoint, r=radius)
    if not ids:
        return np.empty((0, 3), dtype=np.float64)
    local = wire_xyz[np.asarray(ids, dtype=np.int64)]
    s, t = project_xyz(local, frame)
    mask = (
        (s >= -end)
        & (s <= frame["length"] + end)
        & (np.abs(t) <= half)
    )
    return local[mask]


def section_observations(xyz, frame, section_len, cluster_eps, min_cluster_points):
    s, t = project_xyz(xyz, frame)
    z = xyz[:, 2]
    n_sections = max(2, int(math.ceil(frame["length"] / section_len)))
    sections = []
    for section_id in range(n_sections):
        a = section_id * section_len
        b = min(frame["length"], a + section_len)
        mask = (s >= a) & ((s < b) if section_id < n_sections - 1 else (s <= b))
        ids = np.flatnonzero(mask)
        if len(ids) < min_cluster_points:
            sections.append([])
            continue
        tz = np.column_stack((t[ids], z[ids]))
        labels = DBSCAN(
            eps=cluster_eps,
            min_samples=min_cluster_points,
            n_jobs=-1,
        ).fit_predict(tz)
        obs = []
        for lab in sorted(set(int(v) for v in labels if int(v) >= 0)):
            local_ids = ids[labels == lab]
            if len(local_ids) < min_cluster_points:
                continue
            obs.append({
                "section": section_id,
                "s": float(np.median(s[local_ids])),
                "t": float(np.median(t[local_ids])),
                "z": float(np.median(z[local_ids])),
                "points": int(len(local_ids)),
            })
        sections.append(obs)
    return sections, n_sections


def predict_track_state(track, target_s):
    obs = track["obs"]
    if len(obs) < 2:
        last = obs[-1]
        return float(last["t"]), float(last["z"])
    recent = obs[-3:]
    s = np.asarray([o["s"] for o in recent])
    t = np.asarray([o["t"] for o in recent])
    z = np.asarray([o["z"] for o in recent])
    if np.ptp(s) <= 1e-9:
        return float(t[-1]), float(z[-1])
    return (
        float(np.polyval(np.polyfit(s, t, 1), target_s)),
        float(np.polyval(np.polyfit(s, z, 1), target_s)),
    )


def link_tracks(sections, max_t_jump, max_z_jump, max_missed):
    tracks = []
    next_id = 1
    for observations in sections:
        active = [tr for tr in tracks if tr["missed"] <= max_missed]
        used_obs, used_tracks = set(), set()
        if active and observations:
            matrix = np.full((len(active), len(observations)), 1e9)
            for i, tr in enumerate(active):
                for j, cur in enumerate(observations):
                    pred_t, pred_z = predict_track_state(tr, cur["s"])
                    dt = abs(cur["t"] - pred_t)
                    dz = abs(cur["z"] - pred_z)
                    if dt <= max_t_jump and dz <= max_z_jump:
                        matrix[i, j] = math.sqrt(
                            (dt / max(max_t_jump, 1e-9)) ** 2
                            + (dz / max(max_z_jump, 1e-9)) ** 2
                        )
            rows, cols = linear_sum_assignment(matrix)
            for r, c in zip(rows, cols):
                if matrix[r, c] >= 1e8:
                    continue
                tr = active[int(r)]
                tr["obs"].append(observations[int(c)])
                tr["missed"] = 0
                used_tracks.add(id(tr))
                used_obs.add(int(c))
        for tr in active:
            if id(tr) not in used_tracks:
                tr["missed"] += 1
        for j, obs in enumerate(observations):
            if j not in used_obs:
                tracks.append({"id": next_id, "obs": [obs], "missed": 0})
                next_id += 1
    return tracks


def fit_track(
    track,
    frame,
    n_sections,
    unit_to_m,
    min_track_sections,
    min_full_coverage,
    endpoint_fraction,
    max_plan_rmse_m,
    max_z_rmse_m,
):
    obs = sorted(track["obs"], key=lambda row: row["s"])
    if len(obs) < min_track_sections:
        return None, "TOO_FEW_SECTIONS", {}
    s = np.asarray([o["s"] for o in obs])
    t = np.asarray([o["t"] for o in obs])
    z = np.asarray([o["z"] for o in obs])
    if np.ptp(s) <= 1e-9:
        return None, "ZERO_TRACK_EXTENT", {}

    full_coverage = len(set(int(o["section"]) for o in obs)) / max(n_sections, 1)
    start_frac = float(s.min() / max(frame["length"], 1e-9))
    end_frac = float(s.max() / max(frame["length"], 1e-9))
    start_ok = start_frac <= endpoint_fraction
    end_ok = end_frac >= 1.0 - endpoint_fraction

    t_coef = np.polyfit(s, t, 1)
    z_coef = np.polyfit(s, z, 2 if len(obs) >= 3 else 1)
    t_rmse_m = float(np.sqrt(np.mean((t - np.polyval(t_coef, s)) ** 2))) * unit_to_m
    z_rmse_m = float(np.sqrt(np.mean((z - np.polyval(z_coef, s)) ** 2))) * unit_to_m

    sample_count = max(16, min(300, int(frame["length"] * unit_to_m / 2.0) + 2))
    sample_s = np.linspace(0.0, frame["length"], sample_count)
    sample_t = np.polyval(t_coef, sample_s)
    sample_z = np.polyval(z_coef, sample_s)
    xy = (
        frame["origin"][None, :]
        + sample_s[:, None] * frame["u"][None, :]
        + sample_t[:, None] * frame["p"][None, :]
    )
    geom = LineString([
        (float(x), float(y), float(zz))
        for (x, y), zz in zip(xy, sample_z)
    ])

    metrics = {
        "OBS_SECTIONS": len(obs),
        "N_SECTIONS": n_sections,
        "FULL_COVERAGE": full_coverage,
        "START_FRAC": start_frac,
        "END_FRAC": end_frac,
        "START_OK": start_ok,
        "END_OK": end_ok,
        "T_RMSE_M": t_rmse_m,
        "Z_RMSE_M": z_rmse_m,
        "SUPPORT_PTS": int(sum(o["points"] for o in obs)),
        "MED_T_M": float(np.median(t) * unit_to_m),
    }

    if not start_ok:
        return geom, "NO_START_SUPPORT", metrics
    if not end_ok:
        return geom, "NO_END_SUPPORT", metrics
    if full_coverage < min_full_coverage:
        return geom, "LOW_FULL_COVERAGE", metrics
    if t_rmse_m > max_plan_rmse_m:
        return geom, "HIGH_PLAN_RMSE", metrics
    if z_rmse_m > max_z_rmse_m:
        return geom, "HIGH_Z_RMSE", metrics
    return geom, "ACCEPTED", metrics


def build_parser():
    p = argparse.ArgumentParser(
        description="Vectorise LiDAR conductors span-by-span using validated accepted_spans topology."
    )
    p.add_argument("topology", help="01_reference_topology.gpkg")
    p.add_argument("--cache", default=None, help="Optional class_187_xyz.npy cache")
    p.add_argument("--lidar", default=None, help="LAS/LAZ fallback if cache is unavailable")
    p.add_argument("-o", "--output", default="02_conductor_vectorisation_output")
    p.add_argument("--wire-class", type=int, default=187)
    p.add_argument("--corridor-halfwidth-m", type=float, default=18.0)
    p.add_argument("--end-margin-m", type=float, default=12.0)
    p.add_argument("--section-length-m", type=float, default=5.0)
    p.add_argument("--cross-section-eps-m", type=float, default=0.9)
    p.add_argument("--min-cluster-points", type=int, default=2)
    p.add_argument("--max-t-jump-m", type=float, default=2.0)
    p.add_argument("--max-z-jump-m", type=float, default=3.0)
    p.add_argument("--max-missed-sections", type=int, default=2)
    p.add_argument("--min-track-sections", type=int, default=5)
    p.add_argument("--min-full-coverage", type=float, default=0.60)
    p.add_argument("--endpoint-fraction", type=float, default=0.20)
    p.add_argument("--max-plan-rmse-m", type=float, default=0.75)
    p.add_argument("--max-z-rmse-m", type=float, default=1.25)
    return p


def main():
    args = build_parser().parse_args()
    topology = Path(args.topology).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = (
        Path(args.cache).expanduser().resolve()
        if args.cache
        else topology.parent / "_cache" / f"class_{args.wire_class}_xyz.npy"
    )
    lidar = Path(args.lidar).expanduser().resolve() if args.lidar else None

    print("=" * 80)
    print("SPAN-BASED CONDUCTOR VECTORISATION")
    print("=" * 80)

    spans = gpd.read_file(topology, layer="accepted_spans")
    if spans.empty or spans.crs is None:
        raise RuntimeError("accepted_spans is empty or has no CRS.")

    unit_to_m = float(spans.crs.axis_info[0].unit_conversion_factor)
    print(f"Accepted spans: {len(spans):,}")
    print(f"CRS: {spans.crs}")

    if cache.exists():
        print(f"\nLoading wire cache:\n  {cache}")
        wire_xyz = np.load(cache)
        print(f"  wire points: {len(wire_xyz):,} [cache]")
    else:
        if lidar is None:
            raise RuntimeError("Cache not found. Supply --lidar.")
        wire_xyz = load_class_points(find_lidar_files(lidar), args.wire_class)

    tree = cKDTree(wire_xyz[:, :2])
    half = args.corridor_halfwidth_m / unit_to_m
    end = args.end_margin_m / unit_to_m
    section_len = args.section_length_m / unit_to_m
    cluster_eps = args.cross_section_eps_m / unit_to_m
    max_t = args.max_t_jump_m / unit_to_m
    max_z = args.max_z_jump_m / unit_to_m

    final_rows, review_rows, summary_rows = [], [], []
    wire_number = 1

    for span_no, span in enumerate(spans.itertuples(), start=1):
        if span_no == 1 or span_no % 20 == 0 or span_no == len(spans):
            print(f"  span {span_no:,}/{len(spans):,}: {span.SPAN_ID}")

        frame = span_frame(span.geometry)
        if frame is None:
            continue
        local = local_wire_subset(wire_xyz, tree, frame, half, end)
        if len(local) == 0:
            summary_rows.append({
                "SPAN_ID": span.SPAN_ID,
                "LINE_NO": getattr(span, "LINE_NO", ""),
                "WIRES_ACCEPTED": 0,
                "WIRES_REVIEW": 0,
                "STATUS": "NO_LOCAL_WIRE_POINTS",
                "geometry": span.geometry,
            })
            continue

        sections, n_sections = section_observations(
            local, frame, section_len, cluster_eps, args.min_cluster_points
        )
        tracks = link_tracks(
            sections, max_t, max_z, args.max_missed_sections
        )

        n_final = n_review = 0
        for track in tracks:
            geom, reason, metrics = fit_track(
                track,
                frame,
                n_sections,
                unit_to_m,
                args.min_track_sections,
                args.min_full_coverage,
                args.endpoint_fraction,
                args.max_plan_rmse_m,
                args.max_z_rmse_m,
            )
            if geom is None:
                continue

            row = {
                "WIRE_ID": f"W{wire_number:06d}",
                "SPAN_ID": span.SPAN_ID,
                "LINE_NO": getattr(span, "LINE_NO", ""),
                "STRUCTURE_SERIES": getattr(span, "STRUCTURE_SERIES", ""),
                "STRUCT_A": getattr(span, "STRUCT_A", ""),
                "STRUCT_B": getattr(span, "STRUCT_B", ""),
                "LAS_CLASS": args.wire_class,
                "STATUS": "FINAL" if reason == "ACCEPTED" else "REVIEW",
                "QA_REASON": reason,
                **metrics,
                "LENGTH_M": float(geom.length) * unit_to_m,
                "geometry": geom,
            }

            if reason == "ACCEPTED":
                final_rows.append(row)
                wire_number += 1
                n_final += 1
            else:
                row["WIRE_ID"] = f"R{span.SPAN_ID}_{track['id']:03d}"
                review_rows.append(row)
                n_review += 1

        summary_rows.append({
            "SPAN_ID": span.SPAN_ID,
            "LINE_NO": getattr(span, "LINE_NO", ""),
            "STRUCTURE_SERIES": getattr(span, "STRUCTURE_SERIES", ""),
            "STRUCT_A": getattr(span, "STRUCT_A", ""),
            "STRUCT_B": getattr(span, "STRUCT_B", ""),
            "WIRE_POINTS_LOCAL": len(local),
            "N_SECTIONS": n_sections,
            "TRACKS_RAW": len(tracks),
            "WIRES_ACCEPTED": n_final,
            "WIRES_REVIEW": n_review,
            "STATUS": "OK" if n_final else "NO_ACCEPTED_WIRES",
            "geometry": span.geometry,
        })

    wires = rows_to_gdf(final_rows, spans.crs)
    review = rows_to_gdf(review_rows, spans.crs)
    summary = rows_to_gdf(summary_rows, spans.crs)

    gpkg = output / "02_conductor_vectorisation.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    spans.to_file(gpkg, layer="accepted_spans", driver="GPKG", engine="pyogrio")
    summary.to_file(gpkg, layer="span_summary", driver="GPKG", engine="pyogrio")

    if not wires.empty:
        wires.to_file(gpkg, layer="wires_3d", driver="GPKG", engine="pyogrio")
        w2 = wires.copy()
        w2["geometry"] = [
            LineString([(c[0], c[1]) for c in geom.coords])
            for geom in w2.geometry
        ]
        w2.to_file(gpkg, layer="wires_2d", driver="GPKG", engine="pyogrio")

        endpoints = []
        for row in wires.itertuples():
            coords = list(row.geometry.coords)
            for end_name, coord in (("START", coords[0]), ("END", coords[-1])):
                endpoints.append({
                    "WIRE_ID": row.WIRE_ID,
                    "SPAN_ID": row.SPAN_ID,
                    "END": end_name,
                    "geometry": Point(*coord),
                })
        rows_to_gdf(endpoints, spans.crs).to_file(
            gpkg, layer="wire_endpoints", driver="GPKG", engine="pyogrio"
        )
        wires.drop(columns="geometry").to_csv(output / "wire_summary.csv", index=False)

    if not review.empty:
        review.to_file(gpkg, layer="qa_review", driver="GPKG", engine="pyogrio")
        review.drop(columns="geometry").to_csv(output / "qa_review.csv", index=False)

    summary.drop(columns="geometry").to_csv(output / "span_summary.csv", index=False)

    manifest = {
        "stage": "02_span_based_conductor_vectorisation",
        "version": "0.1-dev",
        "topology_input": str(topology),
        "wire_class": args.wire_class,
        "spans": len(spans),
        "wires_final": len(wires),
        "wires_review": len(review),
        "geometry_truth": "LIDAR",
        "topology_role": "VALIDATED_ANALYSIS_SEGMENTS",
    }
    (output / "vectorisation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nResult:")
    print(f"  final wires:   {len(wires):,}")
    print(f"  review tracks: {len(review):,}")
    print(
        f"  spans with final wires: "
        f"{int((summary['WIRES_ACCEPTED'] > 0).sum()) if not summary.empty else 0:,}"
        f"/{len(spans):,}"
    )
    print(f"\nWritten:\n  {gpkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
