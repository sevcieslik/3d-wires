#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import pandas as pd
import geopandas as gpd
from pyproj import CRS

from stage1_core import (
    lidar_files,
    inspect_dataset,
    resolve_sources,
    load_class_points,
    vectorise_source,
    export_outputs,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Agnostic Stage 1 conductor extraction. LAS/LAZ is the geometry truth; "
            "centreline and client source data are not required."
        )
    )
    p.add_argument("input", help="LAS/LAZ file or folder")
    p.add_argument("-o", "--output", default="01_conductor_output")
    p.add_argument("--ptc", default=None, help="Optional PTC classification schema")
    p.add_argument("--corridor-eps-m", type=float, default=25.0)
    p.add_argument("--section-length-m", type=float, default=8.0)
    p.add_argument("--cross-section-eps-m", type=float, default=1.2)
    p.add_argument("--max-t-jump-m", type=float, default=2.5)
    p.add_argument("--max-z-jump-m", type=float, default=4.0)
    p.add_argument("--min-cluster-points", type=int, default=2)
    p.add_argument("--min-track-sections", type=int, default=4)
    p.add_argument(
        "--max-points-per-source",
        type=int,
        default=None,
        help="Optional development cap. Default reads all points from each selected class.",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ptc = Path(args.ptc).expanduser().resolve() if args.ptc else None

    print("=" * 78)
    print("AGNOSTIC STAGE 1 - LIDAR CONDUCTOR EXTRACTION")
    print("=" * 78)

    files = lidar_files(input_path)
    print(f"Input LAS/LAZ: {len(files):,}")

    info = inspect_dataset(files)
    print(f"CRS: {info.crs or 'UNKNOWN'}")
    print(f"CRS status: {info.crs_status}")
    print(f"Source unit: {info.source_unit}")
    print("\nClassification counts:")
    for code, count in info.class_counts.items():
        print(f"  class {code:>3}: {count:,}")

    sources = resolve_sources(info, ptc)
    if not sources:
        print("\nNo known conductor classes were found.")
        if ptc:
            print("The supplied PTC did not identify a populated conductor class.")
        else:
            print("No PTC supplied.")
        print(
            "Stage 1 will exit normally with diagnostics. "
            "Automatic geometry-based class discovery is the next development fallback."
        )
        export_outputs(
            output,
            info,
            [],
            gpd.GeoDataFrame(geometry=[]),
            gpd.GeoDataFrame(geometry=[]),
            [],
        )
        return 0

    print("\nConductor sources:")
    for source in sources:
        print(
            f"  {source.source_id}: class {source.las_class} | {source.label} "
            f"| {source.method} | {source.point_count:,} points"
        )

    crs = CRS.from_user_input(info.crs) if info.crs else None
    all_wires = []
    all_axes = []
    all_stats = []

    for source in sources:
        print(f"\nVectorising {source.source_id} / class {source.las_class}...")
        xyz = load_class_points(
            files,
            source.las_class,
            max_points=args.max_points_per_source,
        )
        print(f"  loaded points: {len(xyz):,}")

        wires, axes, stats = vectorise_source(
            xyz,
            source,
            crs,
            info.unit_to_metre,
            corridor_eps_m=args.corridor_eps_m,
            section_length_m=args.section_length_m,
            cross_section_eps_m=args.cross_section_eps_m,
            max_t_jump_m=args.max_t_jump_m,
            max_z_jump_m=args.max_z_jump_m,
            min_cluster_points=args.min_cluster_points,
            min_track_sections=args.min_track_sections,
        )
        print(f"  networks: {len(axes):,}")
        print(f"  wire tracks: {len(wires):,}")

        if not wires.empty:
            all_wires.append(wires)
        if not axes.empty:
            all_axes.append(axes)
        all_stats.extend(stats)

    if all_wires:
        wires = gpd.GeoDataFrame(
            pd.concat(all_wires, ignore_index=True),
            geometry="geometry",
            crs=crs,
        )
        # Re-number globally because each source starts locally at W000001.
        wires["WIRE_ID"] = [f"W{i:06d}" for i in range(1, len(wires) + 1)]
    else:
        wires = gpd.GeoDataFrame(geometry=[], crs=crs)

    if all_axes:
        axes = gpd.GeoDataFrame(
            pd.concat(all_axes, ignore_index=True),
            geometry="geometry",
            crs=crs,
        )
    else:
        axes = gpd.GeoDataFrame(geometry=[], crs=crs)

    export_outputs(output, info, sources, wires, axes, all_stats)

    print("\nWritten:")
    print(f"  {output / '01_conductor_geometry.gpkg'}")
    print(f"  {output / 'dataset_manifest.json'}")
    print(f"  {output / 'class_summary.csv'}")
    print(f"  {output / 'source_groups.csv'}")
    print(f"\nFinal wires: {(wires['STATUS'] == 'FINAL').sum() if not wires.empty else 0:,}")
    print(f"Review wires: {(wires['STATUS'] != 'FINAL').sum() if not wires.empty else 0:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
