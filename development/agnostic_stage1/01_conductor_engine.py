#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

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
from structure_topology import (
    resolve_structure_class,
    cluster_structures,
    build_span_candidates,
    vectorise_source_by_spans,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Agnostic Stage 1 conductor extraction. LAS/LAZ is geometry truth. "
            "When structure points are available they are used to derive local span axes; "
            "external centrelines are not required."
        )
    )
    p.add_argument("input", help="LAS/LAZ file or folder")
    p.add_argument("-o", "--output", default="01_conductor_output")
    p.add_argument("--ptc", default=None, help="Optional PTC classification schema")

    p.add_argument("--corridor-eps-m", type=float, default=25.0)
    p.add_argument("--structure-cluster-eps-m", type=float, default=5.0)
    p.add_argument("--structure-min-points", type=int, default=3)
    p.add_argument("--min-span-m", type=float, default=12.0)
    p.add_argument("--max-span-m", type=float, default=550.0)
    p.add_argument("--structure-on-line-tol-m", type=float, default=12.0)
    p.add_argument("--span-corridor-halfwidth-m", type=float, default=25.0)
    p.add_argument("--span-evidence-bins", type=int, default=16)
    p.add_argument("--span-min-wire-points", type=int, default=40)
    p.add_argument("--span-min-wire-coverage", type=float, default=0.55)

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
    print("AGNOSTIC STAGE 1 - STRUCTURE-FIRST CONDUCTOR EXTRACTION")
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
        print("Stage 1 exits normally with diagnostics.")
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

    structure_class, structure_label, structure_method = resolve_structure_class(info, ptc)
    structures = gpd.GeoDataFrame(geometry=[], crs=crs)

    if structure_class is not None:
        print(
            f"\nStructure source: class {structure_class} | "
            f"{structure_label} | {structure_method}"
        )
        structure_xyz = load_class_points(files, structure_class)
        print(f"  loaded structure points: {len(structure_xyz):,}")

        structures = cluster_structures(
            structure_xyz,
            crs,
            info.unit_to_metre,
            eps_m=args.structure_cluster_eps_m,
            min_points=args.structure_min_points,
        )
        print(f"  clustered structures: {len(structures):,}")
    else:
        print("\nNo populated structure class identified.")
        print("  Falling back to the wire-only PCA mode. This is lower confidence.")

    all_wires = []
    all_axes = []
    all_stats = []
    all_span_candidates = []

    for source in sources:
        print(f"\nVectorising {source.source_id} / class {source.las_class}...")
        xyz = load_class_points(
            files,
            source.las_class,
            max_points=args.max_points_per_source,
        )
        print(f"  loaded wire points: {len(xyz):,}")

        if not structures.empty:
            spans = build_span_candidates(
                structures,
                xyz,
                crs,
                info.unit_to_metre,
                min_span_m=args.min_span_m,
                max_span_m=args.max_span_m,
                structure_on_line_tol_m=args.structure_on_line_tol_m,
                corridor_halfwidth_m=args.span_corridor_halfwidth_m,
                evidence_bins=args.span_evidence_bins,
                min_wire_points=args.span_min_wire_points,
                min_wire_coverage=args.span_min_wire_coverage,
            )

            print(f"  structure-pair spans confirmed by wires: {len(spans):,}")

            if not spans.empty:
                spans = spans.copy()
                spans["SOURCE_ID"] = source.source_id
                spans["LAS_CLASS"] = int(source.las_class)
                all_span_candidates.append(spans)

            wires, axes, stats = vectorise_source_by_spans(
                xyz,
                source,
                spans,
                crs,
                info.unit_to_metre,
                corridor_halfwidth_m=args.span_corridor_halfwidth_m,
                section_length_m=args.section_length_m,
                cross_section_eps_m=args.cross_section_eps_m,
                max_t_jump_m=args.max_t_jump_m,
                max_z_jump_m=args.max_z_jump_m,
                min_cluster_points=args.min_cluster_points,
                min_track_sections=args.min_track_sections,
            )
        else:
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

        print(f"  span/network axes: {len(axes):,}")
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

    gpkg = output / "01_conductor_geometry.gpkg"

    if not structures.empty:
        structures.to_file(
            gpkg,
            layer="structures",
            driver="GPKG",
            engine="pyogrio",
        )

    if all_span_candidates:
        span_candidates = gpd.GeoDataFrame(
            pd.concat(all_span_candidates, ignore_index=True),
            geometry="geometry",
            crs=crs,
        )
        span_candidates.to_file(
            gpkg,
            layer="span_candidates",
            driver="GPKG",
            engine="pyogrio",
        )

    print("\nWritten:")
    print(f"  {gpkg}")
    print(f"  {output / 'dataset_manifest.json'}")
    print(f"  {output / 'class_summary.csv'}")
    print(f"  {output / 'source_groups.csv'}")
    if not structures.empty:
        print("  GPKG layer: structures")
    if all_span_candidates:
        print("  GPKG layer: span_candidates")

    print(
        f"\nFinal wires: "
        f"{(wires['STATUS'] == 'FINAL').sum() if not wires.empty else 0:,}"
    )
    print(
        f"Review wires: "
        f"{(wires['STATUS'] != 'FINAL').sum() if not wires.empty else 0:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
