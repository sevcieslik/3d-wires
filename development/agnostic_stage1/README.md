# Agnostic Stage 1 - LiDAR Conductor Extraction

Development path for a project-agnostic first stage that derives conductor geometry from LAS/LAZ without requiring centreline, structure, KMZ or client GIS data.

## Principles

1. LiDAR is the geometry truth.
2. Centreline, KMZ and client GIS are optional reference evidence only.
3. Missing optional reference data must never stop Stage 1.
4. Known conductor classes are checked first:
   - 187: Wire
   - 24: Low voltage wire
   - 25: High voltage wire
   - 26: Railroad wire
5. If none of the known classes contain points, a supplied PTC is inspected.
6. If no useful PTC mapping is available, Stage 1 reports that automatic class discovery is required. Geometry discovery beyond known/PTC classes is deliberately kept separate from vectorisation in this first development version.
7. Each input conductor population receives a neutral SOURCE_ID such as C01, C02.
8. Source class is not assumed to be a circuit.
9. Detected geometry is grouped into neutral NETWORK_ID values such as NET_001.
10. NETWORK_AXIS is derived from LiDAR conductor points, not from supplied client linework.

## Current V0.1 workflow

LAS/LAZ
-> inspect CRS and class counts
-> resolve conductor source classes
-> load conductor points
-> coarse network/corridor grouping
-> derive local PCA axis for each network
-> slice along the axis
-> cluster conductor observations in local cross-section (t,z)
-> link observations between slices
-> fit 3D tracks
-> derive 2D wire strings and network axes
-> export GeoPackage + manifest + CSV diagnostics

## Usage

Basic:

    python 01_conductor_engine.py /path/to/lidar

With a PTC:

    python 01_conductor_engine.py /path/to/lidar --ptc mapping.ptc

Output folder:

    python 01_conductor_engine.py /path/to/lidar -o 01_conductor_output

## Outputs

01_conductor_geometry.gpkg

Layers:
- wires_3d
- wires_2d
- network_axes
- source_groups
- wire_endpoints
- qa_review

Other:
- dataset_manifest.json
- class_summary.csv
- source_groups.csv
- wire_summary.csv

## Important

This branch is a development proof of concept. It has not yet been calibrated against a representative production LAS/LAZ sample. Thresholds are intentionally exposed through CLI options and should be validated before production use.

External reference matching is the next module. It will compare derived NETWORK_ID/NETWORK_AXIS geometry against optional KMZ, DXF/DGN, SHP/GPKG or client source data without snapping or reshaping LiDAR-derived wire geometry.
