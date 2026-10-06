from __future__ import annotations

"""
WIRE + STRUCTURE 3D PRODUCTS - CLASS 187 / STRUCTURES 215
=========================================================

Purpose
-------
One script for staged generation of:
    * 3D conductor vectors from LiDAR class 187;
    * tower-top points from LiDAR class 215;
    * tower-bottom points using exact centreline-node XY and local ground Z;
    * tower-bottom ground validation;
    * one combined GeoPackage containing final wires, tower tops and tower bottoms.

Centreline input may be DXF or DGN v8. DGN v8 requires a GDAL/OGR build with
DGNv8 (ODA) support; if unavailable, a same-stem DXF or centrelines.dxf is used
automatically. When --centreline is omitted, the script auto-detects a suitable
DXF/DGN beside the script and prefers DXF.

LiDAR inputs are tiled LAS/LAZ files in ./input by default.

Main classes
------------
    187 - Conductors
    215 - Structures
      2 - Ground by default (override with --ground-class)

Examples
--------
    python wire_structure_products_v3.py --stage all
    python wire_structure_products_v3.py --stage all --epsg 26917
    python wire_structure_products_v3.py --stage wires
    python wire_structure_products_v3.py --stage tops bottoms validate
    python wire_structure_products_v3.py --stage inspect
    python wire_structure_products_v3.py --stage all --centreline "ClusterX_Build_Project_meters.dxf"

CRS behaviour
-------------
If at least one input LAS/LAZ contains a valid CRS, that embedded CRS is used.
If all input tiles have no CRS, use --epsg CODE or the script asks once for the
EPSG code. The resolved CRS is propagated to the working LAZ, GeoPackages and
sidecar CRS files.

Final product GeoPackage
------------------------
wire_structure_products.gpkg contains, when available:
    wires_3d       - final 3D conductor lines with attributes
    tower_tops     - final 3D tower-top points with audit attributes
    tower_bottoms  - final 3D tower-bottom points with audit attributes
"""

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import itertools
import math
import sys
import shutil


import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from shapely.geometry import Point, LineString
from shapely.affinity import scale as scale_geometry

import laspy
import ezdxf
import pyogrio
from pyproj import CRS


# =============================================================================
# PATHS / RUNTIME DEFAULTS
# =============================================================================

BASE_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_FOLDER / "input"
CENTERLINE_PATH = BASE_FOLDER / "centrelines.dxf"
SOURCE_LAZ = BASE_FOLDER / "_working_187_215.laz"
DXF_PATH = CENTERLINE_PATH  # legacy variable name retained by the wire engine

US_SURVEY_FT_PER_M = 3937.0 / 1200.0
SOURCE_TO_INTERNAL_FT = 1.0
INTERNAL_FT_TO_SOURCE = 1.0
SOURCE_UNIT_NAME = "unknown"
ASSUME_SOURCE_UNITS_IF_CRS_MISSING = "metre"
FALLBACK_CRS = None
# Resolved once at startup. Embedded LiDAR CRS wins; --epsg/prompt is only
# a fallback when every input tile is missing CRS metadata.
EFFECTIVE_CRS = None
EFFECTIVE_CRS_SOURCE = "unresolved"
TRACKER_SCRIPT: Path | None = None

OUTPUT_ROOT = BASE_FOLDER / "wire_structure_outputs"
OUTPUT_DIR = OUTPUT_ROOT / "diagnostics"
OUTPUT_GPKG = OUTPUT_DIR / "wire_extraction_187.gpkg"

SPAN_SUMMARY_CSV = OUTPUT_DIR / "span_wire_summary.csv"
HYPOTHESIS_SUMMARY_CSV = OUTPUT_DIR / "hypothesis_summary.csv"
FINAL_WIRE_SUMMARY_CSV = OUTPUT_DIR / "final_wire_summary.csv"
EXTRACTION_SUMMARY_CSV = OUTPUT_DIR / "extraction_summary.csv"
DXF_SPAN_CSV = OUTPUT_DIR / "centreline_exploded_span_summary.csv"
DXF_NODE_CSV = OUTPUT_DIR / "centreline_node_summary.csv"
CANDIDATE_METRICS_CSV = OUTPUT_DIR / "candidate_metrics.csv"
CANDIDATE_ACTIONS_CSV = OUTPUT_DIR / "candidate_validation_actions.csv"
TOPOLOGY_NODE_AUDIT_CSV = OUTPUT_DIR / "topology_node_audit.csv"
TOPOLOGY_STRUCTURE_AUDIT_CSV = OUTPUT_DIR / "topology_structure_audit.csv"
TOPOLOGY_CORRECTIONS_CSV = OUTPUT_DIR / "topology_corrections.csv"
GEOMETRY_QA_CSV = OUTPUT_DIR / "geometry_qa.csv"
CLASS_EXTRACTION_SUMMARY_CSV = OUTPUT_DIR / "class_extraction_summary.csv"
TRACKER_STITCH_ACTIONS_CSV = OUTPUT_DIR / "tracker_stitch_actions.csv"

# Final deliverables: keep the output root deliberately clean.
WIRES_PRODUCT_DGN = OUTPUT_ROOT / "wires_3d.dgn"
WIRES_PRODUCT_PRJ = OUTPUT_ROOT / "wires_3d.prj"
PRODUCT_GPKG = OUTPUT_ROOT / "wire_structure_products.gpkg"
TOWER_TOP_XYZ = OUTPUT_ROOT / "tower_tops.xyz"
TOWER_TOP_PRJ = OUTPUT_ROOT / "tower_tops.prj"
TOWER_BOTTOM_XYZ = OUTPUT_ROOT / "tower_bottoms.xyz"
TOWER_BOTTOM_PRJ = OUTPUT_ROOT / "tower_bottoms.prj"
CRS_TXT = OUTPUT_ROOT / "crs.txt"

# Audit / validation products stay under diagnostics/.
TOWER_TOP_CSV = OUTPUT_DIR / "tower_tops_audit.csv"
TOWER_BOTTOM_CSV = OUTPUT_DIR / "tower_bottoms_audit.csv"
TOWER_BOTTOM_VALIDATION_CSV = OUTPUT_DIR / "tower_bottom_validation.csv"
TOWER_BOTTOM_VALIDATION_GPKG = OUTPUT_DIR / "tower_bottom_validation.gpkg"

# =============================================================================
# LIDAR CLASSES
# =============================================================================

# New dataset: all conductors are carried in one LiDAR class.
CONDUCTOR_CLASS_NAMES = {187: "Conductor_187"}

WIRE_CLASSES = set(CONDUCTOR_CLASS_NAMES)
STRUCTURE_CLASS = 215

# Accept usable linework from every centreline layer/model by default.
# None means: accept usable LINE/LWPOLYLINE/POLYLINE geometry from every DXF
# layer. If a future dataset contains derived linework that must be excluded,
# set this to an explicit set of lowercase layer names instead.
ALLOWED_DXF_LAYERS: set[str] | None = None

LAS_CHUNK_POINTS = 2_000_000


# =============================================================================
# DXF TOPOLOGY
# =============================================================================

DXF_NODE_MERGE_TOLERANCE_FT = 2.0
MIN_DXF_SPAN_LENGTH_FT = 5.0


# =============================================================================
# CLASS 215 QA
# =============================================================================

STRUCTURE_CLUSTER_EPS_FT = 12.0
STRUCTURE_CLUSTER_MIN_SAMPLES = 2
MIN_STRUCTURE_POINTS = 3

NODE_STRUCTURE_GOOD_DISTANCE_FT = 40.0
NODE_STRUCTURE_REVIEW_DISTANCE_FT = 100.0


# =============================================================================
# CLASS 215 <-> DXF TOPOLOGY VALIDATION / CORRECTION
# =============================================================================

ENABLE_TOPOLOGY_CORRECTION = True

# A class-215 cluster within this distance is considered associated with an
# existing DXF node for topology purposes.
TOPO_STRUCTURE_NODE_MATCH_FT = 45.0

# Missing structure candidate inside a DXF span.
TOPO_MISSING_STRUCTURE_MAX_SPAN_DISTANCE_FT = 30.0
TOPO_MISSING_STRUCTURE_MIN_END_CLEARANCE_FT = 60.0
TOPO_MISSING_STRUCTURE_MIN_SECOND_BEST_MARGIN_FT = 10.0
TOPO_MISSING_STRUCTURE_MIN_POINTS = 8

# Suspect intermediate DXF vertex. Automatic collapse is intentionally limited
# to degree-2, same-layer, near-collinear vertices with no class-215 evidence
# and clear classified-conductor continuity through the node.
TOPO_EXTRA_VERTEX_MIN_STRUCTURE_DISTANCE_FT = 70.0
TOPO_EXTRA_VERTEX_MIN_STRAIGHT_ANGLE_DEG = 168.0

TOPO_CONTINUITY_RADIUS_FT = 45.0
TOPO_CONTINUITY_CORRIDOR_FT = 20.0
TOPO_CONTINUITY_SIDE_INNER_FT = 5.0
TOPO_CONTINUITY_SIDE_OUTER_FT = 40.0
TOPO_CONTINUITY_MIN_POINTS_EACH_SIDE = 6

TOPO_MAX_COLLAPSE_PASSES = 1000


# =============================================================================
# SPAN POINT QUERY
# =============================================================================

SPAN_QUERY_HALF_WIDTH_FT = 80.0
SPAN_QUERY_END_EXTENSION_FT = 30.0


# =============================================================================
# BUNDLE COLLAPSE
# =============================================================================

# Bundled conductors may contain several subconductors around one physical phase.
# 0.5-0.7 m apart. The extraction product required here is ONE centre vector
# per physical bundle, not one vector per subconductor.
#
# All geometry engines operate internally in US survey feet, therefore the
# user-facing metric settings below are converted once to internal feet.
ENABLE_BUNDLE_COLLAPSE = True

# Subconductors whose local (lateral, vertical) centres are within this
# distance are treated as members of one bundle. 0.90 m intentionally gives
# headroom above the observed 0.5-0.7 m separation.
BUNDLE_COLLAPSE_DISTANCE_M = 0.90
BUNDLE_COLLAPSE_DISTANCE_FT = (
    BUNDLE_COLLAPSE_DISTANCE_M
    * US_SURVEY_FT_PER_M
)

# Safety gate against transitive DBSCAN chains accidentally combining two
# distinct phase conductors. A proposed bundle wider than this is left split.
BUNDLE_MAX_DIAMETER_M = 1.20
BUNDLE_MAX_DIAMETER_FT = (
    BUNDLE_MAX_DIAMETER_M
    * US_SURVEY_FT_PER_M
)

# A centreline through a bundle must own points on BOTH sides of the centre.
# These values describe the bundle envelope, not the location error of one
# subconductor.
BUNDLE_OWNERSHIP_RADIUS_M = 0.75
BUNDLE_LOOSE_RADIUS_M = 0.90
BUNDLE_MEDIAN_RESIDUAL_LIMIT_M = 0.55
BUNDLE_P90_RESIDUAL_LIMIT_M = 0.85
BUNDLE_TRACKER_AGREEMENT_M = 0.90

BUNDLE_OWNERSHIP_RADIUS_FT = (
    BUNDLE_OWNERSHIP_RADIUS_M
    * US_SURVEY_FT_PER_M
)
BUNDLE_LOOSE_RADIUS_FT = (
    BUNDLE_LOOSE_RADIUS_M
    * US_SURVEY_FT_PER_M
)
BUNDLE_MEDIAN_RESIDUAL_LIMIT_FT = (
    BUNDLE_MEDIAN_RESIDUAL_LIMIT_M
    * US_SURVEY_FT_PER_M
)
BUNDLE_P90_RESIDUAL_LIMIT_FT = (
    BUNDLE_P90_RESIDUAL_LIMIT_M
    * US_SURVEY_FT_PER_M
)
BUNDLE_TRACKER_AGREEMENT_FT = (
    BUNDLE_TRACKER_AGREEMENT_M
    * US_SURVEY_FT_PER_M
)

# Number of samples used when several accepted models still survive and need
# to be replaced with one robust median centreline.
BUNDLE_CENTERLINE_SAMPLES = 41


# =============================================================================
# LEGACY V2.5 AUTOMATIC DX / TX MODE (NOT USED BY V3 MAIN)
# =============================================================================

# Length is only one signal. Close conductor spacing can force DX mode even on
# a somewhat longer span. Ambiguous cases run BOTH engines.
DX_STRONG_MAX_LENGTH_FT = 170.0
TX_STRONG_MIN_LENGTH_FT = 260.0

DX_STRONG_MAX_TERMINAL_SPACING_FT = 4.5
TX_STRONG_MIN_TERMINAL_SPACING_FT = 7.0

# Generic terminal clustering used only to estimate mode.
MODE_TERMINAL_WINDOW_FT = 18.0
MODE_CLUSTER_T_SCALE_FT = 1.0
MODE_CLUSTER_Z_SCALE_FT = 1.0
MODE_DBSCAN_EPS = 1.25
MODE_DBSCAN_MIN_SAMPLES = 3


# =============================================================================
# DX ENGINE
# =============================================================================

DX = {
    "name": "DX",

    "terminal_window_ft": 14.0,
    "terminal_min_points": 3,
    "terminal_t_scale_ft": 0.45,
    "terminal_z_scale_ft": 0.65,
    "terminal_dbscan_eps": 1.15,

    # Tight internal slices: short span + closely spaced conductors.
    "body_fractions": (0.20, 0.40, 0.60, 0.80),
    "body_slice_width_ft": 5.0,
    "body_min_points": 3,
    "body_t_scale_ft": 0.45,
    "body_z_scale_ft": 0.70,
    "body_dbscan_eps": 1.15,

    # A_i -> B_j must pass close to body cluster laterally.
    "body_lateral_gate_ft": 1.40,

    # Implied sag-depth clustering.
    "sag_group_eps_ft": 1.60,
    "max_sag_depth_ft": 18.0,

    # Global point ownership.
    "ownership_tube_ft": 0.80,
    "ownership_margin_ft": 0.10,

    # Final QA.
    "tight_tube_ft": 0.40,
    "loose_tube_ft": 0.80,
    "min_owned_points": 10,
    "min_coverage": 0.55,
    "max_median_residual_ft": 0.38,
    "max_p90_residual_ft": 0.78,

    # Bundle-aware QA. A wider physical conductor/bundle is allowed a wider
    # validation envelope, but only when its terminal clusters themselves show
    # a wider cross-section.
    "bundle_spread_factor": 0.85,
    "max_adaptive_radius_ft": 0.95,
    "adaptive_median_factor": 0.75,
    "adaptive_p90_factor": 1.35,
    "ownership_radius_factor": 1.00,
    "loose_radius_factor": 1.15,
    "ownership_margin_ratio": 0.10,

    # Sparse but distributed evidence can still define a reliable short wire.
    "sparse_min_coverage": 0.18,
    "sparse_min_body_slices": 3,
    "sparse_min_support_zones": 3,
    "sparse_max_gap_fraction": 0.45,

    "coverage_bins": 12,

    "sample_interval_ft": 2.5,

    # Match against V7.
    "v7_agreement_ft": 0.90,
}


# =============================================================================
# TX ENGINE
# =============================================================================

TX = {
    "name": "TX",

    "terminal_window_ft": 28.0,
    "terminal_min_points": 4,
    "terminal_t_scale_ft": 0.90,
    "terminal_z_scale_ft": 1.10,
    "terminal_dbscan_eps": 1.20,

    # Several checks through the span. Missing slices are acceptable.
    "body_fractions": (0.20, 0.40, 0.50, 0.60, 0.80),
    "body_slice_width_ft": 8.0,
    "body_min_points": 3,
    "body_t_scale_ft": 0.85,
    "body_z_scale_ft": 1.15,
    "body_dbscan_eps": 1.20,

    "body_lateral_gate_ft": 4.25,

    "sag_group_eps_ft": 4.0,
    "max_sag_depth_ft": 120.0,

    "ownership_tube_ft": 1.50,
    "ownership_margin_ft": 0.18,

    "tight_tube_ft": 0.70,
    "loose_tube_ft": 1.50,
    "min_owned_points": 14,
    "min_coverage": 0.35,
    "max_median_residual_ft": 0.65,
    "max_p90_residual_ft": 1.40,

    # Bundled HV phase conductors can be physically much wider than shield
    # wires. Do not reject a good centreline because the LiDAR points represent
    # several subconductors around that centreline.
    "bundle_spread_factor": 1.00,
    "max_adaptive_radius_ft": 1.70,
    "adaptive_median_factor": 0.72,
    "adaptive_p90_factor": 1.35,
    "ownership_radius_factor": 1.00,
    "loose_radius_factor": 1.15,
    "ownership_margin_ratio": 0.08,

    # A physical model may bridge classification gaps when both terminals and
    # several independent body slices support the same conductor.
    "sparse_min_coverage": 0.15,
    "sparse_min_body_slices": 3,
    "sparse_min_support_zones": 3,
    "sparse_max_gap_fraction": 0.50,

    "coverage_bins": 20,

    "sample_interval_ft": 5.0,

    "v7_agreement_ft": 1.50,
}



# =============================================================================
# V3 CLASS-SPECIFIC ENGINE CONFIGURATION
# =============================================================================

def _engine_variant(
    base: dict,
    name: str,
    **overrides,
):
    output = dict(base)
    output.update(overrides)
    output["name"] = name
    return output


# The old DX/TX dictionaries remain as geometric templates. These variants
# carry the class-specific calibration inferred from the reclassified dataset.
POA_EARTH = _engine_variant(
    TX,
    "EARTH",
    terminal_window_ft=22.0,
    terminal_min_points=3,
    terminal_t_scale_ft=0.70,
    terminal_z_scale_ft=0.85,
    terminal_dbscan_eps=1.15,
    body_slice_width_ft=7.0,
    body_t_scale_ft=0.70,
    body_z_scale_ft=0.90,
    body_dbscan_eps=1.15,
    body_lateral_gate_ft=2.50,
    ownership_tube_ft=1.00,
    tight_tube_ft=0.55,
    loose_tube_ft=1.00,
    min_owned_points=10,
    min_coverage=0.80,
    max_median_residual_ft=0.45,
    max_p90_residual_ft=0.95,
    sparse_min_coverage=0.35,
    sparse_min_body_slices=3,
    sparse_min_support_zones=1,
    sparse_max_gap_fraction=0.40,
    coverage_bins=16,
    sample_interval_ft=4.0,
    v7_agreement_ft=1.10,
    bundle_enabled=False,
)

POA_DX = _engine_variant(
    DX,
    "DX",
    min_owned_points=8,
    min_coverage=0.70,
    sparse_min_coverage=0.25,
    sparse_min_body_slices=3,
    sparse_min_support_zones=1,
    sparse_max_gap_fraction=0.50,
    coverage_bins=14,
    v7_agreement_ft=0.90,
    bundle_enabled=False,
)

POA_COMMS = _engine_variant(
    DX,
    "COMMS",
    terminal_window_ft=16.0,
    terminal_min_points=2,
    terminal_t_scale_ft=0.55,
    terminal_z_scale_ft=0.80,
    terminal_dbscan_eps=1.15,
    body_slice_width_ft=6.0,
    body_min_points=2,
    body_t_scale_ft=0.55,
    body_z_scale_ft=0.80,
    body_dbscan_eps=1.15,
    body_lateral_gate_ft=1.75,
    ownership_tube_ft=0.90,
    tight_tube_ft=0.45,
    loose_tube_ft=0.90,
    min_owned_points=7,
    min_coverage=0.55,
    max_median_residual_ft=0.45,
    max_p90_residual_ft=0.90,
    sparse_min_coverage=0.25,
    sparse_min_body_slices=2,
    sparse_min_support_zones=1,
    sparse_max_gap_fraction=0.55,
    coverage_bins=14,
    sample_interval_ft=2.5,
    v7_agreement_ft=1.00,
    bundle_enabled=False,
)

POA_TX = _engine_variant(
    TX,
    "TX",
    min_coverage=0.85,
    sparse_min_coverage=0.40,
    sparse_min_support_zones=1,
    sparse_max_gap_fraction=0.40,
    v7_agreement_ft=1.50,
    bundle_enabled=True,
)

POA_345 = _engine_variant(
    TX,
    "HV345",
    terminal_window_ft=32.0,
    terminal_t_scale_ft=1.10,
    terminal_z_scale_ft=1.35,
    terminal_dbscan_eps=1.25,
    body_slice_width_ft=10.0,
    body_t_scale_ft=1.10,
    body_z_scale_ft=1.40,
    body_dbscan_eps=1.25,
    body_lateral_gate_ft=5.50,
    ownership_tube_ft=1.80,
    tight_tube_ft=0.85,
    loose_tube_ft=1.80,
    min_owned_points=16,
    min_coverage=0.90,
    max_median_residual_ft=0.80,
    max_p90_residual_ft=1.70,
    bundle_spread_factor=1.10,
    max_adaptive_radius_ft=2.10,
    sparse_min_coverage=0.45,
    sparse_min_body_slices=4,
    sparse_min_support_zones=1,
    sparse_max_gap_fraction=0.35,
    coverage_bins=22,
    sample_interval_ft=5.0,
    v7_agreement_ft=1.80,
    bundle_enabled=True,
)

CLASS_ENGINE_CONFIG = {
    187: {
        "class_name": "Conductor_187",
        "engine_key": "AUTO",
        "family": "AUTO",
        "query_half_width_ft": 80.0,
    },
}


# =============================================================================
# HYPOTHESIS / SELECTION SETTINGS
# =============================================================================

# Weak gates for an A->B candidate to enter the global assignment.
MIN_HYPOTHESIS_BODY_SLICES = 1
MIN_HYPOTHESIS_COVERAGE = 0.18
MAX_HYPOTHESIS_MEDIAN_RESIDUAL_FT = 2.5

# If terminal counts are <= this value, exact permutation search can be used
# when both sides have equal counts. This permits a global conductor collision
# penalty instead of relying exclusively on Hungarian assignment.
MAX_EXACT_ASSIGNMENT_COUNT = 6

# Hypothesis cost.
COST_COVERAGE_WEIGHT = 35.0
COST_MEDIAN_RESIDUAL_WEIGHT = 8.0
COST_P90_RESIDUAL_WEIGHT = 2.0
COST_BODY_SLICE_WEIGHT = 3.0
COST_TERMINAL_RMSE_WEIGHT = 1.0
COST_POINT_SUPPORT_WEIGHT = 0.20

# Global physical collision penalty between selected wires.
MIN_WIRE_SEPARATION_FT = 0.55
WIRE_COLLISION_PENALTY = 50.0

# Very poor pair.
INVALID_PAIR_COST = 1_000_000.0


# =============================================================================
# CONFIRMED TX BUNDLE RESCUE
# =============================================================================

# Adaptive radius is NOT a generic relaxation. It is enabled only for TX_MAIN
# hypotheses whose terminal cross-sections independently look like a bundle on
# both structures and whose body support is extensive.
TX_BUNDLE_MIN_TERMINAL_SPREAD_FT = 0.42
TX_BUNDLE_MAX_SPREAD_RATIO = 1.80
TX_BUNDLE_MIN_BODY_SLICES = 4
TX_BUNDLE_MIN_INITIAL_COVERAGE = 0.70


# =============================================================================
# TX -> DX UNDERBUILD PASS
# =============================================================================

# After strong TX wires are found, points already explained by those models are
# removed before a second, tighter DX pass. This allows a long transmission span
# to contain both the main HV circuit and a close-spaced distribution underbuild.
ENABLE_TX_RESIDUAL_DX_PASS = True
UNDERBUILD_MIN_RESIDUAL_POINTS = 20

# Claim a slightly wider tube than the final centreline QA envelope so noisy
# points belonging to a confirmed TX bundle do not seed false DX conductors.
TX_CLAIM_MIN_TUBE_FT = 1.50
TX_CLAIM_RADIUS_FACTOR = 1.30

# A DX residual solution is only considered an underbuild if it stays below
# the LOWEST confirmed TX conductor over most of the span.
UNDERBUILD_MIN_VERTICAL_CLEARANCE_FT = 4.0
UNDERBUILD_MIN_BELOW_FRACTION = 0.75
UNDERBUILD_COMPARISON_SAMPLES = 25

# Two or more mutually consistent underbuild wires can confirm one another.
# A solitary candidate is kept out of final unless V7 independently agrees.
UNDERBUILD_MIN_GROUP_SIZE_FOR_AUTO = 2


# =============================================================================
# LEGACY V2.5 V7 TRACKER CONFIG (NOT USED BY V3 MAIN)
# =============================================================================

# TRACKER_OVERRIDES removed in V3.0; CLASS_TRACKER is authoritative.

# Only a clean V7 fit may be used alone as final geometry.
ALLOW_V7_ONLY_FINAL = True

# If a POA result is strong but V7 disagrees, keep POA as final but lower
# confidence. Set False if you prefer every disagreement to go to review.
ALLOW_STRONG_POA_WITHOUT_V7_AGREEMENT = False


# =============================================================================
# INTERNAL TRACKER + NETWORK SECOND PASS
# =============================================================================

# Legacy embedded tracker retained for V2.5 helper compatibility. V3 main uses CLASS_TRACKER.
INTERNAL_TRACKER = {
    "DX": {
        "section_length_ft": 4.0,
        "t_scale_ft": 0.50,
        "z_scale_ft": 0.70,
        "dbscan_eps": 1.15,
        "dbscan_min_samples": 3,
        "max_t_jump_ft": 1.30,
        "max_z_jump_ft": 2.00,
        "max_cost": 1.50,
        "max_missed_sections": 2,
        "min_sections": 4,

        # Normal acceptance path.
        "accept_coverage": 0.45,
        "review_coverage": 0.20,
        "max_t_rmse_ft": 0.60,
        "max_z_rmse_ft": 0.75,

        # V2.4 sparse-data acceptance path.
        # Lower coverage is accepted only when the fit is extremely stable
        # and observations are distributed through the span.
        "adaptive_accept_coverage": 0.30,
        "adaptive_max_t_rmse_ft": 0.20,
        "adaptive_max_z_rmse_ft": 0.30,
        "adaptive_min_observed_sections": 5,
        "adaptive_min_support_extent_fraction": 0.70,
        "adaptive_max_longest_gap_fraction": 0.50,

        "sample_interval_ft": 2.5,
    },
    "TX": {
        "section_length_ft": 10.0,
        "t_scale_ft": 1.00,
        "z_scale_ft": 1.30,
        "dbscan_eps": 1.20,
        "dbscan_min_samples": 3,
        "max_t_jump_ft": 2.50,
        "max_z_jump_ft": 4.00,
        "max_cost": 1.60,
        "max_missed_sections": 5,
        "min_sections": 4,
        "accept_coverage": 0.45,
        "review_coverage": 0.22,
        "max_t_rmse_ft": 0.90,
        "max_z_rmse_ft": 1.15,
        "sample_interval_ft": 5.0,
    },
}

ENABLE_CANDIDATE_VALIDATION_SECOND_PASS = True

# -------------------------------------------------------------------------
# Candidate registry
# -------------------------------------------------------------------------

# Candidate pool is intentionally limited to evidence that showed some value
# against the manual cleanup. REJECTED geometry is not part of V2.
CV_USE_REVIEW = True
CV_USE_POA = True
CV_USE_V7 = True

# Merge geometrically equivalent candidate evidence rather than discarding the
# second engine. POA + V7 agreement is one of the strongest promotion signals.
CV_MERGE_DISTANCE_DX_FT = 0.90
CV_MERGE_DISTANCE_TX_FT = 1.50

# Do not create a candidate that is already represented by a first-pass final.
CV_FINAL_DUPLICATE_DISTANCE_DX_FT = 0.70
CV_FINAL_DUPLICATE_DISTANCE_TX_FT = 1.20

# -------------------------------------------------------------------------
# Residual point ownership
# -------------------------------------------------------------------------

# First-pass finals claim their own points before candidate validation.
CV_FINAL_CLAIM_TUBE_DX_FT = 0.70
CV_FINAL_CLAIM_TUBE_TX_FT = 1.40

# Candidate validation tubes.
CV_CANDIDATE_TUBE_DX_FT = 0.75
CV_CANDIDATE_TUBE_TX_FT = 1.50

# A point belongs to a candidate only when it is the best model and is clearly
# better than the second-best candidate. This is the V4/TerraScan-like idea.
CV_OWNERSHIP_MARGIN_RATIO_DX = 0.12
CV_OWNERSHIP_MARGIN_RATIO_TX = 0.10

# Diagnostics only.
CV_MAX_SUPPORT_POINTS_PER_CANDIDATE = 120
CV_MAX_SUPPORT_POINT_ROWS = 150_000

# -------------------------------------------------------------------------
# Fresh residual-LiDAR QA
# -------------------------------------------------------------------------

# IMPORTANT V2.2 CHANGE
# support_zones counts contiguous groups of occupied bins. A continuous,
# full-span conductor normally has support_zones == 1, so V2's requirement
# support_zones >= 2 was backwards. support_zones remains diagnostic only.

CV_STRONG = {
    "DX": {
        "min_owned_points": 8,
        "min_coverage": 0.35,
        "max_median_residual_ft": 0.40,
        "max_p90_residual_ft": 0.85,
        "max_gap_fraction": 0.55,
        "min_terminal_points_each": 2,
        "min_body_slices": 3,
        "min_unique_fraction": 0.60,
    },
    "TX": {
        "min_owned_points": 12,
        "min_coverage": 0.25,
        "max_median_residual_ft": 0.70,
        "max_p90_residual_ft": 1.50,
        "max_gap_fraction": 0.60,
        "min_terminal_points_each": 3,
        "min_body_slices": 3,
        "min_unique_fraction": 0.60,
    },
}

CV_EXCEPTIONAL = {
    "DX": {
        "min_owned_points": 12,
        "min_coverage": 0.50,
        "max_median_residual_ft": 0.30,
        "max_p90_residual_ft": 0.65,
        "max_gap_fraction": 0.45,
        "min_terminal_points_each": 3,
        "min_body_slices": 3,
        "min_unique_fraction": 0.78,
    },
    "TX": {
        "min_owned_points": 18,
        "min_coverage": 0.40,
        "max_median_residual_ft": 0.55,
        "max_p90_residual_ft": 1.15,
        "max_gap_fraction": 0.50,
        "min_terminal_points_each": 4,
        "min_body_slices": 3,
        "min_unique_fraction": 0.75,
    },
}

# DX requires materially stronger evidence than TX for auto-promotion.
CV_DX_PROMOTION = {
    "min_owned_points": 8,
    "min_coverage": 0.80,
    "max_median_residual_ft": 0.40,
    "max_p90_residual_ft": 0.85,
    "max_gap_fraction": 0.55,
    "min_terminal_points_each": 4,
    "min_body_slices": 4,
    "min_unique_fraction": 0.60,
}

# -------------------------------------------------------------------------
# Network support
# -------------------------------------------------------------------------

CV_NETWORK_SAME_LAYER_ONLY = True

CV_NETWORK_GATE_DX_FT = 2.75
CV_NETWORK_GATE_TX_FT = 5.00

# TX: network remains bonus-only.
# DX: trusted support at BOTH ends is required for automatic promotion.
CV_REQUIRE_BOTH_NETWORK_ENDS_FOR_DX_PROMOTION = True

# -------------------------------------------------------------------------
# Possible vertical duplicate diagnostic
# -------------------------------------------------------------------------

# Vertically stacked conductors can legitimately share XY. Therefore XY
# similarity never deletes a candidate. In DX, when a candidate otherwise
# qualifies for promotion but almost duplicates an existing first-pass XY
# trajectory, keep it in REVIEW as POSSIBLE_VERTICAL_DUPLICATE.
CV_DX_VERTICAL_DUPLICATE_REVIEW_VETO = True
CV_TX_VERTICAL_DUPLICATE_REVIEW_VETO = False

CV_VERTICAL_DUPLICATE_MAX_MEAN_XY_FT_DX = 0.75
CV_VERTICAL_DUPLICATE_MAX_MEAN_XY_FT_TX = 1.00
CV_VERTICAL_DUPLICATE_MIN_OVERLAP = 0.80
CV_VERTICAL_DUPLICATE_MAX_DIRECTION_DIFF_DEG = 12.0
CV_VERTICAL_DUPLICATE_MAX_MEAN_3D_FT = 9.0

# -------------------------------------------------------------------------
# Promotion policy
# -------------------------------------------------------------------------

CV_ENABLE_POA_V7_AUTO_PROMOTE = True

# V2 benchmark did not justify V7-only automatic promotion.
CV_ENABLE_V7_ONLY_AUTO_PROMOTE = False

# POA/REVIEW-only remains review-only.
CV_ENABLE_POA_ONLY_AUTO_PROMOTE = False

CV_PROMOTED_MIN_SEPARATION_DX_FT = 0.60
CV_PROMOTED_MIN_SEPARATION_TX_FT = 1.00

# Crowding is QA only, not a veto by itself.
CV_DX_CROWDING_DISTANCE_FT = 1.60

# A span with first-pass unresolved status and no candidate is surfaced for QA.
CV_UNRESOLVED_STATUSES = {
    "PARTIAL",
    "REVIEW",
    "NO_WIRES",
    "NO_WIRE_POINTS",
}


# =============================================================================
# PHYSICAL GEOMETRY QA
# =============================================================================

ENABLE_GEOMETRY_PHYSICS_QA = True

GEOM_QA_SAMPLE_COUNT = 41
GEOM_QA_MAX_BACKTRACK_FT = 1.0

# Hard review gates. These are deliberately conservative: they are intended to
# catch visible kinks/jumps, not to tune normal catenary shape.
GEOM_QA_DX_MAX_TURN_DEG = 20.0
GEOM_QA_TX_MAX_TURN_DEG = 15.0

GEOM_QA_DX_MAX_LATERAL_LINEAR_RESIDUAL_FT = 1.00
GEOM_QA_TX_MAX_LATERAL_LINEAR_RESIDUAL_FT = 1.50

GEOM_QA_DX_MAX_Z_QUADRATIC_RESIDUAL_FT = 1.50
GEOM_QA_TX_MAX_Z_QUADRATIC_RESIDUAL_FT = 2.50


# =============================================================================
# LOGICAL NODE RECONCILIATION
# =============================================================================

# V2.3 NEVER mutates conductor geometry during node reconciliation.
# This step records likely continuity only.
NODE_JOIN_ONLY_DEGREE_2 = True

NODE_JOIN_MAX_ENDPOINT_XY_FT = 5.0
NODE_JOIN_MAX_ENDPOINT_Z_FT = 5.0
NODE_JOIN_MAX_COST = 3.0
NODE_JOIN_Z_WEIGHT = 1.4


# =============================================================================
# QA SAMPLES
# =============================================================================

MAX_WIRE_SAMPLE = 200_000
MAX_STRUCTURE_SAMPLE = 75_000


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StructureQA:
    structure_id: str
    xyz: np.ndarray
    anchor_xyz: np.ndarray


@dataclass
class DXFNode:
    node_id: str
    xy: np.ndarray
    source_vertex_count: int

    nearest_structure_id: str = ""
    nearest_structure_distance_ft: float = np.nan
    nearest_structure_quality: str = "NO_STRUCTURE"


@dataclass
class DXFSpan:
    span_id: str
    layer_name: str
    geometry: LineString

    node_a: str
    node_b: str

    source_entity_no: int
    source_segment_no: int

    duplicate_count: int
    source_entities: str

    axis_origin_xy: np.ndarray
    axis_u: np.ndarray
    axis_p: np.ndarray
    span_length_ft: float
    midpoint_xy: np.ndarray


@dataclass
class TerminalPOA:
    terminal_id: str
    end: str

    t: float
    z: float

    point_count: int
    s_min: float
    s_max: float

    t_rmse: float
    z_rmse: float

    bundle_member_count: int = 1


@dataclass
class BodyCluster:
    cluster_id: str
    slice_index: int
    fraction: float

    s: float
    t: float
    z: float

    point_count: int
    bundle_member_count: int = 1


@dataclass
class WireHypothesis:
    hypothesis_id: str

    a_id: str
    b_id: str

    t_a: float
    z_a: float
    t_b: float
    z_b: float

    k: float
    sag_depth_ft: float

    body_slice_hits: int
    body_slice_count: int

    support_loose: int
    coverage: float
    median_residual_ft: float
    p90_residual_ft: float

    terminal_rmse: float

    cost: float
    valid: bool

    # Terminal / bundle geometry.
    terminal_a_points: int = 0
    terminal_b_points: int = 0
    terminal_spread_a_ft: float = 0.0
    terminal_spread_b_ft: float = 0.0
    terminal_spread_ft: float = 0.0

    bundle_confirmed: bool = False
    adaptive_radius_ft: float = 0.0

    underbuild_below_fraction: float = np.nan
    underbuild_spatial_ok: bool = False
    underbuild_group_size: int = 0

    # Filled after ownership/refit.
    owned_points: int = 0
    owned_coverage: float = 0.0
    owned_median_residual_ft: float = np.nan
    owned_p90_residual_ft: float = np.nan

    support_zone_count: int = 0
    longest_unsupported_gap_ft: float = np.nan
    longest_unsupported_gap_fraction: float = np.nan

    adaptive_median_limit_ft: float = np.nan
    adaptive_p90_limit_ft: float = np.nan

    quality: str = "UNASSESSED"


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def log(text: str = "") -> None:
    print(text, flush=True)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n  {path}"
        )


def safe_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    return (
        ""
        if text.lower() in {"", "nan", "none", "<na>"}
        else text
    )



def configure_source_units(crs) -> None:
    """Configure source-coordinate scaling to internal US survey feet."""
    global SOURCE_TO_INTERNAL_FT
    global INTERNAL_FT_TO_SOURCE
    global SOURCE_UNIT_NAME

    unit_name = ""

    if crs is not None:
        try:
            axis_info = crs.axis_info
            if axis_info:
                unit_name = safe_text(axis_info[0].unit_name).lower()
        except Exception:
            unit_name = ""

    if not unit_name:
        unit_name = safe_text(
            ASSUME_SOURCE_UNITS_IF_CRS_MISSING
        ).lower()
        log(
            "\nWARNING: CRS unit could not be read."
            f"\n  Assuming source units: {unit_name or 'metre'}"
        )

    if (
        "metre" in unit_name
        or "meter" in unit_name
        or unit_name in {"m", "metres", "meters"}
    ):
        SOURCE_TO_INTERNAL_FT = US_SURVEY_FT_PER_M
        SOURCE_UNIT_NAME = "metre"

    elif "foot" in unit_name or "feet" in unit_name or "ft" == unit_name:
        SOURCE_TO_INTERNAL_FT = 1.0
        SOURCE_UNIT_NAME = unit_name or "foot"

    else:
        # Conservative fallback for projected metric source data.
        SOURCE_TO_INTERNAL_FT = US_SURVEY_FT_PER_M
        SOURCE_UNIT_NAME = unit_name or "metre_assumed"
        log(
            "\nWARNING: Unrecognised CRS linear unit."
            f"\n  Unit text: {unit_name or '<blank>'}"
            "\n  Assuming metres for internal scaling."
        )

    INTERNAL_FT_TO_SOURCE = 1.0 / SOURCE_TO_INTERNAL_FT

    log(
        "\nCoordinate units:"
        f"\n  source unit: {SOURCE_UNIT_NAME}"
        f"\n  source -> internal ft factor: {SOURCE_TO_INTERNAL_FT:.12f}"
        f"\n  internal ft -> source factor: {INTERNAL_FT_TO_SOURCE:.12f}"
    )


def geometry_to_source_units(geometry):
    """Scale an internal-foot Shapely geometry back to source CRS units."""
    if geometry is None:
        return None

    try:
        if geometry.is_empty:
            return geometry
    except Exception:
        return geometry

    if abs(INTERNAL_FT_TO_SOURCE - 1.0) <= 1e-15:
        return geometry

    return scale_geometry(
        geometry,
        xfact=INTERNAL_FT_TO_SOURCE,
        yfact=INTERNAL_FT_TO_SOURCE,
        zfact=INTERNAL_FT_TO_SOURCE,
        origin=(0.0, 0.0, 0.0),
    )

def norm(vector: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(vector, dtype=np.float64)
        )
    )


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = norm(vector)

    if (
        not np.isfinite(length)
        or length <= 1e-9
    ):
        raise ValueError("Cannot normalise zero vector.")

    return vector / length


def make_gdf(
    records: list[dict],
    crs,
) -> gpd.GeoDataFrame:
    if not records:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    output_records = []

    for record in records:
        copied = dict(record)

        if "geometry" in copied:
            copied["geometry"] = geometry_to_source_units(
                copied.get("geometry")
            )

        output_records.append(copied)

    return gpd.GeoDataFrame(
        output_records,
        geometry="geometry",
        crs=crs,
    )


def even_indices(
    count: int,
    maximum: int,
) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)

    return np.linspace(
        0,
        count - 1,
        maximum,
        dtype=np.int64,
    )


def robust_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    max_iter: int = 6,
    sigma: float = 3.0,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < degree + 1:
        raise ValueError("Not enough points for robust_polyfit.")

    mask = np.ones(len(x), dtype=bool)

    for _ in range(max_iter):
        coeff = np.polyfit(
            x[mask],
            y[mask],
            degree,
        )

        prediction = np.polyval(coeff, x)
        residual = y - prediction
        active = residual[mask]

        if len(active) <= degree + 2:
            break

        median = float(np.median(active))
        mad = float(
            np.median(
                np.abs(active - median)
            )
        )

        if mad <= 1e-12:
            break

        robust_sigma = 1.4826 * mad

        new_mask = (
            np.abs(residual - median)
            <= sigma * robust_sigma
        )

        if np.array_equal(mask, new_mask):
            break

        if np.count_nonzero(new_mask) < degree + 1:
            break

        mask = new_mask

    coeff = np.polyfit(
        x[mask],
        y[mask],
        degree,
    )

    residual = (
        y[mask]
        - np.polyval(coeff, x[mask])
    )

    rmse = float(
        np.sqrt(
            np.mean(residual ** 2)
        )
    )

    return coeff, mask, rmse


def robust_weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    finite = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[finite]
    weights = weights[finite]

    if len(values) == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    threshold = 0.5 * float(np.sum(weights))

    index = int(
        np.searchsorted(
            cumulative,
            threshold,
            side="left",
        )
    )

    index = min(
        max(index, 0),
        len(values) - 1,
    )

    return float(values[index])


def coverage_from_s(
    s: np.ndarray,
    span_length: float,
    bins: int,
) -> float:
    if (
        len(s) == 0
        or span_length <= 0
        or bins <= 0
    ):
        return 0.0

    indices = np.floor(
        np.asarray(s, dtype=np.float64)
        / span_length
        * bins
    ).astype(np.int64)

    indices = np.clip(
        indices,
        0,
        bins - 1,
    )

    return float(
        len(np.unique(indices))
        / bins
    )


def support_distribution_metrics(
    s: np.ndarray,
    span_length: float,
    bins: int,
):
    """
    Return:
        support_zone_count
        longest_unsupported_gap_ft
        longest_unsupported_gap_fraction

    A support zone is a contiguous run of occupied longitudinal bins. This is
    deliberately independent of raw point density: three small but distributed
    fragments can be stronger evidence than thousands of points in one place.
    """
    if (
        span_length <= 0
        or bins <= 0
    ):
        return 0, float("nan"), float("nan")

    occupied = np.zeros(
        bins,
        dtype=bool,
    )

    values = np.asarray(
        s,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
        & (values >= 0.0)
        & (values <= span_length)
    ]

    if len(values):
        indices = np.floor(
            values
            / span_length
            * bins
        ).astype(np.int64)

        indices = np.clip(
            indices,
            0,
            bins - 1,
        )

        occupied[
            np.unique(indices)
        ] = True

    zone_count = 0
    in_zone = False

    for value in occupied:
        if value and not in_zone:
            zone_count += 1
            in_zone = True
        elif not value:
            in_zone = False

    longest_empty_bins = 0
    current_empty_bins = 0

    for value in occupied:
        if value:
            longest_empty_bins = max(
                longest_empty_bins,
                current_empty_bins,
            )
            current_empty_bins = 0
        else:
            current_empty_bins += 1

    longest_empty_bins = max(
        longest_empty_bins,
        current_empty_bins,
    )

    gap_ft = (
        longest_empty_bins
        * span_length
        / bins
    )

    gap_fraction = (
        gap_ft
        / span_length
        if span_length > 0
        else float("nan")
    )

    return (
        int(zone_count),
        float(gap_ft),
        float(gap_fraction),
    )


def hypothesis_terminal_spreads(
    a: TerminalPOA,
    b: TerminalPOA,
):
    """
    Return independent physical cross-section spread estimates at A and B.

    A real bundled phase should look broad at BOTH ends. One broad end and one
    narrow end is more likely to be structure clutter, a crossing, or mixed
    conductor evidence.
    """
    spread_a = float(
        math.hypot(
            float(a.t_rmse),
            float(a.z_rmse),
        )
    )

    spread_b = float(
        math.hypot(
            float(b.t_rmse),
            float(b.z_rmse),
        )
    )

    finite = [
        value
        for value in (
            spread_a,
            spread_b,
        )
        if np.isfinite(value)
    ]

    median_spread = (
        float(np.median(finite))
        if finite
        else 0.0
    )

    return (
        spread_a,
        spread_b,
        median_spread,
    )


def terminal_spreads_support_bundle(
    spread_a_ft: float,
    spread_b_ft: float,
) -> bool:
    if not (
        np.isfinite(spread_a_ft)
        and np.isfinite(spread_b_ft)
    ):
        return False

    if (
        spread_a_ft
        < TX_BUNDLE_MIN_TERMINAL_SPREAD_FT
        or spread_b_ft
        < TX_BUNDLE_MIN_TERMINAL_SPREAD_FT
    ):
        return False

    smaller = max(
        min(
            spread_a_ft,
            spread_b_ft,
        ),
        1e-6,
    )

    ratio = (
        max(
            spread_a_ft,
            spread_b_ft,
        )
        / smaller
    )

    return bool(
        ratio
        <= TX_BUNDLE_MAX_SPREAD_RATIO
    )


def adaptive_model_radius(
    terminal_spread_ft: float,
    params: dict,
) -> float:
    base = float(
        params[
            "tight_tube_ft"
        ]
    )

    radius = (
        base
        + float(
            params[
                "bundle_spread_factor"
            ]
        )
        * max(
            float(
                terminal_spread_ft
            ),
            0.0,
        )
    )

    return float(
        np.clip(
            radius,
            base,
            float(
                params[
                    "max_adaptive_radius_ft"
                ]
            ),
        )
    )


def quality_is_strong(
    quality: str,
) -> bool:
    return safe_text(
        quality
    ) in {
        "DENSE_STRONG",
        "SPARSE_STRONG",
    }


# =============================================================================
# PHASE-1 TRACKER
# =============================================================================

def resolve_tracker_script() -> Path:
    """
    Kept only for compatibility with older V5.x code paths.
    SECOND PASS V1 has an embedded tracker and does not require an external file.
    """
    return Path(__file__).resolve()


def load_tracker_module():
    log("Internal tracker: embedded in this script.")
    return None



# =============================================================================
# LIDAR
# =============================================================================

def read_required_lidar(
    path: Path,
):
    """
    Read each reclassified conductor family separately.

    Returns:
        class_xyz: dict[class_code -> Nx3 XYZ]
        structure_xyz: class-215 XYZ
        all_conductor_xyz: all class-187 conductor points, used only for
            topology continuity QA
        crs
    """
    require_path(
        path,
        "Source LAS/LAZ",
    )

    class_parts = {
        class_code: []
        for class_code in sorted(
            CONDUCTOR_CLASS_NAMES
        )
    }

    class_counts = {
        class_code: 0
        for class_code in sorted(
            CONDUCTOR_CLASS_NAMES
        )
    }

    structure_parts = []
    structure_count = 0

    with laspy.open(path) as reader:
        try:
            crs = reader.header.parse_crs()
        except Exception as exc:
            crs = None

            log(
                "\nWARNING: Could not parse embedded LAS/LAZ CRS."
                f"\n  {type(exc).__name__}: {exc}"
            )

        if crs is None:
            crs = FALLBACK_CRS

            log(
                "\nWARNING: LAS/LAZ has no embedded CRS."
                "\n  No CRS identifier will be fabricated."
                "\n  Source coordinates will be treated as metres for unit scaling."
            )

        configure_source_units(
            crs
        )

        log(
            f"\nLiDAR:\n"
            f"  {path}\n"
            f"  CRS: {crs}\n"
            f"  conductor classes: "
            f"{sorted(CONDUCTOR_CLASS_NAMES)}\n"
            f"  structure class: {STRUCTURE_CLASS}"
        )

        for chunk_number, chunk in enumerate(
            reader.chunk_iterator(
                LAS_CHUNK_POINTS
            ),
            start=1,
        ):
            classification = np.asarray(
                chunk.classification
            )

            relevant_mask = np.isin(
                classification,
                [
                    *sorted(
                        CONDUCTOR_CLASS_NAMES
                    ),
                    STRUCTURE_CLASS,
                ],
            )

            if not np.any(relevant_mask):
                continue

            x = (
                np.asarray(
                    chunk.x,
                    dtype=np.float64,
                )
                * SOURCE_TO_INTERNAL_FT
            )
            y = (
                np.asarray(
                    chunk.y,
                    dtype=np.float64,
                )
                * SOURCE_TO_INTERNAL_FT
            )
            z = (
                np.asarray(
                    chunk.z,
                    dtype=np.float64,
                )
                * SOURCE_TO_INTERNAL_FT
            )

            for class_code in sorted(
                CONDUCTOR_CLASS_NAMES
            ):
                mask = (
                    classification
                    == class_code
                )

                if not np.any(mask):
                    continue

                part = np.column_stack(
                    (
                        x[mask],
                        y[mask],
                        z[mask],
                    )
                )

                class_parts[
                    class_code
                ].append(part)

                class_counts[
                    class_code
                ] += len(part)

            structure_mask = (
                classification
                == STRUCTURE_CLASS
            )

            if np.any(structure_mask):
                part = np.column_stack(
                    (
                        x[structure_mask],
                        y[structure_mask],
                        z[structure_mask],
                    )
                )

                structure_parts.append(part)
                structure_count += len(part)

            class_count_text = ", ".join(
                f"{code}={class_counts[code]:,}"
                for code in sorted(
                    CONDUCTOR_CLASS_NAMES
                )
            )

            log(
                f"  chunk {chunk_number}: "
                f"{class_count_text}, "
                f"structures={structure_count:,}"
            )

    class_xyz = {}

    for class_code in sorted(
        CONDUCTOR_CLASS_NAMES
    ):
        parts = class_parts[
            class_code
        ]

        class_xyz[
            class_code
        ] = (
            np.vstack(parts)
            if parts
            else np.empty(
                (0, 3),
                dtype=np.float64,
            )
        )

    if not any(
        len(points)
        for points in class_xyz.values()
    ):
        raise RuntimeError(
            "No points found in the configured conductor classes."
        )

    structure_xyz = (
        np.vstack(structure_parts)
        if structure_parts
        else np.empty(
            (0, 3),
            dtype=np.float64,
        )
    )

    non_empty = [
        points
        for points in class_xyz.values()
        if len(points)
    ]

    all_conductor_xyz = np.vstack(
        non_empty
    )

    log("\nClass totals:")

    for class_code in sorted(
        CONDUCTOR_CLASS_NAMES
    ):
        log(
            f"  {class_code:>3} "
            f"{CONDUCTOR_CLASS_NAMES[class_code]:<8} "
            f"{len(class_xyz[class_code]):>12,}"
        )

    log(
        f"  {STRUCTURE_CLASS:>3} "
        f"Structures "
        f"{len(structure_xyz):>12,}"
    )

    return (
        class_xyz,
        structure_xyz,
        all_conductor_xyz,
        crs,
    )


# =============================================================================
# CLASS 215 QA
# =============================================================================

def cluster_structures_for_qa(
    structure_xyz: np.ndarray,
) -> list[StructureQA]:
    if len(structure_xyz) == 0:
        return []

    labels = DBSCAN(
        eps=STRUCTURE_CLUSTER_EPS_FT,
        min_samples=STRUCTURE_CLUSTER_MIN_SAMPLES,
        n_jobs=1,
    ).fit_predict(
        structure_xyz[:, :2]
    )

    raw = []

    for label in sorted(
        {
            int(value)
            for value in labels
            if int(value) >= 0
        }
    ):
        indices = np.flatnonzero(
            labels == label
        )

        if len(indices) < MIN_STRUCTURE_POINTS:
            continue

        xyz = structure_xyz[indices]

        raw.append(
            (
                np.median(xyz, axis=0),
                xyz,
            )
        )

    raw.sort(
        key=lambda item:
            (
                float(item[0][0]),
                float(item[0][1]),
            )
    )

    return [
        StructureQA(
            structure_id=f"LIDAR_S{index:04d}",
            xyz=xyz,
            anchor_xyz=anchor,
        )
        for index, (anchor, xyz)
        in enumerate(raw, start=1)
    ]


class StructureLookup:
    def __init__(
        self,
        structures: list[StructureQA],
    ):
        self.structures = structures

        if structures:
            xy = np.vstack(
                [
                    item.anchor_xyz[:2]
                    for item in structures
                ]
            )
            self.tree = cKDTree(xy)
        else:
            self.tree = None

    def nearest(
        self,
        xy: np.ndarray,
    ):
        if self.tree is None:
            return None

        distance, index = self.tree.query(
            np.asarray(xy, dtype=np.float64),
            k=1,
        )

        return (
            float(distance),
            self.structures[int(index)],
        )


def node_structure_quality(
    distance: float,
) -> str:
    if not np.isfinite(distance):
        return "NO_STRUCTURE"

    if distance <= NODE_STRUCTURE_GOOD_DISTANCE_FT:
        return "GOOD"

    if distance <= NODE_STRUCTURE_REVIEW_DISTANCE_FT:
        return "REVIEW"

    return "FAR"


# =============================================================================
# DXF TOPOLOGY
# =============================================================================

def dxf_entity_vertices(
    entity,
):
    entity_type = entity.dxftype()

    # Return coordinates in the CAD/source coordinate system. The wire engine
    # performs its source->internal-feet conversion later; product stages need
    # the original XY unchanged.
    if entity_type == "LINE":
        start = entity.dxf.start
        end = entity.dxf.end

        return [
            (
                float(start.x),
                float(start.y),
            ),
            (
                float(end.x),
                float(end.y),
            ),
        ]

    if entity_type == "LWPOLYLINE":
        points = [
            (
                float(point[0]),
                float(point[1]),
            )
            for point in entity.get_points("xy")
        ]

        if (
            len(points) >= 2
            and bool(entity.closed)
            and points[0] != points[-1]
        ):
            points.append(points[0])

        return points

    if entity_type == "POLYLINE":
        points = []

        for vertex in entity.vertices:
            location = vertex.dxf.location
            points.append(
                (
                    float(location.x),
                    float(location.y),
                )
            )

        try:
            closed = bool(entity.is_closed)
        except Exception:
            closed = False

        if (
            len(points) >= 2
            and closed
            and points[0] != points[-1]
        ):
            points.append(points[0])

        return points

    return []


def _geometry_parts(geometry):
    """Yield simple Shapely geometries recursively."""
    if geometry is None or geometry.is_empty:
        return
    geom_type = geometry.geom_type
    if geom_type in {"Point", "LineString", "LinearRing"}:
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _geometry_parts(part)


def read_centerline_sequences(path: Path):
    """
    Read centreline geometry from DXF or DGN v8.

    Returns:
        line_sequences: list[dict(layer_name, points[(x,y), ...])]
        explicit_points: list[(x,y)] from POINT-like CAD elements when available
        resolved_path: actual file used (DGN or DXF fallback)

    DGN v8 requires a GDAL build with the DGNv8/ODA driver. If that driver is
    absent, a same-stem DXF or centrelines.dxf is used automatically when found.
    """
    require_path(path, "Centreline")
    suffix = path.suffix.lower()
    line_sequences = []
    explicit_points = []
    resolved = path

    if suffix == ".dxf":
        document = ezdxf.readfile(path)
        modelspace = document.modelspace()
        for entity in modelspace:
            entity_type = entity.dxftype()
            layer_name = safe_text(getattr(entity.dxf, "layer", ""))
            if entity_type == "POINT":
                p = entity.dxf.location
                explicit_points.append((float(p.x), float(p.y)))
                continue
            if entity_type == "INSERT":
                p = entity.dxf.insert
                explicit_points.append((float(p.x), float(p.y)))
                continue
            points = dxf_entity_vertices(entity)
            if len(points) >= 2:
                line_sequences.append({"layer_name": layer_name, "points": points})
        return line_sequences, explicit_points, resolved

    if suffix != ".dgn":
        raise RuntimeError(f"Unsupported centreline format: {path.suffix}. Use .dgn or .dxf")

    try:
        layers = pyogrio.list_layers(path)
        for layer_info in layers:
            model_name = str(layer_info[0])
            frame = gpd.read_file(path, layer=model_name, engine="pyogrio")
            for _, row in frame.iterrows():
                geometry = row.geometry
                level = safe_text(row.get("Level", "")) if hasattr(row, "get") else ""
                layer_name = model_name if not level else f"{model_name}_L{level}"
                for part in _geometry_parts(geometry):
                    if part.geom_type == "Point":
                        explicit_points.append((float(part.x), float(part.y)))
                    elif part.geom_type in {"LineString", "LinearRing"}:
                        coords = [(float(c[0]), float(c[1])) for c in part.coords]
                        if len(coords) >= 2:
                            line_sequences.append({"layer_name": layer_name, "points": coords})
        if line_sequences:
            return line_sequences, explicit_points, resolved
        raise RuntimeError("DGN opened but contained no usable line geometry.")

    except Exception as exc:
        fallback_candidates = [
            path.with_suffix(".dxf"),
            path.parent / "centrelines.dxf",
            path.parent / "centerlines.dxf",
        ]
        fallback = next((p for p in fallback_candidates if p.exists()), None)
        if fallback is not None:
            log(
                "\nWARNING: DGN v8 could not be read by this GDAL build."
                f"\n  {type(exc).__name__}: {exc}"
                f"\n  Falling back to: {fallback}"
            )
            return read_centerline_sequences(fallback)

        raise RuntimeError(
            "Cannot read the DGN v8 centreline. This requires GDAL/OGR built "
            "with the DGNv8 (ODA) driver. Export the DGN once to DXF and place "
            "it beside the DGN as 'Cluster 8.dxf' or 'centrelines.dxf'.\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def read_and_explode_dxf(path: Path, structure_lookup: StructureLookup):
    """Generic centreline topology reader; legacy function name retained."""
    sequences, _explicit_points, resolved_path = read_centerline_sequences(path)
    scale = SOURCE_TO_INTERNAL_FT

    entities = []
    vertex_occurrences = []
    usable_layer_counts = {}

    for source_entity_no, sequence in enumerate(sequences, start=1):
        source_points = sequence["points"]
        points = [(float(x) * scale, float(y) * scale) for x, y in source_points]
        if len(points) < 2:
            continue
        layer_name = safe_text(sequence.get("layer_name", ""))
        layer_key = layer_name.lower()
        usable_layer_counts[layer_key] = usable_layer_counts.get(layer_key, 0) + 1
        if ALLOWED_DXF_LAYERS is not None and layer_key not in ALLOWED_DXF_LAYERS:
            continue
        start_index = len(vertex_occurrences)
        vertex_occurrences.extend(points)
        entities.append({
            "source_entity_no": source_entity_no,
            "layer_name": layer_key,
            "points": points,
            "vertex_start_index": start_index,
        })

    if not entities:
        raise RuntimeError("Centreline contains no usable line geometry.")

    log(f"\nCentreline used: {resolved_path}")
    if usable_layer_counts:
        log("Centreline linework groups found:")
        for layer, count in sorted(usable_layer_counts.items()):
            log(f"  {layer or '<blank>'}: {count:,} line entities")

    vertex_xy = np.asarray(vertex_occurrences, dtype=np.float64)
    labels = DBSCAN(
        eps=DXF_NODE_MERGE_TOLERANCE_FT,
        min_samples=1,
        n_jobs=1,
    ).fit_predict(vertex_xy)

    clusters = []
    for label in sorted(set(int(value) for value in labels)):
        points = vertex_xy[labels == label]
        clusters.append({
            "label": label,
            "xy": np.mean(points, axis=0),
            "source_vertex_count": len(points),
        })
    clusters.sort(key=lambda row: (float(row["xy"][0]), float(row["xy"][1])))

    label_to_node = {}
    nodes = []
    for number, row in enumerate(clusters, start=1):
        node_id = f"DXF_N{number:04d}"
        label_to_node[row["label"]] = node_id
        nearest = structure_lookup.nearest(row["xy"])
        if nearest is None:
            nearest_id = ""
            nearest_distance = np.nan
        else:
            nearest_distance = nearest[0]
            nearest_id = nearest[1].structure_id
        nodes.append(DXFNode(
            node_id=node_id,
            xy=np.asarray(row["xy"], dtype=np.float64),
            source_vertex_count=int(row["source_vertex_count"]),
            nearest_structure_id=nearest_id,
            nearest_structure_distance_ft=nearest_distance,
            nearest_structure_quality=node_structure_quality(nearest_distance),
        ))

    node_lookup = {node.node_id: node for node in nodes}
    occurrence_node_ids = np.empty(len(vertex_xy), dtype=object)
    for occurrence_index, label in enumerate(labels):
        occurrence_node_ids[occurrence_index] = label_to_node[int(label)]

    spans_by_pair = {}
    next_span_no = 1
    raw_segments = duplicate_segments = ignored_segments = 0

    for entity in entities:
        points = entity["points"]
        start_index = entity["vertex_start_index"]
        node_ids = occurrence_node_ids[start_index:start_index + len(points)]
        for segment_no in range(1, len(points)):
            raw_segments += 1
            node_a = str(node_ids[segment_no - 1])
            node_b = str(node_ids[segment_no])
            if node_a == node_b:
                ignored_segments += 1
                continue
            a_xy = node_lookup[node_a].xy
            b_xy = node_lookup[node_b].xy
            vector = b_xy - a_xy
            length = norm(vector)
            if length < MIN_DXF_SPAN_LENGTH_FT:
                ignored_segments += 1
                continue
            pair_key = tuple(sorted((node_a, node_b)))
            source_descriptor = f"{entity['source_entity_no']}:{segment_no}"
            if pair_key in spans_by_pair:
                existing = spans_by_pair[pair_key]
                existing.duplicate_count += 1
                existing.source_entities += ";" + source_descriptor
                duplicate_segments += 1
                continue
            axis_u = unit(vector)
            axis_p = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)
            geometry = LineString([
                (float(a_xy[0]), float(a_xy[1])),
                (float(b_xy[0]), float(b_xy[1])),
            ])
            span = DXFSpan(
                span_id=f"DXF_SPAN_{next_span_no:04d}",
                layer_name=entity["layer_name"],
                geometry=geometry,
                node_a=node_a,
                node_b=node_b,
                source_entity_no=int(entity["source_entity_no"]),
                source_segment_no=int(segment_no),
                duplicate_count=1,
                source_entities=source_descriptor,
                axis_origin_xy=a_xy.copy(),
                axis_u=axis_u.copy(),
                axis_p=axis_p.copy(),
                span_length_ft=float(length),
                midpoint_xy=(a_xy + b_xy) / 2.0,
            )
            spans_by_pair[pair_key] = span
            next_span_no += 1

    spans = sorted(spans_by_pair.values(), key=lambda item: item.span_id)
    log(
        f"\nCentreline topology:"
        f"\n  source entities: {len(entities):,}"
        f"\n  vertex occurrences: {len(vertex_xy):,}"
        f"\n  unique nodes: {len(nodes):,}"
        f"\n  raw pieces: {raw_segments:,}"
        f"\n  authoritative spans: {len(spans):,}"
        f"\n  duplicate pieces collapsed: {duplicate_segments:,}"
        f"\n  ignored short/degenerate: {ignored_segments:,}"
    )
    return nodes, spans


# =============================================================================
# TOPOLOGY VALIDATION / CORRECTION
# =============================================================================

def _topo_incident_map(
    spans: list[DXFSpan],
):
    incident = {}

    for span in spans:
        incident.setdefault(
            span.node_a,
            [],
        ).append(
            span
        )

        incident.setdefault(
            span.node_b,
            [],
        ).append(
            span
        )

    return incident


def _topo_segment_projection(
    point_xy: np.ndarray,
    a_xy: np.ndarray,
    b_xy: np.ndarray,
):
    vector = (
        b_xy - a_xy
    )

    length = float(
        np.linalg.norm(
            vector
        )
    )

    if length <= 1e-9:
        return {
            "distance_ft":
                np.inf,
            "station_ft":
                0.0,
            "fraction":
                0.0,
            "projected_xy":
                a_xy.copy(),
            "length_ft":
                0.0,
        }

    unit_axis = (
        vector / length
    )

    relative = (
        point_xy - a_xy
    )

    station = float(
        np.dot(
            relative,
            unit_axis,
        )
    )

    clipped = float(
        np.clip(
            station,
            0.0,
            length,
        )
    )

    projected = (
        a_xy
        + clipped
        * unit_axis
    )

    distance = float(
        np.linalg.norm(
            point_xy
            - projected
        )
    )

    return {
        "distance_ft":
            distance,

        "station_ft":
            clipped,

        "fraction":
            (
                clipped / length
                if length > 0.0
                else 0.0
            ),

        "projected_xy":
            projected,

        "length_ft":
            length,
    }


def _topo_make_span(
    span_id: str,
    layer_name: str,
    node_a: DXFNode,
    node_b: DXFNode,
    source_entity_no: int,
    source_segment_no: int,
    duplicate_count: int,
    source_entities: str,
):
    vector = (
        node_b.xy
        - node_a.xy
    )

    length = float(
        np.linalg.norm(
            vector
        )
    )

    if length < MIN_DXF_SPAN_LENGTH_FT:
        return None

    axis_u = unit(
        vector
    )

    axis_p = np.asarray(
        [
            -axis_u[1],
            axis_u[0],
        ],
        dtype=np.float64,
    )

    geometry = LineString(
        [
            (
                float(
                    node_a.xy[0]
                ),
                float(
                    node_a.xy[1]
                ),
            ),
            (
                float(
                    node_b.xy[0]
                ),
                float(
                    node_b.xy[1]
                ),
            ),
        ]
    )

    return DXFSpan(
        span_id=span_id,
        layer_name=layer_name,
        geometry=geometry,

        node_a=node_a.node_id,
        node_b=node_b.node_id,

        source_entity_no=int(
            source_entity_no
        ),
        source_segment_no=int(
            source_segment_no
        ),

        duplicate_count=int(
            duplicate_count
        ),
        source_entities=source_entities,

        axis_origin_xy=node_a.xy.copy(),
        axis_u=axis_u.copy(),
        axis_p=axis_p.copy(),
        span_length_ft=length,
        midpoint_xy=(
            node_a.xy
            + node_b.xy
        ) / 2.0,
    )


def _topo_straight_angle_deg(
    node: DXFNode,
    opposite_a: DXFNode,
    opposite_b: DXFNode,
):
    first = (
        opposite_a.xy
        - node.xy
    )

    second = (
        opposite_b.xy
        - node.xy
    )

    first_length = float(
        np.linalg.norm(
            first
        )
    )

    second_length = float(
        np.linalg.norm(
            second
        )
    )

    if (
        first_length <= 1e-9
        or second_length <= 1e-9
    ):
        return 0.0

    cosine = float(
        np.clip(
            np.dot(
                first / first_length,
                second / second_length,
            ),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def _topo_wire_continuity(
    node: DXFNode,
    opposite_a: DXFNode,
    opposite_b: DXFNode,
    wire_xyz: np.ndarray,
    wire_tree_xy: cKDTree,
):
    indices = wire_tree_xy.query_ball_point(
        node.xy,
        r=TOPO_CONTINUITY_RADIUS_FT,
    )

    if not indices:
        return {
            "continuity_ok":
                False,
            "left_points":
                0,
            "right_points":
                0,
            "corridor_points":
                0,
        }

    points = wire_xyz[
        np.asarray(
            indices,
            dtype=np.int64,
        )
    ]

    axis_vector = (
        opposite_b.xy
        - opposite_a.xy
    )

    axis_length = float(
        np.linalg.norm(
            axis_vector
        )
    )

    if axis_length <= 1e-9:
        return {
            "continuity_ok":
                False,
            "left_points":
                0,
            "right_points":
                0,
            "corridor_points":
                0,
        }

    axis_u = (
        axis_vector
        / axis_length
    )

    axis_p = np.asarray(
        [
            -axis_u[1],
            axis_u[0],
        ],
        dtype=np.float64,
    )

    relative = (
        points[:, :2]
        - node.xy[
            None,
            :
        ]
    )

    station = (
        relative
        @ axis_u
    )

    lateral = np.abs(
        relative
        @ axis_p
    )

    corridor = (
        lateral
        <= TOPO_CONTINUITY_CORRIDOR_FT
    )

    left = (
        corridor
        & (
            station
            <= -TOPO_CONTINUITY_SIDE_INNER_FT
        )
        & (
            station
            >= -TOPO_CONTINUITY_SIDE_OUTER_FT
        )
    )

    right = (
        corridor
        & (
            station
            >= TOPO_CONTINUITY_SIDE_INNER_FT
        )
        & (
            station
            <= TOPO_CONTINUITY_SIDE_OUTER_FT
        )
    )

    left_count = int(
        np.count_nonzero(
            left
        )
    )

    right_count = int(
        np.count_nonzero(
            right
        )
    )

    return {
        "continuity_ok":
            bool(
                left_count
                >= TOPO_CONTINUITY_MIN_POINTS_EACH_SIDE
                and right_count
                >= TOPO_CONTINUITY_MIN_POINTS_EACH_SIDE
            ),

        "left_points":
            left_count,

        "right_points":
            right_count,

        "corridor_points":
            int(
                np.count_nonzero(
                    corridor
                )
            ),
    }


def _topo_renumber_spans(
    spans: list[DXFSpan],
    node_lookup: dict[str, DXFNode],
):
    rebuilt = []

    sorted_spans = sorted(
        spans,
        key=lambda span:
            (
                safe_text(
                    span.layer_name
                ).lower(),
                float(
                    min(
                        node_lookup[
                            span.node_a
                        ].xy[0],
                        node_lookup[
                            span.node_b
                        ].xy[0],
                    )
                ),
                float(
                    min(
                        node_lookup[
                            span.node_a
                        ].xy[1],
                        node_lookup[
                            span.node_b
                        ].xy[1],
                    )
                ),
                span.span_id,
            )
    )

    for number, span in enumerate(
        sorted_spans,
        start=1,
    ):
        new_span = _topo_make_span(
            span_id=(
                f"DXF_SPAN_"
                f"{number:04d}"
            ),

            layer_name=
                span.layer_name,

            node_a=
                node_lookup[
                    span.node_a
                ],

            node_b=
                node_lookup[
                    span.node_b
                ],

            source_entity_no=
                span.source_entity_no,

            source_segment_no=
                span.source_segment_no,

            duplicate_count=
                span.duplicate_count,

            source_entities=
                span.source_entities,
        )

        if new_span is not None:
            rebuilt.append(
                new_span
            )

    return rebuilt


def audit_and_correct_topology(
    nodes: list[DXFNode],
    spans: list[DXFSpan],
    structures: list[StructureQA],
    wire_xyz: np.ndarray,
):
    """
    Bidirectional class-215 / DXF topology audit.

    High-confidence corrections:
      * collapse degree-2 near-collinear DXF vertices with no class-215 support and
        clear classified-conductor continuity
      * split a DXF span at an unmatched class-215 structure when the structure
        lies clearly inside one span

    Ambiguous cases are reported but not modified.
    """
    if not ENABLE_TOPOLOGY_CORRECTION:
        return (
            nodes,
            spans,
            [],
            [],
            [],
        )

    wire_tree_xy = cKDTree(
        wire_xyz[:, :2]
    )

    node_lookup = {
        node.node_id:
            node
        for node in nodes
    }

    working_spans = list(
        spans
    )

    collapsed_node_ids = set()
    correction_records = []

    # ------------------------------------------------------------------
    # 1. Collapse high-confidence non-structure vertices.
    # ------------------------------------------------------------------

    collapse_pass = 0

    while (
        collapse_pass
        < TOPO_MAX_COLLAPSE_PASSES
    ):
        collapse_pass += 1

        incident = _topo_incident_map(
            working_spans
        )

        selected = None

        for node_id in sorted(
            incident
        ):
            if (
                node_id
                not in node_lookup
            ):
                continue

            node = node_lookup[
                node_id
            ]

            attached = incident[
                node_id
            ]

            if len(
                attached
            ) != 2:
                continue

            first_span, second_span = attached

            if (
                safe_text(
                    first_span.layer_name
                ).lower()
                != safe_text(
                    second_span.layer_name
                ).lower()
            ):
                continue

            nearest_distance = (
                float(
                    node.nearest_structure_distance_ft
                )
                if np.isfinite(
                    node.nearest_structure_distance_ft
                )
                else np.inf
            )

            if (
                nearest_distance
                < TOPO_EXTRA_VERTEX_MIN_STRUCTURE_DISTANCE_FT
            ):
                continue

            opposite_a_id = (
                first_span.node_b
                if first_span.node_a
                == node_id
                else first_span.node_a
            )

            opposite_b_id = (
                second_span.node_b
                if second_span.node_a
                == node_id
                else second_span.node_a
            )

            if (
                opposite_a_id
                == opposite_b_id
                or opposite_a_id
                not in node_lookup
                or opposite_b_id
                not in node_lookup
            ):
                continue

            opposite_a = node_lookup[
                opposite_a_id
            ]

            opposite_b = node_lookup[
                opposite_b_id
            ]

            angle = _topo_straight_angle_deg(
                node,
                opposite_a,
                opposite_b,
            )

            if (
                angle
                < TOPO_EXTRA_VERTEX_MIN_STRAIGHT_ANGLE_DEG
            ):
                continue

            continuity = _topo_wire_continuity(
                node,
                opposite_a,
                opposite_b,
                wire_xyz,
                wire_tree_xy,
            )

            if not continuity[
                "continuity_ok"
            ]:
                continue

            selected = {
                "node":
                    node,

                "first_span":
                    first_span,

                "second_span":
                    second_span,

                "opposite_a":
                    opposite_a,

                "opposite_b":
                    opposite_b,

                "angle":
                    angle,

                "continuity":
                    continuity,
            }

            break

        if selected is None:
            break

        node = selected[
            "node"
        ]

        first_span = selected[
            "first_span"
        ]

        second_span = selected[
            "second_span"
        ]

        opposite_a = selected[
            "opposite_a"
        ]

        opposite_b = selected[
            "opposite_b"
        ]

        merged = _topo_make_span(
            span_id=(
                f"TMP_MERGE_"
                f"{collapse_pass:04d}"
            ),

            layer_name=
                first_span.layer_name,

            node_a=
                opposite_a,

            node_b=
                opposite_b,

            source_entity_no=
                min(
                    first_span.source_entity_no,
                    second_span.source_entity_no,
                ),

            source_segment_no=0,

            duplicate_count=
                (
                    first_span.duplicate_count
                    + second_span.duplicate_count
                ),

            source_entities=
                (
                    first_span.source_entities
                    + ";"
                    + second_span.source_entities
                    + f";COLLAPSED@{node.node_id}"
                ),
        )

        if merged is None:
            break

        working_spans = [
            span
            for span in working_spans
            if (
                span is not first_span
                and span is not second_span
            )
        ]

        working_spans.append(
            merged
        )

        collapsed_node_ids.add(
            node.node_id
        )

        correction_records.append(
            {
                "action":
                    "COLLAPSE_NON_STRUCTURE_VERTEX",

                "node_id":
                    node.node_id,

                "structure_id":
                    "",

                "old_span_a":
                    first_span.span_id,

                "old_span_b":
                    second_span.span_id,

                "new_context":
                    merged.source_entities,

                "nearest_structure_distance_ft":
                    node.nearest_structure_distance_ft,

                "straight_angle_deg":
                    selected[
                        "angle"
                    ],

                "wire_left_points":
                    selected[
                        "continuity"
                    ][
                        "left_points"
                    ],

                "wire_right_points":
                    selected[
                        "continuity"
                    ][
                        "right_points"
                    ],

                "geometry":
                    Point(
                        float(
                            node.xy[0]
                        ),
                        float(
                            node.xy[1]
                        ),
                    ),
            }
        )

        node_lookup.pop(
            node.node_id,
            None,
        )

    # ------------------------------------------------------------------
    # 2. Find unmatched class-215 structures inside current spans.
    # ------------------------------------------------------------------

    active_nodes = list(
        node_lookup.values()
    )

    if active_nodes:
        active_node_xy = np.vstack(
            [
                node.xy
                for node in active_nodes
            ]
        )

        active_node_tree = cKDTree(
            active_node_xy
        )
    else:
        active_node_tree = None

    structure_results = []
    accepted_splits = []

    for structure in structures:
        structure_xy = np.asarray(
            structure.anchor_xyz[:2],
            dtype=np.float64,
        )

        if active_node_tree is not None:
            node_distance, node_index = active_node_tree.query(
                structure_xy,
                k=1,
            )

            node_distance = float(
                node_distance
            )

            nearest_node = active_nodes[
                int(
                    node_index
                )
            ]
        else:
            node_distance = np.inf
            nearest_node = None

        if (
            nearest_node is not None
            and node_distance
            <= TOPO_STRUCTURE_NODE_MATCH_FT
        ):
            structure_results.append(
                {
                    "structure_id":
                        structure.structure_id,

                    "status":
                        "MATCHED_DXF_NODE",

                    "nearest_node_id":
                        nearest_node.node_id,

                    "nearest_node_distance_ft":
                        node_distance,

                    "nearest_span_id":
                        "",

                    "nearest_span_distance_ft":
                        np.nan,

                    "station_ft":
                        np.nan,

                    "second_best_margin_ft":
                        np.nan,

                    "structure_points":
                        len(
                            structure.xyz
                        ),

                    "geometry":
                        Point(
                            float(
                                structure.anchor_xyz[0]
                            ),
                            float(
                                structure.anchor_xyz[1]
                            ),
                        ),
                }
            )

            continue

        candidates = []

        for span in working_spans:
            if (
                span.node_a
                not in node_lookup
                or span.node_b
                not in node_lookup
            ):
                continue

            projection = _topo_segment_projection(
                structure_xy,
                node_lookup[
                    span.node_a
                ].xy,
                node_lookup[
                    span.node_b
                ].xy,
            )

            length = projection[
                "length_ft"
            ]

            interior = bool(
                projection[
                    "station_ft"
                ]
                >= TOPO_MISSING_STRUCTURE_MIN_END_CLEARANCE_FT
                and projection[
                    "station_ft"
                ]
                <= (
                    length
                    - TOPO_MISSING_STRUCTURE_MIN_END_CLEARANCE_FT
                )
            )

            candidates.append(
                {
                    "span":
                        span,

                    "projection":
                        projection,

                    "interior":
                        interior,
                }
            )

        candidates.sort(
            key=lambda row:
                row[
                    "projection"
                ][
                    "distance_ft"
                ]
        )

        best = (
            candidates[0]
            if candidates
            else None
        )

        second = (
            candidates[1]
            if len(
                candidates
            ) >= 2
            else None
        )

        if best is None:
            status = (
                "ORPHAN_215_NO_SPAN"
            )

            structure_results.append(
                {
                    "structure_id":
                        structure.structure_id,

                    "status":
                        status,

                    "nearest_node_id":
                        (
                            nearest_node.node_id
                            if nearest_node is not None
                            else ""
                        ),

                    "nearest_node_distance_ft":
                        node_distance,

                    "nearest_span_id":
                        "",

                    "nearest_span_distance_ft":
                        np.nan,

                    "station_ft":
                        np.nan,

                    "second_best_margin_ft":
                        np.nan,

                    "structure_points":
                        len(
                            structure.xyz
                        ),

                    "geometry":
                        Point(
                            float(
                                structure.anchor_xyz[0]
                            ),
                            float(
                                structure.anchor_xyz[1]
                            ),
                        ),
                }
            )

            continue

        best_distance = float(
            best[
                "projection"
            ][
                "distance_ft"
            ]
        )

        second_distance = (
            float(
                second[
                    "projection"
                ][
                    "distance_ft"
                ]
            )
            if second is not None
            else np.inf
        )

        second_margin = (
            second_distance
            - best_distance
        )

        enough_points = bool(
            len(
                structure.xyz
            )
            >= TOPO_MISSING_STRUCTURE_MIN_POINTS
        )

        unambiguous = bool(
            second is None
            or second_margin
            >= TOPO_MISSING_STRUCTURE_MIN_SECOND_BEST_MARGIN_FT
            or second_distance
            > (
                TOPO_MISSING_STRUCTURE_MAX_SPAN_DISTANCE_FT
                + TOPO_MISSING_STRUCTURE_MIN_SECOND_BEST_MARGIN_FT
            )
        )

        auto_split = bool(
            best_distance
            <= TOPO_MISSING_STRUCTURE_MAX_SPAN_DISTANCE_FT

            and best[
                "interior"
            ]

            and enough_points
            and unambiguous
        )

        if auto_split:
            status = (
                "MISSING_DXF_NODE_AUTO_SPLIT"
            )

            accepted_splits.append(
                {
                    "structure":
                        structure,

                    "span":
                        best[
                            "span"
                        ],

                    "projection":
                        best[
                            "projection"
                        ],
                }
            )

        elif (
            best_distance
            <= TOPO_MISSING_STRUCTURE_MAX_SPAN_DISTANCE_FT
        ):
            status = (
                "MISSING_DXF_NODE_REVIEW"
            )

        else:
            status = (
                "ORPHAN_215_NO_SPAN"
            )

        structure_results.append(
            {
                "structure_id":
                    structure.structure_id,

                "status":
                    status,

                "nearest_node_id":
                    (
                        nearest_node.node_id
                        if nearest_node is not None
                        else ""
                    ),

                "nearest_node_distance_ft":
                    node_distance,

                "nearest_span_id":
                    best[
                        "span"
                    ].span_id,

                "nearest_span_distance_ft":
                    best_distance,

                "station_ft":
                    best[
                        "projection"
                    ][
                        "station_ft"
                    ],

                "second_best_margin_ft":
                    second_margin,

                "structure_points":
                    len(
                        structure.xyz
                    ),

                "geometry":
                    Point(
                        float(
                            structure.anchor_xyz[0]
                        ),
                        float(
                            structure.anchor_xyz[1]
                        ),
                    ),
            }
        )

    # ------------------------------------------------------------------
    # 3. Split spans at accepted missing structures.
    # ------------------------------------------------------------------

    split_by_span = {}

    for item in accepted_splits:
        split_by_span.setdefault(
            id(
                item[
                    "span"
                ]
            ),
            [],
        ).append(
            item
        )

    rebuilt_spans = []
    synthetic_counter = 1

    for span in working_spans:
        splits = split_by_span.get(
            id(
                span
            ),
            [],
        )

        if not splits:
            rebuilt_spans.append(
                span
            )
            continue

        splits.sort(
            key=lambda item:
                item[
                    "projection"
                ][
                    "station_ft"
                ]
        )

        chain_node_ids = [
            span.node_a
        ]

        for item in splits:
            structure = item[
                "structure"
            ]

            projected_xy = np.asarray(
                item[
                    "projection"
                ][
                    "projected_xy"
                ],
                dtype=np.float64,
            )

            node_id = (
                f"LIDAR_N"
                f"{synthetic_counter:04d}"
            )

            synthetic_counter += 1

            node = DXFNode(
                node_id=node_id,
                xy=projected_xy,

                source_vertex_count=0,

                nearest_structure_id=
                    structure.structure_id,

                nearest_structure_distance_ft=
                    float(
                        item[
                            "projection"
                        ][
                            "distance_ft"
                        ]
                    ),

                nearest_structure_quality=
                    node_structure_quality(
                        float(
                            item[
                                "projection"
                            ][
                                "distance_ft"
                            ]
                        )
                    ),
            )

            node_lookup[
                node_id
            ] = node

            chain_node_ids.append(
                node_id
            )

            correction_records.append(
                {
                    "action":
                        "INSERT_MISSING_STRUCTURE_NODE",

                    "node_id":
                        node_id,

                    "structure_id":
                        structure.structure_id,

                    "old_span_a":
                        span.span_id,

                    "old_span_b":
                        "",

                    "new_context":
                        span.source_entities,

                    "nearest_structure_distance_ft":
                        item[
                            "projection"
                        ][
                            "distance_ft"
                        ],

                    "straight_angle_deg":
                        np.nan,

                    "wire_left_points":
                        np.nan,

                    "wire_right_points":
                        np.nan,

                    "geometry":
                        Point(
                            float(
                                projected_xy[0]
                            ),
                            float(
                                projected_xy[1]
                            ),
                        ),
                }
            )

        chain_node_ids.append(
            span.node_b
        )

        for segment_no in range(
            1,
            len(
                chain_node_ids
            ),
        ):
            node_a = node_lookup[
                chain_node_ids[
                    segment_no - 1
                ]
            ]

            node_b = node_lookup[
                chain_node_ids[
                    segment_no
                ]
            ]

            new_span = _topo_make_span(
                span_id=(
                    f"TMP_SPLIT_"
                    f"{span.span_id}_"
                    f"{segment_no:02d}"
                ),

                layer_name=
                    span.layer_name,

                node_a=
                    node_a,

                node_b=
                    node_b,

                source_entity_no=
                    span.source_entity_no,

                source_segment_no=
                    span.source_segment_no,

                duplicate_count=
                    span.duplicate_count,

                source_entities=
                    (
                        span.source_entities
                        + ";TOPO_SPLIT"
                    ),
            )

            if new_span is not None:
                rebuilt_spans.append(
                    new_span
                )

    # ------------------------------------------------------------------
    # 4. Final node audit and deterministic span IDs.
    # ------------------------------------------------------------------

    corrected_nodes = sorted(
        node_lookup.values(),
        key=lambda node:
            node.node_id,
    )

    corrected_spans = _topo_renumber_spans(
        rebuilt_spans,
        node_lookup,
    )

    corrected_incident = _topo_incident_map(
        corrected_spans
    )

    node_audit_records = []

    original_by_id = {
        node.node_id:
            node
        for node in nodes
    }

    all_node_ids = sorted(
        set(
            original_by_id
        )
        | set(
            node_lookup
        )
        | collapsed_node_ids
    )

    for node_id in all_node_ids:
        if node_id in collapsed_node_ids:
            original = original_by_id[
                node_id
            ]

            node_audit_records.append(
                {
                    "node_id":
                        node_id,

                    "status":
                        "COLLAPSED_NON_STRUCTURE_VERTEX",

                    "degree":
                        2,

                    "nearest_structure_id":
                        original.nearest_structure_id,

                    "nearest_structure_distance_ft":
                        original.nearest_structure_distance_ft,

                    "source_vertex_count":
                        original.source_vertex_count,

                    "geometry":
                        Point(
                            float(
                                original.xy[0]
                            ),
                            float(
                                original.xy[1]
                            ),
                        ),
                }
            )

            continue

        node = node_lookup.get(
            node_id
        )

        if node is None:
            continue

        nearest_distance = (
            float(
                node.nearest_structure_distance_ft
            )
            if np.isfinite(
                node.nearest_structure_distance_ft
            )
            else np.inf
        )

        if (
            node_id.startswith(
                "LIDAR_N"
            )
        ):
            status = (
                "INSERTED_FROM_215"
            )

        elif (
            nearest_distance
            <= TOPO_STRUCTURE_NODE_MATCH_FT
        ):
            status = (
                "CONFIRMED_215"
            )

        elif (
            nearest_distance
            <= NODE_STRUCTURE_REVIEW_DISTANCE_FT
        ):
            status = (
                "OFFSET_215_REVIEW"
            )

        else:
            status = (
                "DXF_VERTEX_NO_STRUCTURE"
            )

        node_audit_records.append(
            {
                "node_id":
                    node_id,

                "status":
                    status,

                "degree":
                    len(
                        corrected_incident.get(
                            node_id,
                            [],
                        )
                    ),

                "nearest_structure_id":
                    node.nearest_structure_id,

                "nearest_structure_distance_ft":
                    node.nearest_structure_distance_ft,

                "source_vertex_count":
                    node.source_vertex_count,

                "geometry":
                    Point(
                        float(
                            node.xy[0]
                        ),
                        float(
                            node.xy[1]
                        ),
                    ),
            }
        )

    log(
        "\nTopology validation:"
        f"\n  original nodes: {len(nodes):,}"
        f"\n  original spans: {len(spans):,}"
        f"\n  collapsed non-structure vertices: "
        f"{len(collapsed_node_ids):,}"
        f"\n  inserted class-215 nodes: "
        f"{len(accepted_splits):,}"
        f"\n  corrected nodes: {len(corrected_nodes):,}"
        f"\n  corrected spans: {len(corrected_spans):,}"
    )

    status_counts = pd.Series(
        [
            row[
                "status"
            ]
            for row in structure_results
        ]
    ).value_counts()

    for status, count in status_counts.items():
        log(
            f"  class215 {status}: "
            f"{int(count):,}"
        )

    return (
        corrected_nodes,
        corrected_spans,
        node_audit_records,
        structure_results,
        correction_records,
    )


# =============================================================================
# SPAN LOCAL COORDINATES
# =============================================================================

def span_candidate_indices(
    span: DXFSpan,
    wire_xyz: np.ndarray,
    wire_tree_xy: cKDTree,
    half_width_ft: float | None = None,
) -> np.ndarray:
    query_half_width_ft = (
        float(half_width_ft)
        if half_width_ft is not None
        else float(SPAN_QUERY_HALF_WIDTH_FT)
    )

    radius = (
        math.hypot(
            span.span_length_ft / 2.0
            + SPAN_QUERY_END_EXTENSION_FT,
            query_half_width_ft,
        )
        + 5.0
    )

    candidates = wire_tree_xy.query_ball_point(
        span.midpoint_xy,
        r=radius,
    )

    if not candidates:
        return np.empty(
            0,
            dtype=np.int64,
        )

    candidates = np.asarray(
        candidates,
        dtype=np.int64,
    )

    xyz = wire_xyz[candidates]

    relative = (
        xyz[:, :2]
        - span.axis_origin_xy[None, :]
    )

    s = relative @ span.axis_u
    t = relative @ span.axis_p

    mask = (
        (s >= -SPAN_QUERY_END_EXTENSION_FT)
        & (
            s
            <= span.span_length_ft
            + SPAN_QUERY_END_EXTENSION_FT
        )
        & (
            np.abs(t)
            <= query_half_width_ft
        )
    )

    return candidates[mask]


def local_frame(
    span: DXFSpan,
    xyz: np.ndarray,
) -> pd.DataFrame:
    relative = (
        xyz[:, :2]
        - span.axis_origin_xy[None, :]
    )

    s = relative @ span.axis_u
    t = relative @ span.axis_p

    return pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "s": s,
            "t": t,
        }
    )


def local_to_xyz(
    span: DXFSpan,
    s: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    xy = (
        span.axis_origin_xy[None, :]
        + s[:, None] * span.axis_u[None, :]
        + t[:, None] * span.axis_p[None, :]
    )

    return np.column_stack(
        (
            xy,
            z,
        )
    )


# =============================================================================
# GENERIC (t,z) CLUSTERING
# =============================================================================

def cluster_tz(
    frame: pd.DataFrame,
    *,
    t_scale_ft: float,
    z_scale_ft: float,
    eps: float,
    min_samples: int,
):
    if len(frame) < min_samples:
        return []

    t = pd.to_numeric(
        frame["t"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    z = pd.to_numeric(
        frame["z"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    finite = (
        np.isfinite(t)
        & np.isfinite(z)
    )

    if np.count_nonzero(finite) < min_samples:
        return []

    local = frame.loc[finite].copy()

    features = np.column_stack(
        (
            t[finite] / t_scale_ft,
            z[finite] / z_scale_ft,
        )
    )

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=1,
    ).fit_predict(features)

    output = []

    for label in sorted(
        set(int(value) for value in labels)
    ):
        if label < 0:
            continue

        indices = np.flatnonzero(
            labels == label
        )

        if len(indices) < min_samples:
            continue

        output.append(
            local.iloc[indices].copy()
        )

    return output


# =============================================================================
# SPAN MODE
# =============================================================================

def generic_terminal_centres(
    frame: pd.DataFrame,
    span_length: float,
    side: str,
):
    window = min(
        MODE_TERMINAL_WINDOW_FT,
        max(
            6.0,
            0.20 * span_length,
        ),
    )

    if side == "A":
        local = frame[
            (frame["s"] >= 0.0)
            & (frame["s"] <= window)
        ].copy()
    else:
        local = frame[
            (frame["s"] >= span_length - window)
            & (frame["s"] <= span_length)
        ].copy()

    clusters = cluster_tz(
        local,
        t_scale_ft=MODE_CLUSTER_T_SCALE_FT,
        z_scale_ft=MODE_CLUSTER_Z_SCALE_FT,
        eps=MODE_DBSCAN_EPS,
        min_samples=MODE_DBSCAN_MIN_SAMPLES,
    )

    centres = []

    for cluster in clusters:
        centres.append(
            np.asarray(
                [
                    float(np.median(cluster["t"])),
                    float(np.median(cluster["z"])),
                ],
                dtype=np.float64,
            )
        )

    return centres


def median_terminal_spacing(
    centres: list[np.ndarray],
) -> float:
    if len(centres) < 2:
        return float("nan")

    xy = np.vstack(centres)
    tree = cKDTree(xy)

    distances, _ = tree.query(
        xy,
        k=2,
    )

    nearest = distances[:, 1]

    return float(
        np.median(nearest)
    )


def classify_span_mode(
    span: DXFSpan,
    frame: pd.DataFrame,
):
    a_centres = generic_terminal_centres(
        frame,
        span.span_length_ft,
        "A",
    )

    b_centres = generic_terminal_centres(
        frame,
        span.span_length_ft,
        "B",
    )

    spacings = [
        value
        for value in (
            median_terminal_spacing(a_centres),
            median_terminal_spacing(b_centres),
        )
        if np.isfinite(value)
    ]

    spacing = (
        float(np.median(spacings))
        if spacings
        else np.nan
    )

    length = span.span_length_ft

    if (
        length <= DX_STRONG_MAX_LENGTH_FT
        or (
            np.isfinite(spacing)
            and spacing
            <= DX_STRONG_MAX_TERMINAL_SPACING_FT
        )
    ):
        mode = "DX"

    elif (
        length >= TX_STRONG_MIN_LENGTH_FT
        and (
            not np.isfinite(spacing)
            or spacing
            >= TX_STRONG_MIN_TERMINAL_SPACING_FT
        )
    ):
        mode = "TX"

    else:
        mode = "HYBRID"

    return (
        mode,
        spacing,
        len(a_centres),
        len(b_centres),
    )


# =============================================================================
# BUNDLE CROSS-SECTION COLLAPSE
# =============================================================================

def _bundle_pairwise_diameter_ft(
    rows: list[dict],
) -> float:
    if len(rows) < 2:
        return 0.0

    values = np.asarray(
        [
            [
                float(row["t"]),
                float(row["z"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    maximum = 0.0

    for first in range(len(values) - 1):
        distances = np.linalg.norm(
            values[first + 1:] - values[first],
            axis=1,
        )

        if len(distances):
            maximum = max(
                maximum,
                float(np.max(distances)),
            )

    return float(maximum)


def collapse_bundle_cross_section_rows(
    rows: list[dict],
) -> list[dict]:
    """
    Collapse nearby same-section / same-terminal cross-section clusters into
    one weighted bundle centre.

    The function is deliberately called only on rows that already belong to
    the same terminal window, body slice or tracker section. It therefore
    cannot bridge unrelated positions along the span.
    """
    if (
        not ENABLE_BUNDLE_COLLAPSE
        or len(rows) < 2
    ):
        return rows

    features = np.asarray(
        [
            [
                float(row["t"]),
                float(row["z"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    labels = DBSCAN(
        eps=BUNDLE_COLLAPSE_DISTANCE_FT,
        min_samples=1,
        n_jobs=1,
    ).fit_predict(features)

    collapsed = []

    for label in sorted(
        set(int(value) for value in labels)
    ):
        group_indices = np.flatnonzero(
            labels == label
        )

        group = [
            rows[int(index)]
            for index in group_indices
        ]

        if len(group) == 1:
            row = dict(group[0])
            row["bundle_member_count"] = int(
                row.get(
                    "bundle_member_count",
                    1,
                )
            )
            row["bundle_collapsed"] = False
            collapsed.append(row)
            continue

        diameter = _bundle_pairwise_diameter_ft(
            group
        )

        # Reject a transitive chain wider than a plausible physical bundle.
        if diameter > BUNDLE_MAX_DIAMETER_FT:
            for item in group:
                row = dict(item)
                row["bundle_member_count"] = int(
                    row.get(
                        "bundle_member_count",
                        1,
                    )
                )
                row["bundle_collapsed"] = False
                collapsed.append(row)
            continue

        weights = np.asarray(
            [
                max(
                    float(
                        row.get(
                            "point_count",
                            1,
                        )
                    ),
                    1.0,
                )
                for row in group
            ],
            dtype=np.float64,
        )

        total_weight = float(
            np.sum(weights)
        )

        t_values = np.asarray(
            [float(row["t"]) for row in group],
            dtype=np.float64,
        )
        z_values = np.asarray(
            [float(row["z"]) for row in group],
            dtype=np.float64,
        )

        centre_t = float(
            np.average(
                t_values,
                weights=weights,
            )
        )
        centre_z = float(
            np.average(
                z_values,
                weights=weights,
            )
        )

        row = dict(group[0])
        row["t"] = centre_t
        row["z"] = centre_z
        row["point_count"] = int(
            sum(
                int(
                    item.get(
                        "point_count",
                        0,
                    )
                )
                for item in group
            )
        )
        row["bundle_member_count"] = int(
            sum(
                int(
                    item.get(
                        "bundle_member_count",
                        1,
                    )
                )
                for item in group
            )
        )
        row["bundle_collapsed"] = True
        row["bundle_diameter_ft"] = float(
            diameter
        )

        if all("s" in item for item in group):
            row["s"] = float(
                np.average(
                    np.asarray(
                        [
                            float(item["s"])
                            for item in group
                        ],
                        dtype=np.float64,
                    ),
                    weights=weights,
                )
            )

        if all("s_min" in item for item in group):
            row["s_min"] = float(
                min(
                    float(item["s_min"])
                    for item in group
                )
            )

        if all("s_max" in item for item in group):
            row["s_max"] = float(
                max(
                    float(item["s_max"])
                    for item in group
                )
            )

        # Preserve bundle spread in the RMSE fields. This allows the adaptive
        # bundle logic to recognise a broad physical cross-section instead of
        # mistaking the collapsed centroid for a thin single wire.
        if all("t_rmse" in item for item in group):
            component = np.asarray(
                [
                    float(item.get("t_rmse", 0.0))
                    for item in group
                ],
                dtype=np.float64,
            )

            row["t_rmse"] = float(
                math.sqrt(
                    np.sum(
                        weights
                        * (
                            component ** 2
                            + (t_values - centre_t) ** 2
                        )
                    )
                    / max(total_weight, 1e-9)
                )
            )

        if all("z_rmse" in item for item in group):
            component = np.asarray(
                [
                    float(item.get("z_rmse", 0.0))
                    for item in group
                ],
                dtype=np.float64,
            )

            row["z_rmse"] = float(
                math.sqrt(
                    np.sum(
                        weights
                        * (
                            component ** 2
                            + (z_values - centre_z) ** 2
                        )
                    )
                    / max(total_weight, 1e-9)
                )
            )

        collapsed.append(row)

    return collapsed


# =============================================================================
# TERMINAL POA DETECTION
# =============================================================================

def fit_terminal_cluster(
    cluster: pd.DataFrame,
    end: str,
    span_length: float,
):
    s = cluster["s"].to_numpy(dtype=np.float64)
    t = cluster["t"].to_numpy(dtype=np.float64)
    z = cluster["z"].to_numpy(dtype=np.float64)

    target_s = (
        0.0
        if end == "A"
        else span_length
    )

    if (
        len(cluster) >= 3
        and np.ptp(s) > 1e-6
    ):
        try:
            t_coeff, _, t_rmse = robust_polyfit(
                s,
                t,
                degree=1,
            )

            z_coeff, _, z_rmse = robust_polyfit(
                s,
                z,
                degree=1,
            )

            terminal_t = float(
                np.polyval(
                    t_coeff,
                    target_s,
                )
            )

            terminal_z = float(
                np.polyval(
                    z_coeff,
                    target_s,
                )
            )

        except Exception:
            terminal_t = float(
                np.median(t)
            )
            terminal_z = float(
                np.median(z)
            )
            t_rmse = float(
                np.std(t)
            )
            z_rmse = float(
                np.std(z)
            )

    else:
        terminal_t = float(
            np.median(t)
        )
        terminal_z = float(
            np.median(z)
        )
        t_rmse = float(
            np.std(t)
        )
        z_rmse = float(
            np.std(z)
        )

    return (
        terminal_t,
        terminal_z,
        float(t_rmse),
        float(z_rmse),
        float(np.min(s)),
        float(np.max(s)),
    )


def detect_terminal_poas(
    span: DXFSpan,
    frame: pd.DataFrame,
    params: dict,
    end: str,
):
    length = span.span_length_ft

    window = min(
        params["terminal_window_ft"],
        max(
            6.0,
            0.25 * length,
        ),
    )

    if end == "A":
        local = frame[
            (frame["s"] >= 0.0)
            & (frame["s"] <= window)
        ].copy()
    else:
        local = frame[
            (frame["s"] >= length - window)
            & (frame["s"] <= length)
        ].copy()

    clusters = cluster_tz(
        local,
        t_scale_ft=params[
            "terminal_t_scale_ft"
        ],
        z_scale_ft=params[
            "terminal_z_scale_ft"
        ],
        eps=params[
            "terminal_dbscan_eps"
        ],
        min_samples=params[
            "terminal_min_points"
        ],
    )

    raw = []

    for cluster in clusters:
        (
            terminal_t,
            terminal_z,
            t_rmse,
            z_rmse,
            s_min,
            s_max,
        ) = fit_terminal_cluster(
            cluster,
            end,
            length,
        )

        raw.append(
            {
                "t": terminal_t,
                "z": terminal_z,
                "point_count": len(cluster),
                "s_min": s_min,
                "s_max": s_max,
                "t_rmse": t_rmse,
                "z_rmse": z_rmse,
            }
        )

    # Subconductors in one physical bundle are collapsed to one weighted
    # terminal centre before A/B pairing.
    raw = collapse_bundle_cross_section_rows(
        raw
    )

    # Stable deterministic ordering for IDs only.
    raw.sort(
        key=lambda row:
            (
                row["t"],
                row["z"],
            )
    )

    poas = []

    for number, row in enumerate(
        raw,
        start=1,
    ):
        poas.append(
            TerminalPOA(
                terminal_id=(
                    f"{span.span_id}_"
                    f"{params['name']}_"
                    f"{end}{number:02d}"
                ),
                end=end,

                t=float(row["t"]),
                z=float(row["z"]),

                point_count=int(
                    row["point_count"]
                ),
                s_min=float(row["s_min"]),
                s_max=float(row["s_max"]),

                t_rmse=float(
                    row["t_rmse"]
                ),
                z_rmse=float(
                    row["z_rmse"]
                ),

                bundle_member_count=int(
                    row.get(
                        "bundle_member_count",
                        1,
                    )
                ),
            )
        )

    return poas


# =============================================================================
# BODY SLICES
# =============================================================================

def detect_body_clusters(
    span: DXFSpan,
    frame: pd.DataFrame,
    params: dict,
):
    output = []

    half_width = (
        params["body_slice_width_ft"]
        / 2.0
    )

    for slice_index, fraction in enumerate(
        params["body_fractions"],
        start=1,
    ):
        centre_s = (
            fraction
            * span.span_length_ft
        )

        local = frame[
            (frame["s"] >= centre_s - half_width)
            & (frame["s"] <= centre_s + half_width)
        ].copy()

        clusters = cluster_tz(
            local,
            t_scale_ft=params[
                "body_t_scale_ft"
            ],
            z_scale_ft=params[
                "body_z_scale_ft"
            ],
            eps=params[
                "body_dbscan_eps"
            ],
            min_samples=params[
                "body_min_points"
            ],
        )

        raw = []

        for cluster in clusters:
            raw.append(
                {
                    "s":
                        float(
                            np.median(
                                cluster["s"]
                            )
                        ),

                    "t":
                        float(
                            np.median(
                                cluster["t"]
                            )
                        ),

                    "z":
                        float(
                            np.median(
                                cluster["z"]
                            )
                        ),

                    "point_count":
                        len(cluster),
                }
            )

        # Merge same-slice subconductor clusters into one physical bundle
        # observation before sag estimation.
        raw = collapse_bundle_cross_section_rows(
            raw
        )

        raw.sort(
            key=lambda row:
                (
                    row["t"],
                    row["z"],
                )
        )

        for cluster_number, row in enumerate(
            raw,
            start=1,
        ):
            output.append(
                BodyCluster(
                    cluster_id=(
                        f"{span.span_id}_"
                        f"{params['name']}_"
                        f"S{slice_index:02d}_"
                        f"C{cluster_number:02d}"
                    ),
                    slice_index=slice_index,
                    fraction=float(fraction),

                    s=float(row["s"]),
                    t=float(row["t"]),
                    z=float(row["z"]),

                    point_count=int(
                        row["point_count"]
                    ),

                    bundle_member_count=int(
                        row.get(
                            "bundle_member_count",
                            1,
                        )
                    ),
                )
            )

    return output


# =============================================================================
# PHYSICAL WIRE MODEL
# =============================================================================

def model_t(
    s: np.ndarray,
    length: float,
    t_a: float,
    t_b: float,
) -> np.ndarray:
    s = np.asarray(
        s,
        dtype=np.float64,
    )

    ratio = (
        s / length
        if length > 0
        else np.zeros_like(s)
    )

    return (
        t_a
        + ratio * (t_b - t_a)
    )


def endpoint_chord_z(
    s: np.ndarray,
    length: float,
    z_a: float,
    z_b: float,
) -> np.ndarray:
    s = np.asarray(
        s,
        dtype=np.float64,
    )

    ratio = (
        s / length
        if length > 0
        else np.zeros_like(s)
    )

    return (
        z_a
        + ratio * (z_b - z_a)
    )


def model_z(
    s: np.ndarray,
    length: float,
    z_a: float,
    z_b: float,
    k: float,
) -> np.ndarray:
    s = np.asarray(
        s,
        dtype=np.float64,
    )

    return (
        endpoint_chord_z(
            s,
            length,
            z_a,
            z_b,
        )
        + k * s * (s - length)
    )


def sag_depth_from_k(
    length: float,
    k: float,
) -> float:
    return float(
        k
        * length
        * length
        / 4.0
    )


def k_from_sag_depth(
    length: float,
    sag_depth: float,
) -> float:
    if length <= 0:
        return 0.0

    return float(
        4.0
        * sag_depth
        / (
            length
            * length
        )
    )


def model_residuals(
    frame: pd.DataFrame,
    span_length: float,
    t_a: float,
    z_a: float,
    t_b: float,
    z_b: float,
    k: float,
):
    s = frame["s"].to_numpy(
        dtype=np.float64
    )
    t = frame["t"].to_numpy(
        dtype=np.float64
    )
    z = frame["z"].to_numpy(
        dtype=np.float64
    )

    t_prediction = model_t(
        s,
        span_length,
        t_a,
        t_b,
    )

    z_prediction = model_z(
        s,
        span_length,
        z_a,
        z_b,
        k,
    )

    dt = t - t_prediction
    dz = z - z_prediction

    residual = np.sqrt(
        dt ** 2
        + dz ** 2
    )

    return (
        residual,
        dt,
        dz,
    )


def build_wire_geometry(
    span: DXFSpan,
    t_a: float,
    z_a: float,
    t_b: float,
    z_b: float,
    k: float,
    interval_ft: float,
) -> LineString:
    intervals = max(
        2,
        int(
            math.ceil(
                span.span_length_ft
                / interval_ft
            )
        ),
    )

    s = np.linspace(
        0.0,
        span.span_length_ft,
        intervals + 1,
    )

    t = model_t(
        s,
        span.span_length_ft,
        t_a,
        t_b,
    )

    z = model_z(
        s,
        span.span_length_ft,
        z_a,
        z_b,
        k,
    )

    xyz = local_to_xyz(
        span,
        s,
        t,
        z,
    )

    return LineString(
        [
            (
                float(x),
                float(y),
                float(z_value),
            )
            for x, y, z_value
            in xyz
        ]
    )


# =============================================================================
# SAG ESTIMATE FROM BODY CLUSTERS
# =============================================================================

def candidate_sag_observations(
    span: DXFSpan,
    a: TerminalPOA,
    b: TerminalPOA,
    body_clusters: list[BodyCluster],
    params: dict,
):
    observations = []

    for cluster in body_clusters:
        predicted_t = float(
            model_t(
                np.asarray([cluster.s]),
                span.span_length_ft,
                a.t,
                b.t,
            )[0]
        )

        dt = abs(
            cluster.t
            - predicted_t
        )

        if (
            dt
            > params[
                "body_lateral_gate_ft"
            ]
        ):
            continue

        basis = (
            cluster.s
            * (
                cluster.s
                - span.span_length_ft
            )
        )

        if abs(basis) <= 1e-9:
            continue

        chord = float(
            endpoint_chord_z(
                np.asarray([cluster.s]),
                span.span_length_ft,
                a.z,
                b.z,
            )[0]
        )

        k = (
            cluster.z
            - chord
        ) / basis

        sag_depth = sag_depth_from_k(
            span.span_length_ft,
            k,
        )

        # A real unsupported conductor should sag down from the endpoint chord.
        # Tiny negative values are tolerated as noise, large negative curvature
        # is rejected.
        if sag_depth < -1.5:
            continue

        if (
            sag_depth
            > params[
                "max_sag_depth_ft"
            ]
        ):
            continue

        observations.append(
            {
                "cluster":
                    cluster,

                "dt":
                    float(dt),

                "k":
                    float(
                        max(
                            k,
                            0.0,
                        )
                    ),

                "sag_depth_ft":
                    float(
                        max(
                            sag_depth,
                            0.0,
                        )
                    ),
            }
        )

    return observations


def choose_sag_group(
    observations,
    params: dict,
):
    if not observations:
        return (
            0.0,
            [],
        )

    values = np.asarray(
        [
            row["sag_depth_ft"]
            for row in observations
        ],
        dtype=np.float64,
    ).reshape(-1, 1)

    labels = DBSCAN(
        eps=params["sag_group_eps_ft"],
        min_samples=1,
        n_jobs=1,
    ).fit_predict(values)

    best = None

    for label in sorted(
        set(int(value) for value in labels)
    ):
        rows = [
            observations[index]
            for index in np.flatnonzero(
                labels == label
            )
        ]

        distinct_slices = len(
            {
                row[
                    "cluster"
                ].slice_index
                for row in rows
            }
        )

        point_support = sum(
            row[
                "cluster"
            ].point_count
            for row in rows
        )

        mean_dt = float(
            np.mean(
                [
                    row["dt"]
                    for row in rows
                ]
            )
        )

        score = (
            distinct_slices * 1000
            + point_support
            - mean_dt * 10
        )

        if (
            best is None
            or score > best[0]
        ):
            best = (
                score,
                rows,
            )

    selected = best[1]

    weights = np.asarray(
        [
            row[
                "cluster"
            ].point_count
            / (
                1.0
                + row["dt"]
            )
            for row in selected
        ],
        dtype=np.float64,
    )

    sag_depth = robust_weighted_median(
        np.asarray(
            [
                row["sag_depth_ft"]
                for row in selected
            ],
            dtype=np.float64,
        ),
        weights,
    )

    return (
        float(
            max(
                sag_depth,
                0.0,
            )
        ),
        selected,
    )


# =============================================================================
# HYPOTHESIS VALIDATION
# =============================================================================

def evaluate_hypothesis(
    span: DXFSpan,
    frame: pd.DataFrame,
    a: TerminalPOA,
    b: TerminalPOA,
    body_clusters: list[BodyCluster],
    params: dict,
    hypothesis_number: int,
    pass_name: str,
):
    sag_observations = (
        candidate_sag_observations(
            span,
            a,
            b,
            body_clusters,
            params,
        )
    )

    (
        sag_depth,
        selected_body,
    ) = choose_sag_group(
        sag_observations,
        params,
    )

    k = k_from_sag_depth(
        span.span_length_ft,
        sag_depth,
    )

    (
        terminal_spread_a_ft,
        terminal_spread_b_ft,
        terminal_spread_ft,
    ) = hypothesis_terminal_spreads(
        a,
        b,
    )

    # Important: the first validation pass remains fixed-width. This prevents
    # an initially noisy terminal from creating its own broad acceptance tube.
    initial_loose_tube_ft = float(
        params[
            "loose_tube_ft"
        ]
    )

    residual, _, _ = model_residuals(
        frame,
        span.span_length_ft,
        a.t,
        a.z,
        b.t,
        b.z,
        k,
    )

    loose = (
        residual
        <= initial_loose_tube_ft
    )

    support_loose = int(
        np.count_nonzero(loose)
    )

    if support_loose:
        coverage = coverage_from_s(
            frame.loc[
                loose,
                "s",
            ].to_numpy(dtype=np.float64),
            span.span_length_ft,
            params["coverage_bins"],
        )

        median_residual = float(
            np.median(
                residual[loose]
            )
        )

        p90_residual = float(
            np.quantile(
                residual[loose],
                0.90,
            )
        )

    else:
        coverage = 0.0
        median_residual = float("inf")
        p90_residual = float("inf")

    body_slice_hits = len(
        {
            row["cluster"].slice_index
            for row in selected_body
        }
    )

    body_slice_count = len(
        params["body_fractions"]
    )

    bundle_confirmed = bool(
        bool(
            params.get(
                "bundle_enabled",
                pass_name == "TX_MAIN",
            )
        )
        and terminal_spreads_support_bundle(
            terminal_spread_a_ft,
            terminal_spread_b_ft,
        )
        and body_slice_hits
        >= TX_BUNDLE_MIN_BODY_SLICES
        and coverage
        >= TX_BUNDLE_MIN_INITIAL_COVERAGE
    )

    adaptive_radius_ft = (
        adaptive_model_radius(
            terminal_spread_ft,
            params,
        )
        if bundle_confirmed
        else float(
            params[
                "tight_tube_ft"
            ]
        )
    )

    terminal_rmse = float(
        a.t_rmse
        + a.z_rmse
        + b.t_rmse
        + b.z_rmse
    )

    support_reward = math.log1p(
        max(
            support_loose,
            0,
        )
    )

    cost = (
        COST_COVERAGE_WEIGHT
        * (
            1.0
            - coverage
        )

        + COST_MEDIAN_RESIDUAL_WEIGHT
        * (
            median_residual
            if np.isfinite(
                median_residual
            )
            else 20.0
        )

        + COST_P90_RESIDUAL_WEIGHT
        * (
            p90_residual
            if np.isfinite(
                p90_residual
            )
            else 20.0
        )

        + COST_BODY_SLICE_WEIGHT
        * (
            body_slice_count
            - body_slice_hits
        )

        + COST_TERMINAL_RMSE_WEIGHT
        * terminal_rmse

        - COST_POINT_SUPPORT_WEIGHT
        * support_reward
    )

    valid = bool(
        body_slice_hits
        >= MIN_HYPOTHESIS_BODY_SLICES

        and coverage
        >= MIN_HYPOTHESIS_COVERAGE

        and median_residual
        <= MAX_HYPOTHESIS_MEDIAN_RESIDUAL_FT
    )

    if not valid:
        cost = INVALID_PAIR_COST

    return WireHypothesis(
        hypothesis_id=(
            f"{span.span_id}_"
            f"{params['name']}_"
            f"H{hypothesis_number:03d}"
        ),

        a_id=a.terminal_id,
        b_id=b.terminal_id,

        t_a=a.t,
        z_a=a.z,
        t_b=b.t,
        z_b=b.z,

        k=float(k),
        sag_depth_ft=float(sag_depth),

        body_slice_hits=int(
            body_slice_hits
        ),
        body_slice_count=int(
            body_slice_count
        ),

        support_loose=int(
            support_loose
        ),
        coverage=float(
            coverage
        ),
        median_residual_ft=float(
            median_residual
        ),
        p90_residual_ft=float(
            p90_residual
        ),

        terminal_rmse=float(
            terminal_rmse
        ),

        cost=float(cost),
        valid=bool(valid),

        terminal_a_points=int(
            a.point_count
        ),
        terminal_b_points=int(
            b.point_count
        ),
        terminal_spread_a_ft=float(
            terminal_spread_a_ft
        ),
        terminal_spread_b_ft=float(
            terminal_spread_b_ft
        ),
        terminal_spread_ft=float(
            terminal_spread_ft
        ),
        bundle_confirmed=bool(
            bundle_confirmed
        ),
        adaptive_radius_ft=float(
            adaptive_radius_ft
        ),
    )


# =============================================================================
# GLOBAL A/B ASSIGNMENT
# =============================================================================

def hypothesis_curve_local(
    hypothesis: WireHypothesis,
    length: float,
    samples: int = 31,
):
    s = np.linspace(
        0.0,
        length,
        samples,
    )

    t = model_t(
        s,
        length,
        hypothesis.t_a,
        hypothesis.t_b,
    )

    z = model_z(
        s,
        length,
        hypothesis.z_a,
        hypothesis.z_b,
        hypothesis.k,
    )

    return (
        s,
        t,
        z,
    )


def selected_collision_penalty(
    hypotheses: list[WireHypothesis],
    span_length: float,
) -> float:
    if len(hypotheses) < 2:
        return 0.0

    penalty = 0.0

    curves = [
        hypothesis_curve_local(
            hypothesis,
            span_length,
        )
        for hypothesis in hypotheses
    ]

    for first, second in itertools.combinations(
        range(len(curves)),
        2,
    ):
        _, t1, z1 = curves[first]
        _, t2, z2 = curves[second]

        separation = np.sqrt(
            (t1 - t2) ** 2
            + (z1 - z2) ** 2
        )

        minimum = float(
            np.min(separation)
        )

        if minimum < MIN_WIRE_SEPARATION_FT:
            penalty += (
                WIRE_COLLISION_PENALTY
                * (
                    1.0
                    + MIN_WIRE_SEPARATION_FT
                    - minimum
                )
            )

    return penalty


def solve_assignment(
    a_poas: list[TerminalPOA],
    b_poas: list[TerminalPOA],
    hypotheses: list[WireHypothesis],
    span_length: float,
):
    if (
        not a_poas
        or not b_poas
        or not hypotheses
    ):
        return []

    by_pair = {
        (
            hypothesis.a_id,
            hypothesis.b_id,
        ):
            hypothesis
        for hypothesis in hypotheses
    }

    n_a = len(a_poas)
    n_b = len(b_poas)

    # Exact global search for common small phase counts.
    if (
        n_a == n_b
        and n_a <= MAX_EXACT_ASSIGNMENT_COUNT
    ):
        best = None

        for permutation in itertools.permutations(
            range(n_b)
        ):
            selected = []
            invalid = False

            for a_index, b_index in enumerate(
                permutation
            ):
                hypothesis = by_pair.get(
                    (
                        a_poas[a_index].terminal_id,
                        b_poas[b_index].terminal_id,
                    )
                )

                if (
                    hypothesis is None
                    or not hypothesis.valid
                ):
                    invalid = True
                    break

                selected.append(hypothesis)

            if invalid:
                continue

            cost = sum(
                item.cost
                for item in selected
            )

            cost += selected_collision_penalty(
                selected,
                span_length,
            )

            if (
                best is None
                or cost < best[0]
            ):
                best = (
                    cost,
                    selected,
                )

        if best is not None:
            return best[1]

    # General rectangular assignment.
    cost_matrix = np.full(
        (
            n_a,
            n_b,
        ),
        INVALID_PAIR_COST,
        dtype=np.float64,
    )

    for a_index, a in enumerate(a_poas):
        for b_index, b in enumerate(b_poas):
            hypothesis = by_pair.get(
                (
                    a.terminal_id,
                    b.terminal_id,
                )
            )

            if (
                hypothesis is not None
                and hypothesis.valid
            ):
                cost_matrix[
                    a_index,
                    b_index,
                ] = hypothesis.cost

    rows, columns = linear_sum_assignment(
        cost_matrix
    )

    selected = []

    for row, column in zip(
        rows,
        columns,
    ):
        if (
            cost_matrix[
                row,
                column,
            ]
            >= INVALID_PAIR_COST
        ):
            continue

        hypothesis = by_pair[
            (
                a_poas[row].terminal_id,
                b_poas[column].terminal_id,
            )
        ]

        selected.append(hypothesis)

    return selected


# =============================================================================
# POINT OWNERSHIP + REFIT
# =============================================================================

def robust_refit_k(
    s: np.ndarray,
    z: np.ndarray,
    span_length: float,
    z_a: float,
    z_b: float,
    initial_k: float,
):
    s = np.asarray(
        s,
        dtype=np.float64,
    )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    base = endpoint_chord_z(
        s,
        span_length,
        z_a,
        z_b,
    )

    basis = (
        s
        * (
            s - span_length
        )
    )

    valid = (
        np.isfinite(s)
        & np.isfinite(z)
        & np.isfinite(basis)
        & (
            np.abs(basis)
            > 1e-8
        )
    )

    if np.count_nonzero(valid) < 3:
        return float(
            max(
                initial_k,
                0.0,
            )
        )

    s = s[valid]
    z = z[valid]
    base = base[valid]
    basis = basis[valid]

    mask = np.ones(
        len(s),
        dtype=bool,
    )

    k = float(
        max(
            initial_k,
            0.0,
        )
    )

    for _ in range(6):
        denominator = float(
            np.sum(
                basis[mask] ** 2
            )
        )

        if denominator <= 1e-12:
            break

        k = float(
            np.sum(
                basis[mask]
                * (
                    z[mask]
                    - base[mask]
                )
            )
            / denominator
        )

        k = max(
            k,
            0.0,
        )

        residual = (
            z
            - (
                base
                + k * basis
            )
        )

        active = residual[mask]

        if len(active) < 4:
            break

        median = float(
            np.median(active)
        )

        mad = float(
            np.median(
                np.abs(
                    active
                    - median
                )
            )
        )

        if mad <= 1e-12:
            break

        scale = 1.4826 * mad

        new_mask = (
            np.abs(
                residual
                - median
            )
            <= 3.0 * scale
        )

        if np.array_equal(
            mask,
            new_mask,
        ):
            break

        if np.count_nonzero(
            new_mask
        ) < 3:
            break

        mask = new_mask

    return float(
        max(
            k,
            0.0,
        )
    )


def assign_points_to_selected(
    span: DXFSpan,
    frame: pd.DataFrame,
    selected: list[WireHypothesis],
    params: dict,
):
    """
    Compete selected conductors using distance normalised by each conductor's
    physical/adaptive radius.

    This matters for bundles: a wide HV bundle may legitimately have a larger
    raw residual than a single shield wire, without being a worse conductor
    model.
    """
    if not selected:
        return {}

    residual_columns = []
    ownership_tubes = []

    for hypothesis in selected:
        residual, _, _ = model_residuals(
            frame,
            span.span_length_ft,
            hypothesis.t_a,
            hypothesis.z_a,
            hypothesis.t_b,
            hypothesis.z_b,
            hypothesis.k,
        )

        residual_columns.append(
            residual
        )

        if hypothesis.bundle_confirmed:
            ownership_tube = max(
                float(
                    params[
                        "ownership_tube_ft"
                    ]
                ),
                float(
                    hypothesis.adaptive_radius_ft
                )
                * float(
                    params[
                        "ownership_radius_factor"
                    ]
                ),
            )
        else:
            # V5.2-style strict ownership for ordinary TX and ALL DX.
            ownership_tube = float(
                params[
                    "ownership_tube_ft"
                ]
            )

        ownership_tubes.append(
            ownership_tube
        )

    distances = np.column_stack(
        residual_columns
    )

    tubes = np.asarray(
        ownership_tubes,
        dtype=np.float64,
    )

    normalised = (
        distances
        / tubes[
            None,
            :
        ]
    )

    best_index = np.argmin(
        normalised,
        axis=1,
    )

    best_normalised = normalised[
        np.arange(
            len(frame)
        ),
        best_index,
    ]

    if len(selected) > 1:
        partitioned = np.partition(
            normalised,
            kth=1,
            axis=1,
        )

        second_normalised = (
            partitioned[:, 1]
        )
    else:
        second_normalised = np.full(
            len(frame),
            np.inf,
            dtype=np.float64,
        )

    separation = np.full(
        len(frame),
        np.inf,
        dtype=np.float64,
    )

    finite_both = (
        np.isfinite(
            best_normalised
        )
        & np.isfinite(
            second_normalised
        )
    )

    separation[
        finite_both
    ] = (
        second_normalised[
            finite_both
        ]
        - best_normalised[
            finite_both
        ]
    )

    owned = {}

    for model_index, hypothesis in enumerate(
        selected
    ):
        mask = (
            (best_index == model_index)
            & (
                best_normalised
                <= 1.0
            )
            & (
                separation
                >= float(
                    params[
                        "ownership_margin_ratio"
                    ]
                )
            )
        )

        owned[
            hypothesis.hypothesis_id
        ] = frame.loc[
            mask
        ].copy()

    return owned


def refit_and_score_selected(
    span: DXFSpan,
    frame: pd.DataFrame,
    selected: list[WireHypothesis],
    params: dict,
):
    ownership = assign_points_to_selected(
        span,
        frame,
        selected,
        params,
    )

    for hypothesis in selected:
        owned = ownership.get(
            hypothesis.hypothesis_id,
            frame.iloc[
                0:0
            ].copy(),
        )

        hypothesis.owned_points = len(
            owned
        )

        if len(owned) >= 3:
            hypothesis.k = robust_refit_k(
                owned[
                    "s"
                ].to_numpy(
                    dtype=np.float64
                ),
                owned[
                    "z"
                ].to_numpy(
                    dtype=np.float64
                ),
                span.span_length_ft,
                hypothesis.z_a,
                hypothesis.z_b,
                hypothesis.k,
            )

        if not owned.empty:
            owned_residual, _, _ = (
                model_residuals(
                    owned,
                    span.span_length_ft,
                    hypothesis.t_a,
                    hypothesis.z_a,
                    hypothesis.t_b,
                    hypothesis.z_b,
                    hypothesis.k,
                )
            )

            owned_s = owned[
                "s"
            ].to_numpy(
                dtype=np.float64
            )

            hypothesis.owned_coverage = (
                coverage_from_s(
                    owned_s,
                    span.span_length_ft,
                    params[
                        "coverage_bins"
                    ],
                )
            )

            hypothesis.owned_median_residual_ft = float(
                np.median(
                    owned_residual
                )
            )

            hypothesis.owned_p90_residual_ft = float(
                np.quantile(
                    owned_residual,
                    0.90,
                )
            )

            (
                hypothesis.support_zone_count,
                hypothesis.longest_unsupported_gap_ft,
                hypothesis.longest_unsupported_gap_fraction,
            ) = support_distribution_metrics(
                owned_s,
                span.span_length_ft,
                params[
                    "coverage_bins"
                ],
            )

        else:
            hypothesis.owned_coverage = 0.0
            hypothesis.owned_median_residual_ft = np.inf
            hypothesis.owned_p90_residual_ft = np.inf
            hypothesis.support_zone_count = 0
            hypothesis.longest_unsupported_gap_ft = (
                span.span_length_ft
            )
            hypothesis.longest_unsupported_gap_fraction = 1.0

        hypothesis.sag_depth_ft = sag_depth_from_k(
            span.span_length_ft,
            hypothesis.k,
        )

        # A bundle with a broad terminal cross-section is allowed a wider
        # centreline residual. A thin shield wire keeps the original strict
        # thresholds.
        if hypothesis.bundle_confirmed:
            hypothesis.adaptive_median_limit_ft = max(
                float(
                    params[
                        "max_median_residual_ft"
                    ]
                ),
                float(
                    hypothesis.adaptive_radius_ft
                )
                * float(
                    params[
                        "adaptive_median_factor"
                    ]
                ),
            )

            hypothesis.adaptive_p90_limit_ft = max(
                float(
                    params[
                        "max_p90_residual_ft"
                    ]
                ),
                float(
                    hypothesis.adaptive_radius_ft
                )
                * float(
                    params[
                        "adaptive_p90_factor"
                    ]
                ),
            )

        else:
            hypothesis.adaptive_median_limit_ft = float(
                params[
                    "max_median_residual_ft"
                ]
            )

            hypothesis.adaptive_p90_limit_ft = float(
                params[
                    "max_p90_residual_ft"
                ]
            )

        terminal_support_ok = bool(
            hypothesis.terminal_a_points
            >= int(
                params[
                    "terminal_min_points"
                ]
            )
            and hypothesis.terminal_b_points
            >= int(
                params[
                    "terminal_min_points"
                ]
            )
        )

        residual_ok = bool(
            hypothesis.owned_median_residual_ft
            <= hypothesis.adaptive_median_limit_ft
            and hypothesis.owned_p90_residual_ft
            <= hypothesis.adaptive_p90_limit_ft
        )

        enough_points = bool(
            hypothesis.owned_points
            >= int(
                params[
                    "min_owned_points"
                ]
            )
        )

        dense_strong = bool(
            enough_points
            and terminal_support_ok
            and residual_ok
            and hypothesis.owned_coverage
            >= float(
                params[
                    "min_coverage"
                ]
            )
        )

        sparse_strong = bool(
            enough_points
            and terminal_support_ok
            and residual_ok
            and hypothesis.owned_coverage
            >= float(
                params[
                    "sparse_min_coverage"
                ]
            )
            and hypothesis.body_slice_hits
            >= int(
                params[
                    "sparse_min_body_slices"
                ]
            )
            and hypothesis.support_zone_count
            >= int(
                params[
                    "sparse_min_support_zones"
                ]
            )
            and hypothesis.longest_unsupported_gap_fraction
            <= float(
                params[
                    "sparse_max_gap_fraction"
                ]
            )
        )

        weak = bool(
            hypothesis.owned_points >= 3
            and hypothesis.owned_coverage >= 0.10
            and hypothesis.owned_median_residual_ft
            <= max(
                2.0,
                hypothesis.adaptive_median_limit_ft
                * 2.0,
            )
        )

        if dense_strong:
            hypothesis.quality = (
                "DENSE_STRONG"
            )

        elif sparse_strong:
            hypothesis.quality = (
                "SPARSE_STRONG"
            )

        elif weak:
            hypothesis.quality = (
                "REVIEW"
            )

        else:
            hypothesis.quality = (
                "REJECT"
            )

    return selected


def claim_points_explained_by_strong_tx(
    span: DXFSpan,
    frame: pd.DataFrame,
    tx_run: dict,
):
    """
    Remove points already explained by strong TX models before the DX
    underbuild pass.

    Claiming is intentionally based on the physical model tube rather than raw
    clustering, so a classification gap does not split one TX conductor into
    multiple residual fragments.
    """
    strong = [
        hypothesis
        for hypothesis in tx_run[
            "selected"
        ]
        if quality_is_strong(
            hypothesis.quality
        )
    ]

    if (
        not strong
        or frame.empty
    ):
        return (
            frame.iloc[
                0:0
            ].copy(),
            frame.copy(),
        )

    claimed_mask = np.zeros(
        len(frame),
        dtype=bool,
    )

    for hypothesis in strong:
        residual, _, _ = model_residuals(
            frame,
            span.span_length_ft,
            hypothesis.t_a,
            hypothesis.z_a,
            hypothesis.t_b,
            hypothesis.z_b,
            hypothesis.k,
        )

        claim_tube = max(
            TX_CLAIM_MIN_TUBE_FT,
            float(
                hypothesis.adaptive_radius_ft
            )
            * TX_CLAIM_RADIUS_FACTOR,
        )

        claimed_mask |= (
            residual
            <= claim_tube
        )

    claimed = frame.loc[
        claimed_mask
    ].copy()

    residual = frame.loc[
        ~claimed_mask
    ].copy()

    return (
        claimed,
        residual,
    )


def validate_underbuild_run(
    span: DXFSpan,
    tx_run: dict,
    dx_run: dict,
):
    """
    A residual DX model is a credible underbuild only if it remains below the
    lowest strong TX conductor across most of the span.

    After spatial validation, mutually consistent underbuild candidates can
    confirm one another as a group. A single surviving candidate is deliberately
    left for V7 to confirm during fusion.
    """
    strong_tx = [
        hypothesis
        for hypothesis in tx_run[
            "selected"
        ]
        if quality_is_strong(
            hypothesis.quality
        )
    ]

    if not strong_tx:
        for hypothesis in dx_run[
            "selected"
        ]:
            hypothesis.underbuild_below_fraction = 0.0
            hypothesis.underbuild_spatial_ok = False
            hypothesis.underbuild_group_size = 0

            if quality_is_strong(
                hypothesis.quality
            ):
                hypothesis.quality = "REVIEW"

        return dx_run

    s = np.linspace(
        0.05 * span.span_length_ft,
        0.95 * span.span_length_ft,
        UNDERBUILD_COMPARISON_SAMPLES,
    )

    tx_z = np.vstack(
        [
            model_z(
                s,
                span.span_length_ft,
                hypothesis.z_a,
                hypothesis.z_b,
                hypothesis.k,
            )
            for hypothesis in strong_tx
        ]
    )

    # Underbuild must be below the lowest phase/shield conductor.
    lowest_tx_z = np.min(
        tx_z,
        axis=0,
    )

    spatially_valid = []

    for hypothesis in dx_run[
        "selected"
    ]:
        underbuild_z = model_z(
            s,
            span.span_length_ft,
            hypothesis.z_a,
            hypothesis.z_b,
            hypothesis.k,
        )

        vertical_clearance = (
            lowest_tx_z
            - underbuild_z
        )

        below_fraction = float(
            np.mean(
                vertical_clearance
                >= UNDERBUILD_MIN_VERTICAL_CLEARANCE_FT
            )
        )

        hypothesis.underbuild_below_fraction = (
            below_fraction
        )

        hypothesis.underbuild_spatial_ok = bool(
            below_fraction
            >= UNDERBUILD_MIN_BELOW_FRACTION
        )

        if hypothesis.underbuild_spatial_ok:
            spatially_valid.append(
                hypothesis
            )
        elif quality_is_strong(
            hypothesis.quality
        ):
            # Keep it visible, but never auto-final as underbuild.
            hypothesis.quality = "REVIEW"

    group_size = len(
        spatially_valid
    )

    for hypothesis in dx_run[
        "selected"
    ]:
        hypothesis.underbuild_group_size = (
            group_size
            if hypothesis.underbuild_spatial_ok
            else 0
        )

    return dx_run


# =============================================================================
# POA ENGINE
# =============================================================================

def run_poa_engine(
    span: DXFSpan,
    frame: pd.DataFrame,
    params: dict,
    pass_name: str | None = None,
):
    a_poas = detect_terminal_poas(
        span,
        frame,
        params,
        "A",
    )

    b_poas = detect_terminal_poas(
        span,
        frame,
        params,
        "B",
    )

    body_clusters = detect_body_clusters(
        span,
        frame,
        params,
    )

    hypotheses = []
    hypothesis_number = 1

    for a in a_poas:
        for b in b_poas:
            hypothesis = evaluate_hypothesis(
                span,
                frame,
                a,
                b,
                body_clusters,
                params,
                hypothesis_number,
                pass_name=(
                    pass_name
                    or params["name"]
                ),
            )

            hypotheses.append(
                hypothesis
            )

            hypothesis_number += 1

    selected = solve_assignment(
        a_poas,
        b_poas,
        hypotheses,
        span.span_length_ft,
    )

    selected = refit_and_score_selected(
        span,
        frame,
        selected,
        params,
    )

    return {
        "mode":
            params["name"],

        "params":
            params,

        "pass_name":
            (
                pass_name
                or params["name"]
            ),

        "input_points":
            int(
                len(frame)
            ),

        "a_poas":
            a_poas,

        "b_poas":
            b_poas,

        "body_clusters":
            body_clusters,

        "hypotheses":
            hypotheses,

        "selected":
            selected,
    }


# =============================================================================
# V7 ENGINE
# =============================================================================

def _internal_tracker_sections(
    span: DXFSpan,
    frame: pd.DataFrame,
    mode: str,
):
    cfg = INTERNAL_TRACKER[mode]
    section_length = cfg["section_length_ft"]
    n_sections = max(
        1,
        int(math.ceil(span.span_length_ft / section_length)),
    )

    sections = []

    for section_index in range(n_sections):
        start_s = section_index * section_length
        end_s = min(
            span.span_length_ft,
            (section_index + 1) * section_length,
        )

        local = frame[
            (frame["s"] >= start_s)
            & (frame["s"] <= end_s)
        ].copy()

        clusters = cluster_tz(
            local,
            t_scale_ft=cfg["t_scale_ft"],
            z_scale_ft=cfg["z_scale_ft"],
            eps=cfg["dbscan_eps"],
            min_samples=cfg["dbscan_min_samples"],
        )

        observations = []

        for cluster in clusters:
            observations.append(
                {
                    "section_index": section_index,
                    "s": float(np.median(cluster["s"])),
                    "t": float(np.median(cluster["t"])),
                    "z": float(np.median(cluster["z"])),
                    "point_count": int(len(cluster)),
                }
            )

        observations.sort(
            key=lambda row: (
                row["t"],
                row["z"],
            )
        )

        sections.append(observations)

    return sections, n_sections


def _internal_tracker_predict(track: dict):
    obs = track["observations"]

    if len(obs) == 1:
        return obs[-1]["t"], obs[-1]["z"]

    recent = obs[-min(4, len(obs)):]

    s = np.asarray([item["s"] for item in recent], dtype=np.float64)
    t = np.asarray([item["t"] for item in recent], dtype=np.float64)
    z = np.asarray([item["z"] for item in recent], dtype=np.float64)

    next_s = recent[-1]["s"] + max(
        recent[-1]["s"] - recent[-2]["s"],
        1.0,
    )

    try:
        t_coeff = np.polyfit(s, t, 1)
        predicted_t = float(np.polyval(t_coeff, next_s))
    except Exception:
        predicted_t = float(recent[-1]["t"])

    try:
        z_degree = min(2, len(recent) - 1)
        z_coeff = np.polyfit(s, z, z_degree)
        predicted_z = float(np.polyval(z_coeff, next_s))
    except Exception:
        predicted_z = float(recent[-1]["z"])

    return predicted_t, predicted_z


def _internal_tracker_link(
    sections: list[list[dict]],
    mode: str,
):
    cfg = INTERNAL_TRACKER[mode]

    tracks = []
    next_track_id = 1

    for observations in sections:
        active = [
            track
            for track in tracks
            if track["missed"] <= cfg["max_missed_sections"]
        ]

        if not active:
            for observation in observations:
                tracks.append(
                    {
                        "track_id": next_track_id,
                        "observations": [observation],
                        "missed": 0,
                    }
                )
                next_track_id += 1
            continue

        if not observations:
            for track in active:
                track["missed"] += 1
            continue

        matrix = np.full(
            (len(active), len(observations)),
            1_000_000.0,
            dtype=np.float64,
        )

        for row, track in enumerate(active):
            predicted_t, predicted_z = _internal_tracker_predict(track)

            for column, observation in enumerate(observations):
                dt = abs(observation["t"] - predicted_t)
                dz = abs(observation["z"] - predicted_z)

                if (
                    dt > cfg["max_t_jump_ft"]
                    or dz > cfg["max_z_jump_ft"]
                ):
                    continue

                cost = math.sqrt(
                    (dt / cfg["max_t_jump_ft"]) ** 2
                    + (dz / cfg["max_z_jump_ft"]) ** 2
                )

                if cost <= cfg["max_cost"]:
                    matrix[row, column] = cost

        rows, columns = linear_sum_assignment(matrix)

        used_track_ids = set()
        used_observations = set()

        for row, column in zip(rows, columns):
            if matrix[row, column] >= 1_000_000.0:
                continue

            track = active[row]
            track["observations"].append(observations[column])
            track["missed"] = 0

            used_track_ids.add(track["track_id"])
            used_observations.add(column)

        for track in active:
            if track["track_id"] not in used_track_ids:
                track["missed"] += 1

        for column, observation in enumerate(observations):
            if column in used_observations:
                continue

            tracks.append(
                {
                    "track_id": next_track_id,
                    "observations": [observation],
                    "missed": 0,
                }
            )
            next_track_id += 1

    return tracks



def _internal_tracker_support_metrics(
    observations: list[dict],
    n_sections: int,
):
    """
    Section-distribution diagnostics for sparse LiDAR.

    coverage:
        fraction of all span sections containing an observation

    support_extent_fraction:
        fraction of the span covered between the first and last observed
        section. This distinguishes sparse-but-distributed evidence from a
        dense cluster limited to one end of the span.

    longest_gap_fraction:
        longest consecutive run of unobserved sections across the full span,
        expressed as a fraction of all sections.
    """
    if n_sections <= 0:
        return {
            "observed_sections":
                0,

            "coverage":
                0.0,

            "support_extent_fraction":
                0.0,

            "longest_gap_sections":
                0,

            "longest_gap_fraction":
                1.0,
        }

    observed = sorted(
        {
            int(
                item[
                    "section_index"
                ]
            )
            for item in observations
            if "section_index" in item
        }
    )

    observed = [
        index
        for index in observed
        if 0 <= index < n_sections
    ]

    observed_count = len(
        observed
    )

    coverage = (
        observed_count
        / n_sections
    )

    if not observed:
        return {
            "observed_sections":
                0,

            "coverage":
                0.0,

            "support_extent_fraction":
                0.0,

            "longest_gap_sections":
                n_sections,

            "longest_gap_fraction":
                1.0,
        }

    first_index = observed[0]
    last_index = observed[-1]

    support_extent_sections = (
        last_index
        - first_index
        + 1
    )

    support_extent_fraction = (
        support_extent_sections
        / n_sections
    )

    occupied = np.zeros(
        n_sections,
        dtype=bool,
    )

    occupied[
        np.asarray(
            observed,
            dtype=np.int64,
        )
    ] = True

    longest_gap = 0
    current_gap = 0

    for is_occupied in occupied:
        if is_occupied:
            longest_gap = max(
                longest_gap,
                current_gap,
            )
            current_gap = 0
        else:
            current_gap += 1

    longest_gap = max(
        longest_gap,
        current_gap,
    )

    return {
        "observed_sections":
            int(
                observed_count
            ),

        "coverage":
            float(
                coverage
            ),

        "support_extent_fraction":
            float(
                support_extent_fraction
            ),

        "longest_gap_sections":
            int(
                longest_gap
            ),

        "longest_gap_fraction":
            float(
                longest_gap
                / n_sections
            ),
    }


def _internal_tracker_fit(
    span: DXFSpan,
    track: dict,
    mode: str,
    n_sections: int,
):
    cfg = INTERNAL_TRACKER[mode]
    obs = track["observations"]

    if len(obs) < cfg["min_sections"]:
        return None

    s = np.asarray([item["s"] for item in obs], dtype=np.float64)
    t = np.asarray([item["t"] for item in obs], dtype=np.float64)
    z = np.asarray([item["z"] for item in obs], dtype=np.float64)

    try:
        t_coeff, _, t_rmse = robust_polyfit(s, t, 1)
        z_coeff, _, z_rmse = robust_polyfit(s, z, 2)
    except Exception:
        return None

    support = _internal_tracker_support_metrics(
        obs,
        n_sections,
    )

    coverage = support[
        "coverage"
    ]

    normal_accept = bool(
        coverage
        >= cfg[
            "accept_coverage"
        ]

        and t_rmse
        <= cfg[
            "max_t_rmse_ft"
        ]

        and z_rmse
        <= cfg[
            "max_z_rmse_ft"
        ]
    )

    adaptive_accept = False

    if mode == "DX":
        adaptive_accept = bool(
            coverage
            >= cfg[
                "adaptive_accept_coverage"
            ]

            and t_rmse
            <= cfg[
                "adaptive_max_t_rmse_ft"
            ]

            and z_rmse
            <= cfg[
                "adaptive_max_z_rmse_ft"
            ]

            and support[
                "observed_sections"
            ]
            >= cfg[
                "adaptive_min_observed_sections"
            ]

            and support[
                "support_extent_fraction"
            ]
            >= cfg[
                "adaptive_min_support_extent_fraction"
            ]

            and support[
                "longest_gap_fraction"
            ]
            <= cfg[
                "adaptive_max_longest_gap_fraction"
            ]
        )

    if normal_accept:
        status = "AUTO_ACCEPT"
        accept_reason = "NORMAL_COVERAGE"

    elif adaptive_accept:
        status = "AUTO_ACCEPT"
        accept_reason = "LOW_COVERAGE_STRONG_DISTRIBUTED_FIT"

    elif coverage >= cfg["review_coverage"]:
        status = "REVIEW"
        accept_reason = "REVIEW_COVERAGE_OR_FIT"

    else:
        return None

    sample_count = max(
        2,
        int(math.ceil(span.span_length_ft / cfg["sample_interval_ft"])),
    )

    sample_s = np.linspace(
        0.0,
        span.span_length_ft,
        sample_count + 1,
    )

    sample_t = np.polyval(t_coeff, sample_s)
    sample_z = np.polyval(z_coeff, sample_s)

    xyz = local_to_xyz(
        span,
        sample_s,
        sample_t,
        sample_z,
    )

    geometry = LineString(
        [
            (
                float(x),
                float(y),
                float(z_value),
            )
            for x, y, z_value in xyz
        ]
    )

    return {
        "track_id":
            f"{span.span_id}_ITRK_{mode}_{track['track_id']:03d}",

        "status":
            status,

        "review":
            "",

        "coverage_pct":
            float(coverage * 100.0),

        "accept_reason":
            accept_reason,

        "observed_sections":
            int(
                support[
                    "observed_sections"
                ]
            ),

        "support_extent_fraction":
            float(
                support[
                    "support_extent_fraction"
                ]
            ),

        "support_extent_pct":
            float(
                support[
                    "support_extent_fraction"
                ]
                * 100.0
            ),

        "longest_gap_sections":
            int(
                support[
                    "longest_gap_sections"
                ]
            ),

        "longest_gap_fraction":
            float(
                support[
                    "longest_gap_fraction"
                ]
            ),

        "longest_gap_pct":
            float(
                support[
                    "longest_gap_fraction"
                ]
                * 100.0
            ),

        "body_rmse_ft":
            float(z_rmse),

        "switch_risk":
            0,

        "t_rmse_ft":
            float(t_rmse),

        "geometry":
            geometry,
    }


def run_v7_engine(
    span: DXFSpan,
    frame: pd.DataFrame,
    tracker=None,
):
    """
    Independent embedded section tracker.

    It is intentionally simpler than the old external Phase-1 module, but it
    remains independent from POA pairing and therefore still provides the
    agreement signal that manual cleanup showed to be highly valuable.

    V2.4 adds an adaptive DX acceptance path for sparse LiDAR. A DX track below
    the normal 45% section coverage can still be AUTO_ACCEPT when:
        * coverage >= 30%
        * t RMSE <= 0.20 ft
        * Z RMSE <= 0.30 ft
        * >= 5 sections are observed
        * observations span >= 70% of the physical span
        * no unsupported gap exceeds 50% of the span

    TX acceptance is unchanged.
    """
    if frame.empty:
        return {
            "tracks_total": 0,
            "tracks_ok": 0,
            "wires": [],
        }

    mode = (
        "DX"
        if span.span_length_ft <= 190.0
        else "TX"
    )

    sections, n_sections = _internal_tracker_sections(
        span,
        frame,
        mode,
    )

    tracks = _internal_tracker_link(
        sections,
        mode,
    )

    wires = []

    for track in tracks:
        wire = _internal_tracker_fit(
            span,
            track,
            mode,
            n_sections,
        )

        if wire is not None:
            wires.append(wire)

    return {
        "tracks_total": len(tracks),
        "tracks_ok": sum(
            1
            for wire in wires
            if wire["status"] == "AUTO_ACCEPT"
        ),
        "wires": wires,
    }



# =============================================================================
# V3 CLASS-SPECIFIC TRACKER + FRAGMENT STITCHING
# =============================================================================

GENERIC_CLASS_TRACKER = {
    "DX": {
        "name": "AUTO_DX",
        "family": "DX",
        "section_length_ft": 4.0,
        "t_scale_ft": 0.50,
        "z_scale_ft": 0.70,
        "dbscan_eps": 1.15,
        "dbscan_min_samples": 3,
        "max_t_jump_ft": 1.30,
        "max_z_jump_ft": 2.00,
        "max_cost": 1.50,
        "max_missed_sections": 2,
        "min_sections": 3,
        "normal_accept_coverage": 0.70,
        "review_coverage": 0.15,
        "max_t_rmse_ft": 0.60,
        "max_z_rmse_ft": 0.75,
        "adaptive_accept_coverage": 0.25,
        "adaptive_max_t_rmse_ft": 0.20,
        "adaptive_max_z_rmse_ft": 0.30,
        "adaptive_min_observed_sections": 5,
        "adaptive_min_support_extent_fraction": 0.70,
        "adaptive_max_longest_gap_fraction": 0.50,
        "sample_interval_ft": 2.5,
        "stitch_max_gap_sections": 12,
        "stitch_max_gap_fraction": 0.45,
        "stitch_t_gate_ft": 0.85,
        "stitch_z_gate_ft": 1.60,
        "stitch_max_t_rmse_ft": 0.22,
        "stitch_max_z_rmse_ft": 0.38,
    },
    "TX": {
        "name": "AUTO_TX",
        "family": "TX",
        "section_length_ft": 10.0,
        "t_scale_ft": 1.00,
        "z_scale_ft": 1.30,
        "dbscan_eps": 1.20,
        "dbscan_min_samples": 3,
        "max_t_jump_ft": 2.50,
        "max_z_jump_ft": 4.00,
        "max_cost": 1.60,
        "max_missed_sections": 4,
        "min_sections": 3,
        "normal_accept_coverage": 0.85,
        "review_coverage": 0.20,
        "max_t_rmse_ft": 0.90,
        "max_z_rmse_ft": 1.15,
        "adaptive_accept_coverage": 0.50,
        "adaptive_max_t_rmse_ft": 0.45,
        "adaptive_max_z_rmse_ft": 0.70,
        "adaptive_min_observed_sections": 4,
        "adaptive_min_support_extent_fraction": 0.80,
        "adaptive_max_longest_gap_fraction": 0.35,
        "sample_interval_ft": 5.0,
        "stitch_max_gap_sections": 6,
        "stitch_max_gap_fraction": 0.30,
        "stitch_t_gate_ft": 2.00,
        "stitch_z_gate_ft": 3.50,
        "stitch_max_t_rmse_ft": 0.65,
        "stitch_max_z_rmse_ft": 0.90,
    },
}


def _class_tracker_sections(
    span: DXFSpan,
    frame: pd.DataFrame,
    cfg: dict,
):
    section_length = float(
        cfg["section_length_ft"]
    )

    n_sections = max(
        1,
        int(
            math.ceil(
                span.span_length_ft
                / section_length
            )
        ),
    )

    sections = []

    for section_index in range(
        n_sections
    ):
        start_s = (
            section_index
            * section_length
        )

        end_s = min(
            span.span_length_ft,
            (
                section_index + 1
            )
            * section_length,
        )

        local = frame[
            (frame["s"] >= start_s)
            & (frame["s"] <= end_s)
        ].copy()

        clusters = cluster_tz(
            local,
            t_scale_ft=cfg[
                "t_scale_ft"
            ],
            z_scale_ft=cfg[
                "z_scale_ft"
            ],
            eps=cfg[
                "dbscan_eps"
            ],
            min_samples=cfg[
                "dbscan_min_samples"
            ],
        )

        observations = []

        for cluster in clusters:
            observations.append(
                {
                    "section_index":
                        section_index,
                    "s":
                        float(
                            np.median(
                                cluster["s"]
                            )
                        ),
                    "t":
                        float(
                            np.median(
                                cluster["t"]
                            )
                        ),
                    "z":
                        float(
                            np.median(
                                cluster["z"]
                            )
                        ),
                    "point_count":
                        int(
                            len(cluster)
                        ),
                }
            )

        # Collapse nearby subconductor observations in the same tracker
        # section so the tracker follows the physical bundle centre.
        observations = collapse_bundle_cross_section_rows(
            observations
        )

        observations.sort(
            key=lambda row:
                (
                    row["t"],
                    row["z"],
                )
        )

        sections.append(
            observations
        )

    return (
        sections,
        n_sections,
    )


def _class_tracker_link(
    sections: list[list[dict]],
    cfg: dict,
):
    tracks = []
    next_track_id = 1

    for observations in sections:
        active = [
            track
            for track in tracks
            if track["missed"]
            <= cfg[
                "max_missed_sections"
            ]
        ]

        if not active:
            for observation in observations:
                tracks.append(
                    {
                        "track_id":
                            next_track_id,
                        "observations":
                            [observation],
                        "missed":
                            0,
                        "fragment_count":
                            1,
                    }
                )
                next_track_id += 1
            continue

        if not observations:
            for track in active:
                track["missed"] += 1
            continue

        matrix = np.full(
            (
                len(active),
                len(observations),
            ),
            1_000_000.0,
            dtype=np.float64,
        )

        for row, track in enumerate(
            active
        ):
            predicted_t, predicted_z = (
                _internal_tracker_predict(
                    track
                )
            )

            for column, observation in enumerate(
                observations
            ):
                dt = abs(
                    observation["t"]
                    - predicted_t
                )

                dz = abs(
                    observation["z"]
                    - predicted_z
                )

                if (
                    dt
                    > cfg[
                        "max_t_jump_ft"
                    ]
                    or dz
                    > cfg[
                        "max_z_jump_ft"
                    ]
                ):
                    continue

                cost = math.sqrt(
                    (
                        dt
                        / cfg[
                            "max_t_jump_ft"
                        ]
                    )
                    ** 2
                    + (
                        dz
                        / cfg[
                            "max_z_jump_ft"
                        ]
                    )
                    ** 2
                )

                if (
                    cost
                    <= cfg[
                        "max_cost"
                    ]
                ):
                    matrix[
                        row,
                        column,
                    ] = cost

        rows, columns = (
            linear_sum_assignment(
                matrix
            )
        )

        used_track_ids = set()
        used_observations = set()

        for row, column in zip(
            rows,
            columns,
        ):
            if (
                matrix[
                    row,
                    column,
                ]
                >= 1_000_000.0
            ):
                continue

            track = active[
                int(row)
            ]

            track[
                "observations"
            ].append(
                observations[
                    int(column)
                ]
            )

            track["missed"] = 0

            used_track_ids.add(
                track["track_id"]
            )
            used_observations.add(
                int(column)
            )

        for track in active:
            if (
                track["track_id"]
                not in used_track_ids
            ):
                track["missed"] += 1

        for column, observation in enumerate(
            observations
        ):
            if column in used_observations:
                continue

            tracks.append(
                {
                    "track_id":
                        next_track_id,
                    "observations":
                        [observation],
                    "missed":
                        0,
                    "fragment_count":
                        1,
                }
            )
            next_track_id += 1

    return tracks


def _tracker_fit_arrays(
    observations: list[dict],
):
    if len(observations) < 2:
        return None

    observations = sorted(
        observations,
        key=lambda item:
            item["s"],
    )

    s = np.asarray(
        [
            item["s"]
            for item in observations
        ],
        dtype=np.float64,
    )

    t = np.asarray(
        [
            item["t"]
            for item in observations
        ],
        dtype=np.float64,
    )

    z = np.asarray(
        [
            item["z"]
            for item in observations
        ],
        dtype=np.float64,
    )

    try:
        t_degree = 1
        z_degree = (
            2
            if len(observations) >= 3
            else 1
        )

        t_coeff = np.polyfit(
            s,
            t,
            t_degree,
        )
        z_coeff = np.polyfit(
            s,
            z,
            z_degree,
        )

        t_fit = np.polyval(
            t_coeff,
            s,
        )
        z_fit = np.polyval(
            z_coeff,
            s,
        )

        t_rmse = float(
            np.sqrt(
                np.mean(
                    (t - t_fit) ** 2
                )
            )
        )
        z_rmse = float(
            np.sqrt(
                np.mean(
                    (z - z_fit) ** 2
                )
            )
        )

    except Exception:
        return None

    return {
        "s": s,
        "t": t,
        "z": z,
        "t_coeff": t_coeff,
        "z_coeff": z_coeff,
        "t_rmse": t_rmse,
        "z_rmse": z_rmse,
    }


def _tracker_fragment_pair_metrics(
    first: dict,
    second: dict,
    cfg: dict,
    n_sections: int,
):
    first_obs = sorted(
        first["observations"],
        key=lambda item:
            item["s"],
    )
    second_obs = sorted(
        second["observations"],
        key=lambda item:
            item["s"],
    )

    first_last_section = int(
        first_obs[-1][
            "section_index"
        ]
    )
    second_first_section = int(
        second_obs[0][
            "section_index"
        ]
    )

    if (
        second_first_section
        <= first_last_section
    ):
        return None

    gap_sections = max(
        0,
        second_first_section
        - first_last_section
        - 1,
    )

    gap_fraction = (
        gap_sections
        / max(
            n_sections,
            1,
        )
    )

    if (
        gap_sections
        > cfg[
            "stitch_max_gap_sections"
        ]
        or gap_fraction
        > cfg[
            "stitch_max_gap_fraction"
        ]
    ):
        return None

    first_fit = _tracker_fit_arrays(
        first_obs
    )

    if first_fit is None:
        return None

    target_s = float(
        second_obs[0]["s"]
    )

    predicted_t = float(
        np.polyval(
            first_fit["t_coeff"],
            target_s,
        )
    )
    predicted_z = float(
        np.polyval(
            first_fit["z_coeff"],
            target_s,
        )
    )

    dt = abs(
        float(
            second_obs[0]["t"]
        )
        - predicted_t
    )
    dz = abs(
        float(
            second_obs[0]["z"]
        )
        - predicted_z
    )

    if (
        dt
        > cfg[
            "stitch_t_gate_ft"
        ]
        or dz
        > cfg[
            "stitch_z_gate_ft"
        ]
    ):
        return None

    combined_obs = sorted(
        first_obs
        + second_obs,
        key=lambda item:
            item["s"],
    )

    # A track cannot have two observations for the same longitudinal section.
    section_ids = [
        int(
            item[
                "section_index"
            ]
        )
        for item in combined_obs
    ]

    if (
        len(section_ids)
        != len(
            set(section_ids)
        )
    ):
        return None

    combined_fit = _tracker_fit_arrays(
        combined_obs
    )

    if combined_fit is None:
        return None

    if (
        combined_fit[
            "t_rmse"
        ]
        > cfg[
            "stitch_max_t_rmse_ft"
        ]
        or combined_fit[
            "z_rmse"
        ]
        > cfg[
            "stitch_max_z_rmse_ft"
        ]
    ):
        return None

    cost = math.sqrt(
        (
            dt
            / max(
                cfg[
                    "stitch_t_gate_ft"
                ],
                1e-6,
            )
        )
        ** 2
        + (
            dz
            / max(
                cfg[
                    "stitch_z_gate_ft"
                ],
                1e-6,
            )
        )
        ** 2
        + (
            combined_fit[
                "t_rmse"
            ]
            / max(
                cfg[
                    "stitch_max_t_rmse_ft"
                ],
                1e-6,
            )
        )
        ** 2
        + (
            combined_fit[
                "z_rmse"
            ]
            / max(
                cfg[
                    "stitch_max_z_rmse_ft"
                ],
                1e-6,
            )
        )
        ** 2
    )

    return {
        "cost":
            float(cost),
        "gap_sections":
            int(gap_sections),
        "gap_fraction":
            float(gap_fraction),
        "dt_ft":
            float(dt),
        "dz_ft":
            float(dz),
        "combined_t_rmse_ft":
            float(
                combined_fit[
                    "t_rmse"
                ]
            ),
        "combined_z_rmse_ft":
            float(
                combined_fit[
                    "z_rmse"
                ]
            ),
        "combined_observations":
            combined_obs,
    }


def _class_tracker_stitch_fragments(
    span: DXFSpan,
    tracks: list[dict],
    class_code: int,
    cfg: dict,
    n_sections: int,
):
    """
    Iteratively stitch non-overlapping tracker fragments.

    Network continuity is NOT used here. Two fragments are merged only from
    same-span, same-class LiDAR evidence and a physically compatible combined
    t(s) / z(s) fit.
    """
    working = []

    for track in tracks:
        copied = {
            **track,
            "observations":
                list(
                    track[
                        "observations"
                    ]
                ),
            "fragment_count":
                int(
                    track.get(
                        "fragment_count",
                        1,
                    )
                ),
        }
        working.append(copied)

    actions = []
    next_stitch_id = 1

    while True:
        best = None

        for first_index, first in enumerate(
            working
        ):
            for second_index, second in enumerate(
                working
            ):
                if first_index == second_index:
                    continue

                first_last = max(
                    item[
                        "section_index"
                    ]
                    for item in first[
                        "observations"
                    ]
                )
                second_first = min(
                    item[
                        "section_index"
                    ]
                    for item in second[
                        "observations"
                    ]
                )

                if second_first <= first_last:
                    continue

                metrics = (
                    _tracker_fragment_pair_metrics(
                        first,
                        second,
                        cfg,
                        n_sections,
                    )
                )

                if metrics is None:
                    continue

                if (
                    best is None
                    or metrics[
                        "cost"
                    ]
                    < best[
                        "metrics"
                    ][
                        "cost"
                    ]
                ):
                    best = {
                        "first_index":
                            first_index,
                        "second_index":
                            second_index,
                        "metrics":
                            metrics,
                    }

        if best is None:
            break

        first_index = best[
            "first_index"
        ]
        second_index = best[
            "second_index"
        ]

        first = working[
            first_index
        ]
        second = working[
            second_index
        ]
        metrics = best[
            "metrics"
        ]

        merged_track_id = (
            f"STITCH_"
            f"{next_stitch_id:03d}"
        )
        next_stitch_id += 1

        merged = {
            "track_id":
                merged_track_id,
            "observations":
                metrics[
                    "combined_observations"
                ],
            "missed":
                0,
            "fragment_count":
                int(
                    first.get(
                        "fragment_count",
                        1,
                    )
                )
                + int(
                    second.get(
                        "fragment_count",
                        1,
                    )
                ),
        }

        actions.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                "class_code":
                    class_code,
                "class_name":
                    CONDUCTOR_CLASS_NAMES[
                        class_code
                    ],
                "engine_key":
                    cfg[
                        "name"
                    ],
                "first_track":
                    str(
                        first[
                            "track_id"
                        ]
                    ),
                "second_track":
                    str(
                        second[
                            "track_id"
                        ]
                    ),
                "merged_track":
                    merged_track_id,
                "gap_sections":
                    metrics[
                        "gap_sections"
                    ],
                "gap_fraction":
                    metrics[
                        "gap_fraction"
                    ],
                "dt_ft":
                    metrics[
                        "dt_ft"
                    ],
                "dz_ft":
                    metrics[
                        "dz_ft"
                    ],
                "combined_t_rmse_ft":
                    metrics[
                        "combined_t_rmse_ft"
                    ],
                "combined_z_rmse_ft":
                    metrics[
                        "combined_z_rmse_ft"
                    ],
                "fragment_count_after":
                    merged[
                        "fragment_count"
                    ],
            }
        )

        keep = []

        for index, track in enumerate(
            working
        ):
            if index in {
                first_index,
                second_index,
            }:
                continue
            keep.append(track)

        keep.append(merged)
        working = keep

    working.sort(
        key=lambda track:
            (
                min(
                    item[
                        "section_index"
                    ]
                    for item in track[
                        "observations"
                    ]
                ),
                float(
                    np.median(
                        [
                            item[
                                "t"
                            ]
                            for item in track[
                                "observations"
                            ]
                        ]
                    )
                ),
            )
    )

    for track_number, track in enumerate(
        working,
        start=1,
    ):
        track[
            "final_track_number"
        ] = track_number

    return (
        working,
        actions,
    )


def _class_tracker_fit(
    span: DXFSpan,
    track: dict,
    class_code: int,
    cfg: dict,
    n_sections: int,
):
    observations = track[
        "observations"
    ]

    if len(observations) < cfg[
        "min_sections"
    ]:
        return None

    fit = _tracker_fit_arrays(
        observations
    )

    if fit is None:
        return None

    support = _internal_tracker_support_metrics(
        observations,
        n_sections,
    )

    coverage = support[
        "coverage"
    ]

    normal_accept = bool(
        coverage
        >= cfg[
            "normal_accept_coverage"
        ]
        and fit[
            "t_rmse"
        ]
        <= cfg[
            "max_t_rmse_ft"
        ]
        and fit[
            "z_rmse"
        ]
        <= cfg[
            "max_z_rmse_ft"
        ]
    )

    adaptive_accept = bool(
        coverage
        >= cfg[
            "adaptive_accept_coverage"
        ]
        and fit[
            "t_rmse"
        ]
        <= cfg[
            "adaptive_max_t_rmse_ft"
        ]
        and fit[
            "z_rmse"
        ]
        <= cfg[
            "adaptive_max_z_rmse_ft"
        ]
        and support[
            "observed_sections"
        ]
        >= cfg[
            "adaptive_min_observed_sections"
        ]
        and support[
            "support_extent_fraction"
        ]
        >= cfg[
            "adaptive_min_support_extent_fraction"
        ]
        and support[
            "longest_gap_fraction"
        ]
        <= cfg[
            "adaptive_max_longest_gap_fraction"
        ]
    )

    if normal_accept:
        status = "AUTO_ACCEPT"
        accept_reason = (
            "CLASS_NORMAL_COVERAGE"
        )

    elif adaptive_accept:
        status = "AUTO_ACCEPT"
        accept_reason = (
            "CLASS_ADAPTIVE_AFTER_STITCHING"
        )

    elif coverage >= cfg[
        "review_coverage"
    ]:
        status = "REVIEW"
        accept_reason = (
            "CLASS_REVIEW_COVERAGE_OR_FIT"
        )

    else:
        return None

    sample_count = max(
        2,
        int(
            math.ceil(
                span.span_length_ft
                / cfg[
                    "sample_interval_ft"
                ]
            )
        ),
    )

    sample_s = np.linspace(
        0.0,
        span.span_length_ft,
        sample_count + 1,
    )

    sample_t = np.polyval(
        fit[
            "t_coeff"
        ],
        sample_s,
    )
    sample_z = np.polyval(
        fit[
            "z_coeff"
        ],
        sample_s,
    )

    xyz = local_to_xyz(
        span,
        sample_s,
        sample_t,
        sample_z,
    )

    geometry = LineString(
        [
            (
                float(x),
                float(y),
                float(z_value),
            )
            for x, y, z_value
            in xyz
        ]
    )

    track_number = int(
        track.get(
            "final_track_number",
            0,
        )
    )

    return {
        "track_id":
            (
                f"{span.span_id}_"
                f"C{class_code}_"
                f"TRK_{track_number:03d}"
            ),
        "status":
            status,
        "review":
            "",
        "class_code":
            class_code,
        "class_name":
            CONDUCTOR_CLASS_NAMES[
                class_code
            ],
        "class_engine":
            cfg[
                "name"
            ],
        "engine_family":
            cfg[
                "family"
            ],
        "coverage_pct":
            float(
                coverage
                * 100.0
            ),
        "accept_reason":
            accept_reason,
        "observed_sections":
            int(
                support[
                    "observed_sections"
                ]
            ),
        "support_extent_fraction":
            float(
                support[
                    "support_extent_fraction"
                ]
            ),
        "support_extent_pct":
            float(
                support[
                    "support_extent_fraction"
                ]
                * 100.0
            ),
        "longest_gap_sections":
            int(
                support[
                    "longest_gap_sections"
                ]
            ),
        "longest_gap_fraction":
            float(
                support[
                    "longest_gap_fraction"
                ]
            ),
        "longest_gap_pct":
            float(
                support[
                    "longest_gap_fraction"
                ]
                * 100.0
            ),
        "fragment_count":
            int(
                track.get(
                    "fragment_count",
                    1,
                )
            ),
        "stitched":
            bool(
                int(
                    track.get(
                        "fragment_count",
                        1,
                    )
                )
                > 1
            ),
        "body_rmse_ft":
            float(
                fit[
                    "z_rmse"
                ]
            ),
        "t_rmse_ft":
            float(
                fit[
                    "t_rmse"
                ]
            ),
        "switch_risk":
            0,
        "geometry":
            geometry,
    }


def run_class_tracker_engine(
    span: DXFSpan,
    frame: pd.DataFrame,
    class_code: int,
    family: str,
):
    if frame.empty:
        return {
            "tracks_total": 0,
            "tracks_after_stitch": 0,
            "tracks_ok": 0,
            "stitch_actions": [],
            "wires": [],
        }

    family = (
        "DX"
        if safe_text(family).upper() == "DX"
        else "TX"
    )

    cfg = dict(
        GENERIC_CLASS_TRACKER[
            family
        ]
    )

    if ENABLE_BUNDLE_COLLAPSE:
        # Individual subconductors are still detected as local clusters, then
        # collapsed to a section centroid. Two points are enough to preserve a
        # sparse component until that collapse occurs.
        cfg["dbscan_min_samples"] = min(
            int(cfg["dbscan_min_samples"]),
            2,
        )
        cfg["stitch_t_gate_ft"] = max(
            float(cfg["stitch_t_gate_ft"]),
            BUNDLE_COLLAPSE_DISTANCE_FT,
        )
        cfg["stitch_z_gate_ft"] = max(
            float(cfg["stitch_z_gate_ft"]),
            BUNDLE_COLLAPSE_DISTANCE_FT,
        )

    sections, n_sections = (
        _class_tracker_sections(
            span,
            frame,
            cfg,
        )
    )

    raw_tracks = _class_tracker_link(
        sections,
        cfg,
    )

    stitched_tracks, stitch_actions = (
        _class_tracker_stitch_fragments(
            span,
            raw_tracks,
            class_code,
            cfg,
            n_sections,
        )
    )

    wires = []

    for track in stitched_tracks:
        wire = _class_tracker_fit(
            span,
            track,
            class_code,
            cfg,
            n_sections,
        )

        if wire is not None:
            wires.append(wire)

    return {
        "tracks_total":
            len(raw_tracks),
        "tracks_after_stitch":
            len(stitched_tracks),
        "tracks_ok":
            sum(
                1
                for wire in wires
                if wire[
                    "status"
                ]
                == "AUTO_ACCEPT"
            ),
        "stitch_actions":
            stitch_actions,
        "wires":
            wires,
    }

# =============================================================================
# GEOMETRY COMPARISON
# =============================================================================

def line_xyz_samples(
    geometry: LineString,
    count: int = 25,
):
    coords = list(geometry.coords)

    if len(coords) < 2:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    xyz = np.asarray(
        [
            (
                float(coord[0]),
                float(coord[1]),
                float(
                    coord[2]
                    if len(coord) >= 3
                    else 0.0
                ),
            )
            for coord in coords
        ],
        dtype=np.float64,
    )

    segment_vectors = (
        xyz[1:]
        - xyz[:-1]
    )

    segment_lengths = np.linalg.norm(
        segment_vectors,
        axis=1,
    )

    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(segment_lengths),
        )
    )

    total = cumulative[-1]

    if total <= 1e-9:
        return np.repeat(
            xyz[:1],
            count,
            axis=0,
        )

    targets = np.linspace(
        0.0,
        total,
        count,
    )

    samples = []

    segment_index = 0

    for target in targets:
        while (
            segment_index
            < len(segment_lengths) - 1
            and cumulative[
                segment_index + 1
            ] < target
        ):
            segment_index += 1

        start = cumulative[
            segment_index
        ]

        end = cumulative[
            segment_index + 1
        ]

        if end <= start:
            ratio = 0.0
        else:
            ratio = (
                target - start
            ) / (
                end - start
            )

        point = (
            xyz[segment_index]
            + ratio
            * segment_vectors[
                segment_index
            ]
        )

        samples.append(point)

    return np.vstack(samples)


def mean_line_distance_3d(
    first: LineString,
    second: LineString,
):
    a = line_xyz_samples(
        first,
        count=25,
    )

    b = line_xyz_samples(
        second,
        count=25,
    )

    if (
        len(a) == 0
        or len(b) == 0
    ):
        return float("inf")

    direct = float(
        np.mean(
            np.linalg.norm(
                a - b,
                axis=1,
            )
        )
    )

    reverse = float(
        np.mean(
            np.linalg.norm(
                a - b[::-1],
                axis=1,
            )
        )
    )

    return min(
        direct,
        reverse,
    )


# =============================================================================
# FUSION
# =============================================================================

def poa_result_geometry(
    span: DXFSpan,
    hypothesis: WireHypothesis,
    params: dict,
):
    return build_wire_geometry(
        span,
        hypothesis.t_a,
        hypothesis.z_a,
        hypothesis.t_b,
        hypothesis.z_b,
        hypothesis.k,
        params[
            "sample_interval_ft"
        ],
    )


def fuse_span_results(
    span: DXFSpan,
    poa_runs: list[dict],
    v7_result: dict,
):
    poa_candidates = []

    for run in poa_runs:
        params = (
            DX
            if run["mode"] == "DX"
            else TX
        )

        for hypothesis in run["selected"]:
            geometry = poa_result_geometry(
                span,
                hypothesis,
                params,
            )

            poa_candidates.append(
                {
                    "engine_mode":
                        run["mode"],

                    "pass_name":
                        run[
                            "pass_name"
                        ],

                    "hypothesis":
                        hypothesis,

                    "geometry":
                        geometry,
                }
            )

    # If HYBRID ran both engines, remove duplicate POA solutions and keep the
    # stronger candidate.
    poa_candidates.sort(
        key=lambda row:
            (
                {
                    "DENSE_STRONG": 0,
                    "SPARSE_STRONG": 1,
                    "REVIEW": 2,
                    "REJECT": 3,
                }.get(
                    row[
                        "hypothesis"
                    ].quality,
                    4,
                ),

                row[
                    "hypothesis"
                ].owned_median_residual_ft,

                -row[
                    "hypothesis"
                ].owned_coverage,
            )
    )

    unique_poa = []

    for candidate in poa_candidates:
        duplicate = False

        for existing in unique_poa:
            distance = mean_line_distance_3d(
                candidate["geometry"],
                existing["geometry"],
            )

            if distance <= 0.65:
                duplicate = True
                break

        if not duplicate:
            unique_poa.append(
                candidate
            )

    final_records = []
    review_records = []
    rejected_records = []

    used_v7 = set()

    # POA first.
    for candidate in unique_poa:
        hypothesis = candidate[
            "hypothesis"
        ]

        params = (
            DX
            if candidate[
                "engine_mode"
            ]
            == "DX"
            else TX
        )

        nearest_v7_index = None
        nearest_v7_distance = float(
            "inf"
        )

        for index, v7 in enumerate(
            v7_result["wires"]
        ):
            distance = mean_line_distance_3d(
                candidate["geometry"],
                v7["geometry"],
            )

            if distance < nearest_v7_distance:
                nearest_v7_distance = distance
                nearest_v7_index = index

        v7_agrees = bool(
            nearest_v7_index is not None
            and nearest_v7_distance
            <= params[
                "v7_agreement_ft"
            ]
        )

        # V2.5:
        # Agreement does not mean that V7 has already been represented in
        # final geometry. Consume it only if this fused POA/V7 record is
        # actually accepted into FINAL below.

        pass_name = candidate.get(
            "pass_name",
            candidate[
                "engine_mode"
            ],
        )

        is_underbuild = (
            pass_name
            == "DX_UNDERBUILD"
        )

        # ---------------------------------------------------------
        # CONSERVATIVE ACCEPTANCE
        # ---------------------------------------------------------

        if hypothesis.quality == "DENSE_STRONG":
            if is_underbuild:
                if not hypothesis.underbuild_spatial_ok:
                    output_class = "REVIEW"
                    confidence = "LOW_MEDIUM"
                    fusion_reason = (
                        "UNDERBUILD_NOT_CONSISTENTLY_BELOW_TX"
                    )

                elif (
                    hypothesis.underbuild_group_size
                    >= UNDERBUILD_MIN_GROUP_SIZE_FOR_AUTO
                ):
                    if v7_agrees:
                        output_class = "FINAL"
                        confidence = "VERY_HIGH"
                        fusion_reason = (
                            "UNDERBUILD_GROUP_V7_AGREE"
                        )
                    else:
                        output_class = "FINAL"
                        confidence = "HIGH"
                        fusion_reason = (
                            "UNDERBUILD_GROUP_CONFIRMED"
                        )

                elif v7_agrees:
                    output_class = "FINAL"
                    confidence = "HIGH"
                    fusion_reason = (
                        "SINGLE_UNDERBUILD_V7_AGREE"
                    )

                else:
                    output_class = "REVIEW"
                    confidence = "MEDIUM"
                    fusion_reason = (
                        "SINGLE_UNDERBUILD_NEEDS_V7"
                    )

            else:
                if v7_agrees:
                    output_class = "FINAL"
                    confidence = "VERY_HIGH"
                    fusion_reason = (
                        "DENSE_POA_V7_AGREE"
                    )

                elif hypothesis.bundle_confirmed:
                    # Manual cleanup showed confirmed TX bundles to be a
                    # high-precision category even where the independent
                    # section tracker does not resolve the bundle cleanly.
                    output_class = "FINAL"
                    confidence = "HIGH"
                    fusion_reason = (
                        "CONFIRMED_TX_BUNDLE"
                    )

                elif ALLOW_STRONG_POA_WITHOUT_V7_AGREEMENT:
                    output_class = "FINAL"
                    confidence = "HIGH"
                    fusion_reason = (
                        "DENSE_POA_V7_WEAK_OR_MISSING"
                    )

                else:
                    output_class = "REVIEW"
                    confidence = "MEDIUM"
                    fusion_reason = (
                        "DENSE_POA_TRACKER_WEAK_OR_MISSING"
                    )

        elif hypothesis.quality == "SPARSE_STRONG":
            # SECOND PASS V1: sparse support is never enough by itself for final.
            if v7_agrees:
                if (
                    is_underbuild
                    and not hypothesis.underbuild_spatial_ok
                ):
                    output_class = "REVIEW"
                    confidence = "LOW_MEDIUM"
                    fusion_reason = (
                        "SPARSE_UNDERBUILD_BAD_VERTICAL_POSITION"
                    )
                else:
                    output_class = "FINAL"
                    confidence = "HIGH"
                    fusion_reason = (
                        "SPARSE_POA_V7_AGREE"
                    )
            else:
                output_class = "REVIEW"
                confidence = "MEDIUM"
                fusion_reason = (
                    "SPARSE_POA_NEEDS_V7_CONFIRMATION"
                )

        elif hypothesis.quality == "REVIEW":
            output_class = "REVIEW"
            confidence = "MEDIUM"
            fusion_reason = "POA_REVIEW"

        else:
            output_class = "REJECT"
            confidence = "LOW"
            fusion_reason = "POA_REJECT"

        record = {
            "span_id":
                span.span_id,

            "layer_name":
                span.layer_name,

            "source":
                "POA",

            "engine_mode":
                candidate[
                    "engine_mode"
                ],

            "pass_name":
                pass_name,

            "source_id":
                hypothesis.hypothesis_id,

            "confidence":
                confidence,

            "fusion_reason":
                fusion_reason,

            "v7_agreement_ft":
                (
                    nearest_v7_distance
                    if np.isfinite(
                        nearest_v7_distance
                    )
                    else np.nan
                ),

            "poa_quality":
                hypothesis.quality,

            "coverage":
                hypothesis.owned_coverage,

            "median_residual_ft":
                hypothesis.owned_median_residual_ft,

            "p90_residual_ft":
                hypothesis.owned_p90_residual_ft,

            "support_points":
                hypothesis.owned_points,

            "support_zones":
                hypothesis.support_zone_count,

            "longest_gap_ft":
                hypothesis.longest_unsupported_gap_ft,

            "longest_gap_fraction":
                hypothesis.longest_unsupported_gap_fraction,

            "terminal_spread_a_ft":
                hypothesis.terminal_spread_a_ft,

            "terminal_spread_b_ft":
                hypothesis.terminal_spread_b_ft,

            "terminal_spread_ft":
                hypothesis.terminal_spread_ft,

            "bundle_confirmed":
                hypothesis.bundle_confirmed,

            "underbuild_below_fraction":
                hypothesis.underbuild_below_fraction,

            "underbuild_spatial_ok":
                hypothesis.underbuild_spatial_ok,

            "underbuild_group_size":
                hypothesis.underbuild_group_size,

            "adaptive_radius_ft":
                hypothesis.adaptive_radius_ft,

            "adaptive_median_limit_ft":
                hypothesis.adaptive_median_limit_ft,

            "adaptive_p90_limit_ft":
                hypothesis.adaptive_p90_limit_ft,

            "sag_depth_ft":
                hypothesis.sag_depth_ft,

            "locked":
                bool(
                    fusion_reason
                    in {
                        "DENSE_POA_V7_AGREE",
                        "UNDERBUILD_GROUP_V7_AGREE",
                        "SINGLE_UNDERBUILD_V7_AGREE",
                        "SPARSE_POA_V7_AGREE",
                    }
                    or hypothesis.bundle_confirmed
                ),

            "geometry":
                candidate[
                    "geometry"
                ],
        }

        if output_class == "FINAL":
            final_records.append(record)

            # V2.5:
            # The agreeing V7 track is now genuinely represented by a final
            # conductor, so it can be removed from the V7-only recovery pool.
            if (
                v7_agrees
                and nearest_v7_index is not None
            ):
                used_v7.add(
                    nearest_v7_index
                )

        elif output_class == "REVIEW":
            review_records.append(record)

        else:
            rejected_records.append(record)

    # Clean V7 wires can fill a genuine hole where POA did not establish
    # terminal/body evidence.
    if ALLOW_V7_ONLY_FINAL:
        for index, v7 in enumerate(
            v7_result["wires"]
        ):
            if index in used_v7:
                continue

            if (
                v7["status"]
                != "AUTO_ACCEPT"
            ):
                continue

            duplicate = False

            # V2.5:
            # Only accepted final wires can block a V7-only recovery.
            # REVIEW geometry is evidence, not an existing final conductor.
            for record in final_records:
                distance = mean_line_distance_3d(
                    v7["geometry"],
                    record["geometry"],
                )

                if distance <= 0.90:
                    duplicate = True
                    break

            if duplicate:
                continue

            final_records.append(
                {
                    "span_id":
                        span.span_id,

                    "layer_name":
                        span.layer_name,

                    "source":
                        "V7",

                    "engine_mode":
                        "V7",

                    "pass_name":
                        "V7",

                    "source_id":
                        v7["track_id"],

                    "confidence":
                        "MEDIUM_HIGH",

                    "fusion_reason":
                        "V7_ONLY_AUTO_ACCEPT_AFTER_FUSION_FIX",

                    "v7_agreement_ft":
                        0.0,

                    "poa_quality":
                        "",

                    "coverage":
                        v7[
                            "coverage_pct"
                        ]
                        / 100.0,

                    "v7_accept_reason":
                        safe_text(
                            v7.get(
                                "accept_reason",
                                "",
                            )
                        ),

                    "v7_observed_sections":
                        v7.get(
                            "observed_sections",
                            np.nan,
                        ),

                    "v7_support_extent_fraction":
                        v7.get(
                            "support_extent_fraction",
                            np.nan,
                        ),

                    "v7_longest_gap_fraction":
                        v7.get(
                            "longest_gap_fraction",
                            np.nan,
                        ),

                    "median_residual_ft":
                        np.nan,

                    "p90_residual_ft":
                        np.nan,

                    "locked":
                        False,

                    "support_points":
                        np.nan,

                    "sag_depth_ft":
                        np.nan,

                    "geometry":
                        v7[
                            "geometry"
                        ],
                }
            )

    return (
        final_records,
        review_records,
        rejected_records,
        unique_poa,
    )


# =============================================================================
# NODE RECONCILIATION
# =============================================================================

def endpoint_xyz(
    geometry: LineString,
    at_start: bool,
) -> np.ndarray:
    coords = list(
        geometry.coords
    )

    coord = (
        coords[0]
        if at_start
        else coords[-1]
    )

    return np.asarray(
        [
            float(coord[0]),
            float(coord[1]),
            float(
                coord[2]
                if len(coord) >= 3
                else 0.0
            ),
        ],
        dtype=np.float64,
    )


def replace_endpoint(
    geometry: LineString,
    at_start: bool,
    xyz: np.ndarray,
) -> LineString:
    coords = [
        list(coord)
        for coord
        in geometry.coords
    ]

    replacement = (
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
    )

    if at_start:
        coords[0] = list(
            replacement
        )
    else:
        coords[-1] = list(
            replacement
        )

    return LineString(
        [
            (
                float(coord[0]),
                float(coord[1]),
                float(
                    coord[2]
                    if len(coord) >= 3
                    else 0.0
                ),
            )
            for coord in coords
        ]
    )


def reconcile_nodes(
    spans: list[DXFSpan],
    final_records: list[dict],
):
    if not final_records:
        return []

    span_lookup = {
        span.span_id:
            span
        for span in spans
    }

    incident = {}

    for span in spans:
        incident.setdefault(
            span.node_a,
            [],
        ).append(span.span_id)

        incident.setdefault(
            span.node_b,
            [],
        ).append(span.span_id)

    by_span = {}

    for index, record in enumerate(
        final_records
    ):
        by_span.setdefault(
            record["span_id"],
            [],
        ).append(index)

    join_records = []

    for node_id, span_ids in incident.items():
        if (
            NODE_JOIN_ONLY_DEGREE_2
            and len(span_ids) != 2
        ):
            continue

        if len(span_ids) != 2:
            continue

        first_span_id, second_span_id = span_ids

        first_indices = by_span.get(
            first_span_id,
            [],
        )

        second_indices = by_span.get(
            second_span_id,
            [],
        )

        if (
            not first_indices
            or not second_indices
        ):
            continue

        first_span = span_lookup[
            first_span_id
        ]

        second_span = span_lookup[
            second_span_id
        ]

        first_at_start = (
            first_span.node_a
            == node_id
        )

        second_at_start = (
            second_span.node_a
            == node_id
        )

        first_xyz = np.vstack(
            [
                endpoint_xyz(
                    final_records[index][
                        "geometry"
                    ],
                    first_at_start,
                )
                for index
                in first_indices
            ]
        )

        second_xyz = np.vstack(
            [
                endpoint_xyz(
                    final_records[index][
                        "geometry"
                    ],
                    second_at_start,
                )
                for index
                in second_indices
            ]
        )

        cost = np.full(
            (
                len(first_indices),
                len(second_indices),
            ),
            INVALID_PAIR_COST,
            dtype=np.float64,
        )

        for first_no in range(
            len(first_indices)
        ):
            for second_no in range(
                len(second_indices)
            ):
                first_record = final_records[
                    first_indices[first_no]
                ]
                second_record = final_records[
                    second_indices[second_no]
                ]

                first_class = first_record.get(
                    "class_code",
                    None,
                )
                second_class = second_record.get(
                    "class_code",
                    None,
                )

                if (
                    first_class is not None
                    and second_class is not None
                    and first_class != second_class
                ):
                    continue

                delta = (
                    first_xyz[first_no]
                    - second_xyz[second_no]
                )

                xy = float(
                    np.linalg.norm(
                        delta[:2]
                    )
                )

                dz = abs(
                    float(delta[2])
                )

                if (
                    xy
                    > NODE_JOIN_MAX_ENDPOINT_XY_FT
                    or dz
                    > NODE_JOIN_MAX_ENDPOINT_Z_FT
                ):
                    continue

                cost[
                    first_no,
                    second_no,
                ] = (
                    xy
                    + NODE_JOIN_Z_WEIGHT
                    * dz
                )

        rows, columns = linear_sum_assignment(
            cost
        )

        for row, column in zip(
            rows,
            columns,
        ):
            join_cost = float(
                cost[row, column]
            )

            if (
                join_cost
                >= NODE_JOIN_MAX_COST
                or join_cost
                >= INVALID_PAIR_COST
            ):
                continue

            first_index = first_indices[row]
            second_index = second_indices[column]

            common_xyz = (
                first_xyz[row]
                + second_xyz[column]
            ) / 2.0

            # V2.3: logical continuity only.
            # Do NOT move conductor endpoints. Endpoint mutation created
            # visible kinks and non-physical geometry in previous versions.

            join_records.append(
                {
                    "node_id":
                        node_id,

                    "class_code":
                        final_records[
                            first_index
                        ].get(
                            "class_code",
                            np.nan,
                        ),

                    "class_name":
                        safe_text(
                            final_records[
                                first_index
                            ].get(
                                "class_name"
                            )
                        ),

                    "span_a":
                        first_span_id,

                    "span_b":
                        second_span_id,

                    "wire_a":
                        final_records[
                            first_index
                        ][
                            "source_id"
                        ],

                    "wire_b":
                        final_records[
                            second_index
                        ][
                            "source_id"
                        ],

                    "join_cost":
                        join_cost,

                    "endpoint_xy_gap_ft":
                        float(
                            np.linalg.norm(
                                first_xyz[row, :2]
                                - second_xyz[column, :2]
                            )
                        ),

                    "endpoint_z_gap_ft":
                        float(
                            abs(
                                first_xyz[row, 2]
                                - second_xyz[column, 2]
                            )
                        ),

                    "geometry":
                        Point(
                            float(
                                common_xyz[0]
                            ),
                            float(
                                common_xyz[1]
                            ),
                            float(
                                common_xyz[2]
                            ),
                        ),
                }
            )

    return join_records



# =============================================================================
# PHYSICAL GEOMETRY VALIDATION
# =============================================================================

def _geometry_turn_angles_deg(
    geometry: LineString,
):
    coords = _cv_geometry_coords_3d(
        geometry
    )

    if len(coords) < 3:
        return np.empty(
            0,
            dtype=np.float64,
        )

    vectors = (
        coords[1:]
        - coords[:-1]
    )

    lengths = np.linalg.norm(
        vectors,
        axis=1,
    )

    valid = (
        lengths
        > 1e-9
    )

    if np.count_nonzero(
        valid
    ) < 2:
        return np.empty(
            0,
            dtype=np.float64,
        )

    angles = []

    for first, second in zip(
        vectors[:-1],
        vectors[1:],
    ):
        first_length = float(
            np.linalg.norm(
                first
            )
        )

        second_length = float(
            np.linalg.norm(
                second
            )
        )

        if (
            first_length <= 1e-9
            or second_length <= 1e-9
        ):
            continue

        cosine = float(
            np.clip(
                np.dot(
                    first / first_length,
                    second / second_length,
                ),
                -1.0,
                1.0,
            )
        )

        angles.append(
            float(
                np.degrees(
                    np.arccos(
                        cosine
                    )
                )
            )
        )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


def _geometry_local_samples(
    span: DXFSpan,
    geometry: LineString,
):
    samples = line_xyz_samples(
        geometry,
        count=GEOM_QA_SAMPLE_COUNT,
    )

    if len(samples) < 3:
        return None

    relative = (
        samples[:, :2]
        - span.axis_origin_xy[
            None,
            :
        ]
    )

    station = (
        relative
        @ span.axis_u
    )

    lateral = (
        relative
        @ span.axis_p
    )

    z = samples[:, 2].copy()

    # Orient from increasing span station.
    if (
        np.nanmedian(
            np.diff(
                station
            )
        )
        < 0.0
    ):
        station = station[::-1]
        lateral = lateral[::-1]
        z = z[::-1]

    return (
        station,
        lateral,
        z,
    )


def geometry_physics_metrics(
    span: DXFSpan,
    geometry: LineString,
    mode: str,
):
    metrics = {
        "geometry_max_turn_deg":
            np.nan,

        "geometry_max_backtrack_ft":
            np.nan,

        "geometry_lateral_max_residual_ft":
            np.nan,

        "geometry_lateral_p95_residual_ft":
            np.nan,

        "geometry_z_max_residual_ft":
            np.nan,

        "geometry_z_p95_residual_ft":
            np.nan,

        "geometry_station_coverage":
            np.nan,

        "geometry_qa_status":
            "REVIEW",

        "geometry_qa_reason":
            "INSUFFICIENT_GEOMETRY",
    }

    if (
        geometry is None
        or geometry.is_empty
    ):
        return metrics

    local = _geometry_local_samples(
        span,
        geometry,
    )

    if local is None:
        return metrics

    station, lateral, z = local

    finite = (
        np.isfinite(
            station
        )
        & np.isfinite(
            lateral
        )
        & np.isfinite(
            z
        )
    )

    station = station[
        finite
    ]
    lateral = lateral[
        finite
    ]
    z = z[
        finite
    ]

    if len(
        station
    ) < 5:
        return metrics

    ds = np.diff(
        station
    )

    max_backtrack = float(
        max(
            0.0,
            -float(
                np.min(
                    ds
                )
            ),
        )
    )

    station_range = float(
        np.max(
            station
        )
        - np.min(
            station
        )
    )

    coverage = (
        station_range
        / span.span_length_ft
        if span.span_length_ft
        > 1e-9
        else 0.0
    )

    # Lateral position of a conductor should be effectively linear in plan.
    try:
        lateral_coeff = np.polyfit(
            station,
            lateral,
            1,
        )

        lateral_fit = np.polyval(
            lateral_coeff,
            station,
        )

        lateral_residual = np.abs(
            lateral
            - lateral_fit
        )

        lateral_max = float(
            np.max(
                lateral_residual
            )
        )

        lateral_p95 = float(
            np.quantile(
                lateral_residual,
                0.95,
            )
        )

    except Exception:
        lateral_max = np.inf
        lateral_p95 = np.inf

    # A parabola is used only as a smoothness reference. We are not claiming
    # the physical conductor is exactly parabolic.
    try:
        degree = (
            2
            if len(
                np.unique(
                    np.round(
                        station,
                        6,
                    )
                )
            )
            >= 3
            else 1
        )

        z_coeff = np.polyfit(
            station,
            z,
            degree,
        )

        z_fit = np.polyval(
            z_coeff,
            station,
        )

        z_residual = np.abs(
            z
            - z_fit
        )

        z_max = float(
            np.max(
                z_residual
            )
        )

        z_p95 = float(
            np.quantile(
                z_residual,
                0.95,
            )
        )

    except Exception:
        z_max = np.inf
        z_p95 = np.inf

    turn_angles = _geometry_turn_angles_deg(
        geometry
    )

    max_turn = (
        float(
            np.max(
                turn_angles
            )
        )
        if len(
            turn_angles
        )
        else 0.0
    )

    if mode == "TX":
        turn_gate = (
            GEOM_QA_TX_MAX_TURN_DEG
        )

        lateral_gate = (
            GEOM_QA_TX_MAX_LATERAL_LINEAR_RESIDUAL_FT
        )

        z_gate = (
            GEOM_QA_TX_MAX_Z_QUADRATIC_RESIDUAL_FT
        )

    else:
        turn_gate = (
            GEOM_QA_DX_MAX_TURN_DEG
        )

        lateral_gate = (
            GEOM_QA_DX_MAX_LATERAL_LINEAR_RESIDUAL_FT
        )

        z_gate = (
            GEOM_QA_DX_MAX_Z_QUADRATIC_RESIDUAL_FT
        )

    reasons = []

    if (
        max_turn
        > turn_gate
    ):
        reasons.append(
            "INTERNAL_KINK"
        )

    if (
        max_backtrack
        > GEOM_QA_MAX_BACKTRACK_FT
    ):
        reasons.append(
            "STATION_BACKTRACK"
        )

    if (
        lateral_max
        > lateral_gate
    ):
        reasons.append(
            "LATERAL_JUMP_OR_BEND"
        )

    if (
        z_max
        > z_gate
    ):
        reasons.append(
            "VERTICAL_JUMP_OR_BEND"
        )

    metrics.update(
        {
            "geometry_max_turn_deg":
                max_turn,

            "geometry_max_backtrack_ft":
                max_backtrack,

            "geometry_lateral_max_residual_ft":
                lateral_max,

            "geometry_lateral_p95_residual_ft":
                lateral_p95,

            "geometry_z_max_residual_ft":
                z_max,

            "geometry_z_p95_residual_ft":
                z_p95,

            "geometry_station_coverage":
                float(
                    coverage
                ),

            "geometry_qa_status":
                (
                    "PASS"
                    if not reasons
                    else "REVIEW"
                ),

            "geometry_qa_reason":
                (
                    "PASS"
                    if not reasons
                    else "|".join(
                        reasons
                    )
                ),
        }
    )

    return metrics


def validate_final_geometry(
    spans: list[DXFSpan],
    final_records: list[dict],
):
    if not ENABLE_GEOMETRY_PHYSICS_QA:
        return (
            final_records,
            [],
            [],
        )

    span_lookup = {
        span.span_id:
            span
        for span in spans
    }

    accepted = []
    review = []
    qa_records = []

    for record in final_records:
        span_id = safe_text(
            record.get(
                "span_id"
            )
        )

        span = span_lookup.get(
            span_id
        )

        if span is None:
            metrics = {
                "geometry_qa_status":
                    "REVIEW",

                "geometry_qa_reason":
                    "SPAN_NOT_FOUND",
            }

        else:
            mode = safe_text(
                record.get(
                    "engine_mode",
                    ""
                )
            ).upper()

            if mode not in {
                "DX",
                "TX",
            }:
                mode = (
                    "TX"
                    if span.span_length_ft
                    >= 220.0
                    else "DX"
                )

            metrics = geometry_physics_metrics(
                span,
                record.get(
                    "geometry"
                ),
                mode,
            )

        updated = dict(
            record
        )

        updated.update(
            metrics
        )

        qa_records.append(
            {
                "span_id":
                    span_id,

                "class_code":
                    record.get(
                        "class_code",
                        np.nan,
                    ),

                "class_name":
                    safe_text(
                        record.get(
                            "class_name"
                        )
                    ),

                "class_engine":
                    safe_text(
                        record.get(
                            "class_engine"
                        )
                    ),

                "engine_mode":
                    safe_text(
                        record.get(
                            "engine_mode"
                        )
                    ),

                "source_id":
                    safe_text(
                        record.get(
                            "source_id"
                        )
                    ),

                "source":
                    safe_text(
                        record.get(
                            "source"
                        )
                    ),

                "pass_name":
                    safe_text(
                        record.get(
                            "pass_name"
                        )
                    ),

                **metrics,

                "geometry":
                    record.get(
                        "geometry"
                    ),
            }
        )

        if (
            safe_text(
                metrics.get(
                    "geometry_qa_status"
                )
            )
            == "PASS"
        ):
            accepted.append(
                updated
            )

        else:
            updated[
                "review_reason"
            ] = safe_text(
                metrics.get(
                    "geometry_qa_reason"
                )
            )

            review.append(
                updated
            )

    log(
        "\nPhysical geometry QA:"
        f"\n  input final wires: "
        f"{len(final_records):,}"
        f"\n  PASS: {len(accepted):,}"
        f"\n  moved to geometry review: "
        f"{len(review):,}"
    )

    if review:
        reasons = pd.Series(
            [
                safe_text(
                    record.get(
                        "geometry_qa_reason"
                    )
                )
                for record in review
            ]
        ).value_counts()

        for reason, count in reasons.items():
            log(
                f"  {reason}: "
                f"{int(count):,}"
            )

    return (
        accepted,
        review,
        qa_records,
    )



# =============================================================================
# V3 CLASS-SPECIFIC FUSION / SPAN PROCESSING
# =============================================================================

def _annotate_class_record(
    record: dict,
    class_code: int,
    config: dict,
):
    record[
        "class_code"
    ] = class_code
    record[
        "class_name"
    ] = config[
        "class_name"
    ]
    record[
        "class_engine"
    ] = config[
        "engine_key"
    ]

    return record


def _bundle_record_priority(record: dict):
    confidence_rank = {
        "VERY_HIGH": 0,
        "HIGH": 1,
        "MEDIUM_HIGH": 2,
        "MEDIUM": 3,
        "LOW_MEDIUM": 4,
        "LOW": 5,
    }

    source_rank = {
        "POA": 0,
        "CLASS_TRACKER": 1,
        "BUNDLE_CENTER": 2,
    }

    residual = record.get(
        "median_residual_ft",
        np.nan,
    )

    residual = (
        float(residual)
        if np.isfinite(residual)
        else 9999.0
    )

    coverage = record.get(
        "coverage",
        0.0,
    )

    coverage = (
        float(coverage)
        if np.isfinite(coverage)
        else 0.0
    )

    return (
        confidence_rank.get(
            safe_text(
                record.get("confidence")
            ),
            9,
        ),
        source_rank.get(
            safe_text(
                record.get("source")
            ),
            9,
        ),
        residual,
        -coverage,
    )


def _bundle_oriented_samples(
    span: DXFSpan,
    geometry: LineString,
    count: int,
):
    samples = line_xyz_samples(
        geometry,
        count=count,
    )

    if len(samples) == 0:
        return samples

    relative = (
        samples[:, :2]
        - span.axis_origin_xy[None, :]
    )

    station = relative @ span.axis_u

    if (
        len(station) >= 2
        and station[-1] < station[0]
    ):
        samples = samples[::-1].copy()

    return samples


def collapse_final_bundle_records(
    span: DXFSpan,
    class_code: int,
    records: list[dict],
) -> list[dict]:
    """
    Final safety collapse.

    Earlier terminal/body/tracker collapse should normally leave one model per
    bundle. If independent POA/tracker/candidate paths still leave multiple
    accepted models within the physical bundle envelope, replace them with one
    robust median centreline.
    """
    if (
        not ENABLE_BUNDLE_COLLAPSE
        or len(records) < 2
    ):
        for record in records:
            record.setdefault(
                "bundle_collapsed",
                False,
            )
            record.setdefault(
                "bundle_member_count",
                1,
            )
        return records

    count = len(records)
    neighbours = [set() for _ in range(count)]

    for first in range(count - 1):
        for second in range(first + 1, count):
            distance = mean_line_distance_3d(
                records[first]["geometry"],
                records[second]["geometry"],
            )

            if distance <= BUNDLE_COLLAPSE_DISTANCE_FT:
                neighbours[first].add(second)
                neighbours[second].add(first)

    groups = []
    unseen = set(range(count))

    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = set()

        while stack:
            current = stack.pop()

            if current in component:
                continue

            component.add(current)
            unseen.discard(current)

            for neighbour in neighbours[current]:
                if neighbour not in component:
                    stack.append(neighbour)

        groups.append(
            sorted(component)
        )

    output = []
    bundle_number = 1

    for indices in groups:
        group = [
            records[index]
            for index in indices
        ]

        if len(group) == 1:
            record = dict(group[0])
            record["bundle_collapsed"] = False
            record["bundle_member_count"] = int(
                record.get(
                    "bundle_member_count",
                    1,
                )
            )
            output.append(record)
            continue

        pair_distances = []

        for first in range(len(group) - 1):
            for second in range(first + 1, len(group)):
                pair_distances.append(
                    mean_line_distance_3d(
                        group[first]["geometry"],
                        group[second]["geometry"],
                    )
                )

        maximum_width = (
            max(pair_distances)
            if pair_distances
            else 0.0
        )

        # Same safeguard as the cross-section collapse. A chain wider than a
        # physical bundle remains separate rather than swallowing a neighbour.
        if maximum_width > BUNDLE_MAX_DIAMETER_FT:
            for item in group:
                record = dict(item)
                record["bundle_collapsed"] = False
                record["bundle_member_count"] = int(
                    record.get(
                        "bundle_member_count",
                        1,
                    )
                )
                output.append(record)
            continue

        sample_parts = []

        for item in group:
            samples = _bundle_oriented_samples(
                span,
                item["geometry"],
                BUNDLE_CENTERLINE_SAMPLES,
            )

            if len(samples) == BUNDLE_CENTERLINE_SAMPLES:
                sample_parts.append(samples)

        if len(sample_parts) < 2:
            # Cannot safely create a centre geometry. Keep the strongest one.
            group.sort(
                key=_bundle_record_priority
            )
            record = dict(group[0])
            record["bundle_collapsed"] = True
            record["bundle_member_count"] = len(group)
            record["bundle_width_ft"] = float(maximum_width)
            record["bundle_width_m"] = float(
                maximum_width / US_SURVEY_FT_PER_M
            )
            output.append(record)
            continue

        stack = np.stack(
            sample_parts,
            axis=0,
        )

        centre_xyz = np.median(
            stack,
            axis=0,
        )

        centre_geometry = LineString(
            [
                (
                    float(x),
                    float(y),
                    float(z),
                )
                for x, y, z
                in centre_xyz
            ]
        )

        group.sort(
            key=_bundle_record_priority
        )
        template = dict(group[0])

        template["geometry"] = centre_geometry
        template["source"] = "BUNDLE_CENTER"
        template["source_id"] = (
            f"{span.span_id}_"
            f"C{class_code}_"
            f"BUNDLE_{bundle_number:02d}"
        )
        template["fusion_reason"] = (
            "BUNDLE_CENTER_COLLAPSE"
        )
        template["bundle_collapsed"] = True
        template["bundle_member_count"] = int(
            len(group)
        )
        template["bundle_width_ft"] = float(
            maximum_width
        )
        template["bundle_width_m"] = float(
            maximum_width
            / US_SURVEY_FT_PER_M
        )
        template["bundle_member_sources"] = ";".join(
            safe_text(
                item.get("source_id")
            )
            for item in group
        )
        template["locked"] = True

        output.append(template)
        bundle_number += 1

    output.sort(
        key=lambda record: (
            safe_text(record.get("span_id")),
            safe_text(record.get("source_id")),
        )
    )

    return output


def fuse_span_class_results(
    span: DXFSpan,
    class_code: int,
    config: dict,
    poa_run: dict,
    tracker_result: dict,
):
    params = config[
        "poa"
    ]
    family = config[
        "family"
    ]

    poa_candidates = []

    for hypothesis in poa_run[
        "selected"
    ]:
        geometry = poa_result_geometry(
            span,
            hypothesis,
            params,
        )

        poa_candidates.append(
            {
                "hypothesis":
                    hypothesis,
                "geometry":
                    geometry,
            }
        )

    poa_candidates.sort(
        key=lambda row:
            (
                {
                    "DENSE_STRONG": 0,
                    "SPARSE_STRONG": 1,
                    "REVIEW": 2,
                    "REJECT": 3,
                }.get(
                    row[
                        "hypothesis"
                    ].quality,
                    4,
                ),
                row[
                    "hypothesis"
                ].owned_median_residual_ft,
                -row[
                    "hypothesis"
                ].owned_coverage,
            )
    )

    unique_poa = []

    duplicate_distance = (
        BUNDLE_COLLAPSE_DISTANCE_FT
        if ENABLE_BUNDLE_COLLAPSE
        else (
            0.65
            if family == "DX"
            else 1.00
        )
    )

    for candidate in poa_candidates:
        duplicate = False

        for existing in unique_poa:
            distance = mean_line_distance_3d(
                candidate[
                    "geometry"
                ],
                existing[
                    "geometry"
                ],
            )

            if distance <= duplicate_distance:
                duplicate = True
                break

        if not duplicate:
            unique_poa.append(
                candidate
            )

    final_records = []
    review_records = []
    rejected_records = []
    used_tracker = set()

    for candidate in unique_poa:
        hypothesis = candidate[
            "hypothesis"
        ]

        nearest_tracker_index = None
        nearest_tracker_distance = np.inf

        for index, tracker_wire in enumerate(
            tracker_result[
                "wires"
            ]
        ):
            distance = mean_line_distance_3d(
                candidate[
                    "geometry"
                ],
                tracker_wire[
                    "geometry"
                ],
            )

            if distance < nearest_tracker_distance:
                nearest_tracker_distance = distance
                nearest_tracker_index = index

        tracker_agrees = bool(
            nearest_tracker_index
            is not None
            and nearest_tracker_distance
            <= params[
                "v7_agreement_ft"
            ]
        )

        if hypothesis.quality == "DENSE_STRONG":
            if tracker_agrees:
                output_class = "FINAL"
                confidence = "VERY_HIGH"
                fusion_reason = (
                    "CLASS_POA_TRACKER_AGREE"
                )

            elif hypothesis.bundle_confirmed:
                output_class = "FINAL"
                confidence = "HIGH"
                fusion_reason = (
                    "CLASS_CONFIRMED_BUNDLE"
                )

            else:
                output_class = "REVIEW"
                confidence = "MEDIUM"
                fusion_reason = (
                    "CLASS_DENSE_POA_NEEDS_TRACKER"
                )

        elif hypothesis.quality == "SPARSE_STRONG":
            if tracker_agrees:
                output_class = "FINAL"
                confidence = "HIGH"
                fusion_reason = (
                    "CLASS_SPARSE_POA_TRACKER_AGREE"
                )
            else:
                output_class = "REVIEW"
                confidence = "MEDIUM"
                fusion_reason = (
                    "CLASS_SPARSE_POA_NEEDS_TRACKER"
                )

        elif hypothesis.quality == "REVIEW":
            output_class = "REVIEW"
            confidence = "MEDIUM"
            fusion_reason = "CLASS_POA_REVIEW"

        else:
            output_class = "REJECT"
            confidence = "LOW"
            fusion_reason = "CLASS_POA_REJECT"

        record = {
            "span_id":
                span.span_id,
            "layer_name":
                span.layer_name,
            "class_code":
                class_code,
            "class_name":
                config[
                    "class_name"
                ],
            "class_engine":
                config[
                    "engine_key"
                ],
            "source":
                "POA",
            "engine_mode":
                family,
            "pass_name":
                (
                    f"CLASS_{class_code}_"
                    f"{config['engine_key']}"
                ),
            "source_id":
                hypothesis.hypothesis_id,
            "confidence":
                confidence,
            "fusion_reason":
                fusion_reason,
            "v7_agreement_ft":
                (
                    float(
                        nearest_tracker_distance
                    )
                    if np.isfinite(
                        nearest_tracker_distance
                    )
                    else np.nan
                ),
            "poa_quality":
                hypothesis.quality,
            "coverage":
                hypothesis.owned_coverage,
            "median_residual_ft":
                hypothesis.owned_median_residual_ft,
            "p90_residual_ft":
                hypothesis.owned_p90_residual_ft,
            "support_points":
                hypothesis.owned_points,
            "support_zones":
                hypothesis.support_zone_count,
            "longest_gap_ft":
                hypothesis.longest_unsupported_gap_ft,
            "longest_gap_fraction":
                hypothesis.longest_unsupported_gap_fraction,
            "terminal_spread_a_ft":
                hypothesis.terminal_spread_a_ft,
            "terminal_spread_b_ft":
                hypothesis.terminal_spread_b_ft,
            "terminal_spread_ft":
                hypothesis.terminal_spread_ft,
            "bundle_confirmed":
                hypothesis.bundle_confirmed,
            "adaptive_radius_ft":
                hypothesis.adaptive_radius_ft,
            "adaptive_median_limit_ft":
                hypothesis.adaptive_median_limit_ft,
            "adaptive_p90_limit_ft":
                hypothesis.adaptive_p90_limit_ft,
            "sag_depth_ft":
                hypothesis.sag_depth_ft,
            "locked":
                bool(
                    tracker_agrees
                    or hypothesis.bundle_confirmed
                ),
            "geometry":
                candidate[
                    "geometry"
                ],
        }

        if output_class == "FINAL":
            final_records.append(
                record
            )

            if (
                tracker_agrees
                and nearest_tracker_index
                is not None
            ):
                used_tracker.add(
                    nearest_tracker_index
                )

        elif output_class == "REVIEW":
            review_records.append(
                record
            )

        else:
            rejected_records.append(
                record
            )

    # Clean class-specific tracker wires may fill a POA hole. Only FINAL POA
    # geometry can block them; REVIEW geometry cannot consume a valid track.
    if ALLOW_V7_ONLY_FINAL:
        for index, tracker_wire in enumerate(
            tracker_result[
                "wires"
            ]
        ):
            if index in used_tracker:
                continue

            if tracker_wire[
                "status"
            ] != "AUTO_ACCEPT":
                continue

            duplicate = False

            for record in final_records:
                distance = mean_line_distance_3d(
                    tracker_wire[
                        "geometry"
                    ],
                    record[
                        "geometry"
                    ],
                )

                if distance <= duplicate_distance:
                    duplicate = True
                    break

            if duplicate:
                continue

            final_records.append(
                {
                    "span_id":
                        span.span_id,
                    "layer_name":
                        span.layer_name,
                    "class_code":
                        class_code,
                    "class_name":
                        config[
                            "class_name"
                        ],
                    "class_engine":
                        config[
                            "engine_key"
                        ],
                    "source":
                        "CLASS_TRACKER",
                    "engine_mode":
                        family,
                    "pass_name":
                        (
                            f"CLASS_{class_code}_"
                            f"TRACKER"
                        ),
                    "source_id":
                        tracker_wire[
                            "track_id"
                        ],
                    "confidence":
                        "HIGH"
                        if tracker_wire[
                            "accept_reason"
                        ]
                        == "CLASS_NORMAL_COVERAGE"
                        else "MEDIUM_HIGH",
                    "fusion_reason":
                        "CLASS_TRACKER_ONLY_AUTO_ACCEPT",
                    "v7_agreement_ft":
                        0.0,
                    "poa_quality":
                        "",
                    "coverage":
                        tracker_wire[
                            "coverage_pct"
                        ]
                        / 100.0,
                    "v7_accept_reason":
                        tracker_wire[
                            "accept_reason"
                        ],
                    "v7_observed_sections":
                        tracker_wire[
                            "observed_sections"
                        ],
                    "v7_support_extent_fraction":
                        tracker_wire[
                            "support_extent_fraction"
                        ],
                    "v7_longest_gap_fraction":
                        tracker_wire[
                            "longest_gap_fraction"
                        ],
                    "tracker_fragment_count":
                        tracker_wire[
                            "fragment_count"
                        ],
                    "tracker_stitched":
                        tracker_wire[
                            "stitched"
                        ],
                    "tracker_t_rmse_ft":
                        tracker_wire[
                            "t_rmse_ft"
                        ],
                    "tracker_z_rmse_ft":
                        tracker_wire[
                            "body_rmse_ft"
                        ],
                    "median_residual_ft":
                        np.nan,
                    "p90_residual_ft":
                        np.nan,
                    "support_points":
                        np.nan,
                    "sag_depth_ft":
                        np.nan,
                    "bundle_confirmed":
                        False,
                    "locked":
                        False,
                    "geometry":
                        tracker_wire[
                            "geometry"
                        ],
                }
            )

    final_records = collapse_final_bundle_records(
        span,
        class_code,
        final_records,
    )

    review_records = collapse_final_bundle_records(
        span,
        class_code,
        review_records,
    )

    return (
        final_records,
        review_records,
        rejected_records,
        unique_poa,
    )


def _empty_class_span_result(
    span: DXFSpan,
    class_code: int,
    config: dict,
    status: str,
):
    return {
        "span_report": {
            "span_id":
                span.span_id,
            "dxf_layer":
                span.layer_name,
            "node_a":
                span.node_a,
            "node_b":
                span.node_b,
            "span_length_ft":
                span.span_length_ft,
            "class_code":
                class_code,
            "class_name":
                config[
                    "class_name"
                ],
            "class_engine":
                config[
                    "engine_key"
                ],
            "mode":
                config[
                    "family"
                ],
            "detected_mode":
                config.get(
                    "detected_mode",
                    config[
                        "family"
                    ],
                ),
            "terminal_spacing_ft":
                config.get(
                    "terminal_spacing_ft",
                    np.nan,
                ),
            "wire_points":
                0,
            "poa_selected":
                0,
            "poa_candidate_wires":
                0,
            "sparse_final_wires":
                0,
            "v7_tracks":
                0,
            "v7_tracks_after_stitch":
                0,
            "v7_stitch_actions":
                0,
            "v7_tracks_ok":
                0,
            "v7_final":
                0,
            "final_wires":
                0,
            "review_wires":
                0,
            "rejected_wires":
                0,
            "status":
                status,
            "geometry":
                span.geometry,
        },
        "terminal_records": [],
        "body_records": [],
        "hypothesis_records": [],
        "poa_records": [],
        "v7_records": [],
        "stitch_records": [],
        "final_records": [],
        "review_records": [],
        "rejected_records": [],
    }


def process_span_class(
    span: DXFSpan,
    class_code: int,
    wire_xyz: np.ndarray,
    wire_tree_xy: cKDTree,
):
    base_config = CLASS_ENGINE_CONFIG[
        class_code
    ]

    indices = span_candidate_indices(
        span,
        wire_xyz,
        wire_tree_xy,
        half_width_ft=base_config[
            "query_half_width_ft"
        ],
    )

    if len(indices) == 0:
        return _empty_class_span_result(
            span,
            class_code,
            base_config,
            "NO_CLASS_POINTS",
        )

    frame = local_frame(
        span,
        wire_xyz[
            indices
        ],
    )

    frame = frame[
        (frame["s"] >= 0.0)
        & (
            frame["s"]
            <= span.span_length_ft
        )
    ].copy().reset_index(
        drop=True
    )

    if frame.empty:
        return _empty_class_span_result(
            span,
            class_code,
            base_config,
            "NO_CLASS_POINTS",
        )

    (
        detected_mode,
        terminal_spacing_ft,
        generic_a_count,
        generic_b_count,
    ) = classify_span_mode(
        span,
        frame,
    )

    if detected_mode == "DX":
        family = "DX"
    elif detected_mode == "TX":
        family = "TX"
    else:
        # Ambiguous geometry: short spans favour the tighter DX model, while
        # longer spans favour the TX model. The LiDAR class itself never decides.
        family = (
            "DX"
            if span.span_length_ft <= 215.0
            else "TX"
        )

    params = dict(
        DX
        if family == "DX"
        else TX
    )
    params["name"] = (
        f"C{class_code}_{family}"
    )
    params["bundle_enabled"] = bool(
        ENABLE_BUNDLE_COLLAPSE
        or family == "TX"
    )

    if ENABLE_BUNDLE_COLLAPSE:
        # Fit one model through the bundle centroid while allowing the original
        # points to occupy the physical cross-section around that centreline.
        params["ownership_tube_ft"] = max(
            float(params["ownership_tube_ft"]),
            BUNDLE_OWNERSHIP_RADIUS_FT,
        )
        params["tight_tube_ft"] = max(
            float(params["tight_tube_ft"]),
            BUNDLE_OWNERSHIP_RADIUS_FT,
        )
        params["loose_tube_ft"] = max(
            float(params["loose_tube_ft"]),
            BUNDLE_LOOSE_RADIUS_FT,
        )
        params["body_lateral_gate_ft"] = max(
            float(params["body_lateral_gate_ft"]),
            BUNDLE_LOOSE_RADIUS_FT,
        )
        params["max_median_residual_ft"] = max(
            float(params["max_median_residual_ft"]),
            BUNDLE_MEDIAN_RESIDUAL_LIMIT_FT,
        )
        params["max_p90_residual_ft"] = max(
            float(params["max_p90_residual_ft"]),
            BUNDLE_P90_RESIDUAL_LIMIT_FT,
        )
        params["max_adaptive_radius_ft"] = max(
            float(params["max_adaptive_radius_ft"]),
            BUNDLE_LOOSE_RADIUS_FT,
        )
        params["v7_agreement_ft"] = max(
            float(params["v7_agreement_ft"]),
            BUNDLE_TRACKER_AGREEMENT_FT,
        )

    config = {
        **base_config,
        "engine_key": f"AUTO_{family}",
        "family": family,
        "poa": params,
        "detected_mode": detected_mode,
        "terminal_spacing_ft": terminal_spacing_ft,
        "generic_terminal_a_count": generic_a_count,
        "generic_terminal_b_count": generic_b_count,
    }

    poa_run = run_poa_engine(
        span,
        frame,
        params,
        pass_name=(
            f"CLASS_{class_code}_"
            f"{config['engine_key']}"
        ),
    )

    tracker_result = run_class_tracker_engine(
        span,
        frame,
        class_code,
        family,
    )

    (
        final_records,
        review_records,
        rejected_records,
        unique_poa,
    ) = fuse_span_class_results(
        span,
        class_code,
        config,
        poa_run,
        tracker_result,
    )

    terminal_records = []
    body_records = []
    hypothesis_records = []
    poa_records = []
    v7_records = []

    for poa in (
        poa_run[
            "a_poas"
        ]
        + poa_run[
            "b_poas"
        ]
    ):
        s_value = (
            0.0
            if poa.end == "A"
            else span.span_length_ft
        )

        xyz = local_to_xyz(
            span,
            np.asarray(
                [s_value]
            ),
            np.asarray(
                [poa.t]
            ),
            np.asarray(
                [poa.z]
            ),
        )[0]

        terminal_records.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                "class_code":
                    class_code,
                "class_name":
                    config[
                        "class_name"
                    ],
                "class_engine":
                    config[
                        "engine_key"
                    ],
                "engine_mode":
                    config[
                        "family"
                    ],
                "pass_name":
                    poa_run[
                        "pass_name"
                    ],
                "terminal_id":
                    poa.terminal_id,
                "end":
                    poa.end,
                "point_count":
                    poa.point_count,
                "t":
                    poa.t,
                "z":
                    poa.z,
                "t_rmse":
                    poa.t_rmse,
                "z_rmse":
                    poa.z_rmse,
                "bundle_member_count":
                    poa.bundle_member_count,
                "geometry":
                    Point(
                        float(xyz[0]),
                        float(xyz[1]),
                        float(xyz[2]),
                    ),
            }
        )

    for cluster in poa_run[
        "body_clusters"
    ]:
        xyz = local_to_xyz(
            span,
            np.asarray(
                [cluster.s]
            ),
            np.asarray(
                [cluster.t]
            ),
            np.asarray(
                [cluster.z]
            ),
        )[0]

        body_records.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                "class_code":
                    class_code,
                "class_name":
                    config[
                        "class_name"
                    ],
                "class_engine":
                    config[
                        "engine_key"
                    ],
                "engine_mode":
                    config[
                        "family"
                    ],
                "pass_name":
                    poa_run[
                        "pass_name"
                    ],
                "cluster_id":
                    cluster.cluster_id,
                "slice_index":
                    cluster.slice_index,
                "fraction":
                    cluster.fraction,
                "point_count":
                    cluster.point_count,
                "bundle_member_count":
                    cluster.bundle_member_count,
                "s":
                    cluster.s,
                "t":
                    cluster.t,
                "z":
                    cluster.z,
                "geometry":
                    Point(
                        float(xyz[0]),
                        float(xyz[1]),
                        float(xyz[2]),
                    ),
            }
        )

    selected_ids = {
        item.hypothesis_id
        for item in poa_run[
            "selected"
        ]
    }

    for hypothesis in poa_run[
        "hypotheses"
    ]:
        hypothesis_records.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                "class_code":
                    class_code,
                "class_name":
                    config[
                        "class_name"
                    ],
                "class_engine":
                    config[
                        "engine_key"
                    ],
                "engine_mode":
                    config[
                        "family"
                    ],
                "pass_name":
                    poa_run[
                        "pass_name"
                    ],
                "hypothesis_id":
                    hypothesis.hypothesis_id,
                "a_id":
                    hypothesis.a_id,
                "b_id":
                    hypothesis.b_id,
                "selected":
                    hypothesis.hypothesis_id
                    in selected_ids,
                "valid":
                    hypothesis.valid,
                "quality":
                    hypothesis.quality,
                "body_slice_hits":
                    hypothesis.body_slice_hits,
                "coverage_initial":
                    hypothesis.coverage,
                "median_residual_initial_ft":
                    hypothesis.median_residual_ft,
                "cost":
                    hypothesis.cost,
                "owned_points":
                    hypothesis.owned_points,
                "owned_coverage":
                    hypothesis.owned_coverage,
                "owned_median_residual_ft":
                    hypothesis.owned_median_residual_ft,
                "owned_p90_residual_ft":
                    hypothesis.owned_p90_residual_ft,
                "terminal_spread_a_ft":
                    hypothesis.terminal_spread_a_ft,
                "terminal_spread_b_ft":
                    hypothesis.terminal_spread_b_ft,
                "terminal_spread_ft":
                    hypothesis.terminal_spread_ft,
                "bundle_confirmed":
                    hypothesis.bundle_confirmed,
                "adaptive_radius_ft":
                    hypothesis.adaptive_radius_ft,
                "support_zones":
                    hypothesis.support_zone_count,
                "longest_gap_ft":
                    hypothesis.longest_unsupported_gap_ft,
                "longest_gap_fraction":
                    hypothesis.longest_unsupported_gap_fraction,
                "sag_depth_ft":
                    hypothesis.sag_depth_ft,
            }
        )

    for hypothesis in poa_run[
        "selected"
    ]:
        geometry = poa_result_geometry(
            span,
            hypothesis,
            params,
        )

        poa_records.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                "class_code":
                    class_code,
                "class_name":
                    config[
                        "class_name"
                    ],
                "class_engine":
                    config[
                        "engine_key"
                    ],
                "engine_mode":
                    config[
                        "family"
                    ],
                "pass_name":
                    poa_run[
                        "pass_name"
                    ],
                "hypothesis_id":
                    hypothesis.hypothesis_id,
                "quality":
                    hypothesis.quality,
                "owned_points":
                    hypothesis.owned_points,
                "coverage":
                    hypothesis.owned_coverage,
                "median_residual_ft":
                    hypothesis.owned_median_residual_ft,
                "p90_residual_ft":
                    hypothesis.owned_p90_residual_ft,
                "bundle_confirmed":
                    hypothesis.bundle_confirmed,
                "support_zones":
                    hypothesis.support_zone_count,
                "longest_gap_ft":
                    hypothesis.longest_unsupported_gap_ft,
                "longest_gap_fraction":
                    hypothesis.longest_unsupported_gap_fraction,
                "sag_depth_ft":
                    hypothesis.sag_depth_ft,
                "geometry":
                    geometry,
            }
        )

    for tracker_wire in tracker_result[
        "wires"
    ]:
        v7_records.append(
            {
                "span_id":
                    span.span_id,
                "layer_name":
                    span.layer_name,
                **tracker_wire,
            }
        )

    sparse_final_count = sum(
        1
        for record in final_records
        if record.get(
            "poa_quality",
            "",
        )
        == "SPARSE_STRONG"
    )

    unresolved = (
        len(review_records)
        + len(rejected_records)
    )

    if final_records:
        status = (
            "COMPLETE"
            if unresolved == 0
            else "PARTIAL"
        )
    elif review_records:
        status = "REVIEW"
    else:
        status = "NO_WIRES"

    return {
        "span_report": {
            "span_id":
                span.span_id,
            "dxf_layer":
                span.layer_name,
            "node_a":
                span.node_a,
            "node_b":
                span.node_b,
            "span_length_ft":
                span.span_length_ft,
            "class_code":
                class_code,
            "class_name":
                config[
                    "class_name"
                ],
            "class_engine":
                config[
                    "engine_key"
                ],
            "mode":
                config[
                    "family"
                ],
            "wire_points":
                len(frame),
            "poa_selected":
                len(
                    poa_run[
                        "selected"
                    ]
                ),
            "poa_candidate_wires":
                len(unique_poa),
            "sparse_final_wires":
                sparse_final_count,
            "v7_tracks":
                tracker_result[
                    "tracks_total"
                ],
            "v7_tracks_after_stitch":
                tracker_result[
                    "tracks_after_stitch"
                ],
            "v7_stitch_actions":
                len(
                    tracker_result[
                        "stitch_actions"
                    ]
                ),
            "v7_tracks_ok":
                tracker_result[
                    "tracks_ok"
                ],
            "v7_final":
                len(
                    tracker_result[
                        "wires"
                    ]
                ),
            "final_wires":
                len(final_records),
            "bundle_final_vectors":
                int(
                    sum(
                        bool(
                            record.get(
                                "bundle_collapsed",
                                False,
                            )
                            or record.get(
                                "bundle_confirmed",
                                False,
                            )
                        )
                        for record in final_records
                    )
                ),
            "review_wires":
                len(review_records),
            "rejected_wires":
                len(rejected_records),
            "status":
                status,
            "geometry":
                span.geometry,
        },
        "terminal_records":
            terminal_records,
        "body_records":
            body_records,
        "hypothesis_records":
            hypothesis_records,
        "poa_records":
            poa_records,
        "v7_records":
            v7_records,
        "stitch_records":
            tracker_result[
                "stitch_actions"
            ],
        "final_records":
            final_records,
        "review_records":
            review_records,
        "rejected_records":
            rejected_records,
    }

# =============================================================================
# ONE SPAN
# =============================================================================

def process_span(
    span: DXFSpan,
    wire_xyz: np.ndarray,
    wire_tree_xy: cKDTree,
    tracker,
):
    indices = span_candidate_indices(
        span,
        wire_xyz,
        wire_tree_xy,
    )

    if len(indices) == 0:
        return {
            "span_report": {
                "span_id": span.span_id,
                "span_length_ft": span.span_length_ft,
                "mode": "NONE",
                "terminal_spacing_ft": np.nan,
                "wire_points": 0,
                "tx_claimed_points": 0,
                "underbuild_residual_points": 0,
                "poa_selected": 0,
                "poa_candidate_wires": 0,
                "sparse_final_wires": 0,
                "v7_tracks": 0,
                "v7_final": 0,
                "final_wires": 0,
                "review_wires": 0,
                "status": "NO_WIRE_POINTS",
                "geometry": span.geometry,
            },
            "terminal_records": [],
            "body_records": [],
            "hypothesis_records": [],
            "poa_records": [],
            "v7_records": [],
            "final_records": [],
            "review_records": [],
            "rejected_records": [],
        }

    frame = local_frame(
        span,
        wire_xyz[indices],
    )

    frame = frame[
        (frame["s"] >= 0.0)
        & (
            frame["s"]
            <= span.span_length_ft
        )
    ].copy().reset_index(drop=True)

    if frame.empty:
        return {
            "span_report": {
                "span_id": span.span_id,
                "span_length_ft": span.span_length_ft,
                "mode": "NONE",
                "terminal_spacing_ft": np.nan,
                "wire_points": 0,
                "tx_claimed_points": 0,
                "underbuild_residual_points": 0,
                "poa_selected": 0,
                "poa_candidate_wires": 0,
                "sparse_final_wires": 0,
                "v7_tracks": 0,
                "v7_final": 0,
                "final_wires": 0,
                "review_wires": 0,
                "status": "NO_WIRE_POINTS",
                "geometry": span.geometry,
            },
            "terminal_records": [],
            "body_records": [],
            "hypothesis_records": [],
            "poa_records": [],
            "v7_records": [],
            "final_records": [],
            "review_records": [],
            "rejected_records": [],
        }

    (
        mode,
        terminal_spacing,
        generic_a_count,
        generic_b_count,
    ) = classify_span_mode(
        span,
        frame,
    )

    poa_runs = []

    tx_claimed_points = 0
    underbuild_residual_points = 0

    if mode == "DX":
        poa_runs.append(
            run_poa_engine(
                span,
                frame,
                DX,
                pass_name="DX_MAIN",
            )
        )

    else:
        # TX and HYBRID spans first solve the large/main conductor system.
        tx_run = run_poa_engine(
            span,
            frame,
            TX,
            pass_name="TX_MAIN",
        )

        poa_runs.append(
            tx_run
        )

        if ENABLE_TX_RESIDUAL_DX_PASS:
            (
                tx_claimed,
                residual_frame,
            ) = claim_points_explained_by_strong_tx(
                span,
                frame,
                tx_run,
            )

            tx_claimed_points = int(
                len(tx_claimed)
            )

            underbuild_residual_points = int(
                len(residual_frame)
            )

            if (
                len(residual_frame)
                >= UNDERBUILD_MIN_RESIDUAL_POINTS
            ):
                dx_underbuild_run = run_poa_engine(
                    span,
                    residual_frame,
                    DX,
                    pass_name="DX_UNDERBUILD",
                )

                dx_underbuild_run = validate_underbuild_run(
                    span,
                    tx_run,
                    dx_underbuild_run,
                )

                poa_runs.append(
                    dx_underbuild_run
                )

        elif mode == "HYBRID":
            poa_runs.append(
                run_poa_engine(
                    span,
                    frame,
                    DX,
                    pass_name="DX_MAIN",
                )
            )

    v7_result = run_v7_engine(
        span,
        frame,
        tracker,
    )

    (
        final_records,
        review_records,
        rejected_records,
        unique_poa,
    ) = fuse_span_results(
        span,
        poa_runs,
        v7_result,
    )

    terminal_records = []
    body_records = []
    hypothesis_records = []
    poa_records = []
    v7_records = []

    for run in poa_runs:
        params = (
            DX
            if run["mode"] == "DX"
            else TX
        )

        for poa in (
            run["a_poas"]
            + run["b_poas"]
        ):
            s_value = (
                0.0
                if poa.end == "A"
                else span.span_length_ft
            )

            xyz = local_to_xyz(
                span,
                np.asarray([s_value]),
                np.asarray([poa.t]),
                np.asarray([poa.z]),
            )[0]

            terminal_records.append(
                {
                    "span_id":
                        span.span_id,

                    "layer_name":
                        span.layer_name,

                    "engine_mode":
                        run["mode"],

                    "pass_name":
                        run[
                            "pass_name"
                        ],

                    "terminal_id":
                        poa.terminal_id,

                    "end":
                        poa.end,

                    "point_count":
                        poa.point_count,

                    "t":
                        poa.t,

                    "z":
                        poa.z,

                    "t_rmse":
                        poa.t_rmse,

                    "z_rmse":
                        poa.z_rmse,

                    "geometry":
                        Point(
                            float(xyz[0]),
                            float(xyz[1]),
                            float(xyz[2]),
                        ),
                }
            )

        for cluster in run[
            "body_clusters"
        ]:
            xyz = local_to_xyz(
                span,
                np.asarray(
                    [cluster.s]
                ),
                np.asarray(
                    [cluster.t]
                ),
                np.asarray(
                    [cluster.z]
                ),
            )[0]

            body_records.append(
                {
                    "span_id":
                        span.span_id,

                    "engine_mode":
                        run["mode"],

                    "pass_name":
                        run[
                            "pass_name"
                        ],

                    "cluster_id":
                        cluster.cluster_id,

                    "slice_index":
                        cluster.slice_index,

                    "fraction":
                        cluster.fraction,

                    "point_count":
                        cluster.point_count,

                    "s":
                        cluster.s,

                    "t":
                        cluster.t,

                    "z":
                        cluster.z,

                    "geometry":
                        Point(
                            float(xyz[0]),
                            float(xyz[1]),
                            float(xyz[2]),
                        ),
                }
            )

        selected_ids = {
            item.hypothesis_id
            for item in run[
                "selected"
            ]
        }

        for hypothesis in run[
            "hypotheses"
        ]:
            hypothesis_records.append(
                {
                    "span_id":
                        span.span_id,

                    "engine_mode":
                        run["mode"],

                    "pass_name":
                        run[
                            "pass_name"
                        ],

                    "hypothesis_id":
                        hypothesis.hypothesis_id,

                    "a_id":
                        hypothesis.a_id,

                    "b_id":
                        hypothesis.b_id,

                    "selected":
                        hypothesis.hypothesis_id
                        in selected_ids,

                    "valid":
                        hypothesis.valid,

                    "quality":
                        hypothesis.quality,

                    "body_slice_hits":
                        hypothesis.body_slice_hits,

                    "coverage_initial":
                        hypothesis.coverage,

                    "median_residual_initial_ft":
                        hypothesis.median_residual_ft,

                    "cost":
                        hypothesis.cost,

                    "owned_points":
                        hypothesis.owned_points,

                    "owned_coverage":
                        hypothesis.owned_coverage,

                    "owned_median_residual_ft":
                        hypothesis.owned_median_residual_ft,

                    "owned_p90_residual_ft":
                        hypothesis.owned_p90_residual_ft,

                    "terminal_spread_a_ft":
                        hypothesis.terminal_spread_a_ft,

                    "terminal_spread_b_ft":
                        hypothesis.terminal_spread_b_ft,

                    "terminal_spread_ft":
                        hypothesis.terminal_spread_ft,

                    "bundle_confirmed":
                        hypothesis.bundle_confirmed,

                    "underbuild_below_fraction":
                        hypothesis.underbuild_below_fraction,

                    "underbuild_spatial_ok":
                        hypothesis.underbuild_spatial_ok,

                    "underbuild_group_size":
                        hypothesis.underbuild_group_size,

                    "adaptive_radius_ft":
                        hypothesis.adaptive_radius_ft,

                    "adaptive_median_limit_ft":
                        hypothesis.adaptive_median_limit_ft,

                    "adaptive_p90_limit_ft":
                        hypothesis.adaptive_p90_limit_ft,

                    "support_zones":
                        hypothesis.support_zone_count,

                    "longest_gap_ft":
                        hypothesis.longest_unsupported_gap_ft,

                    "longest_gap_fraction":
                        hypothesis.longest_unsupported_gap_fraction,

                    "sag_depth_ft":
                        hypothesis.sag_depth_ft,
                }
            )

        for hypothesis in run[
            "selected"
        ]:
            geometry = poa_result_geometry(
                span,
                hypothesis,
                params,
            )

            poa_records.append(
                {
                    "span_id":
                        span.span_id,

                    "engine_mode":
                        run["mode"],

                    "pass_name":
                        run[
                            "pass_name"
                        ],

                    "hypothesis_id":
                        hypothesis.hypothesis_id,

                    "a_id":
                        hypothesis.a_id,

                    "b_id":
                        hypothesis.b_id,

                    "quality":
                        hypothesis.quality,

                    "owned_points":
                        hypothesis.owned_points,

                    "coverage":
                        hypothesis.owned_coverage,

                    "median_residual_ft":
                        hypothesis.owned_median_residual_ft,

                    "p90_residual_ft":
                        hypothesis.owned_p90_residual_ft,

                    "terminal_spread_a_ft":
                        hypothesis.terminal_spread_a_ft,

                    "terminal_spread_b_ft":
                        hypothesis.terminal_spread_b_ft,

                    "terminal_spread_ft":
                        hypothesis.terminal_spread_ft,

                    "bundle_confirmed":
                        hypothesis.bundle_confirmed,

                    "underbuild_below_fraction":
                        hypothesis.underbuild_below_fraction,

                    "underbuild_spatial_ok":
                        hypothesis.underbuild_spatial_ok,

                    "underbuild_group_size":
                        hypothesis.underbuild_group_size,

                    "adaptive_radius_ft":
                        hypothesis.adaptive_radius_ft,

                    "support_zones":
                        hypothesis.support_zone_count,

                    "longest_gap_ft":
                        hypothesis.longest_unsupported_gap_ft,

                    "longest_gap_fraction":
                        hypothesis.longest_unsupported_gap_fraction,

                    "sag_depth_ft":
                        hypothesis.sag_depth_ft,

                    "geometry":
                        geometry,
                }
            )

    for v7 in v7_result[
        "wires"
    ]:
        v7_records.append(
            {
                "span_id":
                    span.span_id,

                **v7,
            }
        )

    candidate_wire_count = int(
        len(
            unique_poa
        )
    )

    sparse_final_count = sum(
        1
        for record in final_records
        if record.get(
            "poa_quality",
            "",
        ) == "SPARSE_STRONG"
    )

    unresolved_selected = (
        len(
            review_records
        )
        + len(
            rejected_records
        )
    )

    if final_records:
        if unresolved_selected == 0:
            status = (
                "SPARSE_COMPLETE"
                if sparse_final_count
                else "COMPLETE"
            )
        else:
            status = "PARTIAL"

    elif review_records:
        status = "REVIEW"

    else:
        status = "NO_WIRES"

    return {
        "span_report": {
            "span_id":
                span.span_id,

            "dxf_layer":
                span.layer_name,

            "node_a":
                span.node_a,

            "node_b":
                span.node_b,

            "span_length_ft":
                span.span_length_ft,

            "mode":
                mode,

            "terminal_spacing_ft":
                terminal_spacing,

            "generic_terminal_clusters_a":
                generic_a_count,

            "generic_terminal_clusters_b":
                generic_b_count,

            "wire_points":
                len(frame),

            "tx_claimed_points":
                tx_claimed_points,

            "underbuild_residual_points":
                underbuild_residual_points,

            "poa_selected":
                sum(
                    len(
                        run[
                            "selected"
                        ]
                    )
                    for run in poa_runs
                ),

            "poa_candidate_wires":
                candidate_wire_count,

            "sparse_final_wires":
                sparse_final_count,

            "v7_tracks":
                v7_result[
                    "tracks_total"
                ],

            "v7_tracks_ok":
                v7_result[
                    "tracks_ok"
                ],

            "v7_final":
                len(
                    v7_result[
                        "wires"
                    ]
                ),

            "final_wires":
                len(
                    final_records
                ),

            "review_wires":
                len(
                    review_records
                ),

            "rejected_wires":
                len(
                    rejected_records
                ),

            "status":
                status,

            "geometry":
                span.geometry,
        },

        "terminal_records":
            terminal_records,

        "body_records":
            body_records,

        "hypothesis_records":
            hypothesis_records,

        "poa_records":
            poa_records,

        "v7_records":
            v7_records,

        "final_records":
            final_records,

        "review_records":
            review_records,

        "rejected_records":
            rejected_records,
    }


# =============================================================================
# OUTPUT SUPPORT
# =============================================================================

def node_outputs(
    nodes: list[DXFNode],
    crs,
):
    records = []

    for node in nodes:
        records.append(
            {
                "node_id":
                    node.node_id,

                "source_vertex_count":
                    node.source_vertex_count,

                "nearest_structure_id":
                    node.nearest_structure_id,

                "nearest_structure_distance_ft":
                    node.nearest_structure_distance_ft,

                "nearest_structure_quality":
                    node.nearest_structure_quality,

                "geometry":
                    Point(
                        float(node.xy[0]),
                        float(node.xy[1]),
                    ),
            }
        )

    frame = make_gdf(
        records,
        crs,
    )

    return frame


def span_outputs(
    spans: list[DXFSpan],
    crs,
):
    records = []

    for span in spans:
        records.append(
            {
                "span_id":
                    span.span_id,

                "layer_name":
                    span.layer_name,

                "node_a":
                    span.node_a,

                "node_b":
                    span.node_b,

                "source_entity_no":
                    span.source_entity_no,

                "source_segment_no":
                    span.source_segment_no,

                "duplicate_count":
                    span.duplicate_count,

                "source_entities":
                    span.source_entities,

                "length_ft":
                    span.span_length_ft,

                "geometry":
                    span.geometry,
            }
        )

    return make_gdf(
        records,
        crs,
    )


def node_structure_links(
    nodes: list[DXFNode],
    structures: list[StructureQA],
    crs,
):
    structure_lookup = {
        item.structure_id:
            item
        for item in structures
    }

    records = []

    for node in nodes:
        if node.nearest_structure_id not in structure_lookup:
            continue

        structure = structure_lookup[
            node.nearest_structure_id
        ]

        records.append(
            {
                "node_id":
                    node.node_id,

                "nearest_structure_id":
                    node.nearest_structure_id,

                "distance_ft":
                    node.nearest_structure_distance_ft,

                "quality":
                    node.nearest_structure_quality,

                "geometry":
                    LineString(
                        [
                            (
                                float(node.xy[0]),
                                float(node.xy[1]),
                            ),
                            (
                                float(
                                    structure.anchor_xyz[0]
                                ),
                                float(
                                    structure.anchor_xyz[1]
                                ),
                            ),
                        ]
                    ),
            }
        )

    return make_gdf(
        records,
        crs,
    )


def point_sample_gdf(
    xyz: np.ndarray,
    maximum: int,
    classification_label: str,
    crs,
):
    if len(xyz) == 0:
        return make_gdf(
            [],
            crs,
        )

    indices = even_indices(
        len(xyz),
        maximum,
    )

    sample = xyz[
        indices
    ]

    records = []

    for x, y, z in sample:
        records.append(
            {
                "sample_type": classification_label,
                "z_source": float(z) * INTERNAL_FT_TO_SOURCE,
                "geometry": Point(
                    float(x),
                    float(y),
                    float(z),
                ),
            }
        )

    return make_gdf(
        records,
        crs,
    )


def write_layers(
    layers,
):
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    for layer_name, frame in layers:
        if (
            frame is None
            or frame.empty
        ):
            continue

        log(
            f"  writing {layer_name}: "
            f"{len(frame):,}"
        )

        frame.to_file(
            OUTPUT_GPKG,
            layer=layer_name,
            driver="GPKG",
            engine="pyogrio",
        )



# =============================================================================
# CANDIDATE VALIDATION SECOND PASS V2
# =============================================================================

def _cv_span_lookup(
    spans: list[DXFSpan],
):
    return {
        span.span_id: span
        for span in spans
    }


def _cv_incident_map(
    spans: list[DXFSpan],
):
    incident = {}

    for span in spans:
        incident.setdefault(
            span.node_a,
            [],
        ).append(
            span.span_id
        )

        incident.setdefault(
            span.node_b,
            [],
        ).append(
            span.span_id
        )

    return incident


def _cv_mode(
    span: DXFSpan,
    mode_by_span: dict[str, str],
):
    mode = safe_text(
        mode_by_span.get(
            span.span_id,
            ""
        )
    ).upper()

    if mode == "HYBRID":
        return (
            "TX"
            if span.span_length_ft >= 220.0
            else "DX"
        )

    if mode in {
        "DX",
        "TX",
    }:
        return mode

    return (
        "TX"
        if span.span_length_ft >= 220.0
        else "DX"
    )


def _cv_merge_distance(
    mode: str,
):
    return (
        CV_MERGE_DISTANCE_DX_FT
        if mode == "DX"
        else CV_MERGE_DISTANCE_TX_FT
    )


def _cv_final_duplicate_distance(
    mode: str,
):
    return (
        CV_FINAL_DUPLICATE_DISTANCE_DX_FT
        if mode == "DX"
        else CV_FINAL_DUPLICATE_DISTANCE_TX_FT
    )


def _cv_final_claim_tube(
    mode: str,
):
    return (
        CV_FINAL_CLAIM_TUBE_DX_FT
        if mode == "DX"
        else CV_FINAL_CLAIM_TUBE_TX_FT
    )


def _cv_candidate_tube(
    mode: str,
):
    return (
        CV_CANDIDATE_TUBE_DX_FT
        if mode == "DX"
        else CV_CANDIDATE_TUBE_TX_FT
    )


def _cv_margin_ratio(
    mode: str,
):
    return (
        CV_OWNERSHIP_MARGIN_RATIO_DX
        if mode == "DX"
        else CV_OWNERSHIP_MARGIN_RATIO_TX
    )


def _cv_promoted_min_separation(
    mode: str,
):
    return (
        CV_PROMOTED_MIN_SEPARATION_DX_FT
        if mode == "DX"
        else CV_PROMOTED_MIN_SEPARATION_TX_FT
    )


def _cv_geometry_coords_3d(
    geometry: LineString,
):
    if (
        geometry is None
        or geometry.is_empty
    ):
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    coords = np.asarray(
        list(
            geometry.coords
        ),
        dtype=np.float64,
    )

    if (
        coords.ndim != 2
        or len(coords) < 2
    ):
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    if coords.shape[1] >= 3:
        return coords[:, :3]

    return np.column_stack(
        (
            coords[:, :2],
            np.zeros(
                len(coords),
                dtype=np.float64,
            ),
        )
    )


def _cv_point_to_polyline_distance_3d(
    points_xyz: np.ndarray,
    geometry: LineString,
):
    """
    Exact point-to-segment distance in 3D, vectorised over points.

    This avoids the coarse sampled-KD-tree distance that can bias short DX
    conductors when several wires are close together.
    """
    points = np.asarray(
        points_xyz,
        dtype=np.float64,
    )

    coords = _cv_geometry_coords_3d(
        geometry
    )

    if (
        len(points) == 0
        or len(coords) < 2
    ):
        return np.full(
            len(points),
            np.inf,
            dtype=np.float64,
        )

    best = np.full(
        len(points),
        np.inf,
        dtype=np.float64,
    )

    for start, end in zip(
        coords[:-1],
        coords[1:],
    ):
        vector = (
            end - start
        )

        denominator = float(
            np.dot(
                vector,
                vector,
            )
        )

        if denominator <= 1e-12:
            distance = np.linalg.norm(
                points
                - start[
                    None,
                    :
                ],
                axis=1,
            )

        else:
            ratio = (
                (
                    points
                    - start[
                        None,
                        :
                    ]
                )
                @ vector
            ) / denominator

            ratio = np.clip(
                ratio,
                0.0,
                1.0,
            )

            closest = (
                start[
                    None,
                    :
                ]
                + ratio[
                    :,
                    None
                ]
                * vector[
                    None,
                    :
                ]
            )

            distance = np.linalg.norm(
                points
                - closest,
                axis=1,
            )

        best = np.minimum(
            best,
            distance,
        )

    return best


def _cv_is_trusted_first_pass(
    record: dict,
):
    return bool(
        record.get(
            "locked",
            False,
        )
        or safe_text(
            record.get(
                "confidence",
                ""
            )
        )
        == "VERY_HIGH"
        or bool(
            record.get(
                "bundle_confirmed",
                False,
            )
        )
    )


def _cv_record_endpoint(
    span: DXFSpan,
    record: dict,
    node_id: str,
):
    at_start = (
        span.node_a
        == node_id
    )

    return endpoint_xyz(
        record[
            "geometry"
        ],
        at_start,
    )


def _cv_same_layer(
    first: DXFSpan,
    second: DXFSpan,
):
    return (
        safe_text(
            first.layer_name
        ).strip().lower()
        ==
        safe_text(
            second.layer_name
        ).strip().lower()
    )


def _cv_neighbour_anchors(
    target_span: DXFSpan,
    node_id: str,
    incident: dict,
    span_lookup: dict[str, DXFSpan],
    first_final_by_span: dict[str, list[dict]],
):
    anchors = []

    for neighbour_span_id in incident.get(
        node_id,
        [],
    ):
        if (
            neighbour_span_id
            == target_span.span_id
        ):
            continue

        neighbour_span = span_lookup[
            neighbour_span_id
        ]

        if (
            CV_NETWORK_SAME_LAYER_ONLY
            and not _cv_same_layer(
                target_span,
                neighbour_span,
            )
        ):
            continue

        for record in first_final_by_span.get(
            neighbour_span_id,
            [],
        ):
            if not _cv_is_trusted_first_pass(
                record
            ):
                continue

            anchors.append(
                {
                    "neighbour_span_id":
                        neighbour_span_id,

                    "source_id":
                        safe_text(
                            record.get(
                                "source_id"
                            )
                        ),

                    "xyz":
                        _cv_record_endpoint(
                            neighbour_span,
                            record,
                            node_id,
                        ),
                }
            )

    # Different adjacent spans may describe the same physical attachment.
    unique = []

    for anchor in anchors:
        duplicate = False

        for existing in unique:
            if (
                np.linalg.norm(
                    anchor[
                        "xyz"
                    ]
                    - existing[
                        "xyz"
                    ]
                )
                <= 0.75
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(
                anchor
            )

    return unique


def _cv_network_metrics(
    span: DXFSpan,
    geometry: LineString,
    anchors_a: list[dict],
    anchors_b: list[dict],
    mode: str,
):
    start = endpoint_xyz(
        geometry,
        True,
    )

    end = endpoint_xyz(
        geometry,
        False,
    )

    gate = (
        CV_NETWORK_GATE_DX_FT
        if mode == "DX"
        else CV_NETWORK_GATE_TX_FT
    )

    best = None

    for endpoint_a, endpoint_b, reversed_flag in (
        (
            start,
            end,
            False,
        ),
        (
            end,
            start,
            True,
        ),
    ):
        if anchors_a:
            distance_a = min(
                float(
                    np.linalg.norm(
                        endpoint_a
                        - anchor[
                            "xyz"
                        ]
                    )
                )
                for anchor in anchors_a
            )
        else:
            distance_a = np.nan

        if anchors_b:
            distance_b = min(
                float(
                    np.linalg.norm(
                        endpoint_b
                        - anchor[
                            "xyz"
                        ]
                    )
                )
                for anchor in anchors_b
            )
        else:
            distance_b = np.nan

        finite = [
            value
            for value in (
                distance_a,
                distance_b,
            )
            if np.isfinite(
                value
            )
        ]

        score = (
            float(
                np.sum(
                    finite
                )
            )
            if finite
            else np.inf
        )

        if (
            best is None
            or score
            < best[
                "score"
            ]
        ):
            best = {
                "score":
                    score,

                "distance_a_ft":
                    distance_a,

                "distance_b_ft":
                    distance_b,

                "reversed":
                    reversed_flag,
            }

    if best is None:
        best = {
            "score":
                np.inf,

            "distance_a_ft":
                np.nan,

            "distance_b_ft":
                np.nan,

            "reversed":
                False,
        }

    a_ok = bool(
        np.isfinite(
            best[
                "distance_a_ft"
            ]
        )
        and best[
            "distance_a_ft"
        ]
        <= gate
    )

    b_ok = bool(
        np.isfinite(
            best[
                "distance_b_ft"
            ]
        )
        and best[
            "distance_b_ft"
        ]
        <= gate
    )

    if a_ok and b_ok:
        support = "BOTH"

    elif a_ok or b_ok:
        support = "ONE_SIDE"

    else:
        support = "NONE"

    best[
        "support"
    ] = support

    best[
        "both"
    ] = bool(
        a_ok
        and b_ok
    )

    best[
        "one_side"
    ] = bool(
        a_ok
        ^ b_ok
    )

    return best


def _cv_candidate_source_id(
    source_name: str,
    record: dict,
):
    for field in (
        "source_id",
        "hypothesis_id",
        "track_id",
    ):
        value = safe_text(
            record.get(
                field
            )
        )

        if value:
            return value

    return (
        f"{source_name}_"
        f"UNNAMED"
    )


def _cv_candidate_raw_records(
    span: DXFSpan,
    review_records: list[dict],
    poa_records: list[dict],
    v7_records: list[dict],
):
    raw = []

    sources = []

    if CV_USE_REVIEW:
        sources.append(
            (
                "REVIEW",
                review_records,
            )
        )

    if CV_USE_POA:
        sources.append(
            (
                "POA",
                poa_records,
            )
        )

    if CV_USE_V7:
        sources.append(
            (
                "V7",
                v7_records,
            )
        )

    for source_name, source_records in sources:
        for record in source_records:
            if (
                safe_text(
                    record.get(
                        "span_id"
                    )
                )
                != span.span_id
            ):
                continue

            geometry = record.get(
                "geometry"
            )

            if (
                geometry is None
                or geometry.is_empty
            ):
                continue

            candidate = dict(
                record
            )

            candidate[
                "_candidate_source"
            ] = source_name

            candidate[
                "_candidate_source_id"
            ] = _cv_candidate_source_id(
                source_name,
                candidate,
            )

            raw.append(
                candidate
            )

    return raw


def _cv_candidate_representation_rank(
    record: dict,
):
    source_name = safe_text(
        record.get(
            "_candidate_source"
        )
    )

    source_rank = {
        "POA": 0,
        "REVIEW": 1,
        "V7": 2,
    }.get(
        source_name,
        3,
    )

    quality = safe_text(
        record.get(
            "quality",
            record.get(
                "poa_quality",
                ""
            )
        )
    )

    quality_rank = {
        "DENSE_STRONG": 0,
        "SPARSE_STRONG": 1,
        "REVIEW": 2,
        "AUTO_ACCEPT": 2,
    }.get(
        quality,
        3,
    )

    residual = record.get(
        "median_residual_ft",
        np.nan,
    )

    residual_rank = (
        float(
            residual
        )
        if np.isfinite(
            residual
        )
        else 999.0
    )

    coverage = record.get(
        "coverage",
        np.nan,
    )

    if (
        not np.isfinite(
            coverage
        )
        and np.isfinite(
            record.get(
                "coverage_pct",
                np.nan,
            )
        )
    ):
        coverage = (
            float(
                record[
                    "coverage_pct"
                ]
            )
            / 100.0
        )

    coverage_rank = (
        -float(
            coverage
        )
        if np.isfinite(
            coverage
        )
        else 0.0
    )

    return (
        source_rank,
        quality_rank,
        residual_rank,
        coverage_rank,
    )


def _cv_group_candidates(
    span: DXFSpan,
    raw: list[dict],
    first_final: list[dict],
    mode: str,
):
    merge_distance = _cv_merge_distance(
        mode
    )

    final_duplicate_distance = (
        _cv_final_duplicate_distance(
            mode
        )
    )

    filtered = []

    for record in raw:
        if any(
            mean_line_distance_3d(
                record[
                    "geometry"
                ],
                final[
                    "geometry"
                ],
            )
            <= final_duplicate_distance
            for final in first_final
        ):
            continue

        filtered.append(
            record
        )

    filtered.sort(
        key=_cv_candidate_representation_rank
    )

    groups = []

    for record in filtered:
        best_group = None
        best_distance = np.inf

        for group in groups:
            distance = mean_line_distance_3d(
                record[
                    "geometry"
                ],
                group[
                    "geometry"
                ],
            )

            if (
                distance
                <= merge_distance
                and distance
                < best_distance
            ):
                best_group = group
                best_distance = distance

        if best_group is None:
            groups.append(
                {
                    "geometry":
                        record[
                            "geometry"
                        ],

                    "members":
                        [
                            record
                        ],
                }
            )

        else:
            best_group[
                "members"
            ].append(
                record
            )

            # Re-evaluate the representative so POA geometry is preferred when
            # available and a weaker first member cannot dominate the group.
            representative = min(
                best_group[
                    "members"
                ],
                key=_cv_candidate_representation_rank,
            )

            best_group[
                "geometry"
            ] = representative[
                "geometry"
            ]

    output = []

    for number, group in enumerate(
        groups,
        start=1,
    ):
        members = group[
            "members"
        ]

        sources = sorted(
            {
                safe_text(
                    member.get(
                        "_candidate_source"
                    )
                )
                for member in members
                if safe_text(
                    member.get(
                        "_candidate_source"
                    )
                )
            }
        )

        source_ids = sorted(
            {
                safe_text(
                    member.get(
                        "_candidate_source_id"
                    )
                )
                for member in members
                if safe_text(
                    member.get(
                        "_candidate_source_id"
                    )
                )
            }
        )

        poa_members = [
            member
            for member in members
            if safe_text(
                member.get(
                    "_candidate_source"
                )
            )
            in {
                "POA",
                "REVIEW",
            }
        ]

        v7_members = [
            member
            for member in members
            if safe_text(
                member.get(
                    "_candidate_source"
                )
            )
            == "V7"
        ]

        poa_v7_distance = np.nan

        if (
            poa_members
            and v7_members
        ):
            poa_v7_distance = min(
                mean_line_distance_3d(
                    poa[
                        "geometry"
                    ],
                    v7[
                        "geometry"
                    ],
                )
                for poa in poa_members
                for v7 in v7_members
            )

        qualities = [
            safe_text(
                member.get(
                    "quality",
                    member.get(
                        "poa_quality",
                        ""
                    )
                )
            )
            for member in members
        ]

        v7_statuses = [
            safe_text(
                member.get(
                    "status",
                    ""
                )
            )
            for member in v7_members
        ]

        representative = min(
            members,
            key=_cv_candidate_representation_rank,
        )

        output.append(
            {
                "candidate_id":
                    (
                        f"{span.span_id}_"
                        f"CV_{number:03d}"
                    ),

                "span_id":
                    span.span_id,

                "layer_name":
                    span.layer_name,

                "mode":
                    mode,

                "candidate_sources":
                    "|".join(
                        sources
                    ),

                "candidate_source_ids":
                    "|".join(
                        source_ids
                    ),

                "has_review":
                    "REVIEW"
                    in sources,

                "has_poa":
                    bool(
                        poa_members
                    ),

                "has_v7":
                    bool(
                        v7_members
                    ),

                "evidence_count":
                    len(
                        sources
                    ),

                "poa_v7_distance_ft":
                    poa_v7_distance,

                "poa_quality_best":
                    (
                        min(
                            qualities,
                            key=lambda value:
                                {
                                    "DENSE_STRONG": 0,
                                    "SPARSE_STRONG": 1,
                                    "REVIEW": 2,
                                    "": 9,
                                }.get(
                                    value,
                                    8,
                                ),
                        )
                        if qualities
                        else ""
                    ),

                "v7_status_best":
                    (
                        "AUTO_ACCEPT"
                        if "AUTO_ACCEPT"
                        in v7_statuses
                        else (
                            "REVIEW"
                            if "REVIEW"
                            in v7_statuses
                            else ""
                        )
                    ),

                "representative_source":
                    safe_text(
                        representative.get(
                            "_candidate_source"
                        )
                    ),

                "geometry":
                    group[
                        "geometry"
                    ],

                "_members":
                    members,
            }
        )

    return output


def _cv_claim_first_pass_points(
    frame: pd.DataFrame,
    first_final: list[dict],
    mode: str,
):
    if (
        frame.empty
        or not first_final
    ):
        return (
            np.zeros(
                len(
                    frame
                ),
                dtype=bool,
            ),
            np.full(
                len(
                    frame
                ),
                np.inf,
                dtype=np.float64,
            ),
        )

    xyz = frame[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    best = np.full(
        len(
            frame
        ),
        np.inf,
        dtype=np.float64,
    )

    base_tube = _cv_final_claim_tube(
        mode
    )

    claimed = np.zeros(
        len(
            frame
        ),
        dtype=bool,
    )

    for record in first_final:
        distance = _cv_point_to_polyline_distance_3d(
            xyz,
            record[
                "geometry"
            ],
        )

        best = np.minimum(
            best,
            distance,
        )

        tube = base_tube

        if bool(
            record.get(
                "bundle_confirmed",
                False,
            )
        ):
            adaptive = record.get(
                "adaptive_radius_ft",
                np.nan,
            )

            if np.isfinite(
                adaptive
            ):
                tube = max(
                    tube,
                    min(
                        float(
                            adaptive
                        ),
                        1.70,
                    ),
                )

        claimed |= (
            distance
            <= tube
        )

    return (
        claimed,
        best,
    )


def _cv_body_slice_hits(
    owned: pd.DataFrame,
    span: DXFSpan,
    mode: str,
):
    if owned.empty:
        return 0

    params = (
        DX
        if mode == "DX"
        else TX
    )

    minimum_points = (
        2
        if mode == "DX"
        else 3
    )

    half_width = (
        params[
            "body_slice_width_ft"
        ]
        / 2.0
    )

    hits = 0

    for fraction in params[
        "body_fractions"
    ]:
        centre = (
            fraction
            * span.span_length_ft
        )

        count = int(
            np.count_nonzero(
                (
                    owned[
                        "s"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    >= centre
                    - half_width
                )
                & (
                    owned[
                        "s"
                    ].to_numpy(
                        dtype=np.float64
                    )
                    <= centre
                    + half_width
                )
            )
        )

        if count >= minimum_points:
            hits += 1

    return hits



def _cv_line_mean_xy_distance(
    first: LineString,
    second: LineString,
):
    first_samples = line_xyz_samples(
        first,
        count=31,
    )
    second_samples = line_xyz_samples(
        second,
        count=31,
    )

    if (
        len(first_samples) == 0
        or len(second_samples) == 0
    ):
        return float("inf")

    direct = float(
        np.mean(
            np.linalg.norm(
                first_samples[:, :2]
                - second_samples[:, :2],
                axis=1,
            )
        )
    )

    reverse = float(
        np.mean(
            np.linalg.norm(
                first_samples[:, :2]
                - second_samples[::-1, :2],
                axis=1,
            )
        )
    )

    return min(
        direct,
        reverse,
    )


def _cv_line_direction_difference_deg(
    first: LineString,
    second: LineString,
):
    first_coords = np.asarray(
        list(first.coords),
        dtype=np.float64,
    )
    second_coords = np.asarray(
        list(second.coords),
        dtype=np.float64,
    )

    if (
        len(first_coords) < 2
        or len(second_coords) < 2
    ):
        return 180.0

    first_vector = (
        first_coords[-1, :2]
        - first_coords[0, :2]
    )
    second_vector = (
        second_coords[-1, :2]
        - second_coords[0, :2]
    )

    first_length = float(
        np.linalg.norm(first_vector)
    )
    second_length = float(
        np.linalg.norm(second_vector)
    )

    if (
        first_length <= 1e-9
        or second_length <= 1e-9
    ):
        return 180.0

    cosine = float(
        np.clip(
            abs(
                np.dot(
                    first_vector / first_length,
                    second_vector / second_length,
                )
            ),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def _cv_span_station_extent(
    span: DXFSpan,
    geometry: LineString,
):
    samples = line_xyz_samples(
        geometry,
        count=31,
    )

    if len(samples) == 0:
        return (
            np.nan,
            np.nan,
        )

    relative = (
        samples[:, :2]
        - span.axis_origin_xy[
            None,
            :
        ]
    )

    station = (
        relative
        @ span.axis_u
    )

    station = np.clip(
        station,
        0.0,
        span.span_length_ft,
    )

    return (
        float(np.min(station)),
        float(np.max(station)),
    )


def _cv_longitudinal_overlap_fraction(
    span: DXFSpan,
    first: LineString,
    second: LineString,
):
    first_min, first_max = _cv_span_station_extent(
        span,
        first,
    )
    second_min, second_max = _cv_span_station_extent(
        span,
        second,
    )

    if not all(
        np.isfinite(value)
        for value in (
            first_min,
            first_max,
            second_min,
            second_max,
        )
    ):
        return 0.0

    first_length = max(
        0.0,
        first_max - first_min,
    )
    second_length = max(
        0.0,
        second_max - second_min,
    )

    denominator = min(
        first_length,
        second_length,
    )

    if denominator <= 1e-9:
        return 0.0

    overlap = max(
        0.0,
        min(
            first_max,
            second_max,
        )
        - max(
            first_min,
            second_min,
        ),
    )

    return float(
        np.clip(
            overlap / denominator,
            0.0,
            1.0,
        )
    )


def _cv_nearest_first_pass_xy_metrics(
    span: DXFSpan,
    geometry: LineString,
    first_final: list[dict],
    mode: str,
):
    best = {
        "nearest_final_wire_xy_ft":
            np.inf,

        "nearest_final_wire_xy_overlap":
            0.0,

        "nearest_final_wire_direction_diff_deg":
            np.nan,

        "nearest_final_xy_wire_3d_ft":
            np.inf,

        "nearest_final_xy_source_id":
            "",

        "possible_vertical_duplicate":
            False,
    }

    for final in first_final:
        final_geometry = final.get(
            "geometry"
        )

        if (
            final_geometry is None
            or final_geometry.is_empty
        ):
            continue

        mean_xy = _cv_line_mean_xy_distance(
            geometry,
            final_geometry,
        )

        if (
            mean_xy
            >= best[
                "nearest_final_wire_xy_ft"
            ]
        ):
            continue

        overlap = _cv_longitudinal_overlap_fraction(
            span,
            geometry,
            final_geometry,
        )

        direction_diff = _cv_line_direction_difference_deg(
            geometry,
            final_geometry,
        )

        mean_3d = mean_line_distance_3d(
            geometry,
            final_geometry,
        )

        best.update(
            {
                "nearest_final_wire_xy_ft":
                    float(mean_xy),

                "nearest_final_wire_xy_overlap":
                    float(overlap),

                "nearest_final_wire_direction_diff_deg":
                    float(direction_diff),

                "nearest_final_xy_wire_3d_ft":
                    float(mean_3d),

                "nearest_final_xy_source_id":
                    safe_text(
                        final.get(
                            "source_id"
                        )
                    ),
            }
        )

    xy_gate = (
        CV_VERTICAL_DUPLICATE_MAX_MEAN_XY_FT_DX
        if mode == "DX"
        else CV_VERTICAL_DUPLICATE_MAX_MEAN_XY_FT_TX
    )

    best[
        "possible_vertical_duplicate"
    ] = bool(
        np.isfinite(
            best[
                "nearest_final_wire_xy_ft"
            ]
        )
        and best[
            "nearest_final_wire_xy_ft"
        ]
        <= xy_gate

        and best[
            "nearest_final_wire_xy_overlap"
        ]
        >= CV_VERTICAL_DUPLICATE_MIN_OVERLAP

        and np.isfinite(
            best[
                "nearest_final_wire_direction_diff_deg"
            ]
        )
        and best[
            "nearest_final_wire_direction_diff_deg"
        ]
        <= CV_VERTICAL_DUPLICATE_MAX_DIRECTION_DIFF_DEG

        and np.isfinite(
            best[
                "nearest_final_xy_wire_3d_ft"
            ]
        )
        and best[
            "nearest_final_xy_wire_3d_ft"
        ]
        <= CV_VERTICAL_DUPLICATE_MAX_MEAN_3D_FT
    )

    return best


def _cv_dx_promotion_pass(
    metrics: dict,
):
    thresholds = CV_DX_PROMOTION

    required_finite = (
        "coverage",
        "median_residual_ft",
        "p90_residual_ft",
        "longest_gap_fraction",
        "unique_owned_fraction",
    )

    if not all(
        np.isfinite(
            metrics.get(
                field,
                np.nan,
            )
        )
        for field in required_finite
    ):
        return False

    if (
        int(metrics.get("owned_points", 0))
        < thresholds["min_owned_points"]
    ):
        return False

    if (
        float(metrics.get("coverage", 0.0))
        < thresholds["min_coverage"]
    ):
        return False

    if (
        float(
            metrics.get(
                "median_residual_ft",
                np.inf,
            )
        )
        > thresholds[
            "max_median_residual_ft"
        ]
    ):
        return False

    if (
        float(
            metrics.get(
                "p90_residual_ft",
                np.inf,
            )
        )
        > thresholds[
            "max_p90_residual_ft"
        ]
    ):
        return False

    if (
        float(
            metrics.get(
                "longest_gap_fraction",
                1.0,
            )
        )
        > thresholds[
            "max_gap_fraction"
        ]
    ):
        return False

    if (
        int(
            metrics.get(
                "terminal_a_points",
                0,
            )
        )
        < thresholds[
            "min_terminal_points_each"
        ]
        or int(
            metrics.get(
                "terminal_b_points",
                0,
            )
        )
        < thresholds[
            "min_terminal_points_each"
        ]
    ):
        return False

    if (
        int(
            metrics.get(
                "body_slice_hits",
                0,
            )
        )
        < thresholds[
            "min_body_slices"
        ]
    ):
        return False

    if (
        float(
            metrics.get(
                "unique_owned_fraction",
                0.0,
            )
        )
        < thresholds[
            "min_unique_fraction"
        ]
    ):
        return False

    if (
        CV_REQUIRE_BOTH_NETWORK_ENDS_FOR_DX_PROMOTION
        and not bool(
            metrics.get(
                "network_both",
                False,
            )
        )
    ):
        return False

    return True


def _cv_threshold_pass(
    metrics: dict,
    thresholds: dict,
):
    required_finite = (
        "coverage",
        "median_residual_ft",
        "p90_residual_ft",
        "longest_gap_fraction",
        "unique_owned_fraction",
    )

    if not all(
        np.isfinite(
            metrics.get(
                field,
                np.nan,
            )
        )
        for field in required_finite
    ):
        return False

    return bool(
        metrics[
            "owned_points"
        ]
        >= thresholds[
            "min_owned_points"
        ]

        and metrics[
            "coverage"
        ]
        >= thresholds[
            "min_coverage"
        ]

        and metrics[
            "median_residual_ft"
        ]
        <= thresholds[
            "max_median_residual_ft"
        ]

        and metrics[
            "p90_residual_ft"
        ]
        <= thresholds[
            "max_p90_residual_ft"
        ]

        and metrics[
            "longest_gap_fraction"
        ]
        <= thresholds[
            "max_gap_fraction"
        ]

        and metrics[
            "terminal_a_points"
        ]
        >= thresholds[
            "min_terminal_points_each"
        ]

        and metrics[
            "terminal_b_points"
        ]
        >= thresholds[
            "min_terminal_points_each"
        ]

        and metrics[
            "body_slice_hits"
        ]
        >= thresholds[
            "min_body_slices"
        ]

        and metrics[
            "unique_owned_fraction"
        ]
        >= thresholds[
            "min_unique_fraction"
        ]
    )


def _cv_validate_candidates_on_residual(
    span: DXFSpan,
    frame: pd.DataFrame,
    first_final: list[dict],
    candidates: list[dict],
    mode: str,
    anchors_a: list[dict],
    anchors_b: list[dict],
):
    support_point_records = []

    if not candidates:
        return (
            [],
            support_point_records,
            0,
            0,
        )

    (
        final_claimed,
        nearest_final_point_distance,
    ) = _cv_claim_first_pass_points(
        frame,
        first_final,
        mode,
    )

    residual = frame.loc[
        ~final_claimed
    ].copy()

    residual = residual.reset_index(
        drop=True
    )

    claimed_count = int(
        np.count_nonzero(
            final_claimed
        )
    )

    if residual.empty:
        validated = []

        for candidate in candidates:
            row = dict(
                candidate
            )

            row.update(
                {
                    "residual_points":
                        0,

                    "close_points":
                        0,

                    "owned_points":
                        0,

                    "ambiguous_best_points":
                        0,

                    "unique_owned_fraction":
                        0.0,

                    "coverage":
                        0.0,

                    "median_residual_ft":
                        np.inf,

                    "p90_residual_ft":
                        np.inf,

                    "support_zones":
                        0,

                    "longest_gap_ft":
                        span.span_length_ft,

                    "longest_gap_fraction":
                        1.0,

                    "terminal_a_points":
                        0,

                    "terminal_b_points":
                        0,

                    "body_slice_hits":
                        0,

                    "network_a_distance_ft":
                        np.nan,

                    "network_b_distance_ft":
                        np.nan,

                    "network_support":
                        "NONE",

                    "network_both":
                        False,

                    "nearest_final_wire_ft":
                        min(
                            (
                                mean_line_distance_3d(
                                    candidate[
                                        "geometry"
                                    ],
                                    final[
                                        "geometry"
                                    ],
                                )
                                for final in first_final
                            ),
                            default=np.inf,
                        ),

                    **_cv_nearest_first_pass_xy_metrics(
                        span,
                        candidate[
                            "geometry"
                        ],
                        first_final,
                        mode,
                    ),

                    "nearest_candidate_wire_ft":
                        np.inf,

                    "dx_crowded":
                        False,

                    "strong_residual":
                        False,

                    "exceptional_residual":
                        False,

                    "dx_promotion_gate":
                        False,
                }
            )

            validated.append(
                row
            )

        return (
            validated,
            support_point_records,
            claimed_count,
            0,
        )

    xyz = residual[
        [
            "x",
            "y",
            "z",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    tube = _cv_candidate_tube(
        mode
    )

    margin_required = _cv_margin_ratio(
        mode
    )

    distances = np.column_stack(
        [
            _cv_point_to_polyline_distance_3d(
                xyz,
                candidate[
                    "geometry"
                ],
            )
            for candidate in candidates
        ]
    )

    normalised = (
        distances
        / tube
    )

    best_index = np.argmin(
        normalised,
        axis=1,
    )

    best_distance = normalised[
        np.arange(
            len(
                residual
            )
        ),
        best_index,
    ]

    if len(
        candidates
    ) > 1:
        partitioned = np.partition(
            normalised,
            kth=1,
            axis=1,
        )

        second_distance = partitioned[
            :,
            1
        ]

    else:
        second_distance = np.full(
            len(
                residual
            ),
            np.inf,
            dtype=np.float64,
        )

    separation = np.full(
        len(
            residual
        ),
        np.inf,
        dtype=np.float64,
    )

    both_finite = (
        np.isfinite(
            best_distance
        )
        & np.isfinite(
            second_distance
        )
    )

    separation[
        both_finite
    ] = (
        second_distance[
            both_finite
        ]
        - best_distance[
            both_finite
        ]
    )

    validated = []

    for candidate_index, candidate in enumerate(
        candidates
    ):
        candidate_distance = distances[
            :,
            candidate_index
        ]

        close_mask = (
            candidate_distance
            <= tube
        )

        best_mask = (
            (
                best_index
                == candidate_index
            )
            & (
                best_distance
                <= 1.0
            )
        )

        unique_mask = (
            best_mask
            & (
                separation
                >= margin_required
            )
        )

        ambiguous_best_mask = (
            best_mask
            & (
                separation
                < margin_required
            )
        )

        owned = residual.loc[
            unique_mask
        ].copy()

        owned_distances = candidate_distance[
            unique_mask
        ]

        close_points = int(
            np.count_nonzero(
                close_mask
            )
        )

        owned_points = int(
            len(
                owned
            )
        )

        ambiguous_best_points = int(
            np.count_nonzero(
                ambiguous_best_mask
            )
        )

        unique_owned_fraction = (
            float(
                owned_points
                / close_points
            )
            if close_points > 0
            else 0.0
        )

        params = (
            DX
            if mode == "DX"
            else TX
        )

        if owned_points:
            owned_s = owned[
                "s"
            ].to_numpy(
                dtype=np.float64
            )

            coverage = coverage_from_s(
                owned_s,
                span.span_length_ft,
                params[
                    "coverage_bins"
                ],
            )

            (
                support_zones,
                longest_gap_ft,
                longest_gap_fraction,
            ) = support_distribution_metrics(
                owned_s,
                span.span_length_ft,
                params[
                    "coverage_bins"
                ],
            )

            median_residual = float(
                np.median(
                    owned_distances
                )
            )

            p90_residual = float(
                np.quantile(
                    owned_distances,
                    0.90,
                )
            )

            terminal_window = min(
                params[
                    "terminal_window_ft"
                ],
                max(
                    5.0,
                    0.25
                    * span.span_length_ft,
                ),
            )

            terminal_a_points = int(
                np.count_nonzero(
                    owned_s
                    <= terminal_window
                )
            )

            terminal_b_points = int(
                np.count_nonzero(
                    owned_s
                    >= span.span_length_ft
                    - terminal_window
                )
            )

            body_slice_hits = _cv_body_slice_hits(
                owned,
                span,
                mode,
            )

        else:
            coverage = 0.0
            support_zones = 0
            longest_gap_ft = (
                span.span_length_ft
            )
            longest_gap_fraction = 1.0
            median_residual = np.inf
            p90_residual = np.inf
            terminal_a_points = 0
            terminal_b_points = 0
            body_slice_hits = 0

        network = _cv_network_metrics(
            span,
            candidate[
                "geometry"
            ],
            anchors_a,
            anchors_b,
            mode,
        )

        nearest_final_wire = min(
            (
                mean_line_distance_3d(
                    candidate[
                        "geometry"
                    ],
                    final[
                        "geometry"
                    ],
                )
                for final in first_final
            ),
            default=np.inf,
        )

        nearest_candidate_wire = min(
            (
                mean_line_distance_3d(
                    candidate[
                        "geometry"
                    ],
                    other[
                        "geometry"
                    ],
                )
                for other_index, other
                in enumerate(
                    candidates
                )
                if other_index
                != candidate_index
            ),
            default=np.inf,
        )

        xy_duplicate_metrics = (
            _cv_nearest_first_pass_xy_metrics(
                span,
                candidate[
                    "geometry"
                ],
                first_final,
                mode,
            )
        )

        dx_crowded = bool(
            mode == "DX"
            and nearest_candidate_wire
            <= CV_DX_CROWDING_DISTANCE_FT
        )

        metrics = dict(
            candidate
        )

        metrics.update(
            {
                "residual_points":
                    len(
                        residual
                    ),

                "close_points":
                    close_points,

                "owned_points":
                    owned_points,

                "ambiguous_best_points":
                    ambiguous_best_points,

                "unique_owned_fraction":
                    unique_owned_fraction,

                "coverage":
                    float(
                        coverage
                    ),

                "median_residual_ft":
                    float(
                        median_residual
                    ),

                "p90_residual_ft":
                    float(
                        p90_residual
                    ),

                "support_zones":
                    int(
                        support_zones
                    ),

                "longest_gap_ft":
                    float(
                        longest_gap_ft
                    ),

                "longest_gap_fraction":
                    float(
                        longest_gap_fraction
                    ),

                "terminal_a_points":
                    int(
                        terminal_a_points
                    ),

                "terminal_b_points":
                    int(
                        terminal_b_points
                    ),

                "body_slice_hits":
                    int(
                        body_slice_hits
                    ),

                "network_a_distance_ft":
                    network[
                        "distance_a_ft"
                    ],

                "network_b_distance_ft":
                    network[
                        "distance_b_ft"
                    ],

                "network_support":
                    network[
                        "support"
                    ],

                "network_both":
                    bool(
                        network[
                            "both"
                        ]
                    ),

                "nearest_final_wire_ft":
                    float(
                        nearest_final_wire
                    ),

                **xy_duplicate_metrics,

                "nearest_candidate_wire_ft":
                    float(
                        nearest_candidate_wire
                    ),

                "dx_crowded":
                    dx_crowded,
            }
        )

        metrics[
            "strong_residual"
        ] = _cv_threshold_pass(
            metrics,
            CV_STRONG[
                mode
            ],
        )

        metrics[
            "exceptional_residual"
        ] = _cv_threshold_pass(
            metrics,
            CV_EXCEPTIONAL[
                mode
            ],
        )

        metrics[
            "dx_promotion_gate"
        ] = bool(
            mode == "DX"
            and _cv_dx_promotion_pass(
                metrics
            )
        )

        validated.append(
            metrics
        )

        # Diagnostic sample of uniquely owned and ambiguous points.
        unique_indices = np.flatnonzero(
            unique_mask
        )

        ambiguous_indices = np.flatnonzero(
            ambiguous_best_mask
        )

        max_unique = (
            CV_MAX_SUPPORT_POINTS_PER_CANDIDATE
        )

        max_ambiguous = max(
            1,
            CV_MAX_SUPPORT_POINTS_PER_CANDIDATE
            // 2,
        )

        if (
            len(
                unique_indices
            )
            > max_unique
        ):
            unique_indices = unique_indices[
                even_indices(
                    len(
                        unique_indices
                    ),
                    max_unique,
                )
            ]

        if (
            len(
                ambiguous_indices
            )
            > max_ambiguous
        ):
            ambiguous_indices = ambiguous_indices[
                even_indices(
                    len(
                        ambiguous_indices
                    ),
                    max_ambiguous,
                )
            ]

        for point_index in unique_indices:
            point = residual.iloc[
                int(
                    point_index
                )
            ]

            support_point_records.append(
                {
                    "candidate_id":
                        candidate[
                            "candidate_id"
                        ],

                    "span_id":
                        span.span_id,

                    "support_type":
                        "UNIQUE",

                    "distance_ft":
                        float(
                            candidate_distance[
                                int(
                                    point_index
                                )
                            ]
                        ),

                    "geometry":
                        Point(
                            float(
                                point[
                                    "x"
                                ]
                            ),
                            float(
                                point[
                                    "y"
                                ]
                            ),
                            float(
                                point[
                                    "z"
                                ]
                            ),
                        ),
                }
            )

        for point_index in ambiguous_indices:
            point = residual.iloc[
                int(
                    point_index
                )
            ]

            support_point_records.append(
                {
                    "candidate_id":
                        candidate[
                            "candidate_id"
                        ],

                    "span_id":
                        span.span_id,

                    "support_type":
                        "AMBIGUOUS_BEST",

                    "distance_ft":
                        float(
                            candidate_distance[
                                int(
                                    point_index
                                )
                            ]
                        ),

                    "geometry":
                        Point(
                            float(
                                point[
                                    "x"
                                ]
                            ),
                            float(
                                point[
                                    "y"
                                ]
                            ),
                            float(
                                point[
                                    "z"
                                ]
                            ),
                        ),
                }
            )

    return (
        validated,
        support_point_records,
        claimed_count,
        len(
            residual
        ),
    )


def _cv_decide_candidate(
    candidate: dict,
):
    mode = safe_text(
        candidate.get(
            "mode"
        )
    ).upper()

    has_poa = bool(
        candidate.get(
            "has_poa",
            False,
        )
    )

    has_v7 = bool(
        candidate.get(
            "has_v7",
            False,
        )
    )

    strong = bool(
        candidate.get(
            "strong_residual",
            False,
        )
    )

    exceptional = bool(
        candidate.get(
            "exceptional_residual",
            False,
        )
    )

    network_both = bool(
        candidate.get(
            "network_both",
            False,
        )
    )

    dx_crowded = bool(
        candidate.get(
            "dx_crowded",
            False,
        )
    )

    possible_vertical_duplicate = bool(
        candidate.get(
            "possible_vertical_duplicate",
            False,
        )
    )

    poa_v7_distance = candidate.get(
        "poa_v7_distance_ft",
        np.nan,
    )

    agreement_gate = (
        CV_MERGE_DISTANCE_DX_FT
        if mode == "DX"
        else CV_MERGE_DISTANCE_TX_FT
    )

    poa_v7_agrees = bool(
        has_poa
        and has_v7
        and np.isfinite(
            poa_v7_distance
        )
        and poa_v7_distance
        <= agreement_gate
    )

    # TX: strong LiDAR evidence + POA/V7 agreement.
    # Network is bonus evidence only.
    if (
        mode == "TX"
        and CV_ENABLE_POA_V7_AUTO_PROMOTE
        and poa_v7_agrees
        and strong
    ):
        if (
            CV_TX_VERTICAL_DUPLICATE_REVIEW_VETO
            and possible_vertical_duplicate
        ):
            return (
                "REVIEW",
                "POSSIBLE_VERTICAL_DUPLICATE",
                "MEDIUM",
            )

        return (
            "AUTO_PROMOTE",
            "TX_POA_V7_RESIDUAL_STRONG",
            "VERY_HIGH",
        )

    # DX: strict independent evidence and BOTH network ends.
    if (
        mode == "DX"
        and CV_ENABLE_POA_V7_AUTO_PROMOTE
        and poa_v7_agrees
        and bool(
            candidate.get(
                "dx_promotion_gate",
                False,
            )
        )
    ):
        if (
            CV_DX_VERTICAL_DUPLICATE_REVIEW_VETO
            and possible_vertical_duplicate
        ):
            return (
                "REVIEW",
                "POSSIBLE_VERTICAL_DUPLICATE",
                "MEDIUM",
            )

        return (
            "AUTO_PROMOTE",
            "DX_POA_V7_STRICT_GATE",
            "VERY_HIGH",
        )

    # Disabled by default in V2.2, retained only as an explicit experiment.
    if (
        CV_ENABLE_V7_ONLY_AUTO_PROMOTE
        and has_v7
        and not has_poa
        and exceptional
        and network_both
    ):
        return (
            "AUTO_PROMOTE",
            "V7_ONLY_EXCEPTIONAL_NETWORK_BOTH",
            "HIGH",
        )

    if (
        CV_ENABLE_POA_ONLY_AUTO_PROMOTE
        and has_poa
        and not has_v7
        and exceptional
        and network_both
        and not dx_crowded
    ):
        return (
            "AUTO_PROMOTE",
            "POA_ONLY_EXCEPTIONAL_NETWORK_BOTH",
            "HIGH",
        )

    reasons = []

    if has_poa and not has_v7:
        reasons.append(
            "POA_NO_V7"
        )

    if has_v7 and not has_poa:
        reasons.append(
            "V7_ONLY"
        )

    if (
        has_poa
        and has_v7
        and not poa_v7_agrees
    ):
        reasons.append(
            "POA_V7_DISAGREE"
        )

    if mode == "TX":
        if not strong:
            reasons.append(
                "TX_RESIDUAL_NOT_STRONG"
            )

    elif mode == "DX":
        if not bool(
            candidate.get(
                "dx_promotion_gate",
                False,
            )
        ):
            reasons.append(
                "DX_STRICT_GATE_FAIL"
            )

        if (
            CV_REQUIRE_BOTH_NETWORK_ENDS_FOR_DX_PROMOTION
            and not network_both
        ):
            reasons.append(
                "DX_NETWORK_NOT_BOTH"
            )

    elif not strong:
        reasons.append(
            "RESIDUAL_NOT_STRONG"
        )

    if possible_vertical_duplicate:
        reasons.append(
            "POSSIBLE_VERTICAL_DUPLICATE"
        )

    if safe_text(
        candidate.get(
            "network_support"
        )
    ) == "ONE_SIDE":
        reasons.append(
            "NETWORK_ONE_SIDE_ONLY"
        )

    if dx_crowded:
        reasons.append(
            "DX_CROWDED"
        )

    if not reasons:
        reasons.append(
            "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        )

    reasons = list(
        dict.fromkeys(
            reasons
        )
    )

    return (
        "REVIEW",
        "|".join(
            reasons
        ),
        "MEDIUM",
    )


def _cv_public_candidate_record(
    candidate: dict,
):
    hidden = {
        "_members",
    }

    return {
        key:
            value
        for key, value
        in candidate.items()
        if key not in hidden
    }


def _cv_candidate_to_final_record(
    candidate: dict,
):
    return {
        "span_id":
            candidate[
                "span_id"
            ],

        "layer_name":
            candidate[
                "layer_name"
            ],

        "source":
            "CANDIDATE_VALIDATION_V2",

        "engine_mode":
            candidate[
                "mode"
            ],

        "pass_name":
            "CANDIDATE_VALIDATION_V2",

        "source_id":
            candidate[
                "candidate_id"
            ],

        "confidence":
            candidate[
                "decision_confidence"
            ],

        "fusion_reason":
            candidate[
                "decision_reason"
            ],

        "v7_agreement_ft":
            candidate.get(
                "poa_v7_distance_ft",
                np.nan,
            ),

        "poa_quality":
            candidate.get(
                "poa_quality_best",
                "",
            ),

        "coverage":
            candidate.get(
                "coverage",
                np.nan,
            ),

        "median_residual_ft":
            candidate.get(
                "median_residual_ft",
                np.nan,
            ),

        "p90_residual_ft":
            candidate.get(
                "p90_residual_ft",
                np.nan,
            ),

        "support_points":
            candidate.get(
                "owned_points",
                0,
            ),

        "support_zones":
            candidate.get(
                "support_zones",
                0,
            ),

        "longest_gap_ft":
            candidate.get(
                "longest_gap_ft",
                np.nan,
            ),

        "longest_gap_fraction":
            candidate.get(
                "longest_gap_fraction",
                np.nan,
            ),

        "candidate_sources":
            candidate.get(
                "candidate_sources",
                "",
            ),

        "terminal_a_points":
            candidate.get(
                "terminal_a_points",
                0,
            ),

        "terminal_b_points":
            candidate.get(
                "terminal_b_points",
                0,
            ),

        "body_slice_hits":
            candidate.get(
                "body_slice_hits",
                0,
            ),

        "unique_owned_fraction":
            candidate.get(
                "unique_owned_fraction",
                0.0,
            ),

        "network_support":
            candidate.get(
                "network_support",
                "NONE",
            ),

        "network_a_distance_ft":
            candidate.get(
                "network_a_distance_ft",
                np.nan,
            ),

        "network_b_distance_ft":
            candidate.get(
                "network_b_distance_ft",
                np.nan,
            ),

        "dx_crowded":
            bool(
                candidate.get(
                    "dx_crowded",
                    False,
                )
            ),

        "nearest_final_wire_xy_ft":
            candidate.get(
                "nearest_final_wire_xy_ft",
                np.nan,
            ),

        "nearest_final_wire_xy_overlap":
            candidate.get(
                "nearest_final_wire_xy_overlap",
                np.nan,
            ),

        "nearest_final_wire_direction_diff_deg":
            candidate.get(
                "nearest_final_wire_direction_diff_deg",
                np.nan,
            ),

        "nearest_final_xy_wire_3d_ft":
            candidate.get(
                "nearest_final_xy_wire_3d_ft",
                np.nan,
            ),

        "possible_vertical_duplicate":
            bool(
                candidate.get(
                    "possible_vertical_duplicate",
                    False,
                )
            ),

        "dx_promotion_gate":
            bool(
                candidate.get(
                    "dx_promotion_gate",
                    False,
                )
            ),

        "bundle_confirmed":
            False,

        "underbuild_spatial_ok":
            False,

        "locked":
            False,

        "geometry":
            candidate[
                "geometry"
            ],
    }


def _cv_remove_promoted_from_review(
    review_records: list[dict],
    promoted_records: list[dict],
    mode_by_span: dict[str, str],
    span_lookup: dict[str, DXFSpan],
):
    if not promoted_records:
        return review_records

    promoted_by_span = {}

    for record in promoted_records:
        promoted_by_span.setdefault(
            record[
                "span_id"
            ],
            [],
        ).append(
            record
        )

    output = []

    for review in review_records:
        span_id = safe_text(
            review.get(
                "span_id"
            )
        )

        promoted = promoted_by_span.get(
            span_id,
            [],
        )

        if not promoted:
            output.append(
                review
            )
            continue

        span = span_lookup.get(
            span_id
        )

        if span is None:
            output.append(
                review
            )
            continue

        mode = _cv_mode(
            span,
            mode_by_span,
        )

        threshold = _cv_merge_distance(
            mode
        )

        if any(
            mean_line_distance_3d(
                review[
                    "geometry"
                ],
                promoted_record[
                    "geometry"
                ],
            )
            <= threshold
            for promoted_record in promoted
        ):
            continue

        output.append(
            review
        )

    return output


def ensure_final_record_ids(
    records: list[dict],
):
    """
    Keep node reconciliation/output robust across all source types.
    """
    for index, record in enumerate(
        records,
        start=1,
    ):
        source_id = safe_text(
            record.get(
                "source_id"
            )
        )

        if not source_id:
            source_id = safe_text(
                record.get(
                    "hypothesis_id"
                )
            )

        if not source_id:
            source_id = safe_text(
                record.get(
                    "track_id"
                )
            )

        if not source_id:
            source_id = (
                f"{safe_text(record.get('span_id')) or 'SPAN'}_"
                f"FINAL_{index:05d}"
            )

        record[
            "source_id"
        ] = source_id

    return records


def run_candidate_validation_second_pass(
    spans: list[DXFSpan],
    wire_xyz: np.ndarray,
    wire_tree_xy: cKDTree,
    first_pass_final: list[dict],
    review_records: list[dict],
    poa_records: list[dict],
    v7_records: list[dict],
    span_reports: list[dict],
    class_code: int | None = None,
):
    """
    Candidate-validation second pass.

    Network geometry is frozen and used only as a bonus signal. No expected
    conductor count is inferred and no new geometry is generated.
    """
    if not ENABLE_CANDIDATE_VALIDATION_SECOND_PASS:
        return (
            list(
                first_pass_final
            ),
            [],
            [],
            [],
            [],
            [],
            review_records,
            span_reports,
        )

    span_lookup = _cv_span_lookup(
        spans
    )

    incident = _cv_incident_map(
        spans
    )

    first_final_by_span = {}

    for record in first_pass_final:
        first_final_by_span.setdefault(
            record[
                "span_id"
            ],
            [],
        ).append(
            record
        )

    mode_by_span = {
        safe_text(
            row.get(
                "span_id"
            )
        ):
            safe_text(
                row.get(
                    "mode"
                )
            )
        for row in span_reports
    }

    report_by_span = {
        safe_text(
            row.get(
                "span_id"
            )
        ):
            row
        for row in span_reports
    }

    candidate_records = []
    promoted_candidates = []
    candidate_review = []
    support_point_records = []
    action_records = []

    total = len(
        spans
    )

    log(
        "\nCandidate validation second pass..."
    )

    for span_number, span in enumerate(
        spans,
        start=1,
    ):
        mode = _cv_mode(
            span,
            mode_by_span,
        )

        first_final = list(
            first_final_by_span.get(
                span.span_id,
                [],
            )
        )

        raw = _cv_candidate_raw_records(
            span,
            review_records,
            poa_records,
            v7_records,
        )

        candidates = _cv_group_candidates(
            span,
            raw,
            first_final,
            mode,
        )

        candidate_query_half_width = (
            CLASS_ENGINE_CONFIG[
                class_code
            ][
                "query_half_width_ft"
            ]
            if class_code in CLASS_ENGINE_CONFIG
            else SPAN_QUERY_HALF_WIDTH_FT
        )

        indices = span_candidate_indices(
            span,
            wire_xyz,
            wire_tree_xy,
            half_width_ft=candidate_query_half_width,
        )

        if len(
            indices
        ):
            frame = local_frame(
                span,
                wire_xyz[
                    indices
                ],
            )

            frame = frame[
                (
                    frame[
                        "s"
                    ]
                    >= 0.0
                )
                & (
                    frame[
                        "s"
                    ]
                    <= span.span_length_ft
                )
            ].copy().reset_index(
                drop=True
            )

        else:
            frame = pd.DataFrame(
                columns=[
                    "x",
                    "y",
                    "z",
                    "s",
                    "t",
                ]
            )

        anchors_a = _cv_neighbour_anchors(
            span,
            span.node_a,
            incident,
            span_lookup,
            first_final_by_span,
        )

        anchors_b = _cv_neighbour_anchors(
            span,
            span.node_b,
            incident,
            span_lookup,
            first_final_by_span,
        )

        (
            validated,
            support_rows,
            first_claimed_points,
            residual_points,
        ) = _cv_validate_candidates_on_residual(
            span,
            frame,
            first_final,
            candidates,
            mode,
            anchors_a,
            anchors_b,
        )

        support_point_records.extend(
            support_rows
        )

        # Decide first, then apply a final geometry-collision gate.
        for candidate in validated:
            (
                decision,
                decision_reason,
                decision_confidence,
            ) = _cv_decide_candidate(
                candidate
            )

            candidate[
                "decision"
            ] = decision

            candidate[
                "decision_reason"
            ] = decision_reason

            candidate[
                "decision_confidence"
            ] = decision_confidence

        promotable = [
            candidate
            for candidate in validated
            if candidate[
                "decision"
            ]
            == "AUTO_PROMOTE"
        ]

        # Strongest first.
        promotable.sort(
            key=lambda candidate:
                (
                    0
                    if (
                        candidate.get(
                            "has_poa",
                            False,
                        )
                        and candidate.get(
                            "has_v7",
                            False,
                        )
                    )
                    else 1,

                    -float(
                        candidate.get(
                            "coverage",
                            0.0,
                        )
                    ),

                    float(
                        candidate.get(
                            "median_residual_ft",
                            np.inf,
                        )
                    ),

                    -float(
                        candidate.get(
                            "unique_owned_fraction",
                            0.0,
                        )
                    ),
                )
        )

        promoted_here = []

        for candidate in promotable:
            minimum_separation = (
                _cv_promoted_min_separation(
                    mode
                )
            )

            collision = any(
                mean_line_distance_3d(
                    candidate[
                        "geometry"
                    ],
                    existing[
                        "geometry"
                    ],
                )
                <= minimum_separation
                for existing in (
                    first_final
                    + promoted_here
                )
            )

            if collision:
                candidate[
                    "decision"
                ] = "REVIEW"

                candidate[
                    "decision_reason"
                ] = (
                    "COLLISION_WITH_FINAL_OR_PROMOTED"
                )

                candidate[
                    "decision_confidence"
                ] = "MEDIUM"

                continue

            final_record = (
                _cv_candidate_to_final_record(
                    candidate
                )
            )

            promoted_here.append(
                final_record
            )

            promoted_candidates.append(
                final_record
            )

            action_records.append(
                {
                    "span_id":
                        span.span_id,

                    "layer_name":
                        span.layer_name,

                    "candidate_id":
                        candidate[
                            "candidate_id"
                        ],

                    "candidate_sources":
                        candidate[
                            "candidate_sources"
                        ],

                    "decision":
                        "AUTO_PROMOTE",

                    "reason":
                        candidate[
                            "decision_reason"
                        ],

                    "coverage":
                        candidate[
                            "coverage"
                        ],

                    "median_residual_ft":
                        candidate[
                            "median_residual_ft"
                        ],

                    "p90_residual_ft":
                        candidate[
                            "p90_residual_ft"
                        ],

                    "unique_owned_fraction":
                        candidate[
                            "unique_owned_fraction"
                        ],

                    "network_support":
                        candidate[
                            "network_support"
                        ],

                    "body_slice_hits":
                        candidate.get(
                            "body_slice_hits",
                            0,
                        ),

                    "terminal_a_points":
                        candidate.get(
                            "terminal_a_points",
                            0,
                        ),

                    "terminal_b_points":
                        candidate.get(
                            "terminal_b_points",
                            0,
                        ),

                    "nearest_final_wire_xy_ft":
                        candidate.get(
                            "nearest_final_wire_xy_ft",
                            np.nan,
                        ),

                    "nearest_final_wire_xy_overlap":
                        candidate.get(
                            "nearest_final_wire_xy_overlap",
                            np.nan,
                        ),

                    "nearest_final_xy_wire_3d_ft":
                        candidate.get(
                            "nearest_final_xy_wire_3d_ft",
                            np.nan,
                        ),

                    "possible_vertical_duplicate":
                        bool(
                            candidate.get(
                                "possible_vertical_duplicate",
                                False,
                            )
                        ),

                    "dx_promotion_gate":
                        bool(
                            candidate.get(
                                "dx_promotion_gate",
                                False,
                            )
                        ),
                }
            )

        for candidate in validated:
            public = _cv_public_candidate_record(
                candidate
            )

            candidate_records.append(
                public
            )

            if (
                candidate[
                    "decision"
                ]
                != "AUTO_PROMOTE"
            ):
                candidate_review.append(
                    public
                )

                action_records.append(
                    {
                        "span_id":
                            span.span_id,

                        "layer_name":
                            span.layer_name,

                        "candidate_id":
                            candidate[
                                "candidate_id"
                            ],

                        "candidate_sources":
                            candidate[
                                "candidate_sources"
                            ],

                        "decision":
                            "REVIEW",

                        "reason":
                            candidate[
                                "decision_reason"
                            ],

                        "coverage":
                            candidate[
                                "coverage"
                            ],

                        "median_residual_ft":
                            candidate[
                                "median_residual_ft"
                            ],

                        "p90_residual_ft":
                            candidate[
                                "p90_residual_ft"
                            ],

                        "unique_owned_fraction":
                            candidate[
                                "unique_owned_fraction"
                            ],

                        "network_support":
                            candidate[
                                "network_support"
                            ],

                        "body_slice_hits":
                            candidate.get(
                                "body_slice_hits",
                                0,
                            ),

                        "terminal_a_points":
                            candidate.get(
                                "terminal_a_points",
                                0,
                            ),

                        "terminal_b_points":
                            candidate.get(
                                "terminal_b_points",
                                0,
                            ),

                        "nearest_final_wire_xy_ft":
                            candidate.get(
                                "nearest_final_wire_xy_ft",
                                np.nan,
                            ),

                        "nearest_final_wire_xy_overlap":
                            candidate.get(
                                "nearest_final_wire_xy_overlap",
                                np.nan,
                            ),

                        "nearest_final_xy_wire_3d_ft":
                            candidate.get(
                                "nearest_final_xy_wire_3d_ft",
                                np.nan,
                            ),

                        "possible_vertical_duplicate":
                            bool(
                                candidate.get(
                                    "possible_vertical_duplicate",
                                    False,
                                )
                            ),

                        "dx_promotion_gate":
                            bool(
                                candidate.get(
                                    "dx_promotion_gate",
                                    False,
                                )
                            ),
                    }
                )

        report = report_by_span.get(
            span.span_id
        )

        if report is not None:
            report[
                "first_pass_final_wires"
            ] = int(
                report.get(
                    "final_wires",
                    len(
                        first_final
                    ),
                )
            )

            report[
                "candidate_count"
            ] = len(
                validated
            )

            report[
                "candidate_auto_promoted"
            ] = len(
                promoted_here
            )

            report[
                "candidate_review"
            ] = (
                len(
                    validated
                )
                - len(
                    promoted_here
                )
            )

            report[
                "candidate_first_pass_claimed_points"
            ] = first_claimed_points

            report[
                "candidate_residual_points"
            ] = residual_points

            report[
                "final_wires_after_v2"
            ] = (
                len(
                    first_final
                )
                + len(
                    promoted_here
                )
            )

            if promoted_here:
                report[
                    "candidate_status"
                ] = "PROMOTED"

            elif validated:
                report[
                    "candidate_status"
                ] = "REVIEW"

            else:
                report[
                    "candidate_status"
                ] = "NO_CANDIDATE"

        if (
            span_number == 1
            or span_number % 25 == 0
            or span_number == total
        ):
            log(
                f"  candidate spans "
                f"{span_number}/{total}"
                f" | candidates="
                f"{len(candidate_records):,}"
                f" | promoted="
                f"{len(promoted_candidates):,}"
            )

    final_records = (
        list(
            first_pass_final
        )
        + promoted_candidates
    )

    review_records_clean = (
        _cv_remove_promoted_from_review(
            review_records,
            promoted_candidates,
            mode_by_span,
            span_lookup,
        )
    )

    # Limit point-support diagnostics deterministically if necessary.
    if (
        len(
            support_point_records
        )
        > CV_MAX_SUPPORT_POINT_ROWS
    ):
        selected = even_indices(
            len(
                support_point_records
            ),
            CV_MAX_SUPPORT_POINT_ROWS,
        )

        support_point_records = [
            support_point_records[
                int(
                    index
                )
            ]
            for index in selected
        ]

    return (
        final_records,
        promoted_candidates,
        candidate_records,
        candidate_review,
        support_point_records,
        action_records,
        review_records_clean,
        span_reports,
    )


# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# V3 OUTPUT / CLASS HELPERS
# =============================================================================

def _apply_class_metadata(
    records: list[dict],
    class_code: int,
):
    config = CLASS_ENGINE_CONFIG[
        class_code
    ]

    for record in records:
        record[
            "class_code"
        ] = class_code
        record[
            "class_name"
        ] = config[
            "class_name"
        ]
        if not safe_text(
            record.get(
                "class_engine"
            )
        ):
            record[
                "class_engine"
            ] = config[
                "engine_key"
            ]

        if not safe_text(
            record.get(
                "engine_mode"
            )
        ):
            record[
                "engine_mode"
            ] = config[
                "family"
            ]

    return records


def _prefix_candidate_outputs(
    class_code: int,
    promoted: list[dict],
    candidates: list[dict],
    candidate_review: list[dict],
    support: list[dict],
    actions: list[dict],
):
    id_map = {}

    for candidate in candidates:
        old = safe_text(
            candidate.get(
                "candidate_id"
            )
        )

        if not old:
            continue

        new = (
            f"C{class_code}_"
            f"{old}"
        )
        id_map[old] = new
        candidate[
            "candidate_id"
        ] = new

    for collection in (
        candidate_review,
        support,
        actions,
    ):
        for record in collection:
            old = safe_text(
                record.get(
                    "candidate_id"
                )
            )
            if old in id_map:
                record[
                    "candidate_id"
                ] = id_map[old]

    for record in promoted:
        old_source = safe_text(
            record.get(
                "source_id"
            )
        )

        if old_source in id_map:
            record[
                "source_id"
            ] = id_map[
                old_source
            ]

    for collection in (
        promoted,
        candidates,
        candidate_review,
        support,
        actions,
    ):
        _apply_class_metadata(
            collection,
            class_code,
        )


def conductor_class_sample_gdf(
    class_xyz: dict[int, np.ndarray],
    crs,
):
    records = []

    per_class_maximum = max(
        1,
        int(
            MAX_WIRE_SAMPLE
            / max(
                len(
                    CONDUCTOR_CLASS_NAMES
                ),
                1,
            )
        ),
    )

    for class_code in sorted(
        CONDUCTOR_CLASS_NAMES
    ):
        xyz = class_xyz[
            class_code
        ]

        if len(xyz) == 0:
            continue

        indices = even_indices(
            len(xyz),
            per_class_maximum,
        )

        for point in xyz[
            indices
        ]:
            records.append(
                {
                    "class_code":
                        class_code,
                    "class_name":
                        CONDUCTOR_CLASS_NAMES[
                            class_code
                        ],
                    "z_source":
                        float(
                            point[2]
                        )
                        * INTERNAL_FT_TO_SOURCE,
                    "geometry":
                        Point(
                            float(
                                point[0]
                            ),
                            float(
                                point[1]
                            ),
                            float(
                                point[2]
                            ),
                        ),
                }
            )

    return make_gdf(
        records,
        crs,
    )


def run_wires() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "=" * 80
    )
    log(
        "WIRE EXTRACTION - CLASS 187 / STRUCTURES 215"
    )
    log(
        "=" * 80
    )

    if ENABLE_BUNDLE_COLLAPSE:
        log(
            "\nBundle mode:"
            f"\n  collapse distance: {BUNDLE_COLLAPSE_DISTANCE_M:.2f} m"
            f"\n  maximum bundle diameter: {BUNDLE_MAX_DIAMETER_M:.2f} m"
            f"\n  ownership radius: {BUNDLE_OWNERSHIP_RADIUS_M:.2f} m"
            f"\n  loose model radius: {BUNDLE_LOOSE_RADIUS_M:.2f} m"
            "\n  output: one centre vector per detected bundle"
        )

    require_path(
        SOURCE_LAZ,
        "LiDAR",
    )
    require_path(
        DXF_PATH,
        "Centreline",
    )

    (
        class_xyz,
        structure_xyz,
        all_conductor_xyz,
        crs,
    ) = read_required_lidar(
        SOURCE_LAZ
    )

    structures = cluster_structures_for_qa(
        structure_xyz
    )

    structure_lookup = StructureLookup(
        structures
    )

    (
        original_nodes,
        original_spans,
    ) = read_and_explode_dxf(
        DXF_PATH,
        structure_lookup,
    )

    (
        nodes,
        spans,
        topology_node_audit_records,
        topology_structure_audit_records,
        topology_correction_records,
    ) = audit_and_correct_topology(
        original_nodes,
        original_spans,
        structures,
        all_conductor_xyz,
    )

    span_reports = []
    terminal_records = []
    body_records = []
    hypothesis_records = []
    poa_records = []
    v7_records = []
    stitch_records = []

    first_pass_final_records = []
    final_records = []
    review_records = []
    rejected_records = []

    candidate_promoted = []
    candidate_records = []
    candidate_review_records = []
    candidate_support_records = []
    candidate_actions = []

    class_processing_rows = []

    total_spans = len(
        spans
    )

    # ---------------------------------------------------------------------
    # Each class is a physically independent extraction problem.
    # ---------------------------------------------------------------------

    for class_code in sorted(
        CONDUCTOR_CLASS_NAMES
    ):
        config = CLASS_ENGINE_CONFIG[
            class_code
        ]
        xyz = class_xyz[
            class_code
        ]

        log(
            "\n"
            + "-" * 80
        )
        log(
            f"CLASS {class_code} "
            f"{config['class_name']} "
            f"[{config['engine_key']} / "
            f"{config['family']}]"
        )
        log(
            f"points: {len(xyz):,}"
        )

        if len(xyz) == 0:
            continue

        tree = cKDTree(
            xyz[:, :2]
        )

        class_span_reports = []
        class_terminal_records = []
        class_body_records = []
        class_hypothesis_records = []
        class_poa_records = []
        class_v7_records = []
        class_stitch_records = []
        class_final_records = []
        class_review_records = []
        class_rejected_records = []

        for span_number, span in enumerate(
            spans,
            start=1,
        ):
            try:
                result = process_span_class(
                    span,
                    class_code,
                    xyz,
                    tree,
                )

            except Exception as exc:
                log(
                    f"  ERROR class {class_code} "
                    f"{span.span_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

                result = _empty_class_span_result(
                    span,
                    class_code,
                    config,
                    "ERROR",
                )

                result[
                    "span_report"
                ][
                    "error"
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            class_span_reports.append(
                result[
                    "span_report"
                ]
            )
            class_terminal_records.extend(
                result[
                    "terminal_records"
                ]
            )
            class_body_records.extend(
                result[
                    "body_records"
                ]
            )
            class_hypothesis_records.extend(
                result[
                    "hypothesis_records"
                ]
            )
            class_poa_records.extend(
                result[
                    "poa_records"
                ]
            )
            class_v7_records.extend(
                result[
                    "v7_records"
                ]
            )
            class_stitch_records.extend(
                result[
                    "stitch_records"
                ]
            )
            class_final_records.extend(
                result[
                    "final_records"
                ]
            )
            class_review_records.extend(
                result[
                    "review_records"
                ]
            )
            class_rejected_records.extend(
                result[
                    "rejected_records"
                ]
            )

            if (
                span_number == 1
                or span_number % 50 == 0
                or span_number == total_spans
            ):
                log(
                    f"  spans "
                    f"{span_number}/{total_spans}"
                    f" | first-pass final="
                    f"{len(class_final_records):,}"
                    f" | review="
                    f"{len(class_review_records):,}"
                    f" | stitched="
                    f"{len(class_stitch_records):,}"
                )

        class_first_pass = list(
            class_final_records
        )

        # Candidate validation uses ONLY points and candidates from this class.
        (
            class_final_records,
            class_candidate_promoted,
            class_candidate_records,
            class_candidate_review,
            class_candidate_support,
            class_candidate_actions,
            class_review_records,
            class_span_reports,
        ) = run_candidate_validation_second_pass(
            spans,
            xyz,
            tree,
            class_first_pass,
            class_review_records,
            class_poa_records,
            class_v7_records,
            class_span_reports,
            class_code=class_code,
        )

        # Candidate validation can re-introduce a close model that represents
        # another subconductor of an already accepted bundle. Collapse once
        # more after the second pass so the class product is guaranteed to
        # contain one accepted vector per physical bundle.
        class_final_by_span = {}

        for record in class_final_records:
            class_final_by_span.setdefault(
                safe_text(
                    record.get("span_id")
                ),
                [],
            ).append(record)

        collapsed_class_final = []

        span_lookup_for_bundle = {
            span.span_id: span
            for span in spans
        }

        for span_id, records_for_span in class_final_by_span.items():
            span_for_bundle = span_lookup_for_bundle.get(
                span_id
            )

            if span_for_bundle is None:
                collapsed_class_final.extend(
                    records_for_span
                )
                continue

            collapsed_class_final.extend(
                collapse_final_bundle_records(
                    span_for_bundle,
                    class_code,
                    records_for_span,
                )
            )

        class_final_records = collapsed_class_final

        _apply_class_metadata(
            class_first_pass,
            class_code,
        )
        _apply_class_metadata(
            class_final_records,
            class_code,
        )
        _apply_class_metadata(
            class_review_records,
            class_code,
        )
        _apply_class_metadata(
            class_rejected_records,
            class_code,
        )

        _prefix_candidate_outputs(
            class_code,
            class_candidate_promoted,
            class_candidate_records,
            class_candidate_review,
            class_candidate_support,
            class_candidate_actions,
        )

        span_reports.extend(
            class_span_reports
        )
        terminal_records.extend(
            class_terminal_records
        )
        body_records.extend(
            class_body_records
        )
        hypothesis_records.extend(
            class_hypothesis_records
        )
        poa_records.extend(
            class_poa_records
        )
        v7_records.extend(
            class_v7_records
        )
        stitch_records.extend(
            class_stitch_records
        )

        first_pass_final_records.extend(
            class_first_pass
        )
        final_records.extend(
            class_final_records
        )
        review_records.extend(
            class_review_records
        )
        rejected_records.extend(
            class_rejected_records
        )

        candidate_promoted.extend(
            class_candidate_promoted
        )
        candidate_records.extend(
            class_candidate_records
        )
        candidate_review_records.extend(
            class_candidate_review
        )
        candidate_support_records.extend(
            class_candidate_support
        )
        candidate_actions.extend(
            class_candidate_actions
        )

        class_processing_rows.append(
            {
                "class_code":
                    class_code,
                "class_name":
                    config[
                        "class_name"
                    ],
                "class_engine":
                    config[
                        "engine_key"
                    ],
                "family":
                    config[
                        "family"
                    ],
                "points":
                    len(xyz),
                "spans_with_points":
                    int(
                        sum(
                            int(
                                report.get(
                                    "wire_points",
                                    0,
                                )
                            )
                            > 0
                            for report in class_span_reports
                        )
                    ),
                "tracker_raw_tracks":
                    int(
                        sum(
                            int(
                                report.get(
                                    "v7_tracks",
                                    0,
                                )
                            )
                            for report in class_span_reports
                        )
                    ),
                "tracker_tracks_after_stitch":
                    int(
                        sum(
                            int(
                                report.get(
                                    "v7_tracks_after_stitch",
                                    0,
                                )
                            )
                            for report in class_span_reports
                        )
                    ),
                "stitch_actions":
                    len(
                        class_stitch_records
                    ),
                "first_pass_final":
                    len(
                        class_first_pass
                    ),
                "candidate_promoted":
                    len(
                        class_candidate_promoted
                    ),
                "final_before_geometry_qa":
                    len(
                        class_final_records
                    ),
                "bundle_final_vectors":
                    int(
                        sum(
                            bool(
                                record.get(
                                    "bundle_collapsed",
                                    False,
                                )
                                or record.get(
                                    "bundle_confirmed",
                                    False,
                                )
                            )
                            for record in class_final_records
                        )
                    ),
                "review":
                    len(
                        class_review_records
                    ),
                "rejected":
                    len(
                        class_rejected_records
                    ),
            }
        )

        log(
            f"  candidate validation: "
            f"candidates="
            f"{len(class_candidate_records):,} "
            f"promoted="
            f"{len(class_candidate_promoted):,}"
        )

    final_records = ensure_final_record_ids(
        final_records
    )

    # ---------------------------------------------------------------------
    # Safety checkpoint before geometry QA / node continuity.
    # ---------------------------------------------------------------------

    checkpoint_gpkg = (
        OUTPUT_DIR
        / "_checkpoint_before_node_reconciliation.gpkg"
    )

    if checkpoint_gpkg.exists():
        checkpoint_gpkg.unlink()

    checkpoint_gdf = make_gdf(
        [
            {
                "span_id":
                    safe_text(
                        record.get(
                            "span_id"
                        )
                    ),
                "class_code":
                    record.get(
                        "class_code",
                        np.nan,
                    ),
                "class_name":
                    safe_text(
                        record.get(
                            "class_name"
                        )
                    ),
                "source_id":
                    safe_text(
                        record.get(
                            "source_id"
                        )
                    ),
                "source":
                    safe_text(
                        record.get(
                            "source"
                        )
                    ),
                "pass_name":
                    safe_text(
                        record.get(
                            "pass_name"
                        )
                    ),
                "confidence":
                    safe_text(
                        record.get(
                            "confidence"
                        )
                    ),
                "fusion_reason":
                    safe_text(
                        record.get(
                            "fusion_reason"
                        )
                    ),
                "geometry":
                    record[
                        "geometry"
                    ],
            }
            for record in final_records
            if record.get(
                "geometry"
            ) is not None
        ],
        crs,
    )

    if not checkpoint_gdf.empty:
        checkpoint_gdf.to_file(
            checkpoint_gpkg,
            layer="wires_final_prejoin",
            driver="GPKG",
            engine="pyogrio",
        )

    # ---------------------------------------------------------------------
    # Physical geometry QA retained from V2.5.
    # ---------------------------------------------------------------------

    (
        final_records,
        geometry_review_records,
        geometry_qa_records,
    ) = validate_final_geometry(
        spans,
        final_records,
    )

    final_records = ensure_final_record_ids(
        final_records
    )

    # Logical continuity is class-aware and never mutates geometry.
    node_join_records = reconcile_nodes(
        spans,
        final_records,
    )

    # Update final class counts after physics QA.
    for row in class_processing_rows:
        class_code = row[
            "class_code"
        ]
        row[
            "final_after_geometry_qa"
        ] = int(
            sum(
                record.get(
                    "class_code"
                )
                == class_code
                for record in final_records
            )
        )
        row[
            "geometry_review"
        ] = int(
            sum(
                record.get(
                    "class_code"
                )
                == class_code
                for record in geometry_review_records
            )
        )

    # ---------------------------------------------------------------------
    # Frames.
    # ---------------------------------------------------------------------

    dxf_nodes_original = node_outputs(
        original_nodes,
        crs,
    )
    dxf_spans_original = span_outputs(
        original_spans,
        crs,
    )
    dxf_nodes = node_outputs(
        nodes,
        crs,
    )
    dxf_spans = span_outputs(
        spans,
        crs,
    )

    structure_clusters_gdf = make_gdf(
        [
            {
                "structure_id":
                    structure.structure_id,
                "structure_class":
                    STRUCTURE_CLASS,
                "point_count":
                    len(
                        structure.xyz
                    ),
                "geometry":
                    Point(
                        float(
                            structure.anchor_xyz[0]
                        ),
                        float(
                            structure.anchor_xyz[1]
                        ),
                        float(
                            structure.anchor_xyz[2]
                        ),
                    ),
            }
            for structure in structures
        ],
        crs,
    )

    topology_node_audit_gdf = make_gdf(
        topology_node_audit_records,
        crs,
    )
    topology_structure_audit_gdf = make_gdf(
        topology_structure_audit_records,
        crs,
    )
    topology_corrections_gdf = make_gdf(
        topology_correction_records,
        crs,
    )
    geometry_review_gdf = make_gdf(
        geometry_review_records,
        crs,
    )
    geometry_qa_gdf = make_gdf(
        geometry_qa_records,
        crs,
    )

    node_links = node_structure_links(
        nodes,
        structures,
        crs,
    )

    span_analysis = make_gdf(
        span_reports,
        crs,
    )
    terminal_gdf = make_gdf(
        terminal_records,
        crs,
    )
    body_gdf = make_gdf(
        body_records,
        crs,
    )
    poa_gdf = make_gdf(
        poa_records,
        crs,
    )
    v7_gdf = make_gdf(
        v7_records,
        crs,
    )
    final_gdf = make_gdf(
        final_records,
        crs,
    )
    review_gdf = make_gdf(
        review_records,
        crs,
    )
    rejected_gdf = make_gdf(
        rejected_records,
        crs,
    )
    first_pass_final_gdf = make_gdf(
        first_pass_final_records,
        crs,
    )
    candidate_promoted_gdf = make_gdf(
        candidate_promoted,
        crs,
    )
    candidates_all_gdf = make_gdf(
        candidate_records,
        crs,
    )
    candidates_review_gdf = make_gdf(
        candidate_review_records,
        crs,
    )
    candidate_support_gdf = make_gdf(
        candidate_support_records,
        crs,
    )
    node_joins_gdf = make_gdf(
        node_join_records,
        crs,
    )

    unresolved_span_records = []

    for report in span_reports:
        if (
            safe_text(
                report.get(
                    "status"
                )
            )
            in CV_UNRESOLVED_STATUSES
            and int(
                report.get(
                    "candidate_count",
                    0,
                )
            )
            == 0
        ):
            unresolved_span_records.append(
                dict(
                    report
                )
            )

    unresolved_spans_gdf = make_gdf(
        unresolved_span_records,
        crs,
    )

    conductor_sample = conductor_class_sample_gdf(
        class_xyz,
        crs,
    )

    structure_sample = point_sample_gdf(
        structure_xyz,
        MAX_STRUCTURE_SAMPLE,
        "CLASS215",
        crs,
    )

    # ---------------------------------------------------------------------
    # CSV outputs.
    # ---------------------------------------------------------------------

    dxf_nodes.drop(
        columns=["geometry"],
        errors="ignore",
    ).to_csv(
        DXF_NODE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    dxf_spans.drop(
        columns=["geometry"],
        errors="ignore",
    ).to_csv(
        DXF_SPAN_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    span_analysis.drop(
        columns=["geometry"],
        errors="ignore",
    ).to_csv(
        SPAN_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        hypothesis_records
    ).to_csv(
        HYPOTHESIS_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                key:
                    value
                for key, value
                in record.items()
                if key != "geometry"
            }
            for record in (
                final_records
                + geometry_review_records
                + review_records
                + rejected_records
            )
        ]
    ).to_csv(
        FINAL_WIRE_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                key:
                    value
                for key, value
                in record.items()
                if key != "geometry"
            }
            for record in candidate_records
        ]
    ).to_csv(
        CANDIDATE_METRICS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        candidate_actions
    ).to_csv(
        CANDIDATE_ACTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        stitch_records
    ).to_csv(
        TRACKER_STITCH_ACTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        class_processing_rows
    ).to_csv(
        CLASS_EXTRACTION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    for records, path in (
        (
            topology_node_audit_records,
            TOPOLOGY_NODE_AUDIT_CSV,
        ),
        (
            topology_structure_audit_records,
            TOPOLOGY_STRUCTURE_AUDIT_CSV,
        ),
        (
            topology_correction_records,
            TOPOLOGY_CORRECTIONS_CSV,
        ),
        (
            geometry_qa_records,
            GEOMETRY_QA_CSV,
        ),
    ):
        pd.DataFrame(
            [
                {
                    key:
                        value
                    for key, value
                    in record.items()
                    if key != "geometry"
                }
                for record in records
            ]
        ).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

    summary = pd.DataFrame(
        [
            {
                "conductor_classes":
                    ",".join(
                        str(value)
                        for value in sorted(
                            CONDUCTOR_CLASS_NAMES
                        )
                    ),
                "conductor_points":
                    len(
                        all_conductor_xyz
                    ),
                "structure_class":
                    STRUCTURE_CLASS,
                "structure_points":
                    len(
                        structure_xyz
                    ),
                "structure_clusters":
                    len(
                        structures
                    ),
                "original_dxf_nodes":
                    len(
                        original_nodes
                    ),
                "original_dxf_spans":
                    len(
                        original_spans
                    ),
                "corrected_dxf_nodes":
                    len(
                        nodes
                    ),
                "corrected_dxf_spans":
                    len(
                        spans
                    ),
                "topology_collapsed_vertices":
                    int(
                        sum(
                            record.get(
                                "action"
                            )
                            == "COLLAPSE_NON_STRUCTURE_VERTEX"
                            for record in topology_correction_records
                        )
                    ),
                "topology_inserted_structure_nodes":
                    int(
                        sum(
                            record.get(
                                "action"
                            )
                            == "INSERT_MISSING_STRUCTURE_NODE"
                            for record in topology_correction_records
                        )
                    ),
                "tracker_stitch_actions":
                    len(
                        stitch_records
                    ),
                "poa_selected_models":
                    len(
                        poa_records
                    ),
                "tracker_models":
                    len(
                        v7_records
                    ),
                "first_pass_final_wires":
                    len(
                        first_pass_final_records
                    ),
                "candidate_registry":
                    len(
                        candidate_records
                    ),
                "candidate_auto_promoted":
                    len(
                        candidate_promoted
                    ),
                "final_wires":
                    len(
                        final_records
                    ),
                "geometry_review_wires":
                    len(
                        geometry_review_records
                    ),
                "review_wires":
                    len(
                        review_records
                    ),
                "rejected_wires":
                    len(
                        rejected_records
                    ),
                "node_joins":
                    len(
                        node_join_records
                    ),
            }
        ]
    )

    summary.to_csv(
        EXTRACTION_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------------------
    # GPKG.
    # ---------------------------------------------------------------------

    layer_list = [
        (
            "dxf_nodes_original",
            dxf_nodes_original,
        ),
        (
            "dxf_spans_original",
            dxf_spans_original,
        ),
        (
            "structure215_clusters",
            structure_clusters_gdf,
        ),
        (
            "topology_node_audit",
            topology_node_audit_gdf,
        ),
        (
            "topology_structure_audit",
            topology_structure_audit_gdf,
        ),
        (
            "topology_corrections",
            topology_corrections_gdf,
        ),
        (
            "dxf_nodes",
            dxf_nodes,
        ),
        (
            "dxf_spans_exploded",
            dxf_spans,
        ),
        (
            "dxf_node_to_215_links",
            node_links,
        ),
        (
            "terminal_poa_candidates",
            terminal_gdf,
        ),
        (
            "body_slice_clusters",
            body_gdf,
        ),
        (
            "poa_wires",
            poa_gdf,
        ),
        (
            "class_tracker_wires",
            v7_gdf,
        ),
        # Alias retained for old visual checks / comparator code.
        (
            "v7_wires",
            v7_gdf,
        ),
        (
            "wires_first_pass_final",
            first_pass_final_gdf,
        ),
        (
            "wires_promoted_second_pass",
            candidate_promoted_gdf,
        ),
        (
            "candidates_all",
            candidates_all_gdf,
        ),
        (
            "candidates_auto_promoted",
            candidate_promoted_gdf,
        ),
        (
            "candidates_review",
            candidates_review_gdf,
        ),
        (
            "candidate_point_support",
            candidate_support_gdf,
        ),
        (
            "spans_unresolved_no_candidate",
            unresolved_spans_gdf,
        ),
        (
            "wire_geometry_qa",
            geometry_qa_gdf,
        ),
        (
            "wires_geometry_review",
            geometry_review_gdf,
        ),
        (
            "wires_3d",
            final_gdf,
        ),
        (
            "wires_review",
            review_gdf,
        ),
        (
            "wires_rejected",
            rejected_gdf,
        ),
        (
            "node_joins",
            node_joins_gdf,
        ),
        (
            "span_analysis",
            span_analysis,
        ),
        (
            "conductor_class_sample",
            conductor_sample,
        ),
        (
            "structure215_sample",
            structure_sample,
        ),
    ]

    # Class-specific final layers make QGIS review much faster.
    if not final_gdf.empty:
        for class_code in sorted(
            CONDUCTOR_CLASS_NAMES
        ):
            class_name = (
                CONDUCTOR_CLASS_NAMES[
                    class_code
                ]
                .replace(
                    " ",
                    "_",
                )
            )

            class_frame = final_gdf[
                final_gdf[
                    "class_code"
                ]
                == class_code
            ].copy()

            layer_list.append(
                (
                    f"wires_3d_"
                    f"{class_code}_"
                    f"{class_name}",
                    class_frame,
                )
            )

    log(
        "\nWriting GeoPackage..."
    )

    write_layers(
        layer_list
    )

    log(
        "\n"
        + "=" * 80
    )
    log(
        "CLASS 187 WIRE EXTRACTION COMPLETE"
    )
    log(
        "=" * 80
    )

    log(
        "\nOverall summary:\n"
        + summary.to_string(
            index=False
        )
    )

    class_summary = pd.DataFrame(
        class_processing_rows
    )

    if not class_summary.empty:
        log(
            "\nClass summary:\n"
            + class_summary.to_string(
                index=False
            )
        )

    log(
        "\nFirst QGIS check:"
        "\n  dxf_spans_original"
        "\n  structure215_clusters"
        "\n  topology_corrections"
        "\n  dxf_spans_exploded"
        "\n  conductor_class_sample"
        "\n  class_tracker_wires  [coverage + stitching fields]"
        "\n  poa_wires"
        "\n  wires_first_pass_final"
        "\n  candidates_all"
        "\n  wires_promoted_second_pass"
        "\n  wire_geometry_qa"
        "\n  wires_geometry_review"
        "\n  wires_3d"
    )

    log(
        f"\nOutputs:"
        f"\n  {OUTPUT_GPKG}"
        f"\n  {SPAN_SUMMARY_CSV}"
        f"\n  {HYPOTHESIS_SUMMARY_CSV}"
        f"\n  {FINAL_WIRE_SUMMARY_CSV}"
        f"\n  {EXTRACTION_SUMMARY_CSV}"
        f"\n  {CLASS_EXTRACTION_SUMMARY_CSV}"
        f"\n  {TRACKER_STITCH_ACTIONS_CSV}"
        f"\n  {CANDIDATE_METRICS_CSV}"
        f"\n  {CANDIDATE_ACTIONS_CSV}"
        f"\n  {TOPOLOGY_NODE_AUDIT_CSV}"
        f"\n  {TOPOLOGY_STRUCTURE_AUDIT_CSV}"
        f"\n  {TOPOLOGY_CORRECTIONS_CSV}"
        f"\n  {GEOMETRY_QA_CSV}"
    )



# =============================================================================
# WIRE / STRUCTURE PRODUCT PIPELINE
# =============================================================================

GROUND_CLASS_DEFAULT = 2
STRUCTURE_MATCH_MAX_M_DEFAULT = 15.0
STRUCTURE_CLUSTER_EPS_M = 4.0
GROUND_PRIMARY_RADIUS_M_DEFAULT = 1.5
GROUND_FALLBACK_RADIUS_M_DEFAULT = 5.0
GROUND_VALIDATION_TOLERANCE_M_DEFAULT = 0.25
TOWER_TOP_Z_PERCENTILE = 99.5


@dataclass
class ProductStructure:
    structure_id: str
    anchor_xyz: np.ndarray
    xyz: np.ndarray


@dataclass
class CentrelineNodeCandidate:
    node_id: str
    xy: np.ndarray
    source_kind: str
    source_count: int


def _resolve_runtime_path(value: str | Path, base: Path = BASE_FOLDER) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_centreline_path(value: str | Path | None) -> Path:
    """Resolve an explicit centreline or auto-detect one beside the script.

    DXF is preferred because it avoids optional proprietary DGNv8/ODA support.
    If multiple plausible files exist, require --centreline rather than guessing.
    """
    if value is not None and str(value).strip():
        path = _resolve_runtime_path(value)
        if not path.exists():
            raise FileNotFoundError(f"Centreline not found:\n  {path}")
        return path

    preferred_names = (
        "centrelines.dxf",
        "centerlines.dxf",
        "centreline.dxf",
        "centerline.dxf",
        "centrelines.dgn",
        "centerlines.dgn",
        "centreline.dgn",
        "centerline.dgn",
    )
    for name in preferred_names:
        path = BASE_FOLDER / name
        if path.exists():
            return path.resolve()

    dxf_files = sorted(
        [p for p in BASE_FOLDER.iterdir() if p.is_file() and p.suffix.lower() == ".dxf"],
        key=lambda p: p.name.lower(),
    )
    if len(dxf_files) == 1:
        return dxf_files[0].resolve()
    if len(dxf_files) > 1:
        names = "\n".join(f"  {p.name}" for p in dxf_files)
        raise RuntimeError(
            "Multiple DXF files were found beside the script. Specify the centreline "
            "explicitly with --centreline.\n" + names
        )

    dgn_files = sorted(
        [p for p in BASE_FOLDER.iterdir() if p.is_file() and p.suffix.lower() == ".dgn"],
        key=lambda p: p.name.lower(),
    )
    if len(dgn_files) == 1:
        return dgn_files[0].resolve()
    if len(dgn_files) > 1:
        names = "\n".join(f"  {p.name}" for p in dgn_files)
        raise RuntimeError(
            "Multiple DGN files were found beside the script. Specify the centreline "
            "explicitly with --centreline.\n" + names
        )

    raise FileNotFoundError(
        "No centreline DXF/DGN was found beside the script. "
        "Place it there or use --centreline <path>."
    )


def configure_runtime_paths(input_folder: Path, centreline: Path, output_root: Path) -> None:
    global INPUT_FOLDER, CENTERLINE_PATH, SOURCE_LAZ, DXF_PATH
    global OUTPUT_ROOT, OUTPUT_DIR, OUTPUT_GPKG
    global SPAN_SUMMARY_CSV, HYPOTHESIS_SUMMARY_CSV, FINAL_WIRE_SUMMARY_CSV
    global EXTRACTION_SUMMARY_CSV, DXF_SPAN_CSV, DXF_NODE_CSV
    global CANDIDATE_METRICS_CSV, CANDIDATE_ACTIONS_CSV
    global TOPOLOGY_NODE_AUDIT_CSV, TOPOLOGY_STRUCTURE_AUDIT_CSV
    global TOPOLOGY_CORRECTIONS_CSV, GEOMETRY_QA_CSV
    global CLASS_EXTRACTION_SUMMARY_CSV, TRACKER_STITCH_ACTIONS_CSV
    global WIRES_PRODUCT_DGN, WIRES_PRODUCT_PRJ, PRODUCT_GPKG
    global TOWER_TOP_XYZ, TOWER_TOP_PRJ, TOWER_TOP_CSV
    global TOWER_BOTTOM_XYZ, TOWER_BOTTOM_PRJ, TOWER_BOTTOM_CSV
    global CRS_TXT
    global TOWER_BOTTOM_VALIDATION_CSV, TOWER_BOTTOM_VALIDATION_GPKG

    INPUT_FOLDER = input_folder
    CENTERLINE_PATH = centreline
    DXF_PATH = centreline
    OUTPUT_ROOT = output_root
    OUTPUT_DIR = OUTPUT_ROOT / "diagnostics"
    # Temporary merged LiDAR belongs with diagnostics, not with deliverables.
    SOURCE_LAZ = OUTPUT_DIR / "_working_187_215.laz"
    OUTPUT_GPKG = OUTPUT_DIR / "wire_extraction_187.gpkg"

    SPAN_SUMMARY_CSV = OUTPUT_DIR / "span_wire_summary.csv"
    HYPOTHESIS_SUMMARY_CSV = OUTPUT_DIR / "hypothesis_summary.csv"
    FINAL_WIRE_SUMMARY_CSV = OUTPUT_DIR / "final_wire_summary.csv"
    EXTRACTION_SUMMARY_CSV = OUTPUT_DIR / "extraction_summary.csv"
    DXF_SPAN_CSV = OUTPUT_DIR / "centreline_exploded_span_summary.csv"
    DXF_NODE_CSV = OUTPUT_DIR / "centreline_node_summary.csv"
    CANDIDATE_METRICS_CSV = OUTPUT_DIR / "candidate_metrics.csv"
    CANDIDATE_ACTIONS_CSV = OUTPUT_DIR / "candidate_validation_actions.csv"
    TOPOLOGY_NODE_AUDIT_CSV = OUTPUT_DIR / "topology_node_audit.csv"
    TOPOLOGY_STRUCTURE_AUDIT_CSV = OUTPUT_DIR / "topology_structure_audit.csv"
    TOPOLOGY_CORRECTIONS_CSV = OUTPUT_DIR / "topology_corrections.csv"
    GEOMETRY_QA_CSV = OUTPUT_DIR / "geometry_qa.csv"
    CLASS_EXTRACTION_SUMMARY_CSV = OUTPUT_DIR / "class_extraction_summary.csv"
    TRACKER_STITCH_ACTIONS_CSV = OUTPUT_DIR / "tracker_stitch_actions.csv"

    WIRES_PRODUCT_DGN = OUTPUT_ROOT / "wires_3d.dgn"
    WIRES_PRODUCT_PRJ = OUTPUT_ROOT / "wires_3d.prj"
    PRODUCT_GPKG = OUTPUT_ROOT / "wire_structure_products.gpkg"
    TOWER_TOP_XYZ = OUTPUT_ROOT / "tower_tops.xyz"
    TOWER_TOP_PRJ = OUTPUT_ROOT / "tower_tops.prj"
    TOWER_BOTTOM_XYZ = OUTPUT_ROOT / "tower_bottoms.xyz"
    TOWER_BOTTOM_PRJ = OUTPUT_ROOT / "tower_bottoms.prj"
    CRS_TXT = OUTPUT_ROOT / "crs.txt"

    TOWER_TOP_CSV = OUTPUT_DIR / "tower_tops_audit.csv"
    TOWER_BOTTOM_CSV = OUTPUT_DIR / "tower_bottoms_audit.csv"
    TOWER_BOTTOM_VALIDATION_CSV = OUTPUT_DIR / "tower_bottom_validation.csv"
    TOWER_BOTTOM_VALIDATION_GPKG = OUTPUT_DIR / "tower_bottom_validation.gpkg"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse a working LAZ from V1, but move it out of the deliverables folder.
    legacy_working = OUTPUT_ROOT / "_working_187_215.laz"
    if legacy_working.exists() and not SOURCE_LAZ.exists():
        legacy_working.replace(SOURCE_LAZ)
    elif legacy_working.exists() and SOURCE_LAZ.exists():
        legacy_working.unlink()

    # Remove known V1 product files from the root so only the three final
    # deliverables remain after the new pipeline is run. Diagnostics are retained.
    for legacy_name in (
        "wires_3d.gpkg",
        "tower_tops.gpkg",
        "tower_tops.laz",
        "tower_tops.csv",
        "tower_bottoms.gpkg",
        "tower_bottoms.laz",
        "tower_bottoms.csv",
        "tower_bottom_validation.csv",
        "tower_bottom_validation.gpkg",
    ):
        legacy_path = OUTPUT_ROOT / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def find_input_lidar_files() -> list[Path]:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found:\n  {INPUT_FOLDER}")
    files = sorted(
        [
            p for p in INPUT_FOLDER.iterdir()
            if p.is_file() and p.suffix.lower() in {".las", ".laz"}
        ],
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise FileNotFoundError(
            f"No LAS/LAZ files found in:\n  {INPUT_FOLDER}\n"
            "Place the tiled point cloud there or pass --input-folder."
        )
    return files


def _extra_signature(header) -> tuple:
    return tuple(
        (dimension.name, str(dimension.dtype))
        for dimension in header.point_format.extra_dimensions
    )


def _parse_header_crs(header):
    try:
        return header.parse_crs()
    except Exception:
        return None


def _crs_equivalent(first, second) -> bool:
    if first is None or second is None:
        return False
    try:
        return bool(first.equals(second, ignore_axis_order=True))
    except TypeError:
        try:
            return bool(first.equals(second))
        except Exception:
            return first == second
    except Exception:
        return first == second


def _normalise_epsg(value) -> tuple[int, CRS]:
    text_value = str(value).strip().upper()
    if text_value.startswith("EPSG:"):
        text_value = text_value.split(":", 1)[1].strip()
    if not text_value or not text_value.isdigit():
        raise ValueError(f"Invalid EPSG code: {value!r}")
    code = int(text_value)
    crs = CRS.from_epsg(code)
    if crs is None:
        raise ValueError(f"EPSG:{code} could not be resolved.")
    if not crs.is_projected:
        raise ValueError(
            f"EPSG:{code} is not a projected CRS. This workflow uses linear XY "
            "distances, so enter the projected CRS used by the LiDAR/centreline."
        )
    return code, crs


def _crs_label(crs) -> str:
    if crs is None:
        return "<missing CRS>"
    try:
        authority = crs.to_authority()
    except Exception:
        authority = None
    if authority:
        return f"{authority[0]}:{authority[1]} | {crs.name}"
    try:
        return crs.name or crs.to_string()
    except Exception:
        return str(crs)


def resolve_effective_crs(files: list[Path], epsg_fallback=None):
    """Resolve one authoritative CRS for the whole run.

    Embedded LiDAR CRS is authoritative. --epsg/prompt is only a fallback when
    every input tile is missing CRS metadata.
    """
    global EFFECTIVE_CRS, EFFECTIVE_CRS_SOURCE, FALLBACK_CRS

    valid = []
    missing = []
    for path in files:
        with laspy.open(path) as reader:
            crs = _parse_header_crs(reader.header)
        if crs is None:
            missing.append(path)
        else:
            valid.append((path, crs))

    if valid:
        reference_path, reference_crs = valid[0]
        for path, current_crs in valid[1:]:
            if not _crs_equivalent(reference_crs, current_crs):
                raise RuntimeError(
                    "Input LAS/LAZ tiles contain non-equivalent embedded CRSs.\n"
                    f"  reference: {reference_path.name}: {_crs_label(reference_crs)}\n"
                    f"  mismatch:  {path.name}: {_crs_label(current_crs)}\n"
                    "Fix the source metadata before running the workflow."
                )

        if epsg_fallback is not None:
            code, supplied = _normalise_epsg(epsg_fallback)
            if not _crs_equivalent(reference_crs, supplied):
                log(
                    f"\nNOTE: --epsg {code} was supplied, but an embedded LiDAR CRS "
                    "already exists. The embedded CRS is authoritative and --epsg "
                    "will not override it."
                )

        EFFECTIVE_CRS = reference_crs
        EFFECTIVE_CRS_SOURCE = f"embedded LiDAR metadata ({reference_path.name})"
        FALLBACK_CRS = reference_crs

        log(f"\nCRS: {_crs_label(reference_crs)}")
        log(f"  source: {EFFECTIVE_CRS_SOURCE}")
        if missing:
            log(
                f"  warning: {len(missing):,} / {len(files):,} input tiles have no "
                "readable CRS metadata; they are assumed to already use the same coordinates."
            )
        return reference_crs

    if epsg_fallback is not None:
        code, resolved = _normalise_epsg(epsg_fallback)
        source = "--epsg fallback"
    else:
        log("\nLAS/LAZ has no CRS metadata.")
        log("Enter the projected EPSG code used by BOTH the LiDAR and centreline.")
        while True:
            try:
                entered = input("EPSG code: ").strip()
            except EOFError as exc:
                raise RuntimeError(
                    "LiDAR CRS is missing and no interactive input is available. "
                    "Run again with --epsg <code>."
                ) from exc
            try:
                code, resolved = _normalise_epsg(entered)
                break
            except Exception as exc:
                log(f"  Invalid EPSG: {exc}")
        source = "interactive EPSG fallback"

    EFFECTIVE_CRS = resolved
    EFFECTIVE_CRS_SOURCE = source
    FALLBACK_CRS = resolved

    log(f"\nCRS: {_crs_label(resolved)}")
    log(f"  source: {source}")
    log("  source LAS/LAZ files are not modified; CRS is applied to generated products.")
    return resolved


def first_source_header_and_crs(files: list[Path]):
    first_header = None
    first_embedded_crs = None
    for path in files:
        with laspy.open(path) as reader:
            if first_header is None:
                first_header = reader.header.copy()
            if first_embedded_crs is None:
                first_embedded_crs = _parse_header_crs(reader.header)
    if first_header is None:
        raise RuntimeError("No LiDAR header could be read.")

    crs = first_embedded_crs if first_embedded_crs is not None else EFFECTIVE_CRS
    return first_header, crs


def write_prj(path: Path, crs) -> None:
    if crs is None:
        return
    try:
        wkt = crs.to_wkt(version="WKT1_ESRI")
    except Exception:
        wkt = crs.to_wkt()
    path.write_text(wkt + "\n", encoding="utf-8")


def write_crs_txt(crs) -> None:
    if crs is None:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        authority = crs.to_authority()
    except Exception:
        authority = None
    lines = []
    if authority:
        lines.append(f"{authority[0]}:{authority[1]}")
    else:
        try:
            lines.append(crs.to_string())
        except Exception:
            lines.append(str(crs))
    try:
        if crs.name:
            lines.append(crs.name)
    except Exception:
        pass
    lines.append(f"CRS source: {EFFECTIVE_CRS_SOURCE}")
    CRS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_unit_to_metre_factor(crs) -> float:
    if crs is None:
        return 1.0
    try:
        axis_info = crs.axis_info
        if axis_info:
            factor = float(axis_info[0].unit_conversion_factor)
            if np.isfinite(factor) and factor > 0.0:
                return factor
    except Exception:
        pass
    return 1.0


def prepare_wire_working_laz(
    files: list[Path],
    effective_crs=None,
    force: bool = False,
) -> Path:
    """Merge only classes 187 and 215 into one working LAZ for the legacy engine."""
    if SOURCE_LAZ.exists() and not force:
        output_mtime = SOURCE_LAZ.stat().st_mtime
        source_current = all(path.stat().st_mtime <= output_mtime for path in files)
        with laspy.open(SOURCE_LAZ) as reader:
            cached_crs = _parse_header_crs(reader.header)
        crs_current = (
            effective_crs is None and cached_crs is None
        ) or (
            effective_crs is not None
            and cached_crs is not None
            and _crs_equivalent(effective_crs, cached_crs)
        )
        if source_current and crs_current:
            log(f"\nReusing working LAZ:\n  {SOURCE_LAZ}")
            return SOURCE_LAZ
        if source_current and not crs_current:
            log(
                "\nWorking LAZ CRS is missing/different from the resolved run CRS; "
                "rebuilding it so CRS metadata is embedded correctly."
            )

    log("\nPreparing wire working LAZ (classes 187 + 215)...")
    reference_header = None
    reference_pf = None
    reference_extra = None
    reference_crs = None
    scales = []
    mins = []

    for path in files:
        with laspy.open(path) as reader:
            header = reader.header
            if reference_header is None:
                reference_header = header.copy()
                reference_pf = header.point_format.id
                reference_extra = _extra_signature(header)
            else:
                if header.point_format.id != reference_pf:
                    raise RuntimeError(
                        "Input tiles use different LAS point formats. "
                        f"Reference={reference_pf}, {path.name}={header.point_format.id}."
                    )
                if _extra_signature(header) != reference_extra:
                    raise RuntimeError(
                        "Input tiles use different LAS Extra Bytes layouts. "
                        f"Problem file: {path.name}"
                    )
            current_crs = _parse_header_crs(header)
            if reference_crs is None and current_crs is not None:
                reference_crs = current_crs
            scales.append(np.asarray(header.scales, dtype=np.float64))
            mins.append(np.asarray(header.mins, dtype=np.float64))

    out_scales = np.min(np.vstack(scales), axis=0)
    global_min = np.min(np.vstack(mins), axis=0)
    out_offsets = np.floor(global_min / out_scales) * out_scales
    out_header = reference_header.copy()
    out_header.scales = out_scales
    out_header.offsets = out_offsets
    out_header.point_count = 0
    output_crs = reference_crs if reference_crs is not None else effective_crs
    if output_crs is not None:
        try:
            out_header.add_crs(output_crs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not write resolved CRS to working LAZ: {type(exc).__name__}: {exc}"
            ) from exc

    if SOURCE_LAZ.exists():
        SOURCE_LAZ.unlink()
    SOURCE_LAZ.parent.mkdir(parents=True, exist_ok=True)

    total_187 = 0
    total_215 = 0
    with laspy.open(SOURCE_LAZ, mode="w", header=out_header, do_compress=True) as writer:
        for number, path in enumerate(files, start=1):
            file_187 = 0
            file_215 = 0
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(LAS_CHUNK_POINTS):
                    cls = np.asarray(chunk.classification)
                    mask = (cls == 187) | (cls == 215)
                    if not np.any(mask):
                        continue
                    selected = chunk[np.flatnonzero(mask)]
                    selected.change_scaling(scales=out_scales, offsets=out_offsets)
                    writer.write_points(selected)
                    kept_cls = cls[mask]
                    file_187 += int(np.count_nonzero(kept_cls == 187))
                    file_215 += int(np.count_nonzero(kept_cls == 215))
            total_187 += file_187
            total_215 += file_215
            log(
                f"  [{number}/{len(files)}] {path.name}: "
                f"class187={file_187:,}, class215={file_215:,}"
            )

    if total_187 == 0:
        raise RuntimeError("No conductor points found in class 187.")
    if total_215 == 0:
        raise RuntimeError("No structure points found in class 215.")

    log(
        f"Working LAZ written:\n  {SOURCE_LAZ}\n"
        f"  class 187: {total_187:,}\n"
        f"  class 215: {total_215:,}\n"
        f"  CRS: {_crs_label(output_crs)}"
    )
    return SOURCE_LAZ


def read_product_structures(files: list[Path]) -> tuple[list[ProductStructure], object, object, float]:
    header, crs = first_source_header_and_crs(files)
    unit_to_m = source_unit_to_metre_factor(crs)
    parts = []
    for path in files:
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(LAS_CHUNK_POINTS):
                cls = np.asarray(chunk.classification)
                mask = cls == 215
                if not np.any(mask):
                    continue
                parts.append(np.column_stack((
                    np.asarray(chunk.x, dtype=np.float64)[mask],
                    np.asarray(chunk.y, dtype=np.float64)[mask],
                    np.asarray(chunk.z, dtype=np.float64)[mask],
                )))
    if not parts:
        raise RuntimeError("No class-215 points found in the input tiles.")
    xyz = np.vstack(parts)
    eps_source = STRUCTURE_CLUSTER_EPS_M / unit_to_m
    labels = DBSCAN(eps=eps_source, min_samples=2, n_jobs=1).fit_predict(xyz[:, :2])
    raw = []
    for label in sorted(set(int(v) for v in labels if int(v) >= 0)):
        current = xyz[labels == label]
        if len(current) < 3:
            continue
        raw.append((np.median(current, axis=0), current))
    raw.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
    structures = [
        ProductStructure(
            structure_id=f"LIDAR_S{index:04d}",
            anchor_xyz=np.asarray(anchor, dtype=np.float64),
            xyz=current,
        )
        for index, (anchor, current) in enumerate(raw, start=1)
    ]
    if not structures:
        raise RuntimeError("Class 215 was found, but no valid structure clusters were created.")
    return structures, header, crs, unit_to_m


def build_centreline_node_candidates(path: Path, unit_to_m: float) -> tuple[list[CentrelineNodeCandidate], Path]:
    sequences, explicit_points, resolved = read_centerline_sequences(path)
    records = []
    for x, y in explicit_points:
        records.append((float(x), float(y), "EXPLICIT"))
    for sequence in sequences:
        for x, y in sequence["points"]:
            records.append((float(x), float(y), "LINE_VERTEX"))
    if not records:
        raise RuntimeError("No point/node candidates could be derived from the centreline.")

    xy = np.asarray([(r[0], r[1]) for r in records], dtype=np.float64)
    merge_tol_source = 0.05 / unit_to_m
    labels = DBSCAN(eps=merge_tol_source, min_samples=1, n_jobs=1).fit_predict(xy)
    nodes = []
    for label in sorted(set(int(v) for v in labels)):
        indices = np.flatnonzero(labels == label)
        current_xy = xy[indices]
        kinds = [records[int(i)][2] for i in indices]
        explicit_indices = [int(i) for i in indices if records[int(i)][2] == "EXPLICIT"]
        if explicit_indices:
            # Preserve the exact XY of the source CAD node. Do not average/move it.
            chosen_xy = xy[explicit_indices[0]].copy()
            source_kind = "EXPLICIT"
        else:
            # Same rule for topology vertices: retain an actual source vertex.
            chosen_xy = current_xy[0].copy()
            source_kind = "LINE_VERTEX"
        nodes.append((chosen_xy, source_kind, len(indices)))
    nodes.sort(key=lambda item: (float(item[0][0]), float(item[0][1])))
    output = [
        CentrelineNodeCandidate(
            node_id=f"CL_N{index:04d}",
            xy=np.asarray(node_xy, dtype=np.float64),
            source_kind=source_kind,
            source_count=int(source_count),
        )
        for index, (node_xy, source_kind, source_count) in enumerate(nodes, start=1)
    ]
    return output, resolved


def match_structures_to_nodes(
    structures: list[ProductStructure],
    nodes: list[CentrelineNodeCandidate],
    unit_to_m: float,
    max_distance_m: float,
):
    if not structures or not nodes:
        return []
    node_xy = np.vstack([node.xy for node in nodes])
    tree = cKDTree(node_xy)
    max_source = max_distance_m / unit_to_m
    candidate_pairs = []
    for s_index, structure in enumerate(structures):
        nearby = tree.query_ball_point(structure.anchor_xyz[:2], r=max_source)
        for n_index in nearby:
            distance_source = float(np.linalg.norm(structure.anchor_xyz[:2] - nodes[int(n_index)].xy))
            candidate_pairs.append((distance_source, s_index, int(n_index)))
    candidate_pairs.sort(key=lambda row: row[0])
    used_structures = set()
    used_nodes = set()
    matches = []
    for distance_source, s_index, n_index in candidate_pairs:
        if s_index in used_structures or n_index in used_nodes:
            continue
        used_structures.add(s_index)
        used_nodes.add(n_index)
        matches.append({
            "structure": structures[s_index],
            "node": nodes[n_index],
            "match_distance_m": distance_source * unit_to_m,
        })
    matches.sort(key=lambda row: row["structure"].structure_id)
    return matches


def write_point_product_laz(
    frame: pd.DataFrame,
    path: Path,
    source_header,
    crs,
    user_data_value: int,
    classification: int = 215,
) -> None:
    if path.exists():
        path.unlink()
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.asarray(source_header.scales, dtype=np.float64)
    product_min = np.asarray([
        float(frame["x"].min()),
        float(frame["y"].min()),
        float(frame["z"].min()),
    ], dtype=np.float64)
    header.offsets = np.floor(product_min / header.scales) * header.scales
    if crs is not None:
        try:
            header.add_crs(crs)
        except Exception:
            pass
    las = laspy.LasData(header)
    count = len(frame)
    las.x = frame["x"].to_numpy(dtype=np.float64)
    las.y = frame["y"].to_numpy(dtype=np.float64)
    las.z = frame["z"].to_numpy(dtype=np.float64)
    las.classification = np.full(count, classification, dtype=np.uint8)
    las.user_data = np.full(count, user_data_value, dtype=np.uint8)
    if "structure_number" in frame.columns:
        las.point_source_id = np.clip(
            frame["structure_number"].to_numpy(dtype=np.int64), 0, 65535
        ).astype(np.uint16)
    las.write(path)


def write_xyz(frame: pd.DataFrame, path: Path) -> None:
    """Write a clean headerless X Y Z ASCII deliverable."""
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    if len(xyz) == 0:
        raise RuntimeError(f"No finite XYZ records available for {path.name}.")
    if path.exists():
        path.unlink()
    np.savetxt(
        path,
        xyz,
        fmt="%.3f",
        delimiter=" ",
        newline="\n",
    )


def run_tower_tops(files: list[Path], structure_match_m: float) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    structures, source_header, crs, unit_to_m = read_product_structures(files)

    # Tower tops are a class-215 product and must not disappear merely because
    # a centreline node cannot be matched. Centreline proximity is diagnostic only.
    nodes, resolved = build_centreline_node_candidates(CENTERLINE_PATH, unit_to_m)
    node_tree = cKDTree(np.vstack([node.xy for node in nodes])) if nodes else None
    match_gate_source = structure_match_m / unit_to_m

    rows = []
    matched_count = 0
    for number, structure in enumerate(structures, start=1):
        z = np.asarray(structure.xyz[:, 2], dtype=np.float64)
        selected_z = float(np.percentile(z, TOWER_TOP_Z_PERCENTILE))
        raw_max_z = float(np.max(z))
        anchor = structure.anchor_xyz

        nearest_node_id = ""
        nearest_node_source = ""
        nearest_node_x = np.nan
        nearest_node_y = np.nan
        nearest_node_distance_m = np.nan
        centreline_match_status = "NO_CENTRELINE_NODE"

        if node_tree is not None:
            distance_source, node_index = node_tree.query(anchor[:2], k=1)
            node = nodes[int(node_index)]
            nearest_node_id = node.node_id
            nearest_node_source = node.source_kind
            nearest_node_x = float(node.xy[0])
            nearest_node_y = float(node.xy[1])
            nearest_node_distance_m = float(distance_source * unit_to_m)
            if float(distance_source) <= match_gate_source:
                centreline_match_status = "MATCHED_WITHIN_GATE"
                matched_count += 1
            else:
                centreline_match_status = "NEAREST_NODE_OUTSIDE_GATE"

        rows.append({
            "structure_id": structure.structure_id,
            "structure_number": number,
            "assigned_215_points": int(len(structure.xyz)),
            "selection_method": f"CENTROID_XY_P{TOWER_TOP_Z_PERCENTILE:g}_Z",
            "x": float(anchor[0]),
            "y": float(anchor[1]),
            "z": selected_z,
            "nearest_node_id": nearest_node_id,
            "nearest_node_source": nearest_node_source,
            "nearest_node_x": nearest_node_x,
            "nearest_node_y": nearest_node_y,
            "nearest_node_distance_m": nearest_node_distance_m,
            "centreline_match_gate_m": float(structure_match_m),
            "centreline_match_status": centreline_match_status,
            "raw_max_z": raw_max_z,
            "raw_max_above_selected_m": float((raw_max_z - selected_z) * unit_to_m),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(TOWER_TOP_CSV, index=False, encoding="utf-8-sig")
    write_xyz(frame, TOWER_TOP_XYZ)
    write_prj(TOWER_TOP_PRJ, crs)
    log(
        f"\nTower tops complete: {len(frame):,}"
        f"\n  within {structure_match_m:.2f} m of a centreline node: {matched_count:,}"
        f"\n  centreline: {resolved}"
        f"\n  XYZ: {TOWER_TOP_XYZ}"
        f"\n  audit: {TOWER_TOP_CSV}"
    )

def collect_ground_points_for_nodes(
    files: list[Path],
    node_xy: np.ndarray,
    ground_class: int,
    max_radius_m: float,
    unit_to_m: float,
):
    support = [[] for _ in range(len(node_xy))]
    tree = cKDTree(node_xy)
    max_source = max_radius_m / unit_to_m
    total_ground = 0
    assigned = 0

    for path in files:
        with laspy.open(path) as reader:
            for chunk in reader.chunk_iterator(LAS_CHUNK_POINTS):
                cls = np.asarray(chunk.classification)
                mask = cls == ground_class
                count = int(np.count_nonzero(mask))
                if count == 0:
                    continue
                total_ground += count
                xyz = np.column_stack((
                    np.asarray(chunk.x, dtype=np.float64)[mask],
                    np.asarray(chunk.y, dtype=np.float64)[mask],
                    np.asarray(chunk.z, dtype=np.float64)[mask],
                ))
                distance, index = tree.query(xyz[:, :2], k=1)
                keep = distance <= max_source
                if not np.any(keep):
                    continue
                xyz_keep = xyz[keep]
                index_keep = np.asarray(index[keep], dtype=np.int64)
                assigned += len(xyz_keep)
                for node_index in np.unique(index_keep):
                    local = index_keep == node_index
                    support[int(node_index)].append(xyz_keep[local])

    if total_ground == 0:
        raise RuntimeError(
            f"No points found in ground class {ground_class}. "
            "Use --ground-class with the correct ground classification."
        )
    output = [
        np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64)
        for parts in support
    ]
    log(
        f"\nGround support: class={ground_class}, source points={total_ground:,}, "
        f"assigned within {max_radius_m:.2f} m={assigned:,}"
    )
    return output


def _robust_plane_at_node(node_xy: np.ndarray, xyz: np.ndarray):
    dx = xyz[:, 0] - float(node_xy[0])
    dy = xyz[:, 1] - float(node_xy[1])
    z = xyz[:, 2]
    A = np.column_stack((dx, dy, np.ones(len(xyz), dtype=np.float64)))
    mask = np.ones(len(xyz), dtype=bool)
    beta = None
    for _ in range(6):
        if np.count_nonzero(mask) < 3:
            break
        beta, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        residual = z - A @ beta
        active = residual[mask]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        if mad <= 1e-9:
            break
        sigma = 1.4826 * mad
        new_mask = np.abs(residual - median) <= 3.0 * sigma
        if np.count_nonzero(new_mask) < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if beta is None:
        return None
    residual = z[mask] - A[mask] @ beta
    rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else np.nan
    return float(beta[2]), int(np.count_nonzero(mask)), rmse


def estimate_ground_at_node(
    node_xy: np.ndarray,
    xyz: np.ndarray,
    unit_to_m: float,
    primary_radius_m: float,
    fallback_radius_m: float,
):
    if len(xyz) == 0:
        return None
    distance_m = np.linalg.norm(xyz[:, :2] - node_xy[None, :], axis=1) * unit_to_m
    primary = distance_m <= primary_radius_m
    if np.count_nonzero(primary) >= 4:
        chosen = xyz[primary]
        used_radius_m = primary_radius_m
    else:
        fallback = distance_m <= fallback_radius_m
        if not np.any(fallback):
            return None
        chosen = xyz[fallback]
        used_radius_m = fallback_radius_m

    chosen_distance_m = np.linalg.norm(chosen[:, :2] - node_xy[None, :], axis=1) * unit_to_m
    nearest_index = int(np.argmin(chosen_distance_m))
    nearest = chosen[nearest_index]

    if len(chosen) >= 6:
        plane = _robust_plane_at_node(node_xy, chosen)
    else:
        plane = None

    if plane is not None:
        selected_z, used_points, plane_rmse_source = plane
        method = "ROBUST_LOCAL_GROUND_PLANE"
        plane_rmse_m = plane_rmse_source * unit_to_m
    else:
        z = chosen[:, 2]
        median = float(np.median(z))
        mad = float(np.median(np.abs(z - median)))
        if mad > 1e-9:
            keep = np.abs(z - median) <= 3.0 * 1.4826 * mad
            clipped = z[keep] if np.any(keep) else z
        else:
            clipped = z
        selected_z = float(np.mean(clipped))
        used_points = int(len(clipped))
        method = "ROBUST_LOCAL_GROUND_MEAN"
        plane_rmse_m = np.nan

    z_values = chosen[:, 2]
    return {
        "selected_z": selected_z,
        "method": method,
        "used_points": used_points,
        "support_points": int(len(chosen)),
        "support_radius_m": float(used_radius_m),
        "nearest_ground_distance_m": float(chosen_distance_m[nearest_index]),
        "nearest_ground_z": float(nearest[2]),
        "local_ground_z_p10": float(np.quantile(z_values, 0.10)),
        "local_ground_z_median": float(np.median(z_values)),
        "local_ground_z_p90": float(np.quantile(z_values, 0.90)),
        "local_ground_spread_m": float((np.quantile(z_values, 0.90) - np.quantile(z_values, 0.10)) * unit_to_m),
        "plane_rmse_m": float(plane_rmse_m),
    }


def build_matched_structure_nodes(
    files: list[Path],
    structure_match_m: float,
):
    structures, source_header, crs, unit_to_m = read_product_structures(files)
    nodes, resolved = build_centreline_node_candidates(CENTERLINE_PATH, unit_to_m)
    matches = match_structures_to_nodes(structures, nodes, unit_to_m, structure_match_m)
    if not matches:
        raise RuntimeError("No structure-to-centreline node matches found.")
    log(
        f"\nStructure/node matching: {len(matches):,} matched / "
        f"{len(structures):,} class-215 clusters / {len(nodes):,} centreline nodes "
        f"(gate {structure_match_m:.2f} m)"
    )
    return matches, source_header, crs, unit_to_m, resolved


def run_tower_bottoms(
    files: list[Path],
    ground_class: int,
    structure_match_m: float,
    primary_radius_m: float,
    fallback_radius_m: float,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    matches, source_header, crs, unit_to_m, resolved = build_matched_structure_nodes(
        files, structure_match_m
    )
    node_xy = np.vstack([match["node"].xy for match in matches])
    ground_support = collect_ground_points_for_nodes(
        files,
        node_xy,
        ground_class,
        fallback_radius_m,
        unit_to_m,
    )

    rows = []
    missing = 0
    for index, (match, support) in enumerate(zip(matches, ground_support), start=1):
        structure = match["structure"]
        node = match["node"]
        estimate = estimate_ground_at_node(
            node.xy,
            support,
            unit_to_m,
            primary_radius_m,
            fallback_radius_m,
        )

        base = {
            "structure_id": structure.structure_id,
            "structure_number": index,
            "node_id": node.node_id,
            "node_source": node.source_kind,
            "match_distance_m": float(match["match_distance_m"]),
            "ground_class": int(ground_class),
            # Critical rule: exact source centreline-node XY is retained.
            "x": float(node.xy[0]),
            "y": float(node.xy[1]),
        }

        if estimate is None:
            missing += 1
            rows.append({
                **base,
                "generation_status": "NO_GROUND_SUPPORT",
                "z": np.nan,
                "z_method": "",
                "ground_points_used": 0,
                "ground_support_points": int(len(support)),
                "ground_support_radius_m": np.nan,
                "nearest_ground_distance_m": np.nan,
                "nearest_ground_z": np.nan,
                "local_ground_z_p10": np.nan,
                "local_ground_z_median": np.nan,
                "local_ground_z_p90": np.nan,
                "local_ground_spread_m": np.nan,
                "plane_rmse_m": np.nan,
            })
            continue

        rows.append({
            **base,
            "generation_status": "OK",
            "z": float(estimate["selected_z"]),
            "z_method": estimate["method"],
            "ground_points_used": int(estimate["used_points"]),
            "ground_support_points": int(estimate["support_points"]),
            "ground_support_radius_m": float(estimate["support_radius_m"]),
            "nearest_ground_distance_m": float(estimate["nearest_ground_distance_m"]),
            "nearest_ground_z": float(estimate["nearest_ground_z"]),
            "local_ground_z_p10": float(estimate["local_ground_z_p10"]),
            "local_ground_z_median": float(estimate["local_ground_z_median"]),
            "local_ground_z_p90": float(estimate["local_ground_z_p90"]),
            "local_ground_spread_m": float(estimate["local_ground_spread_m"]),
            "plane_rmse_m": float(estimate["plane_rmse_m"]),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(TOWER_BOTTOM_CSV, index=False, encoding="utf-8-sig")

    valid = frame[np.isfinite(pd.to_numeric(frame["z"], errors="coerce"))].copy()
    if valid.empty:
        raise RuntimeError(
            "No tower bottoms could be assigned a ground elevation. "
            f"Audit written to {TOWER_BOTTOM_CSV}"
        )

    write_xyz(valid, TOWER_BOTTOM_XYZ)
    write_prj(TOWER_BOTTOM_PRJ, crs)
    log(
        f"\nTower bottoms complete: {len(valid):,} / {len(frame):,} matched structures"
        f"\n  no-ground support: {missing:,}"
        f"\n  XY source: exact centreline node ({resolved})"
        f"\n  XYZ: {TOWER_BOTTOM_XYZ}"
        f"\n  audit: {TOWER_BOTTOM_CSV}"
    )


def run_tower_bottom_validation(
    files: list[Path],
    ground_class: int,
    fallback_radius_m: float,
    tolerance_m: float,
) -> None:
    if not TOWER_BOTTOM_CSV.exists():
        raise FileNotFoundError(
            f"Tower bottom CSV not found:\n  {TOWER_BOTTOM_CSV}\nRun --stage bottoms first."
        )
    bottoms = pd.read_csv(TOWER_BOTTOM_CSV)
    if bottoms.empty:
        raise RuntimeError("Tower bottom CSV is empty.")
    _, crs = first_source_header_and_crs(files)
    unit_to_m = source_unit_to_metre_factor(crs)
    node_xy = bottoms[["x", "y"]].to_numpy(dtype=np.float64)
    support = collect_ground_points_for_nodes(
        files, node_xy, ground_class, fallback_radius_m, unit_to_m
    )

    validation_rows = []
    for row, xyz in zip(bottoms.to_dict("records"), support):
        bottom_z = pd.to_numeric(pd.Series([row.get("z")]), errors="coerce").iloc[0]
        if not np.isfinite(bottom_z):
            nearest_distance_m = np.nan
            nearest_z = np.nan
            delta_m = np.nan
            status = "MISSING_BOTTOM"
            support_count = int(len(xyz))
        elif len(xyz) == 0:
            nearest_distance_m = np.nan
            nearest_z = np.nan
            delta_m = np.nan
            status = "NO_GROUND_SUPPORT"
            support_count = 0
        else:
            distances_m = np.linalg.norm(
                xyz[:, :2] - np.asarray([row["x"], row["y"]], dtype=np.float64)[None, :],
                axis=1,
            ) * unit_to_m
            nearest_index = int(np.argmin(distances_m))
            nearest_distance_m = float(distances_m[nearest_index])
            nearest_z = float(xyz[nearest_index, 2])
            delta_m = float((float(bottom_z) - nearest_z) * unit_to_m)
            support_count = int(len(xyz))
            if abs(delta_m) <= tolerance_m:
                status = "ON_GROUND"
            elif delta_m < -tolerance_m:
                status = "BELOW_GROUND"
            else:
                status = "ABOVE_GROUND"

        output = dict(row)
        output.update({
            "validation_ground_class": int(ground_class),
            "validation_support_points": support_count,
            "validation_nearest_ground_distance_m": nearest_distance_m,
            "validation_nearest_ground_z": nearest_z,
            "bottom_minus_nearest_ground_m": delta_m,
            "validation_tolerance_m": float(tolerance_m),
            "validation_status": status,
        })
        validation_rows.append(output)

    frame = pd.DataFrame(validation_rows)
    frame.to_csv(TOWER_BOTTOM_VALIDATION_CSV, index=False, encoding="utf-8-sig")
    if TOWER_BOTTOM_VALIDATION_GPKG.exists():
        TOWER_BOTTOM_VALIDATION_GPKG.unlink()
    validation_geometry = []
    for r in frame.itertuples():
        z_value = pd.to_numeric(pd.Series([getattr(r, "z", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(z_value):
            validation_geometry.append(Point(float(r.x), float(r.y), float(z_value)))
        else:
            validation_geometry.append(Point(float(r.x), float(r.y)))
    gdf = gpd.GeoDataFrame(
        frame.copy(),
        geometry=validation_geometry,
        crs=crs,
    )
    gdf.to_file(
        TOWER_BOTTOM_VALIDATION_GPKG,
        layer="tower_bottom_validation",
        driver="GPKG",
        engine="pyogrio",
    )
    counts = frame["validation_status"].value_counts().to_dict()
    log("\nTower bottom validation:")
    for status, count in counts.items():
        log(f"  {status}: {int(count):,}")
    log(f"  {TOWER_BOTTOM_VALIDATION_CSV}")
    log(f"  {TOWER_BOTTOM_VALIDATION_GPKG}")



def _point_product_gdf_from_csv(path: Path, crs, *, require_ok: bool = False) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    if frame.empty or not {"x", "y", "z"}.issubset(frame.columns):
        return None

    x = pd.to_numeric(frame["x"], errors="coerce")
    y = pd.to_numeric(frame["y"], errors="coerce")
    z = pd.to_numeric(frame["z"], errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if require_ok and "generation_status" in frame.columns:
        finite &= frame["generation_status"].astype(str).str.upper().eq("OK")

    frame = frame.loc[finite].copy()
    if frame.empty:
        return None

    frame["x"] = x.loc[finite].to_numpy(dtype=np.float64)
    frame["y"] = y.loc[finite].to_numpy(dtype=np.float64)
    frame["z"] = z.loc[finite].to_numpy(dtype=np.float64)

    geometry = [
        Point(float(px), float(py), float(pz))
        for px, py, pz in frame[["x", "y", "z"]].itertuples(index=False, name=None)
    ]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=crs)


def refresh_product_gpkg() -> list[str]:
    """Rebuild one clean GeoPackage from the currently available final products."""
    layers: list[tuple[str, gpd.GeoDataFrame]] = []

    if OUTPUT_GPKG.exists():
        try:
            wires = gpd.read_file(OUTPUT_GPKG, layer="wires_3d", engine="pyogrio")
            if not wires.empty:
                if wires.crs is None and EFFECTIVE_CRS is not None:
                    wires = wires.set_crs(EFFECTIVE_CRS, allow_override=True)
                layers.append(("wires_3d", wires))
        except Exception as exc:
            log(f"\nWARNING: could not add wires_3d to product GPKG: {exc}")

    tops = _point_product_gdf_from_csv(TOWER_TOP_CSV, EFFECTIVE_CRS)
    if tops is not None and not tops.empty:
        layers.append(("tower_tops", tops))

    bottoms = _point_product_gdf_from_csv(
        TOWER_BOTTOM_CSV,
        EFFECTIVE_CRS,
        require_ok=True,
    )
    if bottoms is not None and not bottoms.empty:
        layers.append(("tower_bottoms", bottoms))

    if not layers:
        return []

    if PRODUCT_GPKG.exists():
        PRODUCT_GPKG.unlink()

    for index, (layer_name, gdf) in enumerate(layers):
        gdf.to_file(
            PRODUCT_GPKG,
            layer=layer_name,
            driver="GPKG",
            engine="pyogrio",
            mode="w" if index == 0 else "a",
        )

    # Read back layer names and geometry types. This catches accidental 2D export
    # or a failed append before the file is presented as a final product.
    written = pyogrio.list_layers(PRODUCT_GPKG)
    written_names = [str(row[0]) for row in written]
    expected_names = [name for name, _ in layers]
    missing = [name for name in expected_names if name not in written_names]
    if missing:
        raise RuntimeError(
            "Combined product GPKG verification failed. Missing layers: "
            + ", ".join(missing)
        )

    for layer_name, _ in layers:
        check = gpd.read_file(PRODUCT_GPKG, layer=layer_name, engine="pyogrio")
        if check.empty:
            raise RuntimeError(
                f"Combined product GPKG verification failed: {layer_name} is empty."
            )
        if any(
            geometry is None
            or geometry.is_empty
            or not getattr(geometry, "has_z", False)
            for geometry in check.geometry
        ):
            raise RuntimeError(
                f"Combined product GPKG verification failed: {layer_name} contains non-3D geometry."
            )

    log(
        f"\nCombined product GeoPackage: {PRODUCT_GPKG}"
        f"\n  layers: {', '.join(expected_names)}"
    )
    return expected_names


def export_wires_product() -> None:
    """Export final 3D wire geometry to a Bentley-compatible DGN V7 file.

    The standard GDAL DGN driver writes classic DGN (V7). This deliberately
    avoids the unavailable proprietary DGNv8/ODA driver. MicroStation can open
    the result directly. DGN V7 does not accept the arbitrary attribute schema
    from the diagnostic GeoPackage, so the final deliverable contains geometry
    only; all attributes also remain in diagnostics/wire_extraction_187.gpkg.
    """
    if not OUTPUT_GPKG.exists():
        raise FileNotFoundError(f"Wire diagnostic GPKG not found:\n  {OUTPUT_GPKG}")

    wires = gpd.read_file(OUTPUT_GPKG, layer="wires_3d", engine="pyogrio")
    if wires.empty:
        raise RuntimeError("The wires_3d diagnostic layer is empty.")

    # DGN driver cannot create the arbitrary fields carried by the GPKG.
    # Write geometry only and explicitly verify that every final line is 3D.
    geometry_only = gpd.GeoDataFrame(
        geometry=wires.geometry.copy(),
        crs=wires.crs,
    )
    two_d = [
        index
        for index, geometry in enumerate(geometry_only.geometry)
        if geometry is None or geometry.is_empty or not getattr(geometry, "has_z", False)
    ]
    if two_d:
        raise RuntimeError(
            f"Cannot export final DGN: {len(two_d):,} wire geometries are missing Z."
        )

    driver_mode = pyogrio.list_drivers().get("DGN", "")
    if "w" not in driver_mode:
        raise RuntimeError(
            "The GDAL DGN (V7) writer is not available in this Python/GDAL build. "
            "This is separate from the DGNv8/ODA reader used for the source centreline."
        )

    if WIRES_PRODUCT_DGN.exists():
        WIRES_PRODUCT_DGN.unlink()

    pyogrio.write_dataframe(
        geometry_only,
        WIRES_PRODUCT_DGN,
        driver="DGN",
    )

    # Read back once to make sure Z survived the format conversion.
    check = pyogrio.read_dataframe(WIRES_PRODUCT_DGN)
    if len(check) != len(geometry_only):
        raise RuntimeError(
            f"DGN verification failed: wrote {len(geometry_only):,} wires but "
            f"read back {len(check):,}."
        )
    if any(not getattr(g, "has_z", False) for g in check.geometry):
        raise RuntimeError("DGN verification failed: one or more wire elements lost Z.")

    product_crs = wires.crs if wires.crs is not None else EFFECTIVE_CRS
    write_prj(WIRES_PRODUCT_PRJ, product_crs)

    log(
        f"\nStandalone 3D wires: {len(wires):,}"
        f"\n  DGN V7: {WIRES_PRODUCT_DGN}"
        f"\n  attributes/QA: {OUTPUT_GPKG}"
    )


def inspect_inputs(files: list[Path]) -> None:
    counts = {}
    crs_values = []
    for path in files:
        with laspy.open(path) as reader:
            try:
                crs = reader.header.parse_crs()
            except Exception:
                crs = None
            crs_values.append(str(crs))
            for chunk in reader.chunk_iterator(LAS_CHUNK_POINTS):
                values, n = np.unique(np.asarray(chunk.classification), return_counts=True)
                for value, count in zip(values, n):
                    counts[int(value)] = counts.get(int(value), 0) + int(count)
    log("\nInput LiDAR class counts:")
    log(f"Resolved run CRS: {_crs_label(EFFECTIVE_CRS)}")
    for code, count in sorted(counts.items()):
        log(f"  class {code:>3}: {count:,}")
    log(f"\nCentreline: {CENTERLINE_PATH}")
    try:
        sequences, points, resolved = read_centerline_sequences(CENTERLINE_PATH)
        log(f"  resolved: {resolved}")
        log(f"  line sequences: {len(sequences):,}")
        log(f"  explicit points: {len(points):,}")
    except Exception as exc:
        log(f"  ERROR: {type(exc).__name__}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3D wires, tower tops and tower bottoms, plus a combined "
            "GeoPackage, with QA retained under diagnostics/."
        )
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", "inspect", "wires", "tops", "bottoms", "validate"],
        help="One or more stages to run. Default: all.",
    )
    parser.add_argument("--input-folder", default="input")
    parser.add_argument("--centreline", default=None, help="Centreline DXF/DGN. If omitted, auto-detect beside the script and prefer DXF.")
    parser.add_argument("--output-folder", default="wire_structure_outputs")
    parser.add_argument(
        "--epsg",
        default=None,
        help=(
            "Fallback EPSG code used only when every input LAS/LAZ is missing CRS. "
            "Example: --epsg 26917. Embedded LiDAR CRS is never overridden."
        ),
    )
    parser.add_argument("--ground-class", type=int, default=GROUND_CLASS_DEFAULT)
    parser.add_argument(
        "--structure-match-m", type=float, default=STRUCTURE_MATCH_MAX_M_DEFAULT
    )
    parser.add_argument(
        "--ground-radius-m", type=float, default=GROUND_PRIMARY_RADIUS_M_DEFAULT
    )
    parser.add_argument(
        "--ground-fallback-radius-m",
        type=float,
        default=GROUND_FALLBACK_RADIUS_M_DEFAULT,
    )
    parser.add_argument(
        "--ground-validation-tolerance-m",
        type=float,
        default=GROUND_VALIDATION_TOLERANCE_M_DEFAULT,
    )
    parser.add_argument(
        "--force-working-laz",
        action="store_true",
        help="Rebuild the temporary class-187/215 working LAZ even if it is current.",
    )
    return parser


def pipeline_main() -> None:
    args = build_arg_parser().parse_args()
    input_folder = _resolve_runtime_path(args.input_folder)
    centreline = resolve_centreline_path(args.centreline)
    output_root = _resolve_runtime_path(args.output_folder)
    configure_runtime_paths(input_folder, centreline, output_root)

    files = find_input_lidar_files()
    stages = args.stage
    if "all" in stages:
        stages = ["wires", "tops", "bottoms", "validate"]

    effective_crs = resolve_effective_crs(files, args.epsg)
    if any(stage != "inspect" for stage in stages):
        write_crs_txt(effective_crs)

    log("=" * 80)
    log("WIRE + STRUCTURE 3D PRODUCTS - CLASS 187 / STRUCTURES 215")
    log("=" * 80)
    log(f"Input tiles: {len(files):,}")
    log(f"Centreline: {CENTERLINE_PATH}")
    log(f"Output: {OUTPUT_ROOT}")
    log(f"Stages: {', '.join(stages)}")
    log(f"CRS: {_crs_label(effective_crs)}")

    for stage in stages:
        log("\n" + "=" * 80)
        log(f"STAGE: {stage.upper()}")
        log("=" * 80)
        if stage == "inspect":
            inspect_inputs(files)
        elif stage == "wires":
            prepare_wire_working_laz(
                files, effective_crs=effective_crs, force=args.force_working_laz
            )
            run_wires()
            export_wires_product()
        elif stage == "tops":
            run_tower_tops(files, args.structure_match_m)
        elif stage == "bottoms":
            run_tower_bottoms(
                files,
                args.ground_class,
                args.structure_match_m,
                args.ground_radius_m,
                args.ground_fallback_radius_m,
            )
        elif stage == "validate":
            run_tower_bottom_validation(
                files,
                args.ground_class,
                args.ground_fallback_radius_m,
                args.ground_validation_tolerance_m,
            )

    product_layers = []
    if any(stage in {"wires", "tops", "bottoms"} for stage in stages):
        product_layers = refresh_product_gpkg()

    log("\n" + "=" * 80)
    log("PIPELINE COMPLETE")
    log("=" * 80)
    log(
        f"Final deliverables:\n  {WIRES_PRODUCT_DGN}\n  {TOWER_TOP_XYZ}\n"
        f"  {TOWER_BOTTOM_XYZ}\n  {PRODUCT_GPKG}"
    )
    if product_layers:
        log(f"Product GPKG layers: {', '.join(product_layers)}")
    log(
        f"CRS metadata:\n  {CRS_TXT}\n  {WIRES_PRODUCT_PRJ}\n"
        f"  {TOWER_TOP_PRJ}\n  {TOWER_BOTTOM_PRJ}"
    )
    log(f"Diagnostics: {OUTPUT_DIR}")


if __name__ == "__main__":
    pipeline_main()
