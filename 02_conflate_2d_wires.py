#!/usr/bin/env python3
"""
02_conflate_2d_wires.py
=======================
Project-specific conflation stage for the 2D Wires product.

Purpose
-------
Take the survey-derived RAW 2D wires produced by 01_build_2d_wires.py and add
client/project attribution without changing the survey-derived geometry.

Inputs
------
1. 01_raw_2d_wires.gpkg (preferred; uses span_edges layer)
2. client source ZIP/folder containing transmission line source data
3. Project SoW KMZ
4. capture year

Truth hierarchy
---------------
GEOMETRY truth:
    raw 2D wires from stage 01

ATTRIBUTE truth:
    client source data (LINE_NO, VOLTAGE, COMPANY, GLOBALID...)

PROJECT / RoW truth:
    Project SoW KMZ hierarchy

Important: client source geometry and KMZ geometry are NEVER used to snap,
shift, reshape or replace the stage-01 wire geometry. They are matching and
attribution references only.

Current NMIP26068 assumptions
-----------------------------
* Client source line data contains LINE_NO and VOLTAGE.
* SoW KMZ hierarchy is Circuits -> <RoW> -> <Circuit> -> Placemarks.
* A circuit may occur in more than one RoW. In that case each raw span is
  assigned to the nearest KMZ section of the SAME circuit. This makes the
  method tolerant of a consistent spatial offset between client/KMZ data and
  LiDAR-derived NM geometry.
* Voltage comes from client source where available. KMZ voltage is fallback and
  a cross-check only.

Outputs
-------
02_final_2d_wires.gpkg
    wires_2d           final conflated product
    span_edges         attributed raw span segments; geometry unchanged
    kmz_reference      project reference used for RoW assignment
    client_reference   client line source geometry (reference only)

02_conflation_audit.csv
02_unmatched.csv
shp/<Circuit>_Wires_<CaptureYear>.shp

Dependencies
------------
pip install geopandas shapely pyproj pyogrio pandas numpy
"""

from __future__ import annotations

import argparse
import html
import math
import re
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, transform, unary_union


KML_NS = "http://www.opengis.net/kml/2.2"
K = "{" + KML_NS + "}"


def info(msg: str = "") -> None:
    print(msg, flush=True)


def txt(v: Any) -> str:
    return "" if v is None else str(v).strip()


def clean_path(value: str) -> Path:
    s = txt(value)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1]
    return Path(s).expanduser()


def circuit_key(v: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", txt(v).upper())


def normalise_voltage(v: Any) -> str:
    s = txt(v)
    if not s:
        return ""
    m = re.search(r"\d+(?:\.\d+)?", s)
    if m:
        x = float(m.group(0))
        return str(int(x)) if x.is_integer() else str(x)
    return s


def is_line(g) -> bool:
    return g is not None and not g.is_empty and g.geom_type in {"LineString", "MultiLineString"}


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", txt(s)).strip("._")
    return s or "UNKNOWN"


def prompt_path(label: str, extensions: set[str] | None = None, allow_dir: bool = False) -> Path:
    while True:
        p = clean_path(input(label))
        if not p.exists():
            info(f"Path does not exist: {p}")
            continue
        if p.is_dir() and allow_dir:
            return p
        if extensions and p.suffix.lower() not in extensions:
            info("Expected: " + ", ".join(sorted(extensions)))
            continue
        return p


def prompt_year() -> int:
    while True:
        raw = input("Capture year [2026]: ").strip() or "2026"
        if raw.isdigit() and 2000 <= int(raw) <= 2100:
            return int(raw)
        info("Enter a four-digit year.")


# -----------------------------------------------------------------------------
# Stage 01 input
# -----------------------------------------------------------------------------

def read_raw_stage01(path: Path) -> tuple[gpd.GeoDataFrame, str]:
    if path.suffix.lower() == ".gpkg":
        layers = [txt(x[0]) for x in pyogrio.list_layers(path)]
        layer = "span_edges" if "span_edges" in layers else ("wires_2d" if "wires_2d" in layers else layers[0])
        g = gpd.read_file(path, layer=layer, engine="pyogrio")
        return g, layer
    g = gpd.read_file(path, engine="pyogrio")
    return g, path.stem


def find_circuit_field(gdf: gpd.GeoDataFrame) -> str:
    lookup = {str(c).lower(): c for c in gdf.columns}
    for name in ("circuit", "circuit_id", "circuitid", "line_no", "lineno"):
        if name in lookup:
            return lookup[name]
    raise RuntimeError("Raw 2D wires do not contain a CIRCUIT field")


# -----------------------------------------------------------------------------
# Client source data
# -----------------------------------------------------------------------------

def locate_client_line_shp(root: Path) -> Path:
    candidates = list(root.rglob("*.shp"))
    scored = []
    for p in candidates:
        try:
            schema = pyogrio.read_info(p)
            fields = {str(f).upper() for f in schema.get("fields", [])}
        except Exception:
            continue
        score = 0
        if "LINE_NO" in fields:
            score += 10
        if "VOLTAGE" in fields:
            score += 10
        if "GLOBALID" in fields:
            score += 2
        if "COMPANY" in fields:
            score += 2
        if "OH_LINE" in p.name.upper() or "LINES" in p.name.upper():
            score += 3
        if score:
            scored.append((score, p))
    if not scored:
        raise RuntimeError("Could not find client line shapefile containing LINE_NO / VOLTAGE")
    return max(scored, key=lambda x: x[0])[1]


def load_client_lines(source: Path, temp_dir: Path) -> gpd.GeoDataFrame:
    if source.is_dir():
        root = source
    elif source.suffix.lower() == ".zip":
        root = temp_dir / "client_source"
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as zf:
            zf.extractall(root)
    elif source.suffix.lower() == ".shp":
        root = source.parent
        shp = source
        g = gpd.read_file(shp, engine="pyogrio")
        return g
    else:
        raise RuntimeError("Client source must be a ZIP, folder or SHP")

    shp = locate_client_line_shp(root)
    info(f"Client line source: {shp.name}")
    return gpd.read_file(shp, engine="pyogrio")


def client_lookup(g: gpd.GeoDataFrame) -> dict[str, dict[str, Any]]:
    cols = {str(c).upper(): c for c in g.columns}
    if "LINE_NO" not in cols:
        raise RuntimeError("Client source does not contain LINE_NO")
    line_col = cols["LINE_NO"]
    volt_col = cols.get("VOLTAGE")
    company_col = cols.get("COMPANY")
    gid_col = cols.get("GLOBALID")

    lookup: dict[str, dict[str, Any]] = {}
    for key, sg in g.groupby(g[line_col].apply(circuit_key), dropna=False):
        if not key:
            continue
        labels = sorted({txt(v) for v in sg[line_col] if txt(v)})
        voltages = sorted({normalise_voltage(v) for v in sg[volt_col] if normalise_voltage(v)}) if volt_col else []
        companies = sorted({txt(v) for v in sg[company_col] if txt(v)}) if company_col else []
        gids = sorted({txt(v) for v in sg[gid_col] if txt(v)}) if gid_col else []
        lookup[key] = {
            "line_no": labels[0] if labels else "",
            "voltages": voltages,
            "companies": companies,
            "globalids": gids,
            "n_features": len(sg),
        }
    return lookup


# -----------------------------------------------------------------------------
# KMZ parsing: Circuits -> RoW -> Circuit
# -----------------------------------------------------------------------------

@dataclass
class KmzRef:
    row: str
    circuit: str
    voltage: str
    company: str
    globalid: str
    geometry_wgs84: LineString
    geometry: LineString | None = None


def load_kml_root(kmz: Path) -> ET.Element:
    with zipfile.ZipFile(kmz) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise RuntimeError("No KML found inside KMZ")
        return ET.fromstring(zf.read(names[0]))


def folder_name(f: ET.Element) -> str:
    return txt(f.findtext(K + "name"))


def parse_description(desc: str) -> dict[str, str]:
    if not desc:
        return {}
    s = html.unescape(desc)
    # Typical KML: <B>VOLTAGE</B> = 115kV<BR>
    pairs = re.findall(r"<B>\s*([^<]+?)\s*</B>\s*=\s*([^<\r\n]+)", s, flags=re.I)
    out = {txt(k).upper(): txt(v) for k, v in pairs}
    if out:
        return out
    # Fallback for plain text descriptions.
    for line in re.split(r"<BR\s*/?>|\r?\n", s, flags=re.I):
        if "=" in line:
            k, v = line.split("=", 1)
            k = re.sub(r"<[^>]+>", "", k).strip().upper()
            v = re.sub(r"<[^>]+>", "", v).strip()
            if k:
                out[k] = v
    return out


def parse_linestring(ls: ET.Element) -> LineString | None:
    raw = txt(ls.findtext(K + "coordinates"))
    coords = []
    for token in raw.split():
        p = token.split(",")
        if len(p) < 2:
            continue
        try:
            coords.append((float(p[0]), float(p[1])))
        except ValueError:
            pass
    return LineString(coords) if len(coords) >= 2 else None


def find_named_folder(root: ET.Element, name: str) -> ET.Element | None:
    for f in root.iter(K + "Folder"):
        if folder_name(f).lower() == name.lower():
            return f
    return None


def parse_sow_kmz(kmz: Path) -> list[KmzRef]:
    root = load_kml_root(kmz)
    circuits_folder = find_named_folder(root, "Circuits")
    if circuits_folder is None:
        raise RuntimeError("Could not find Circuits folder in SoW KMZ")

    refs: list[KmzRef] = []
    for row_folder in circuits_folder.findall(K + "Folder"):
        row_name = folder_name(row_folder)
        if not row_name:
            continue
        for circuit_folder in row_folder.findall(K + "Folder"):
            circuit_name = folder_name(circuit_folder)
            if not circuit_name:
                continue
            for pm in circuit_folder.findall(K + "Placemark"):
                attrs = parse_description(txt(pm.findtext(K + "description")))
                ckt = txt(attrs.get("LINE_NO")) or circuit_name
                voltage = normalise_voltage(attrs.get("VOLTAGE"))
                company = txt(attrs.get("COMPANY"))
                gid = txt(attrs.get("GLOBALID"))
                for ls in pm.iter(K + "LineString"):
                    geom = parse_linestring(ls)
                    if geom is not None:
                        refs.append(KmzRef(row_name, ckt, voltage, company, gid, geom))
    if not refs:
        raise RuntimeError("No circuit linework found in SoW KMZ")
    return refs


def project_kmz(refs: list[KmzRef], target_crs) -> list[KmzRef]:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    out = []
    for r in refs:
        out.append(KmzRef(
            r.row, r.circuit, r.voltage, r.company, r.globalid,
            r.geometry_wgs84,
            transform(transformer.transform, r.geometry_wgs84),
        ))
    return out


def kmz_lookup(refs: list[KmzRef]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_circuit: dict[str, list[KmzRef]] = defaultdict(list)
    for r in refs:
        by_circuit[circuit_key(r.circuit)].append(r)
    for k, items in by_circuit.items():
        by_row: dict[str, list[LineString]] = defaultdict(list)
        voltages = set()
        companies = set()
        gids = set()
        for r in items:
            if r.geometry is not None:
                by_row[r.row].append(r.geometry)
            if r.voltage:
                voltages.add(r.voltage)
            if r.company:
                companies.add(r.company)
            if r.globalid:
                gids.add(r.globalid)
        out[k] = {
            "rows": {row: unary_union(gs) for row, gs in by_row.items()},
            "voltages": sorted(voltages),
            "companies": sorted(companies),
            "globalids": sorted(gids),
        }
    return out


# -----------------------------------------------------------------------------
# Conflation
# -----------------------------------------------------------------------------

def midpoint(geom):
    if geom is None or geom.is_empty:
        return Point()
    try:
        return geom.interpolate(0.5, normalized=True)
    except Exception:
        return geom.centroid


def choose_row(geom, kmz_rec: dict[str, Any] | None) -> tuple[str, float | None, int]:
    if not kmz_rec or not kmz_rec.get("rows"):
        return "", None, 0
    rows = kmz_rec["rows"]
    if len(rows) == 1:
        row, ref = next(iter(rows.items()))
        return row, float(midpoint(geom).distance(ref)), 1
    p = midpoint(geom)
    scored = sorted((float(p.distance(ref)), row) for row, ref in rows.items())
    return scored[0][1], scored[0][0], len(rows)


def aggregate_final(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.empty:
        return gpd.GeoDataFrame(columns=["ROW", "CIRCUIT", "VOLTAGE", "COMPANY", "N_SPANS", "LENGTH", "geometry"], geometry="geometry", crs=edges.crs)
    rows = []
    for (row, circuit, voltage, company), sg in edges.groupby(["ROW", "CIRCUIT", "VOLTAGE", "COMPANY"], dropna=False, sort=False):
        u = unary_union(list(sg.geometry))
        if u.geom_type == "MultiLineString":
            geom = linemerge(u)
        else:
            geom = u
        rows.append({
            "ROW": txt(row),
            "CIRCUIT": txt(circuit),
            "VOLTAGE": txt(voltage),
            "COMPANY": txt(company),
            "N_SPANS": int(len(sg)),
            "LENGTH": float(sum(sg.geometry.length)),
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)


def write_layer(gdf: gpd.GeoDataFrame, gpkg: Path, layer: str) -> None:
    if gdf is not None and not gdf.empty:
        gdf.to_file(gpkg, layer=layer, driver="GPKG", engine="pyogrio")


def short_gid(gids: list[str]) -> str:
    if not gids:
        return ""
    return gids[0][:38]


def run(raw_path: Path, client_source: Path, kmz_path: Path, capture_year: int, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    shp_dir = output / "shp"
    shp_dir.mkdir(parents=True, exist_ok=True)

    info("Reading stage-01 raw wires...")
    raw, raw_layer = read_raw_stage01(raw_path)
    if raw.empty:
        raise RuntimeError("Raw 2D wires are empty")
    if raw.crs is None:
        raise RuntimeError("Raw 2D wires have no CRS. Stage 01 must be run with the correct CRS.")
    raw = raw[raw.geometry.apply(is_line)].copy()
    cfield = find_circuit_field(raw)
    raw["CIRCUIT"] = raw[cfield].apply(txt)
    raw["CKEY"] = raw["CIRCUIT"].apply(circuit_key)
    info(f"  layer={raw_layer}, features={len(raw):,}, circuits={raw['CKEY'].nunique():,}")

    with tempfile.TemporaryDirectory(prefix="wires2d_02_") as td:
        td = Path(td)
        info("Reading client source...")
        client = load_client_lines(client_source, td)
        clook = client_lookup(client)
        info(f"  client circuits={len(clook):,}")

        info("Reading Project SoW KMZ...")
        refs_wgs = parse_sow_kmz(kmz_path)
        refs = project_kmz(refs_wgs, raw.crs)
        klook = kmz_lookup(refs)
        info(f"  KMZ circuit/RoW references={len(klook):,}")

        edge_rows = []
        audit_rows = []
        unmatched_rows = []

        for ckey, sg in raw.groupby("CKEY", sort=False):
            circuit_label = txt(sg["CIRCUIT"].iloc[0])
            crec = clook.get(ckey)
            krec = klook.get(ckey)

            source_voltages = crec["voltages"] if crec else []
            kmz_voltages = krec["voltages"] if krec else []
            voltage = source_voltages[0] if len(source_voltages) == 1 else (kmz_voltages[0] if len(kmz_voltages) == 1 else "")
            companies = crec["companies"] if crec else (krec["companies"] if krec else [])
            company = companies[0] if len(companies) == 1 else (companies[0] if companies else "")
            gids = crec["globalids"] if crec else (krec["globalids"] if krec else [])

            row_counts: dict[str, int] = defaultdict(int)
            dists = []
            row_options = len(krec["rows"]) if krec else 0

            for _, r in sg.iterrows():
                row, dist, nrows = choose_row(r.geometry, krec)
                if row:
                    row_counts[row] += 1
                if dist is not None:
                    dists.append(dist)
                rec = r.to_dict()
                rec.update({
                    "ROW": row,
                    "CIRCUIT": circuit_label,
                    "VOLTAGE": voltage,
                    "COMPANY": company,
                    "SRC_GID": short_gid(gids),
                    "ROW_DIST": dist,
                    "N_ROW_OPT": nrows,
                })
                edge_rows.append(rec)

            voltage_conflict = bool(source_voltages and kmz_voltages and set(source_voltages) != set(kmz_voltages))
            audit_rows.append({
                "CIRCUIT": circuit_label,
                "CLIENT_MATCH": int(crec is not None),
                "KMZ_MATCH": int(krec is not None),
                "CLIENT_VOLTAGE": ";".join(source_voltages),
                "KMZ_VOLTAGE": ";".join(kmz_voltages),
                "FINAL_VOLTAGE": voltage,
                "VOLTAGE_CONFLICT": int(voltage_conflict),
                "ROW_OPTIONS": row_options,
                "ROWS_ASSIGNED": ";".join(sorted(row_counts)),
                "N_FEATURES": len(sg),
                "MED_ROW_DIST": float(np.median(dists)) if dists else None,
                "MAX_ROW_DIST": float(max(dists)) if dists else None,
                "COMPANY": company,
                "N_CLIENT_FEATURES": int(crec["n_features"]) if crec else 0,
            })
            if crec is None or krec is None:
                unmatched_rows.append({
                    "CIRCUIT": circuit_label,
                    "CLIENT_MATCH": int(crec is not None),
                    "KMZ_MATCH": int(krec is not None),
                })

        edges = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=raw.crs)
        # Drop internal matching key from deliverable tables.
        if "CKEY" in edges.columns:
            edges = edges.drop(columns=["CKEY"])
        final = aggregate_final(edges)

        gpkg = output / "02_final_2d_wires.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        write_layer(final, gpkg, "wires_2d")
        write_layer(edges, gpkg, "span_edges")

        # Reference layers are explicitly labelled as reference-only.
        kmz_rows = []
        for r in refs:
            if r.geometry is not None:
                kmz_rows.append({
                    "ROW": r.row,
                    "CIRCUIT": r.circuit,
                    "VOLTAGE": r.voltage,
                    "COMPANY": r.company,
                    "GLOBALID": r.globalid,
                    "geometry": r.geometry,
                })
        kmz_gdf = gpd.GeoDataFrame(kmz_rows, geometry="geometry", crs=raw.crs)
        write_layer(kmz_gdf, gpkg, "kmz_reference")

        client_ref = client.to_crs(raw.crs) if client.crs and CRS.from_user_input(client.crs) != CRS.from_user_input(raw.crs) else client.copy()
        # Keep all client attributes in GPKG; this is reference-only.
        write_layer(client_ref, gpkg, "client_reference")

        audit = pd.DataFrame(audit_rows).sort_values("CIRCUIT")
        audit.to_csv(output / "02_conflation_audit.csv", index=False)
        pd.DataFrame(unmatched_rows).to_csv(output / "02_unmatched.csv", index=False)

        # Delivery entity = Circuit. One SHP per circuit; if the circuit spans
        # multiple RoWs, those RoW portions are separate features in the same SHP.
        for ckey, sg in final.groupby(final["CIRCUIT"].apply(circuit_key), sort=False):
            label = txt(sg["CIRCUIT"].iloc[0])
            name = f"{safe_filename(label)}_Wires_{capture_year}.shp"
            # Shapefile field names are kept <=10 chars by using already-short fields.
            out_cols = [c for c in ["ROW", "CIRCUIT", "VOLTAGE", "COMPANY", "N_SPANS", "LENGTH", "geometry"] if c in sg.columns]
            sg[out_cols].to_file(shp_dir / name, driver="ESRI Shapefile", engine="pyogrio")

        info("\nWritten:")
        info(f"  {gpkg}")
        info(f"  {output / '02_conflation_audit.csv'}")
        info(f"  {output / '02_unmatched.csv'}")
        info(f"  {shp_dir}")
        return gpkg


def main() -> int:
    parser = argparse.ArgumentParser(description="Conflate raw 2D wires with client source + Project SoW KMZ")
    parser.add_argument("raw", nargs="?", help="01_raw_2d_wires.gpkg (preferred) or line SHP/GPKG")
    parser.add_argument("client_source", nargs="?", help="Client source ZIP/folder/SHP")
    parser.add_argument("sow_kmz", nargs="?", help="Project SoW KMZ")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("-o", "--output", default="02_2d_wires_output")
    args = parser.parse_args()

    raw = clean_path(args.raw) if args.raw else prompt_path("Stage 01 raw wires (.gpkg/.shp): ", {".gpkg", ".shp"})
    if args.client_source:
        client = clean_path(args.client_source)
    else:
        client = prompt_path("Client source (.zip/.shp or folder): ", {".zip", ".shp"}, allow_dir=True)
    kmz = clean_path(args.sow_kmz) if args.sow_kmz else prompt_path("Project SoW KMZ (.kmz): ", {".kmz"})
    year = args.year or prompt_year()

    try:
        run(raw, client, kmz, year, Path(args.output))
        return 0
    except Exception as exc:
        info(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
