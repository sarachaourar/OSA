#!/usr/bin/env python3
"""
OSA -- Oil Spill Analysis pipeline
===================================

Converted from the OSA.ipynb notebook. Given an area of interest and a date
range, this script:

  1. Connects to the Copernicus Data Space Ecosystem openEO backend.
  2. Fetches a Sentinel-2 (true-colour) and a Sentinel-1 (SAR, VV) image for
     the AOI / time window as batch jobs, waiting for both to finish.
  3. Converts the Sentinel-1 image to dB and runs a segmentation model
     (DeepLabV3+) on it to produce a binary oil-spill mask.
  4. Vectorises the mask into a georeferenced polygon (Shapefile).
  5. Generates PNG previews, a PDF report, and zips everything up.

Required companion files (NOT included in this handoff, see README.md):
  - eoapi_preprocessor.py   (local module providing EOAPIPreprocessor)
  - a trained model checkpoint (.pth), path set in config.toml [paths].model_path

Usage:
    python osa_pipeline.py
    python osa_pipeline.py --start-date 2024-03-01 --end-date 2024-03-15
    python osa_pipeline.py --config my_config.toml

All non-secret settings live in config.toml. Secrets (openEO client
credentials) live in a `.env` file -- see .env.example.
"""

from __future__ import annotations

import sys
import argparse
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: this runs as a script, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import geopandas as gpd
import openeo
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from shapely.geometry import Polygon
from skimage import measure
from skimage.transform import resize
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =============================================================================
# Job orchestration (batch job submission / polling against the openEO backend)
# =============================================================================

class NoScenesAvailableError(RuntimeError):
    """Raised when a batch job fails because the backend found no matching
    scenes for the requested spatial/temporal extent (e.g. the Sentinel-2
    case: 'no item in the collection'). Kept separate from RuntimeError so
    callers can catch it specifically and react differently (e.g. prompt for
    a wider date range) instead of just seeing a generic crash.

    Note: this only covers jobs the backend reports as status "error" or
    "canceled". A job that "succeeds" but returns an empty/NaN-filled raster
    (the Sentinel-1 case, where the job finishes normally but the pixel
    stats are 0) will NOT raise this -- that failure mode has to be checked
    separately on the downloaded output, since no exception occurs on the
    backend side.
    """


class EmptyImageError(RuntimeError):
    """Raised when a downloaded Sentinel-1/Sentinel-2 GeoTIFF has no valid
    (non-NaN) pixels anywhere in it. This is the failure mode described in
    NoScenesAvailableError's docstring: the batch job finishes normally (so
    no backend error occurs), but the AOI/time window turned out to have no
    usable scene, so the output is just a NaN-filled raster. Checked right
    after download, before that raster is used for anything downstream
    (dB conversion, previews, etc.).
    """


# Substrings seen in openEO backend error messages when a query matches
# nothing. Check the real message text your backend returns and extend this
# list if it phrases "no data" differently.
_NO_DATA_MARKERS = (
    "no data",
    "nodataavailable",
    "no items found",
    "empty datacube",
    "no product",
    "no scene",
)


def _job_error_message(job) -> str:
    """Best-effort fetch of the backend's error message for a failed job."""
    try:
        logs = job.logs()
        msgs = [entry.get("message", "") for entry in logs if entry.get("level", "").lower() == "error"]
        return " | ".join(msgs) if msgs else ""
    except Exception:
        return ""


def _start_job_with_retry(job, max_retries=5, base_delay=5, max_delay=60):
    """
    Call job.start(), automatically retrying on HTTP 429 (Too Many Requests).

    The openEO client already retries a handful of times internally, but on a
    busy backend that budget can run out before the rate limit clears. This
    wraps job.start() with its own backoff loop on top of that, so a 429 is
    handled quietly instead of crashing the whole pipeline.

    The user only sees a message if we're actually about to retry; if all
    retries are exhausted, the original error is re-raised (with a note
    telling the user to just re-run the script).
    """
    attempt = 0
    while True:
        try:
            job.start()
            return
        except openeo.rest.OpenEoApiPlainError as e:
            status = getattr(e, "http_status_code", None)
            attempt += 1
            if status == 429 and attempt <= max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                print(
                    f"\n\n[rate limited] Backend returned : '429 Too Many Requests while starting' "
                    f"job '{job.job_id}'.\nRetrying automatically in {delay:.0f}s "
                    f"\n(attempt {attempt}/{max_retries})..."
                )
                time.sleep(delay)
                continue
            if status == 429:
                print(
                    f"[rate limited] Job '{job.job_id}' still hitting 429 Too Many Requests "
                    f"after {max_retries} automatic retries. Please just re-run the script -- "
                    f"jobs that already started/finished won't need to be redone."
                )
            raise


def run_jobs(job_specs, max_concurrent=3, initial_interval=10, max_interval=60):
    """
    Create, start, and wait for a list of batch jobs, without exceeding
    `max_concurrent` jobs running/queued on the backend at the same time.

    job_specs: list of (cube, create_job_kwargs) tuples.
    Returns a list of finished BatchJob objects, in the same order as job_specs.

    Raises NoScenesAvailableError if a job fails because no matching scenes
    were found. Raises RuntimeError for any other backend failure (auth,
    quota, bugs).
    """
    jobs = [None] * len(job_specs)
    titles = [kwargs.get("title", f"job_{i}") for i, (_, kwargs) in enumerate(job_specs)]
    pending_indices = list(range(len(job_specs)))
    active = []  # list of (index, job)
    interval = initial_interval

    def _cancel_remaining(skip_job_id):
        # Best-effort cleanup so one failed job doesn't leave siblings running/billing.
        for _, other_job in active:
            if other_job.job_id != skip_job_id:
                try:
                    other_job.stop()
                except Exception:
                    pass

    while pending_indices or active:
        while pending_indices and len(active) < max_concurrent:
            idx = pending_indices.pop(0)
            cube, kwargs = job_specs[idx]
            job = cube.create_job(**kwargs)
            _start_job_with_retry(job)
            jobs[idx] = job
            active.append((idx, job))
            print(f"Started job '{job.job_id}' ({titles[idx]})")

        still_active = []
        for idx, job in active:
            status = job.status()
            print(f"Job '{job.job_id}': {status}")
            if status == "finished":
                continue
            elif status in ("error", "canceled"):
                message = _job_error_message(job)
                _cancel_remaining(job.job_id)
                if status == "error" and any(marker in message.lower() for marker in _NO_DATA_MARKERS):
                    raise NoScenesAvailableError(
                        f"Job '{job.job_id}' ({titles[idx]}) found no matching scenes "
                        f"for the requested date range/AOI. Backend message: {message or '(none)'}"
                    )
                raise RuntimeError(
                    f"Job '{job.job_id}' ({titles[idx]}) did not finish successfully "
                    f"(status: {status}). Backend message: {message or '(none)'}"
                )
            else:
                still_active.append((idx, job))
        active = still_active

        if active or pending_indices:
            time.sleep(interval)
            interval = min(interval * 1.5, max_interval)

    return jobs


def cancel_stale_jobs(connection) -> None:
    """Cancel any leftover jobs from previous runs still queued/running on the
    backend, so they don't keep consuming your job quota."""
    non_terminal_statuses = {"created", "queued", "running"}
    for job_meta in connection.list_jobs():
        if job_meta.get("status") in non_terminal_statuses:
            job = connection.job(job_meta["id"])
            print(f"Cancelling leftover job '{job_meta['id']}' (status: {job_meta['status']})")
            try:
                job.stop()
            except Exception as e:
                print(f"  Could not stop job: {e}")


def extract_timestamp_str(metadata: dict) -> str:
    """Pull acquisition timestamp(s) out of an openEO job-results.json's
    'derived_from' links."""
    derived = [link["href"] for link in metadata.get("links", []) if link.get("rel") == "derived_from"]
    timestamps = set()
    for product_id in derived:
        filename = os.path.basename(product_id)
        match = re.search(r"\d{8}T\d{6}", filename)
        if not match:
            raise ValueError(f"No timestamp found in: {filename}")
        timestamps.add(datetime.strptime(match.group(), "%Y%m%dT%H%M%S"))
    return ", ".join(sorted(t.strftime("%Y-%m-%d %H:%M:%S UTC") for t in timestamps))


# =============================================================================
# Pipeline steps
# =============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def prepare_output_dirs(config: dict) -> None:
    """Empty the Sentinel-1 / Sentinel-2 output folders before a fresh run."""
    for folder_name in (config["paths"]["sentinel1_dir"], config["paths"]["sentinel2_dir"]):
        folder_path = Path(folder_name)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Emptied '{folder_name}/'")


def connect_openeo(config: dict):
    """Connect and authenticate against the openEO backend using OIDC client
    credentials. Reads OPENEO_AUTH_CLIENT_ID / OPENEO_AUTH_CLIENT_SECRET (and
    optionally OPENEO_AUTH_PROVIDER_ID) from the environment -- make sure
    load_dotenv() has been called first."""
    connection = openeo.connect(config["openeo"]["backend_url"])
    connection.authenticate_oidc_client_credentials()
    return connection


def build_datacubes(connection, config: dict, start_date: str, start_time: str, end_date: str):
    """Build the (not-yet-executed) Sentinel-2 and Sentinel-1 datacubes for
    the configured AOI and the given date range."""
    aoi = config["aoi"]
    spatial_extent = {"west": aoi["west"], "south": aoi["south"], "east": aoi["east"], "north": aoi["north"]}
    temporal_extent = [f"{start_date}T{start_time}Z", f"{end_date}T23:59:59Z"]

    s2_cfg = config["sentinel2"]
    s2_cube = connection.load_collection(
        s2_cfg["collection"],
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=s2_cfg["bands"],
    )
    s2_cube = s2_cube.min_time()
    result_s2 = s2_cube.save_result(format="GTiff")

    s1_cfg = config["sentinel1"]
    s1_cube = connection.load_collection(
        s1_cfg["collection"],
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=s1_cfg["bands"],
    )
    s1_cube = s1_cube.min_time()
    s1_cube = s1_cube.sar_backscatter(coefficient=s1_cfg["backscatter_coefficient"])

    return result_s2, s1_cube


def _assert_raster_has_data(tif_path: str, label: str, require_positive: bool = False) -> None:
    """Raise EmptyImageError if the GeoTIFF at tif_path has no valid
    (non-NaN, finite) pixels in any band, or -- when require_positive is
    True -- no finite pixels greater than zero.

    Batch jobs can finish with status "finished" and still hand back a
    raster that's entirely NaN/no-data -- typically because there was no
    usable scene for the chosen AOI/date range (e.g. persistent cloud cover
    for Sentinel-2, or a coverage gap for Sentinel-1). That's not something
    the openEO backend flags as a job error, so it has to be caught here,
    before the raster gets used downstream.

    require_positive=True is used for the Sentinel-1 image: the dB
    conversion step fills every non-positive pixel with
    ``np.nanmin(data[data > 0])``, which crashes with "zero-size array to
    reduction operation fmin" if there isn't at least one strictly positive
    pixel anywhere in the image. A raster that's all zeros (or a mix of NaN
    and non-positive values) passes the plain "any finite pixel" check but
    still triggers that crash, so Sentinel-1 needs this stricter check.
    """
    with rasterio.open(tif_path) as src:
        data = src.read().astype("float32")

    if not np.isfinite(data).any():
        raise EmptyImageError(
            f"The downloaded {label} image ('{Path(tif_path).name}') has no valid data -- "
            f"it's entirely NaN/no-data. This usually means there was no usable {label} scene "
            f"for the chosen bounding box and date range (e.g. persistent cloud cover, or a "
            f"coverage gap). Please choose a different bounding box or a different/wider time "
            f"frame and re-run."
        )

    if require_positive and not (data[np.isfinite(data)] > 0).any():
        raise EmptyImageError(
            f"The downloaded {label} image ('{Path(tif_path).name}') has no pixels greater than "
            f"zero, so it can't be converted to dB. This usually means there was no usable "
            f"{label} scene for the chosen bounding box and date range. Please choose a "
            f"different bounding box or a different/wider time frame and re-run."
        )


def convert_sar_to_db(s1_dir: str) -> str:
    """Convert the raw Sentinel-1 GeoTIFF to dB using the project's local
    EOAPIPreprocessor helper. Imported lazily so the rest of this script
    (e.g. --help) still works if that module isn't on disk yet."""
    from eoapi_preprocessor import EOAPIPreprocessor  # local project module -- not on PyPI

    raw_path = str(Path(s1_dir) / "openEO.tif")
    db_path = str(Path(s1_dir) / "Sentinel1_converted_dB.tif")

    preprocessor = EOAPIPreprocessor(raw_path, db_path)
    preprocessor.convert_to_db()
    return db_path


def _save_side_by_side(image: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Diagnostic preview: original SAR image next to the predicted mask."""
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original SAR Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Predicted Oil Spill Mask")
    plt.axis("off")
    plt.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved SAR/mask preview to: {out_path}")


def run_inference(db_path: str, config: dict) -> str:
    """Run the segmentation model on the dB-converted SAR image and write out
    a georeferenced binary mask GeoTIFF."""
    model_cfg = config["model"]

    with rasterio.open(db_path) as src:
        image = src.read(1).astype(np.float32)
        profile = src.profile

    transform_fn = A.Compose(
        [
            A.Resize(model_cfg["input_size"], model_cfg["input_size"]),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ]
    )
    augmented = transform_fn(image=image)
    input_tensor = augmented["image"].unsqueeze(0)  # add batch dimension

    model_cls = getattr(smp, model_cfg["architecture"])
    model = model_cls(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["num_classes"],
    )
    model.load_state_dict(torch.load(config["paths"]["model_path"], map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        pred_mask = (pred > model_cfg["threshold"]).float().squeeze().numpy()

    resized_mask = resize(pred_mask, image.shape, preserve_range=True, anti_aliasing=True)
    binary_mask = (resized_mask > model_cfg["threshold"]).astype(np.uint8)

    mask_path = str(Path(db_path).parent / "georeferenced_segmentation_mask.tif")
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(binary_mask, 1)
    print(f"Saved segmented mask to: {mask_path}")

    _save_side_by_side(image, binary_mask, Path(db_path).parent / "sar_vs_mask_preview.png")

    return mask_path


def export_polygon(mask_path: str, db_path: str, output_dir: str) -> str:
    """Vectorise the binary mask into a georeferenced polygon and save it as
    an ESRI Shapefile.

    NOTE ON A NOTEBOOK BUG FIXED HERE: the original notebook wrote this layer
    with driver="GeoJSON" to a `.geojson` path, but a later cell tried to
    read it back from a `.shp` path that was never created (a guaranteed
    crash). Since a Shapefile is what the rest of the pipeline -- and the
    final zip -- expects, this function now writes an actual Shapefile.
    """
    with rasterio.open(db_path) as src:
        transform = src.transform
        crs = src.crs

    with rasterio.open(mask_path) as mask_src:
        binary_mask = mask_src.read(1)

    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []
    for contour in contours:
        coords = [rasterio.transform.xy(transform, row, col) for row, col in contour]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if poly.is_valid:
            polygons.append(poly)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    shp_path = str(Path(output_dir) / "georreferenced_segmentation.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    print(f"Saved polygon shapefile to: {shp_path}")
    return shp_path


def plot_overlay(db_path: str, mask_path: str, shp_path: str, output_dir: str) -> str:
    """Save a diagnostic overlay of the SAR image + predicted mask."""
    with rasterio.open(db_path) as src:
        image = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    # Loaded and reprojected for parity with the original notebook, though
    # (as in the notebook) the polygon outline itself isn't drawn below --
    # only the raster mask is. Add `gdf.boundary.plot(ax=ax, color="red")`
    # if you want the vector outline overlaid too.
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(crs)  # noqa: F841

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image, extent=extent, origin="upper", cmap="gray")
    ax.imshow(mask, extent=extent, origin="upper", alpha=0.3)
    ax.set_title("Sentinel-1 with the extracted polygon")
    ax.set_axis_off()

    overlay_path = str(Path(output_dir) / "sentinel1_mask_overlay.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay figure to: {overlay_path}")
    return overlay_path


def extract_timestamps(s1_dir: str, s2_dir: str) -> tuple[str, str]:
    with open(Path(s2_dir) / "job-results.json") as f:
        metadata_s2 = json.load(f)
    with open(Path(s1_dir) / "job-results.json") as f:
        metadata_s1 = json.load(f)

    timestamp_s2 = extract_timestamp_str(metadata_s2)
    timestamp_s1 = extract_timestamp_str(metadata_s1)
    print("S2:", timestamp_s2)
    print("S1:", timestamp_s1)
    return timestamp_s1, timestamp_s2


def generate_preview_images(directories) -> list[Path]:
    """Normalise every GeoTIFF in the given directories to a viewable PNG."""
    generated = []
    for folder in directories:
        for tif_path in Path(folder).glob("*.tif"):
            with rasterio.open(tif_path) as src:
                img = src.read()

            if img.shape[0] == 1:
                img, cmap = img[0], "gray"
            else:
                img, cmap = img[:3].transpose(1, 2, 0), None

            img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

            fig = plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap=cmap)
            plt.axis("off")
            png_path = tif_path.with_suffix(".png")
            plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)
            generated.append(png_path)
            print(f"Saved preview: {png_path}")
    return generated


# =============================================================================
# PDF report
# =============================================================================

_DARK = colors.HexColor("#004F59")
_LIGHT = colors.HexColor("#EAF5F5")
_GREY = colors.HexColor("#667085")
_BORDER = colors.HexColor("#D8E2E7")
_TEXT = colors.HexColor("#1F2937")

_PAGE_WIDTH, _PAGE_HEIGHT = A4

_base_styles = getSampleStyleSheet()

_STYLE_BODY = ParagraphStyle(
    "OsaBody", parent=_base_styles["Normal"], fontName="Helvetica", fontSize=10,
    leading=15, textColor=_TEXT, spaceAfter=6,
)
_STYLE_CAPTION = ParagraphStyle(
    "OsaCaption", parent=_base_styles["Normal"], fontName="Helvetica-Oblique", fontSize=9,
    leading=12, textColor=_GREY, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
)
_STYLE_SECTION = ParagraphStyle(
    "OsaSection", parent=_base_styles["Normal"], fontName="Helvetica-Bold", fontSize=16,
    leading=16, textColor=_TEXT, spaceBefore=12, spaceAfter=8,
)


def _draw_header(canvas, doc, title: str, pageinfo: str) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 20)
    canvas.setFillColor(_DARK)
    canvas.drawString(doc.leftMargin, _PAGE_HEIGHT - 0.85 * inch, title)
    canvas.setStrokeColor(_LIGHT)
    canvas.setLineWidth(1)
    canvas.line(doc.leftMargin, _PAGE_HEIGHT - 0.95 * inch, _PAGE_WIDTH - doc.rightMargin, _PAGE_HEIGHT - 0.95 * inch)
    canvas.drawString(inch, 0.75 * inch, "Page %d | %s" % (doc.page, pageinfo))
    canvas.restoreState()


def _draw_later_pages(canvas, doc, pageinfo: str) -> None:
    canvas.saveState()
    canvas.setFillColor(_GREY)
    canvas.setFont("Helvetica", 12)
    canvas.drawString(inch, 0.55 * inch, pageinfo)
    canvas.drawString(inch, 0.75 * inch, "Page %d | %s" % (doc.page, pageinfo))
    canvas.restoreState()


def generate_pdf_report(
    timestamp_s1: str,
    timestamp_s2: str,
    s1_png: str,
    s2_png: str,
    title: str = "Oil Spill Analysis Report",
    output_dir: str = ".",
) -> str:
    """Build the PDF report bundling the two image previews and acquisition
    metadata. Returns the path to the generated PDF."""
    pageinfo = "OSA report"

    s1_day, s1_time = timestamp_s1[:10], timestamp_s1[11:19]
    s2_day, s2_time = timestamp_s2[:10], timestamp_s2[11:19]

    def _safe(ts: str) -> str:
        return ts.replace("-", "").replace(":", "").replace(" ", "_")

    filename = str(
        Path(output_dir) / f"OSA_S1_{_safe(s1_day + '_' + s1_time)}_S2_{_safe(s2_day + '_' + s2_time)}.pdf"
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=1.1 * inch,
        bottomMargin=0.9 * inch,
    )

    usable_width = _PAGE_WIDTH - 2 * inch
    usable_height = _PAGE_HEIGHT * 0.45

    story = []

    story.append(Paragraph("Sentinel-1 SAR Backscatter", _STYLE_SECTION))
    story.append(HRFlowable(width="100%", thickness=1, color=_LIGHT, spaceAfter=10))
    s1_image = RLImage(s1_png, width=usable_width, height=usable_height)
    s1_image.hAlign = "CENTER"
    story.append(s1_image)
    story.append(
        Paragraph(
            f"Figure 1 &mdash; Sentinel-1 SAR (VV, dB) image acquired on {s1_day} at {s1_time} UTC. "
            "Potential oil spill extent is visible as dark, low-backscatter patches on the water surface.",
            _STYLE_CAPTION,
        )
    )

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Sentinel-2 RGB Composite", _STYLE_SECTION))
    story.append(HRFlowable(width="100%", thickness=1, color=_LIGHT, spaceAfter=10))
    s2_image = RLImage(s2_png, width=usable_width, height=usable_height)
    s2_image.hAlign = "CENTER"
    story.append(s2_image)
    story.append(
        Paragraph(
            f"Figure 2 &mdash; Sentinel-2 true-colour composite acquired on {s2_day} at {s2_time} UTC. "
            "Potential oil spill extent is visible as dark, low-reflectance patches on the water surface.",
            _STYLE_CAPTION,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Acquisition Metadata", _STYLE_SECTION))
    story.append(HRFlowable(width="100%", thickness=1, color=_LIGHT, spaceAfter=10))

    table_data = [
        ["Parameter", "Sentinel-1 (SAR)", "Sentinel-2 (Optical)"],
        ["Acquisition Date", s1_day, s2_day],
        ["Acquisition Time (UTC)", s1_time, s2_time],
    ]
    col_widths = [usable_width * 0.38, usable_width * 0.31, usable_width * 0.31]
    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), _LIGHT),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), _BORDER),
                ("BACKGROUND", (0, 2), (-1, 2), _LIGHT),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("TEXTCOLOR", (0, 1), (-1, -1), _DARK),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ]
        )
    )
    story.append(tbl)

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Observation Notes", _STYLE_SECTION))
    story.append(HRFlowable(width="100%", thickness=1, color=_LIGHT, spaceAfter=8))
    note = (
        f"The Sentinel-1 SAR image acquired on <b>{s1_day}</b> at <b>{s1_time} UTC</b> was used "
        "for initial spill detection based on backscatter anomalies. "
        f"The Sentinel-2 optical image acquired on <b>{s2_day}</b> at <b>{s2_time} UTC</b> "
        "was used to visually confirm the extent of the potential spill."
    )
    story.append(Paragraph(note, _STYLE_BODY))

    doc.build(
        story,
        onFirstPage=lambda c, d: _draw_header(c, d, title, pageinfo),
        onLaterPages=lambda c, d: _draw_later_pages(c, d, pageinfo),
    )
    print(f"PDF saved: {filename}")
    return filename


# =============================================================================
# Zip outputs
# =============================================================================

def zip_osa_outputs(directories, pdf_path: str, zip_name: str | None = None) -> str:
    """Bundle the sentinel output folders together with the PDF report into a
    single zip archive."""
    if zip_name is None:
        zip_name = str(Path(pdf_path).with_suffix(".zip"))

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in directories:
            for file_path in Path(folder).rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.as_posix())
        zf.write(pdf_path, arcname=Path(pdf_path).name)

    print(f"Created archive: {zip_name}")
    return zip_name


# =============================================================================
# CLI / orchestration
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oil Spill Analysis (OSA) pipeline.")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml (default: %(default)s)")
    parser.add_argument("--start-date", help="Override [search].start_date, format YYYY-MM-DD")
    parser.add_argument("--start-time", help="Override [search].start_time, format HH:MM:SS")
    parser.add_argument("--end-date", help="Override [search].end_date, format YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()  # populates OPENEO_AUTH_CLIENT_ID / SECRET from a local .env file
    config = load_config(args.config)

    start_date = args.start_date or config["search"]["start_date"]
    start_time = args.start_time or config["search"]["start_time"]
    end_date = args.end_date or config["search"]["end_date"]

    s1_dir = config["paths"]["sentinel1_dir"]
    s2_dir = config["paths"]["sentinel2_dir"]

    prepare_output_dirs(config)

    connection = connect_openeo(config)
    cancel_stale_jobs(connection)

    result_s2, s1_cube = build_datacubes(connection, config, start_date, start_time, end_date)

    try:
        job_s2, job_s1 = run_jobs(
            [
                (result_s2, dict(title="Sentinel-2_image")),
                (s1_cube, dict(out_format="GTIFF", title="Sentinel-1_image")),
            ],
            max_concurrent=config["jobs"]["max_concurrent_jobs"],
            initial_interval=config["jobs"]["initial_poll_interval_sec"],
            max_interval=config["jobs"]["max_poll_interval_sec"],
        )
    except NoScenesAvailableError as e:
        print(f"No scenes found: {e}")
        print("Try widening the date range or double-check the AOI, then re-run.")
        raise

    job_s2.get_results().download_files(s2_dir)
    job_s1.get_results().download_files(s1_dir)

    try:
        _assert_raster_has_data(str(Path(s2_dir) / "openEO.tif"), "Sentinel-2")
        _assert_raster_has_data(str(Path(s1_dir) / "openEO.tif"), "Sentinel-1", require_positive=True)
    except EmptyImageError as e:
        print(f"No usable image data: {e}")
        sys.exit(1)
        #raise

    db_path = convert_sar_to_db(s1_dir)
    mask_path = run_inference(db_path, config)
    shp_path = export_polygon(mask_path, db_path, s1_dir)
    plot_overlay(db_path, mask_path, shp_path, s1_dir)

    timestamp_s1, timestamp_s2 = extract_timestamps(s1_dir, s2_dir)
    generate_preview_images([s1_dir, s2_dir])

    pdf_path = generate_pdf_report(
        timestamp_s1,
        timestamp_s2,
        s1_png=str(Path(s1_dir) / "Sentinel1_converted_dB.png"),
        s2_png=str(Path(s2_dir) / "openEO.png"),
        title=config["report"]["title"],
    )

    zip_path = zip_osa_outputs([s1_dir, s2_dir], pdf_path)
    print(f"\nDone. Archive created at: {zip_path}")


if __name__ == "__main__":
    main()
