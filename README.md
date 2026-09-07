# OSA Pipeline (Oil Spill Analyser)

This is a script that fetches a Sentinel-1 and Sentinel-2 image from the openEO library for 
a specified area and time, runs a segmentation model on
the Sentinel-1 image to detect a potential oil spill, 
and packages the imagery, mask, shapefile, 
and a PDF report into a zip file.

## Files

| File | Purpose |
|---|---|
| `osa_pipeline.py` | The pipeline. Run this. |
| `eoapi_preprocessor.py` | Main file for sentinel-1 image processing. |
| `deepLabV3_resnet_oilspill_final.pth` | AI model for running inference. |
| `config.toml` | All non-secret settings (AOI, dates, model params, paths). Edit freely. |
| `.env.example` | Template for your openEO credentials. Copy to `.env` and fill in. |
| `requirements.txt` | Python packages to install. |
| `README.md` | This file. |

## Setup

### 1. System dependencies

`rasterio` and `geopandas` depend on **GDAL**. 
Installing them with **conda/mamba** is the most reliable route for this stack:
  ```bash
  conda create -n osa python=3.12  #create a virtual env
  conda activate osa
  conda install -c conda-forge rasterio geopandas
  ```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt #or 'requirements.lock.txt' to install the exact libary versions used by dev
```

### 3. Credentials

```bash
cp .env.example .env
```

Then edit `.env` and fill in your Copernicus Data Space Ecosystem openEO
client credentials (`OPENEO_AUTH_CLIENT_ID`, `OPENEO_AUTH_CLIENT_SECRET`).
See the comments in `.env.example` for how to obtain these. 

### 4. Configuration

Open `config.toml` and check:
- `[aoi]` — your area of interest (defaults to Al-Qibliyah island, Oman).
- `[search]` — default date range (can be overridden per-run).
- `[paths].model_path` — path to the `.pth` checkpoint.
- `[jobs].max_concurrent_jobs` — how many openEO batch jobs to run at once; match this to your account's limits.

## Running it

With the defaults from `config.toml`:
```bash
python osa_pipeline.py
```

Overriding the date range for a specific run without editing the file:
```bash
python osa_pipeline.py --start-date 2024-03-01 --start-time 00:00:00 --end-date 2024-03-15
```

Using a different config file entirely:
```bash
python osa_pipeline.py --config other_config.toml
```

## What it produces

In the working directory you'll get:
- `sent1/`, `sent2/` — downloaded GeoTIFFs, PNG previews, the segmentation mask, and the Shapefile (`.shp`/`.shx`/`.dbf`/`.prj`).
- `OSA_S1_<timestamp>_S2_<timestamp>.pdf` — the report.
- `OSA_S1_<timestamp>_S2_<timestamp>.zip` — everything above, zipped together.
