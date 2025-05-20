'''
This script aggregates the gridded data sets to the sub-national level (GADM 3.6 GID_1).
It is designed to be run on a SLURM cluster with multiple cores.
'''

import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import re
from shapely.geometry import Point
from rasterio.mask import mask
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from rtree import index
import sys

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),  # Save to file
        logging.StreamHandler(sys.stdout)  # Print to stdout (important for SLURM logs!)
    ]
)

logging.info("Script started...")
# Setup Logging
LOG_FILE = "logs/aggregate.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths (modify as needed)
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp" # needed for matching with DOSE
DATA_DIR = "/p/projects/impactee/Josh/chen_agg/chen_GDP/data/" # Directory containing GDP rasters
OUTPUT_CSV = "/p/projects/impactee/Josh/chen_agg/chen_GDP/output/aggregated_results.csv"

# Detect available cores
NUM_WORKERS = min(16, multiprocessing.cpu_count())  # Adjust based on HPC resources

# Load GADM shapefile
logging.info("Loading GADM shapefile...")
gadm = gpd.read_file(GADM_PATH)
logging.info(f"Loaded {len(gadm)} regions.")

for handler in logging.getLogger().handlers:
    handler.flush()

# Ensure GADM projection matches raster projection
raster_sample = next((f for f in os.listdir(DATA_DIR) if f.endswith(".tif")), None)
if raster_sample:
    with rasterio.open(os.path.join(DATA_DIR, raster_sample)) as src:
        if gadm.crs != src.crs:
            logging.info("Reprojecting GADM shapefile to match raster CRS...")
            gadm = gadm.to_crs(src.crs)

for handler in logging.getLogger().handlers:
    handler.flush()

# Create an R-tree index for spatial lookup
logging.info("Creating R-tree index for region lookups...")
idx = index.Index()
for i, region in gadm.iterrows():
    idx.insert(i, region.geometry.bounds, obj=region.geometry)

def process_region_wrapper(args):
    return process_region(*args)  # Unpack tuple before calling

def process_region(region, raster_path, year):
    gid = region["GID_1"]
    logging.info(f"Processing region: {gid} for year: {year}")

    try:
        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata if src.nodata is not None else 0
            out_image, out_transform = mask(src, [region.geometry], crop=True, nodata=nodata_value)

        out_image = out_image[0]

        if out_image.size == 0 or np.all(out_image == nodata_value):
            logging.info(f"Region {gid}: Masking resulted in an empty raster.")
            return gid, year, 0

        grp_count = np.sum(out_image[out_image > 0])  # Faster sum calculation

        logging.info(f"Finished processing region: {gid}, GRP: {grp_count}")
        return gid, year, grp_count

    except Exception as e:
        logging.error(f"Error processing region {gid} for year {year}: {e}")
        return gid, year, -1  # Mark as failed

def process_all_rasters(gadm, data_dir):
    results = []

    for raster_filename in os.listdir(data_dir):
        if raster_filename.endswith("GDP.tif"):
            year_match = re.search(r'(\d{4})GDP.tif', raster_filename)
            if not year_match:
                continue
            year = int(year_match.group(1))
            raster_path = os.path.join(data_dir, raster_filename)
            logging.info(f"Processing raster: {raster_filename} for year {year}")

            args_list = [(region, raster_path, year) for _, region in gadm.iterrows()]

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                region_results = list(executor.map(process_region_wrapper, args_list))

            results.extend(region_results)

    return results

# Run Processing
logging.info("Starting processing of all rasters...")
results = process_all_rasters(gadm, DATA_DIR)
logging.info("Finished processing all rasters.")

for handler in logging.getLogger().handlers:
    handler.flush()

# Save results
df = pd.DataFrame(results, columns=["GID_1", "year", "grp"])
df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Results saved to {OUTPUT_CSV}")