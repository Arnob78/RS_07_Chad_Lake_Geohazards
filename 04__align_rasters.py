import os
import numpy as np
from tqdm import tqdm
import re
import time

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    print("Error: The 'rasterio' library is required for this script.")
    print("Please install it by running: pip install rasterio")
    exit()

def create_output_directory(script_path):
    """Creates a directory with the same name as the script to save outputs."""
    script_name = os.path.basename(script_path)
    folder_name = os.path.splitext(script_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def align_raster(source_path, reference_path, output_path):
    """Aligns a source raster to a reference raster."""
    print(f"\nAligning {os.path.basename(source_path)} to {os.path.basename(reference_path)}...")

    with rasterio.open(reference_path) as ref_src:
        ref_profile = ref_src.profile

    with rasterio.open(source_path) as src:
        if ref_src.count != src.count:
            raise ValueError(
                f"Mismatch in number of annual bands! "
                f"Reference has {ref_src.count} years, "
                f"Source has {src.count} years. "
            )

        output_nodata = -9999.0
        out_profile = ref_profile.copy()
        out_profile.update(count=src.count, nodata=output_nodata)

        with rasterio.open(output_path, 'w', **out_profile) as dst:
            for i in tqdm(range(1, src.count + 1), desc=f"Reprojecting {os.path.basename(source_path)}"):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile['transform'],
                    dst_crs=ref_profile['crs'],
                    resampling=Resampling.bilinear,
                    num_threads=os.cpu_count(),
                    src_nodata=src.nodata,
                    dst_nodata=output_nodata,
                )
    print(f"Successfully created aligned raster: {os.path.basename(output_path)}")

if __name__ == '__main__':
    output_folder = create_output_directory(__file__)
    print(f"Output folder '{output_folder}' created.")

    # --- Reference Raster ---
    reference_dir = "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/03__create_annual_mosaics"
    ndvi_files = [f for f in os.listdir(reference_dir) if f.lower().startswith('ndvi') and f.lower().endswith('annual_composite.tif')] # noqa
    if not ndvi_files:
        raise FileNotFoundError(f"No NDVI annual composite file found in {reference_dir}")
    reference_raster_path = os.path.join(reference_dir, ndvi_files[0])
    print(f"Found reference raster: {ndvi_files[0]}")

    # --- Source Rasters ---
    precip_source_path = "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/Precipitation/LakeChad_Precipitation_Annual_2000_2023.tif"
    lst_source_path = "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/03__create_annual_mosaics/lst_day_annual_composite.tif"

    # --- Output Paths ---
    year_range = "2000_2023"
    precip_output_path = os.path.join(output_folder, f"Precipitation_annual_aligned_{year_range}.tif")
    lst_output_path = os.path.join(output_folder, f"LST_annual_aligned_{year_range}.tif")

    # --- Align Rasters ---
    align_raster(precip_source_path, reference_raster_path, precip_output_path)
    align_raster(lst_source_path, reference_raster_path, lst_output_path)

    print("\n--- All rasters aligned successfully! ---")
