import os
import numpy as np
from tqdm import tqdm
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import glob

def create_output_directory(script_path):
    """Creates a directory with the same name as the script to save outputs."""
    script_name = os.path.basename(script_path)
    folder_name = os.path.splitext(script_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def find_yearly_files():
    """Find all yearly NDVI and LST files in the data directory structure."""
    base_path = '/srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster'
    
    print("Searching for yearly files...")
    
    # Find NDVI files
    ndvi_pattern = os.path.join(base_path, 'NDVI', 'LakeChad_NDVI_*_SINGLE_FILE.tif')
    ndvi_files = glob.glob(ndvi_pattern)
    ndvi_files.sort()  # Sort to ensure chronological order
    
    # Find LST files - exact pattern matching your files
    lst_pattern = os.path.join(base_path, 'LST', 'LakeChad_LST_*_SINGLE_FILE.tif')
    lst_files = glob.glob(lst_pattern)
    lst_files.sort()
    
    print(f"Found {len(ndvi_files)} NDVI files:")
    for file in ndvi_files:
        print(f"  - {os.path.basename(file)}")
    
    print(f"Found {len(lst_files)} LST files:")
    for file in lst_files:
        print(f"  - {os.path.basename(file)}")
    
    return ndvi_files, lst_files

def convert_lst_to_celsius(lst_data):
    """
    Convert MODIS LST data from scaled Kelvin to Celsius.
    
    MODIS LST formula: 
    LST = (DN * 0.02) - 273.15  [to get Celsius]
    Where DN is the digital number from the GeoTIFF
    """
    # Convert from digital number to Kelvin, then to Celsius
    kelvin = lst_data * 0.02
    celsius = kelvin - 273.15
    return celsius

def convert_ndvi_to_standard(ndvi_data):
    """
    Convert MODIS NDVI data to standard range (-1 to 1).
    
    MODIS NDVI formula:
    NDVI = DN * 0.0001
    Where DN is the digital number from the GeoTIFF
    """
    return ndvi_data * 0.0001

def get_band_info(file_path):
    """Get information about bands in a GeoTIFF file."""
    try:
        with rasterio.open(file_path) as src:
            return {
                'count': src.count,
                'descriptions': src.descriptions,
                'dtype': src.dtypes[0],
                'shape': src.shape,
                'nodata': src.nodata
            }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def create_annual_composite_from_yearly_files(yearly_files, output_path, variable_name, band_index=0):
    """
    Creates an annual composite from individual yearly files.
    
    Args:
        yearly_files (list): List of paths to yearly GeoTIFF files
        output_path (str): Path to save the output composite
        variable_name (str): Name of the variable for logging
        band_index (int): Which band to extract (0 for NDVI, 0 for LST_Day, 1 for LST_Night)
    
    Returns:
        list: A list of the mean values for each year
    """
    print(f"\nCreating annual composite for {variable_name} from {len(yearly_files)} yearly files...")
    
    annual_means = []
    
    # Read first file to get profile and check data
    first_file = yearly_files[0]
    print(f"Checking first {variable_name} file: {os.path.basename(first_file)}")
    
    band_info = get_band_info(first_file)
    if band_info:
        print(f"  Bands: {band_info['count']}")
        print(f"  Band descriptions: {band_info['descriptions']}")
        print(f"  Data type: {band_info['dtype']}")
        print(f"  Shape: {band_info['shape']}")
        print(f"  NoData value: {band_info['nodata']}")
    
    with rasterio.open(first_file) as src:
        first_data = src.read(band_index + 1)  # Bands are 1-indexed
        print(f"First file - Raw data range: {np.nanmin(first_data):.2f} to {np.nanmax(first_data):.2f}")
        
        if variable_name.startswith("LST"):
            converted_data = convert_lst_to_celsius(first_data)
            print(f"First file - Converted LST range: {np.nanmin(converted_data):.1f} to {np.nanmax(converted_data):.1f} °C")
        
        profile = src.profile
        profile.update(count=len(yearly_files), dtype=rasterio.float32)
        output_nodata = src.nodata if src.nodata is not None else -9999.0
        profile.update(nodata=output_nodata)
    
    # Create output file
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, file_path in enumerate(tqdm(yearly_files, desc=f"Processing {variable_name}")):
            try:
                with rasterio.open(file_path) as src:
                    # Read the specific band (LST files now have multiple bands)
                    data = src.read(band_index + 1)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Apply conversions based on variable type
                    if variable_name.startswith("LST"):
                        # Convert LST from scaled Kelvin to Celsius
                        data_converted = convert_lst_to_celsius(data)
                        # Filter reasonable temperature range for Lake Chad
                        data_converted = np.where((data_converted >= -50) & (data_converted <= 60), data_converted, np.nan)
                    elif variable_name == "NDVI":
                        # Convert NDVI to standard range
                        data_converted = convert_ndvi_to_standard(data)
                        data_converted = np.where((data_converted >= -1) & (data_converted <= 1), data_converted, np.nan)
                    else:
                        data_converted = data
                    
                    # Calculate mean for this year (only valid pixels)
                    valid_mask = ~np.isnan(data_converted)
                    if np.any(valid_mask):
                        year_mean = np.nanmean(data_converted)
                    else:
                        year_mean = np.nan
                    annual_means.append(year_mean)
                    
                    # Write converted data to output
                    data_filled = np.where(np.isnan(data_converted), output_nodata, data_converted)
                    dst.write(data_filled.astype(rasterio.float32), i + 1)
                    
                    # Set band description to year
                    year = os.path.basename(file_path).split('_')[2]  # Extract year from filename
                    dst.set_band_description(i + 1, f"Year_{year}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Write empty data for failed years
                empty_data = np.full((profile['height'], profile['width']), output_nodata)
                dst.write(empty_data.astype(rasterio.float32), i + 1)
                annual_means.append(np.nan)
    
    print(f"Successfully created {variable_name} composite with {len(yearly_files)} years.")
    print(f"{variable_name} annual means range: {np.nanmin(annual_means):.2f} to {np.nanmax(annual_means):.2f}")
    return annual_means

def plot_final_year(raster_path, variable_name, output_folder):
    """Plot the final year of the annual composite with proper units and dynamic range."""
    try:
        with rasterio.open(raster_path) as src:
            final_year_data = src.read(src.count)  # Get the last band
            
            # Replace nodata values with NaN for plotting
            if src.nodata is not None:
                final_year_data = np.where(final_year_data == src.nodata, np.nan, final_year_data)
            
            # Calculate actual data range (ignoring NaN)
            valid_data = final_year_data[~np.isnan(final_year_data)]
            
            if len(valid_data) == 0:
                print(f"Warning: No valid data found for {variable_name} in {raster_path}")
                # Create an empty plot with warning message
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, f'No valid data for {variable_name}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title(f'{variable_name} - Year {2000 + src.count - 1} - NO DATA')
                plot_path = os.path.join(output_folder, f'{variable_name.lower()}_final_year.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return
            
            # Set appropriate color maps and dynamic ranges
            if variable_name.startswith("LST"):
                cmap = 'coolwarm'
                # Use percentiles to avoid outliers
                vmin = np.nanpercentile(final_year_data, 2)  # 2nd percentile
                vmax = np.nanpercentile(final_year_data, 98)  # 98th percentile
                units = '°C'
                title = f'{variable_name} - Year {2000 + src.count - 1} ({units})'
                
                print(f"{variable_name} data range: {np.nanmin(final_year_data):.1f} to {np.nanmax(final_year_data):.1f} {units}")
                print(f"{variable_name} plot range: {vmin:.1f} to {vmax:.1f} {units}")
                
            else:  # NDVI
                cmap = 'viridis'
                vmin = np.nanpercentile(final_year_data, 2)
                vmax = np.nanpercentile(final_year_data, 98)
                units = 'NDVI'
                title = f'NDVI - Year {2000 + src.count - 1}'
                
                print(f"NDVI data range: {np.nanmin(final_year_data):.3f} to {np.nanmax(final_year_data):.3f}")
                print(f"NDVI plot range: {vmin:.3f} to {vmax:.3f}")
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            im = plt.imshow(final_year_data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, label=f'{variable_name} ({units})', shrink=0.8)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Add statistics text
            stats_text = f'Min: {np.nanmin(final_year_data):.1f}{units}\nMax: {np.nanmax(final_year_data):.1f}{units}\nMean: {np.nanmean(final_year_data):.1f}{units}'
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plot_path = os.path.join(output_folder, f'{variable_name.lower()}_final_year.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved {variable_name} plot to: {plot_path}")
            
    except Exception as e:
        print(f"Error plotting {variable_name}: {e}")

def main():
    """Main function to process yearly NDVI and LST data."""
    try:
        # Create output directory
        output_folder = create_output_directory(__file__)
        print(f"Output folder: {output_folder}")
        
        # Find yearly files
        ndvi_files, lst_files = find_yearly_files()
        
        if not ndvi_files:
            raise FileNotFoundError("No NDVI files found. Check the path: /srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/NDVI/")
        if not lst_files:
            raise FileNotFoundError("No LST files found. Check the path: /srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/LST/")
        
        # Check the structure of the first LST file
        print("\n=== CHECKING LST FILE STRUCTURE ===")
        first_lst_file = lst_files[0]
        band_info = get_band_info(first_lst_file)
        if band_info:
            print(f"LST files have {band_info['count']} bands")
            print(f"Band descriptions: {band_info['descriptions']}")
        
        # Ensure we have the same number of years for both datasets
        min_years = min(len(ndvi_files), len(lst_files))
        if len(ndvi_files) != len(lst_files):
            print(f"Warning: Different number of years - NDVI: {len(ndvi_files)}, LST: {len(lst_files)}")
            print(f"Using first {min_years} years for both datasets")
            ndvi_files = ndvi_files[:min_years]
            lst_files = lst_files[:min_years]
        
        # Define output paths
        ndvi_composite_path = os.path.join(output_folder, 'ndvi_annual_composite.tif')
        lst_day_composite_path = os.path.join(output_folder, 'lst_day_annual_composite.tif')
        lst_night_composite_path = os.path.join(output_folder, 'lst_night_annual_composite.tif')
        
        # Create annual composites from yearly files
        print("\nApplying conversions:")
        print("NDVI: DN * 0.0001 → Standard NDVI (-1 to 1)")
        print("LST: (DN * 0.02) - 273.15 → Celsius")
        
        # Process NDVI (single band files)
        ndvi_annual_means = create_annual_composite_from_yearly_files(ndvi_files, ndvi_composite_path, "NDVI", band_index=0)
        
        # Process LST Day (band 0 in multi-band LST files)
        lst_day_annual_means = create_annual_composite_from_yearly_files(lst_files, lst_day_composite_path, "LST_Day", band_index=0)
        
        # Process LST Night (band 1 in multi-band LST files)
        lst_night_annual_means = create_annual_composite_from_yearly_files(lst_files, lst_night_composite_path, "LST_Night", band_index=1)
        
        print("\nAll annual composites created successfully.")
        
        # --- Plotting ---
        print("\nGenerating plots...")
        plot_final_year(ndvi_composite_path, "NDVI", output_folder)
        plot_final_year(lst_day_composite_path, "LST_Day", output_folder)
        plot_final_year(lst_night_composite_path, "LST_Night", output_folder)
        
        # --- CSV Generation ---
        print("\nGenerating annual statistics CSV...")
        years = list(range(2000, 2000 + len(ndvi_annual_means)))
        
        stats_df = pd.DataFrame({
            'Year': years,
            'Mean_NDVI': ndvi_annual_means,
            'Mean_LST_Day_Celsius': lst_day_annual_means,
            'Mean_LST_Night_Celsius': lst_night_annual_means
        })
        
        csv_path = os.path.join(output_folder, 'annual_summary_stats.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved annual stats to: {csv_path}")
        
        # Print summary
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Output folder: {output_folder}")
        print(f"NDVI composite: {ndvi_composite_path} ({len(ndvi_files)} years)")
        print(f"LST Day composite: {lst_day_composite_path} ({len(lst_files)} years)")
        print(f"LST Night composite: {lst_night_composite_path} ({len(lst_files)} years)")
        print(f"Statistics CSV: {csv_path}")
        print(f"Years processed: {len(years)} ({years[0]} - {years[-1]})")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()