import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def create_hotspot_map(input_tiff_path, output_folder):
    """
    Analyzes a Transfer Entropy (TE) map to identify and plot causal hotspots.
    """
    if not os.path.exists(input_tiff_path):
        print(f"Error: Input file not found at {input_tiff_path}")
        return

    print(f"Reading raster data from: {input_tiff_path}")
    with rasterio.open(input_tiff_path) as src:
        te_map = src.read(1)
        nodata_value = src.nodata
        
        # Get geographic extent
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Mask out no-data values for statistical analysis
    te_valid_data = te_map[te_map != nodata_value]

    if te_valid_data.size == 0:
        print("Error: No valid data found in the raster.")
        return

    # 1. Statistical Characterization
    print("--- Statistical Characterization ---")
    min_val = np.min(te_valid_data)
    max_val = np.max(te_valid_data)
    mean_val = np.mean(te_valid_data)
    median_val = np.median(te_valid_data)
    std_val = np.std(te_valid_data)
    p25 = np.percentile(te_valid_data, 25)
    p75 = np.percentile(te_valid_data, 75)

    print(f"  Minimum: {min_val:.4f}")
    print(f"  Maximum: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Standard Deviation: {std_val:.4f}")
    print(f"  25th Percentile: {p25:.4f}")
    print(f"  75th Percentile: {p75:.4f}")

    # 2. Causal Hotspot Classification
    print("\n--- Classifying Causal Hotspots ---")
    # Create a new raster for classified data
    classified_map = np.full(te_map.shape, 0, dtype=np.int8) # 0 for No Data

    # Classify based on percentiles
    classified_map[te_map > p75] = 3  # High (Hotspot)
    classified_map[(te_map > p25) & (te_map <= p75)] = 2 # Medium
    classified_map[te_map <= p25] = 1 # Low
    
    # Mask the original nodata values
    classified_map[te_map == nodata_value] = 0

    # 3. Plotting the Classified Map
    print("--- Generating Hotspot Map ---")
    
    # Define a discrete colormap
    # 0: No Data (transparent), 1: Low (blue), 2: Medium (yellow), 3: High (red)
    cmap = ListedColormap(['#00000000', 'blue', 'yellow', 'red'])

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    im = ax.imshow(classified_map, cmap=cmap, extent=extent, interpolation='nearest')
    
    # Create a custom legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label=f'High (Hotspot) > {p75:.3f}'),
        Patch(facecolor='yellow', edgecolor='black', label=f'Medium > {p25:.3f}'),
        Patch(facecolor='blue', edgecolor='black', label=f'Low <= {p25:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Causal Influence', fontsize=12, title_fontsize=14)

    # Extract base name for title
    base_name = os.path.basename(input_tiff_path)
    if "Precipitation" in base_name:
        title = "Causal Hotspots: Precipitation to NDVI"
    elif "LST" in base_name:
        title = "Causal Hotspots: LST to NDVI"
    else:
        title = "Causal Hotspots"

    ax.set_title(title, pad=20)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.tight_layout()
    
    # --- Save the plot ---
    output_plot_filename = os.path.splitext(base_name)[0] + "_hotspots.png"
    output_plot_path = os.path.join(output_folder, output_plot_filename)
    print(f"\nSaving plot to: {output_plot_path}")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    
    print("\nHotspot analysis complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify and plot causal hotspots from a Transfer Entropy map.")
    parser.add_argument("input_file", type=str, help="Path to the input TE map GeoTIFF file.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the output plot will be saved.")
    
    args = parser.parse_args()
    
    create_hotspot_map(args.input_file, args.output_folder)
