#!/usr/bin/env python3
import os
import glob
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import re

# =============================================================================
# Configuration
# =============================================================================

data_dirs = {
    "LST": "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/LST/",
    "NDVI": "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/NDVI/",
    "Precipitation": "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/data/raster/Precipitation/"
}

file_patterns = {
    "LST": "LakeChad_LST_*.tif",
    "NDVI": "LakeChad_NDVI_*.tif",
    "Precipitation": "LakeChad_Precipitation_Annual_2000_2023.tif"
}

# Use script filename to create output directory
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(os.getcwd(), script_name)

os.makedirs(output_dir, exist_ok=True)
print(f"Output folder: {output_dir}")

# =============================================================================
# Function to plot data
# =============================================================================

def plot_data(data_type, data_dir, file_pattern):
    """Plot LST, NDVI, or Precipitation data with consistent styling"""
    print(f"\nProcessing {data_type} data...")
    
    if data_type == "Precipitation":
        # Handle single stacked precipitation file
        tif_files = [os.path.join(data_dir, file_pattern)]
        if not os.path.exists(tif_files[0]):
            print(f"Precipitation file not found: {tif_files[0]}")
            return
    else:
        # Handle multiple individual files for LST and NDVI
        tif_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))

    if not tif_files:
        print(f"No {data_type} TIFF files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(tif_files)} {data_type} file(s):")
    for f in tif_files:
        print(f"  - {os.path.basename(f)}")

    # =========================================================================
    # Plotting
    # =========================================================================

    if data_type == "Precipitation":
        # For precipitation, we need to handle the stacked file
        plot_stacked_precipitation(tif_files[0], data_type, output_dir)
    else:
        # For LST and NDVI, use individual files
        plot_individual_files(tif_files, data_type, output_dir)

def plot_individual_files(tif_files, data_type, output_dir):
    """Plot individual files for LST and NDVI"""
    n = len(tif_files)
    cols = 4
    rows = int(np.ceil(n / cols))

    # Increase figure size for better spacing
    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 6))
    if rows == 1:
        axes = [axes] if cols == 1 else axes.flatten()
    else:
        axes = axes.flatten()

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    for idx, tif in enumerate(tif_files):
        try:
            with rasterio.open(tif) as src:
                data = src.read(1)
                
                # Remove fill values
                if data_type == "LST":
                    data = np.where(data < -1000, np.nan, data)
                else:  # NDVI
                    data = np.where(data < -10000, np.nan, data)

                ax = axes[idx]
                
                # Apply scaling based on data type
                if data_type == "LST":
                    # For MODIS LST, apply scaling and convert to Celsius
                    data_scaled = data * 0.02 - 273.15
                    vmin, vmax = -20, 40
                    cmap = "coolwarm"
                    unit = "°C"
                    variable_name = "LST"
                else:  # NDVI
                    # For MODIS NDVI, apply scaling (typically 0.0001)
                    data_scaled = data * 0.0001
                    # Clip to valid NDVI range
                    data_scaled = np.clip(data_scaled, -1.0, 1.0)
                    vmin, vmax = 0.0, 1.0  # Focus on positive NDVI values
                    cmap = "YlGn"  # Green color scheme for vegetation
                    unit = ""  # Empty string for NDVI
                    variable_name = "NDVI"
                
                # Extract year from filename using regex
                filename = os.path.basename(tif)
                year_match = re.search(r'(\d{4})', filename)
                year = year_match.group(1) if year_match else "Unknown"
                
                # Create the plot with proper geographic extent
                im = ax.imshow(data_scaled, cmap=cmap, 
                              vmin=vmin, vmax=vmax,
                              extent=[src.bounds.left, src.bounds.right, 
                                     src.bounds.bottom, src.bounds.top])
                
                # Set simplified title with larger font
                ax.set_title(f'{year} ({variable_name})', fontsize=18, fontweight='bold', pad=15)
                
                # Increase number of ticks and spacing
                ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
                
                # Set more ticks for both axes
                from matplotlib.ticker import AutoMinorLocator, MultipleLocator
                
                # Calculate reasonable tick intervals based on data extent
                x_range = src.bounds.right - src.bounds.left
                y_range = src.bounds.top - src.bounds.bottom
                
                # Set major ticks every 2 degrees (adjust as needed)
                x_major_interval = max(1, round(x_range / 6))
                y_major_interval = max(1, round(y_range / 6))
                
                ax.xaxis.set_major_locator(MultipleLocator(x_major_interval))
                ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
                
                # Add minor ticks for more detailed grid
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                
                # Add grid with lighter appearance
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
                
                # Only show axis labels for some plots to reduce clutter
                if idx >= (rows - 1) * cols:  # Bottom row
                    ax.set_xlabel("Longitude", fontsize=16, labelpad=10)
                else:
                    ax.set_xticklabels([])
                    
                if idx % cols == 0:  # Left column
                    ax.set_ylabel("Latitude", fontsize=16, labelpad=10)
                else:
                    ax.set_yticklabels([])
                
                # Format coordinate labels to be more readable
                ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
                
                # Add properly sized colorbar with larger font
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(unit, fontsize=16, rotation=0, labelpad=15)
                cbar.ax.tick_params(labelsize=14)
                
                # Print data statistics for debugging
                valid_data = data_scaled[~np.isnan(data_scaled)]
                if len(valid_data) > 0:
                    print(f"  {year} {variable_name}: min={np.min(valid_data):.3f}, max={np.max(valid_data):.3f}, mean={np.mean(valid_data):.3f}")
                
        except Exception as e:
            print(f"Error processing {tif}: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error\n{str(e)}", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Error', fontsize=16)

    # Clean empty axes
    for j in range(len(tif_files), len(axes)):
        fig.delaxes(axes[j])

    # Set appropriate title based on data type
    if data_type == "LST":
        title = "Land Surface Temperature - Lake Chad Basin (2000-2023)"
    else:
        title = "Normalized Difference Vegetation Index - Lake Chad Basin (2000-2023)"
    
    # Increase supertitle font size and padding
    fig.suptitle(title, fontsize=22, y=0.98, fontweight='bold')

    # Increase spacing between subplots
    plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)

    # Save result
    output_png = os.path.join(output_dir, f"{data_type}_preview_all.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    print(f"Plot saved: {output_png}")

def plot_stacked_precipitation(tif_file, data_type, output_dir):
    """Plot stacked precipitation file with multiple bands (years)"""
    try:
        with rasterio.open(tif_file) as src:
            print(f"Precipitation file has {src.count} bands (years)")
            
            # Assuming 24 bands for 2000-2023
            num_years = src.count
            years = list(range(2000, 2000 + num_years))
            
            cols = 4
            rows = int(np.ceil(num_years / cols))

            # Increase figure size for better spacing
            fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 6))
            if rows == 1:
                axes = [axes] if cols == 1 else axes.flatten()
            else:
                axes = axes.flatten()

            # Set global font sizes
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })

            for idx in range(num_years):
                data = src.read(idx + 1)  # Bands are 1-indexed
                data = np.where(data < -1000, np.nan, data)  # Remove fill values
                
                ax = axes[idx]
                
                # Precipitation data - no scaling needed typically
                data_scaled = data
                vmin, vmax = 0, np.nanpercentile(data, 98)  # Use 98th percentile to avoid outliers
                cmap = "Blues"
                unit = "mm"
                variable_name = "Precipitation"
                year = years[idx]
                
                # Create the plot with proper geographic extent
                im = ax.imshow(data_scaled, cmap=cmap, 
                              vmin=vmin, vmax=vmax,
                              extent=[src.bounds.left, src.bounds.right, 
                                     src.bounds.bottom, src.bounds.top])
                
                # Set simplified title with larger font
                ax.set_title(f'{year} ({variable_name})', fontsize=18, fontweight='bold', pad=15)
                
                # Increase number of ticks and spacing
                ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
                
                # Set more ticks for both axes
                from matplotlib.ticker import AutoMinorLocator, MultipleLocator
                
                # Calculate reasonable tick intervals based on data extent
                x_range = src.bounds.right - src.bounds.left
                y_range = src.bounds.top - src.bounds.bottom
                
                # Set major ticks every 2 degrees (adjust as needed)
                x_major_interval = max(1, round(x_range / 6))
                y_major_interval = max(1, round(y_range / 6))
                
                ax.xaxis.set_major_locator(MultipleLocator(x_major_interval))
                ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
                
                # Add minor ticks for more detailed grid
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                
                # Add grid with lighter appearance
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
                
                # Only show axis labels for some plots to reduce clutter
                if idx >= (rows - 1) * cols:  # Bottom row
                    ax.set_xlabel("Longitude", fontsize=16, labelpad=10)
                else:
                    ax.set_xticklabels([])
                    
                if idx % cols == 0:  # Left column
                    ax.set_ylabel("Latitude", fontsize=16, labelpad=10)
                else:
                    ax.set_yticklabels([])
                
                # Format coordinate labels to be more readable
                ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
                
                # Add properly sized colorbar with larger font
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.15)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(unit, fontsize=16, rotation=0, labelpad=15)
                cbar.ax.tick_params(labelsize=14)
                
                # Print data statistics for debugging
                valid_data = data_scaled[~np.isnan(data_scaled)]
                if len(valid_data) > 0:
                    print(f"  {year} Precipitation: min={np.min(valid_data):.1f}, max={np.max(valid_data):.1f}, mean={np.mean(valid_data):.1f} mm")

            # Clean empty axes
            for j in range(num_years, len(axes)):
                fig.delaxes(axes[j])

            # Set title for precipitation
            title = "Annual Precipitation - Lake Chad Basin (2000-2023)"
            
            # Increase supertitle font size and padding
            fig.suptitle(title, fontsize=22, y=0.98, fontweight='bold')

            # Increase spacing between subplots
            plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)

            # Save result
            output_png = os.path.join(output_dir, f"{data_type}_preview_all.png")
            plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor='white')
            plt.close()

            print(f"Plot saved: {output_png}")

    except Exception as e:
        print(f"Error processing precipitation file: {e}")

# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    # Plot all three datasets
    plot_data("LST", data_dirs["LST"], file_patterns["LST"])
    plot_data("NDVI", data_dirs["NDVI"], file_patterns["NDVI"])
    plot_data("Precipitation", data_dirs["Precipitation"], file_patterns["Precipitation"])
    
    print(f"\nAll processing completed. Check output folder: {output_dir}")
