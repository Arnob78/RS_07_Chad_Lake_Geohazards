import os
import numpy as np
import tensorflow as tf
import joblib
from tqdm import tqdm
import time
import pandas as pd

# SUPPRESS ALL WARNINGS
import warnings
warnings.filterwarnings('ignore')  # This catches ALL warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

try:
    import rasterio
except ImportError:
    print("Error: The 'rasterio' library is required for this script.")
    print("Please install it by running: pip install rasterio")
    exit()

def create_output_directory(script_path):
    script_name = os.path.basename(script_path)
    folder_name = os.path.splitext(script_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def find_raster_file(directory, pattern):
    """Find raster file in directory matching pattern (case-insensitive)"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file in os.listdir(directory):
        if pattern.lower() in file.lower() and file.endswith('.tif'):
            return os.path.join(directory, file)
    
    raise FileNotFoundError(f"No file matching '{pattern}' found in {directory}")

def get_sequence_statistics(sequences, M):
    X_next, X_current, Y_current = sequences
    length = len(X_next)
    
    m_abc = np.zeros((M, M, M))
    m_ab = np.zeros((M, M))
    m_bc = np.zeros((M, M))
    m_b = np.zeros(M)

    for i in range(length):
        a, b, c = X_next[i], X_current[i], Y_current[i]
        m_abc[a, b, c] += 1
        m_ab[a, b] += 1
        m_bc[b, c] += 1
        m_b[b] += 1
        
    features = np.concatenate([
        m_abc.flatten(),
        m_b,
        m_ab.flatten(),
        m_bc.flatten()
    ])
    return features

def normalize_ndvi(ndvi_series):
    """Normalize NDVI from scaled values to typical -1 to 1 range"""
    return ndvi_series / 10000.0

def normalize_precipitation(precip_series):
    """Normalize precipitation to 0-1 range"""
    return precip_series / 2500.0

def discretize_timeseries_simple(ts):
    """
    Simple but robust discretization using median split
    """
    if np.all(ts == ts[0]) or np.std(ts) < 1e-6:
        return np.zeros_like(ts, dtype=int)
    
    median_val = np.median(ts)
    discrete = (ts > median_val).astype(int)
    
    if np.all(discrete == discrete[0]):
        mean_val = np.mean(ts)
        discrete = (ts > mean_val).astype(int)
        
    return discrete

def is_valid_pixel_data(pixel_data, nodata_value):
    """Check if pixel data is valid for processing."""
    if nodata_value is not None:
        if np.any(pixel_data == nodata_value):
            return False
    
    if np.any(~np.isfinite(pixel_data)):
        return False
        
    if np.std(pixel_data) < 1e-9:
        return False
        
    return True

if __name__ == '__main__':
    start_time = time.time()
    # --- Setup ---
    output_folder = create_output_directory(__file__)
    print(f"Output folder '{output_folder}' created.")

    # --- Load Trained Model and Scaler ---
    model_path = os.path.join('05__train_ml_te_model', 'ml_te_model.h5')
    scaler_path = os.path.join('05__train_ml_te_model', 'scaler.joblib')
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # --- File Paths ---
    # Find the precipitation raster dynamically
    precip_dir = "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/04__align_rasters"
    precip_path = find_raster_file(precip_dir, "Precipitation_annual_aligned")
    print(f"Found precipitation file: {os.path.basename(precip_path)}")
    
    # Find the NDVI raster dynamically
    ndvi_dir = "/srv/AB_YU/RS_07_Chad_Lake_Geohazards/03__create_annual_mosaics"
    ndvi_path = find_raster_file(ndvi_dir, "annual_composite")
    print(f"Found NDVI file: {os.path.basename(ndvi_path)}")
    
    # --- Raster Processing ---
    print("Opening raster files...")
    with rasterio.open(precip_path) as y_src, rasterio.open(ndvi_path) as x_src:
        # Basic validation
        if x_src.profile['count'] != 24 or y_src.profile['count'] != 24:
            raise ValueError("Input rasters must have 24 bands (years 2000-2023).")
        if x_src.profile['height'] != y_src.profile['height'] or x_src.profile['width'] != y_src.profile['width']:
            raise ValueError("Input rasters must have the same dimensions.")

        # Read all data
        x_data = x_src.read().astype(np.float32)
        y_data = y_src.read().astype(np.float32)
        
        # Get metadata for the output raster
        profile = x_src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32)
        
        # Handle no-data values
        x_nodata = x_src.nodata
        y_nodata = y_src.nodata
        output_nodata = -9999.0
        
        print(f"X no-data: {x_nodata}, Y no-data: {y_nodata}")
        
        # Create output array
        te_map = np.full((x_src.height, x_src.width), output_nodata, dtype=np.float32)
        
        # --- Batch Processing ---
        print("\nPreparing data for batch processing...")
        
        # Get all pixel coordinates
        rows, cols = np.indices((x_src.height, x_src.width))
        pixel_coords = np.stack((rows.flatten(), cols.flatten()), axis=1)
        
        # Extract time series for all pixels
        x_all_pixels = x_data.reshape(x_data.shape[0], -1).T
        y_all_pixels = y_data.reshape(y_data.shape[0], -1).T
        
        # Identify valid pixels
        valid_mask = np.ones(x_all_pixels.shape[0], dtype=bool)
        if x_nodata is not None:
            valid_mask &= ~np.any(x_all_pixels == x_nodata, axis=1)
        if y_nodata is not None:
            valid_mask &= ~np.any(y_all_pixels == y_nodata, axis=1)
            
        valid_mask &= np.all(np.isfinite(x_all_pixels), axis=1)
        valid_mask &= np.all(np.isfinite(y_all_pixels), axis=1)
        
        valid_mask &= (np.std(x_all_pixels, axis=1) >= 1e-9)
        valid_mask &= (np.std(y_all_pixels, axis=1) >= 1e-9)

        valid_coords = pixel_coords[valid_mask]
        x_valid = x_all_pixels[valid_mask]
        y_valid = y_all_pixels[valid_mask]
        
        print(f"Found {len(x_valid)} valid pixels to process.")

        if len(x_valid) > 0:
            # --- Feature Engineering in Batch ---
            print("Calculating features for valid pixels...")
            
            # Normalize
            x_normalized = normalize_ndvi(x_valid)
            y_normalized = normalize_precipitation(y_valid)
            
            # Discretize
            x_discrete = (x_normalized > np.median(x_normalized, axis=1, keepdims=True)).astype(int)
            y_discrete = (y_normalized > np.median(y_normalized, axis=1, keepdims=True)).astype(int)

            # Prepare sequences
            X_next = x_discrete[:, 1:]
            X_current = x_discrete[:, :-1]
            Y_current = y_discrete[:, :-1]
            
            # Calculate statistics for each pixel
            features_list = []
            for i in tqdm(range(len(x_valid)), desc="Calculating statistics"):
                sequences = (X_next[i], X_current[i], Y_current[i])
                features = get_sequence_statistics(sequences, M=2)
                features_list.append(features)
            
            features_array = np.array(features_list)
            
            # --- Scaling and Prediction in Batch ---
            print("Scaling features and predicting with the model (this will use the GPU)...")
            features_scaled = scaler.transform(features_array)
            predicted_te = model.predict(features_scaled, batch_size=1024, verbose=1)
            
            # --- Populate Output Map ---
            print("Populating the output TE map...")
            te_map[valid_coords[:, 0], valid_coords[:, 1]] = predicted_te.flatten()

            # --- Create and Save CSV Table ---
            print("Creating CSV table of results...")
            # Get the geographic coordinates for the valid pixels
            lons, lats = rasterio.transform.xy(x_src.transform, valid_coords[:, 0], valid_coords[:, 1])

            # Create a pandas DataFrame
            results_df = pd.DataFrame({
                'latitude': lats,
                'longitude': lons,
                'predicted_te': predicted_te.flatten()
            })

            # Save the DataFrame to a CSV file
            csv_output_path = os.path.join(output_folder, "te_results.csv")
            results_df.to_csv(csv_output_path, index=False)
            print(f"Results table saved to: {csv_output_path}")
            
            # --- Final Statistics ---
            print(f"\n=== PROCESSING SUMMARY ===")
            print(f"Total pixels: {x_src.height * x_src.width}")
            print(f"Valid pixels processed: {len(x_valid)}")
            
            te_values = predicted_te.flatten()
            print(f"TE Statistics:")
            print(f"  Min: {np.min(te_values):.4f}")
            print(f"  Max: {np.max(te_values):.4f}")
            print(f"  Mean: {np.mean(te_values):.4f}")
            print(f"  Std: {np.std(te_values):.4f}")
        else:
            print("❌ WARNING: No valid pixels found to process!")

        # Save output
        output_path = os.path.join(output_folder, "TE_Precipitation_to_NDVI_GPU.tif")
        print(f"\nSaving to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(te_map, 1)
            dst.nodata = output_nodata

    print("Processing complete.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")