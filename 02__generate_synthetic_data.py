import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def create_output_directory(script_path):
    """Creates a directory with the same name as the script to save outputs."""
    script_name = os.path.basename(script_path)
    folder_name = os.path.splitext(script_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def calculate_theoretical_te(p_abc):
    """
    Calculates the theoretical transfer entropy (TTE) for a given joint probability distribution.
    
    Args:
        p_abc (np.ndarray): A 3D numpy array of shape (M, M, M) representing the 
                            joint probability distribution p(A, B, C), where A is the next state of X,
                            B is the current state of X, and C is the current state of Y.
                            M is the number of bins.

    Returns:
        float: The theoretical transfer entropy value.
    """
    M = p_abc.shape[0]
    
    # Calculate marginal probabilities
    p_b = np.sum(p_abc, axis=(0, 2))
    p_ab = np.sum(p_abc, axis=2)
    p_bc = np.sum(p_abc, axis=0)

    # Add a small epsilon to prevent log(0)
    epsilon = 1e-15
    p_abc = p_abc + epsilon
    p_b = p_b + epsilon
    p_ab = p_ab + epsilon
    p_bc = p_bc + epsilon

    # Calculate TTE using the formula: TTE = sum(p(a,b,c) * log2( (p(a,b,c)*p(b)) / (p(a,b)*p(b,c)) ))
    te = 0.0
    for a in range(M):
        for b in range(M):
            for c in range(M):
                if p_abc[a, b, c] > epsilon: # Process only if there's a non-zero probability
                    term = (p_abc[a, b, c] * p_b[b]) / (p_ab[a, b] * p_bc[b, c])
                    te += p_abc[a, b, c] * np.log2(term)
    
    return te

def generate_synthetic_timeseries(p_abc, length):
    """
    Generates a synthetic time series of a given length based on a joint probability distribution.

    Args:
        p_abc (np.ndarray): The joint probability distribution p(A, B, C).
        length (int): The desired length of the time series.

    Returns:
        tuple: A tuple containing three numpy arrays (X_next, X_current, Y_current).
    """
    M = p_abc.shape[0]
    
    # Flatten the probability distribution for sampling
    flat_p = p_abc.flatten()
    
    # Generate random indices based on the flattened probability distribution
    # Each index corresponds to a unique (a, b, c) state
    indices = np.random.choice(M*M*M, size=length, p=flat_p)
    
    # Convert flat indices back to 3D indices (a, b, c)
    a_indices, b_indices, c_indices = np.unravel_index(indices, (M, M, M))
    
    # The states correspond to the indices
    X_next = a_indices
    X_current = b_indices
    Y_current = c_indices
    
    return X_next, X_current, Y_current

def get_sequence_statistics(sequences, M):
    """
    Calculates the occurrence counts of events from generated sequences.
    These counts are the features for the ML model.
    
    Args:
        sequences (tuple): Tuple of (X_next, X_current, Y_current) arrays.
        M (int): The number of bins.

    Returns:
        dict: A dictionary containing the counts m_abc, m_b, m_ab, m_bc.
    """
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
        
    return {
        "m_abc": m_abc.flatten(),
        "m_b": m_b,
        "m_ab": m_ab.flatten(),
        "m_bc": m_bc.flatten()
    }

def generate_training_dataset(num_samples, M, sequence_length):
    """
    Generates the full training dataset.

    Args:
        num_samples (int): The number of synthetic datasets to generate.
        M (int): The number of bins (e.g., 2 for binary states).
        sequence_length (int): The length of each synthetic time series.

    Returns:
        pandas.DataFrame: A DataFrame containing the features (event counts) and the target (TTE).
    """
    features_list = []
    
    print(f"Generating {num_samples} synthetic samples for M={M} and length={sequence_length}...")
    
    for _ in tqdm(range(num_samples)):
        # 1. Generate a random probability distribution
        p_abc_flat = np.random.rand(M*M*M)
        p_abc_flat /= np.sum(p_abc_flat) # Normalize to make it a valid probability distribution
        p_abc = p_abc_flat.reshape((M, M, M))
        
        # 2. Calculate the theoretical TE for this distribution (our target variable)
        tte = calculate_theoretical_te(p_abc)
        
        # 3. Generate a short time series based on this distribution
        sequences = generate_synthetic_timeseries(p_abc, sequence_length)
        
        # 4. Calculate the statistics (event counts) from the generated sequence (our features)
        stats = get_sequence_statistics(sequences, M)
        
        # 5. Store features and target
        feature_row = {
            'tte': tte,
            'sequence_length': sequence_length,
            'M': M
        }
        # Flatten the count arrays and add them to the feature row
        for i, count in enumerate(stats['m_abc']):
            feature_row[f'm_abc_{i}'] = count
        for i, count in enumerate(stats['m_b']):
            feature_row[f'm_b_{i}'] = count
        for i, count in enumerate(stats['m_ab']):
            feature_row[f'm_ab_{i}'] = count
        for i, count in enumerate(stats['m_bc']):
            feature_row[f'm_bc_{i}'] = count
            
        features_list.append(feature_row)
        
    return pd.DataFrame(features_list)

if __name__ == '__main__':
    # --- Setup ---
    # Create the output directory based on the script name
    output_folder = create_output_directory(__file__)
    print(f"Output folder '{output_folder}' created.")

    # --- Parameters ---
    # As per the paper, we generate a large number of synthetic samples.
    # Let's start with a reasonable number.
    NUM_SAMPLES = 10000 
    
    # The paper tests various sequence lengths. Our real data has length 24.
    # We should train on data of the same length.
    SEQUENCE_LENGTH = 24 
    
    # Number of bins for discretizing the data. Let's start with 2 (binary).
    # The paper suggests this is a critical parameter.
    NUM_BINS = 2
    
    # --- Execution ---
    training_data = generate_training_dataset(NUM_SAMPLES, NUM_BINS, SEQUENCE_LENGTH)
    
    # --- Save the dataset ---
    output_filename = f'synthetic_training_data_M{NUM_BINS}_L{SEQUENCE_LENGTH}.csv'
    output_path = os.path.join(output_folder, output_filename)
    training_data.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully generated and saved {len(training_data)} training samples to '{output_path}'")
    print("\nDataset preview:")
    print(training_data.head())

