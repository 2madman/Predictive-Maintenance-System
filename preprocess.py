import pandas as pd
import numpy as np

def sample_and_preprocess_csv(input_filepath, output_filepath, sample_size=300000, random_state=42):
    """
    Sample and preprocess a large CSV file
    
    Args:
        input_filepath (str): Path to the original large CSV
        output_filepath (str): Path to save the sampled and preprocessed CSV
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducibility
    """
    # Read the large CSV file in chunks to manage memory
    print("Reading input CSV...")
    chunks = pd.read_csv(input_filepath, low_memory=False, chunksize=100000)
    
    # Concatenate chunks
    df = pd.concat(chunks)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Select relevant columns (adjust as needed)
    essential_columns = [
        'event_id', 'status', 'severity', 'category', 'source', 
        'creation_date', 'node_id', 'subcategory', 'eti_type', 
        'eti_value', 'monitor_uuid', 'severity_id', 'occurance','information'
    ]
    
    # Keep only essential columns and drop rows with missing status
    df_filtered = df[essential_columns].dropna(subset=['status'])
    
    # Sample the data
    if len(df_filtered) > sample_size:
        df_sampled = df_filtered.sample(n=sample_size, random_state=random_state)
    else:
        df_sampled = df_filtered
    
    print(f"Sampled dataset shape: {df_sampled.shape}")
    
    # Additional preprocessing
    # Convert categorical columns to category type
    categorical_columns = ['status', 'category', 'source', 'subcategory']
    for col in categorical_columns:
        df_sampled[col] = df_sampled[col].astype('category')
    
    '''
    # Convert date column
    df_sampled['creation_date'] = pd.to_datetime(df_sampled['creation_date'])
    
    # Extract additional features from date
    df_sampled['month'] = df_sampled['creation_date'].dt.month
    df_sampled['day_of_week'] = df_sampled['creation_date'].dt.dayofweek
    '''
    
    # Save the processed sample
    df_sampled.to_csv(output_filepath, index=False)
    
    print(f"Processed dataset saved to {output_filepath}")
    
    # Print some basic information about the sampled dataset
    print("\nDataset Information:")
    print(df_sampled['status'].value_counts())
    print("\nColumn Types:")
    print(df_sampled.dtypes)

def main():
    input_file = 'history.csv'
    output_file = 'history5.csv'
    
    sample_and_preprocess_csv(input_file, output_file)

if __name__ == "__main__":
    main()