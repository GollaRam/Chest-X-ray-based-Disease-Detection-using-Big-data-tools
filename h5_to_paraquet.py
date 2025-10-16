import numpy as np
import h5py
import ast
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

train_h5_path = "hdfs://localhost:9000/chestxray/embeddings_final1.h5"
output_parquet_path = "hdfs://localhost:9000/chestxray/embeddings_final1.parquet "


label_fields = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

def parse_embedding(cell_bytes: bytes, token_dim=768) -> np.ndarray:
    arr = np.array(ast.literal_eval(cell_bytes.decode("utf-8")), dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == token_dim:
        return arr
    elif arr.ndim == 2 and arr.shape[1] == token_dim:
        return arr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected embedding shape {arr.shape}")

def load_h5_to_dataframe(path, label_fields):
    print("Loading HDF5 file...")
    
    with h5py.File(path, "r") as f:
        table = f["embeddings"]["table"]
        N = table.shape[0]
        
        print(f"Processing {N} records...")
        
        # Parse embeddings
        emb = np.zeros((N, 768), dtype=np.float32)
        for i, cell in enumerate(table["image_embedding"]):
            if i % 1000 == 0:
                print(f"  Processed {i}/{N} embeddings...")
            emb[i] = parse_embedding(cell)
        
        # Create DataFrame with embeddings
        embedding_cols = [f"emb_{i}" for i in range(768)]
        df = pd.DataFrame(emb, columns=embedding_cols)
        
        # Add Sex and Age
        df['Sex'] = np.array(table["Sex"], dtype=np.float32)
        df['Age'] = np.array(table["Age"], dtype=np.float32)
        
        # Add labels
        for name in label_fields:
            col = np.array(table[name], dtype=np.float32)
            # Replace -1 with 0
            col = np.where(col == -1.0, 0.0, col)
            df[name] = col
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

print("Starting HDF5 to Parquet conversion...")
df = load_h5_to_dataframe(train_h5_path, label_fields)

print("\nDataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}... (showing first 10)")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nLabel distribution:")
for label in label_fields:
    positive_count = (df[label] == 1.0).sum()
    print(f"  {label}: {positive_count} positive samples")

print(f"\nSaving to Parquet: {output_parquet_path}")
table = pa.Table.from_pandas(df)
pq.write_table(table, output_parquet_path, compression='snappy')

print("Conversion complete!")
print(f"You can now use this file in Spark: {output_parquet_path}")
try:
    from google.colab import files
    print("\nDownloading Parquet file...")
    files.download(output_parquet_path)
except ImportError:

    print("\nNot in Colab environment, skipping download.")
