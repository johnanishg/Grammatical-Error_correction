import pandas as pd

def parquet_to_tsv_pandas(parquet_file, tsv_file):
  """Converts a Parquet file to TSV using Pandas.

  Args:
    parquet_file: Path to the input Parquet file.
    tsv_file: Path to the output TSV file.
  """
  try:
      df = pd.read_parquet(parquet_file)
      df.to_csv(tsv_file, sep='\t', index=False)
      print(f"Successfully converted {parquet_file} to {tsv_file}")
  except Exception as e:
      print(f"Error converting file: {e}")

# Example usage:
parquet_to_tsv_pandas('datasets/coedit/train-00000-of-00001-d4ec08f29d3eda5a.parquet', 'datasets/coedit/train-00000-of-00001-d4ec08f29d3eda5a.tsv')