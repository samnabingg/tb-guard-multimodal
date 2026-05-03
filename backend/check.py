
import os, pandas as pd, pyarrow.parquet as pq

data_dir = 'data'
for f in os.listdir(data_dir):
    path = os.path.join(data_dir, f)
    print(f'\n=== {f} ===')
    try:
        if f.endswith('.csv'):
            df = pd.read_csv(path, nrows=3)
        elif f.endswith('.parquet'):
            df = pd.read_parquet(path).head(3)
        else:
            print('(skipped)')
            continue
        print('Shape:', df.shape)
        print('Columns:', list(df.columns))
        print(df.head(2).to_string())
    except Exception as e:
        print('Error:', e)
