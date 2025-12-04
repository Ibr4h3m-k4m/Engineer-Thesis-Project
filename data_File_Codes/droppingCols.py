import os
import pandas as pd

# the feature indices to keep
FEATURE_INDICES = [1,3,7,8,12,13,14]

# input file names
data_files = {
    'X_train': 'X_train.csv',
    'X_val':   'X_val.csv',
    'X_test':  'X_test.csv',
    'y_train': 'y_train.csv',
    'y_val':   'y_val.csv',
    'y_test':  'y_test.csv',
}

# output folder
out_dir = 'selected_data2'
os.makedirs(out_dir, exist_ok=True)

# load, select, and save X datasets
def process_X(name, fname):
    # read header row, disable low-memory
    df = pd.read_csv(fname, header=0, low_memory=False)
    # select by integer position, then convert to numeric
    df_sel = df.iloc[:, FEATURE_INDICES].apply(pd.to_numeric, errors='coerce')
    out_path = os.path.join(out_dir, f"{name}_selected.csv")
    df_sel.to_csv(out_path, index=False)
    return df_sel

# load and save y datasets (unchanged values)
def process_y(name, fname):
    df = pd.read_csv(fname, header=0)
    # assume first column is label
    series = df.iloc[:, 0]
    out_path = os.path.join(out_dir, f"{name}.csv")
    series.to_frame().to_csv(out_path, index=False)
    return series

# process all files
X_train = process_X('X_train', data_files['X_train'])
X_val   = process_X('X_val',   data_files['X_val'])
X_test  = process_X('X_test',  data_files['X_test'])

y_train = process_y('y_train', data_files['y_train'])
y_val   = process_y('y_val',   data_files['y_val'])
y_test  = process_y('y_test',  data_files['y_test'])

# combine all y for overall distribution
y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)

# print shapes and overall distribution
print(f"Saved selected CSVs into folder: {out_dir}\n")
print("--- Data Shapes ---")
print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_all:   {y_all.shape}\n")

print("--- Overall Class Distribution (all splits) ---")
dist = y_all.value_counts().sort_index()
pct = (dist / len(y_all) * 100).round(2)
for cls, cnt in dist.items():
    print(f"Class {cls}: {cnt} samples ({pct[cls]}%)")
