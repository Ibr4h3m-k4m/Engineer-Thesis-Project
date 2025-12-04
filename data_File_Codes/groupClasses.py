import pandas as pd
import numpy as np
import os

# Define the class mapping
class_mapping = {
    0: 0,  # Normal stays as 0
    1: 1,  # Position-related faults -> 1
    2: 1,  # Position-related faults -> 1 (assuming class 2 is similar to other position faults)
    3: 1,  # Position-related faults -> 1
    4: 1,  # Position-related faults -> 1
    5: 2,  # DoS-related attacks -> 2
    6: 2,  # DoS-related attacks -> 2
    7: 2,  # DoS-related attacks -> 2
    8: 3,  # Data/Message manipulation attacks -> 3
    9: 3,  # Data/Message manipulation attacks -> 3
    10: 2, # DoS-related attacks -> 2
    11: 2, # DoS-related attacks -> 2
    12: 2, # DoS-related attacks -> 2
    13: 4, # Eventual Stop -> 4 (keeping separate as requested)
    14: 5, # Speed-related faults -> 5
    15: 5, # Speed-related faults -> 5
    16: 5, # Speed-related faults -> 5
    17: 3, # Data/Message manipulation attacks -> 3
    18: 5, # Speed-related faults -> 5
    19: 5, # Speed-related faults -> 5
}

# Class names mapping for the output display
class_names = {
    0: "Normal",
    1: "Position-related faults",
    2: "DoS-related attacks",
    3: "Data/Message manipulation attacks",
    4: "Eventual Stop",
    5: "Speed-related faults"
}

# Files to process
files = {
    'X_train': 'X_train.csv',
    'X_test': 'X_test.csv',
    'X_val': 'X_val.csv',
    'y_train': 'y_train.csv',
    'y_test': 'y_test.csv',
    'y_val': 'y_val.csv'
}

# Load datasets
data = {}
for key, file in files.items():
    try:
        data[key] = pd.read_csv(file)
        print(f"Loaded {file}, shape: {data[key].shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")
        exit(1)

# Apply class mapping to y files
for y_file in ['y_train', 'y_test', 'y_val']:
    print(f"Applying class mapping to {y_file}...")
    # Depending on the format of your y files, choose the appropriate method
    if len(data[y_file].shape) == 1 or data[y_file].shape[1] == 1:
        # If y is a single column
        data[y_file] = data[y_file].applymap(lambda x: class_mapping[x]) if data[y_file].shape[1] == 1 else data[y_file].map(lambda x: class_mapping[x])
    else:
        # If y is in one-hot encoding format, need to convert to numerical first
        y_numeric = data[y_file].idxmax(axis=1)
        y_mapped = y_numeric.map(lambda x: class_mapping[x])
        # Convert back to one-hot encoding
        data[y_file] = pd.get_dummies(y_mapped)

# Save the modified datasets
for key, df in data.items():
    output_file = f"merged_{key}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

# Calculate and print the new class distributions
print("\nNew class distributions:")
for y_file in ['y_train', 'y_test', 'y_val']:
    print(f"\n{y_file} distribution:")
    
    # Get the distribution based on the format of the y files
    if len(data[y_file].shape) == 1 or data[y_file].shape[1] == 1:
        # If y is a single column
        if data[y_file].shape[1] == 1:
            distribution = data[y_file].iloc[:, 0].value_counts()
        else:
            distribution = data[y_file].value_counts()
    else:
        # If y is in one-hot encoding format
        distribution = data[y_file].sum(axis=0)
    
    total = distribution.sum()
    
    # Print distribution
    print(f"{'Class':<5} {'Class Name':<30} {'Count':<10} {'Percentage':>10}")
    print("-" * 60)
    
    for class_id in sorted(distribution.index):
        class_name = class_names.get(class_id, f"Class {class_id}")
        count = distribution[class_id]
        percentage = (count / total) * 100
        print(f"{class_id:<5} {class_name:<30} {count:<10} {percentage:>10.2f}%")
    
    print(f"Total: {total}")

# Calculate and print combined distribution across all datasets
print("\nCombined distribution across all datasets:")
combined_counts = {}

for y_file in ['y_train', 'y_test', 'y_val']:
    if len(data[y_file].shape) == 1 or data[y_file].shape[1] == 1:
        # If y is a single column
        if data[y_file].shape[1] == 1:
            current = data[y_file].iloc[:, 0].value_counts().to_dict()
        else:
            current = data[y_file].value_counts().to_dict()
    else:
        # If y is in one-hot encoding format
        current = data[y_file].sum(axis=0).to_dict()
    
    for class_id, count in current.items():
        if class_id in combined_counts:
            combined_counts[class_id] += count
        else:
            combined_counts[class_id] = count

total_combined = sum(combined_counts.values())

print(f"{'Class':<5} {'Class Name':<30} {'Count':<10} {'Percentage':>10}")
print("-" * 60)

for class_id in sorted(combined_counts.keys()):
    class_name = class_names.get(class_id, f"Class {class_id}")
    count = combined_counts[class_id]
    percentage = (count / total_combined) * 100
    print(f"{class_id:<5} {class_name:<30} {count:<10} {percentage:>10.2f}%")

print(f"Total: {total_combined}")
