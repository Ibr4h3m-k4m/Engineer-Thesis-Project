import pandas as pd

# Define the class mapping dictionary
class_mapping = {
    0: 0,  # Normal stays as 0
    1: 1,  # Position-related faults -> 1
    2: 1,  # Position-related faults -> 1
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
    13: 4, # Eventual Stop -> 4
    14: 5, # Speed-related faults -> 5
    15: 5, # Speed-related faults -> 5
    16: 5, # Speed-related faults -> 5
    17: 3, # Data/Message manipulation attacks -> 3
    18: 5, # Speed-related faults -> 5
    19: 5, # Speed-related faults -> 5
}

# Load your dataset
df = pd.read_csv('dataset.csv')

# Apply the class mapping to create a new column
df['class'] = df['class'].map(class_mapping)

print("\nMerged class distribution:")
print(df['class'].value_counts().sort_index())

# Save the updated dataset
df.to_csv('dataset_merged_classes.csv', index=False)
