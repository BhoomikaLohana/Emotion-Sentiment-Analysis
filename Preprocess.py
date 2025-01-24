import pandas as pd

# Load the dataset
data = pd.read_csv('C:\\Users\\PMLS\\Desktop\\Sentiment_analysis\\Data\\Text.csv')

# Inspect the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling them with empty strings.")
    data.fillna("", inplace=True)

# Display dataset information
print("\nDataset Info:")
print(data.info())

# Check for duplicates
if data.duplicated().sum() > 0:
    print("\nDuplicates detected. Removing them...")
    data = data.drop_duplicates()

# Define emotion mapping
emotion_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}

# Replace labels with emotions
data['label'] = data['label'].map(emotion_mapping)
print("\nCleaned Dataset:")
print(data.head())

# Drop the 'Unnamed: 0' column
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Inspect the dataset after dropping
print("Dataset after dropping unnecessary columns:")
print(data.head())
cleaned_file_path = 'data/cleaned_data.csv'
data.to_csv(cleaned_file_path, index=False)

# Inspect the dataset after saving
print(f"Cleaned dataset saved to '{cleaned_file_path}' successfully.")
print(data.head())

import pandas as pd
import os
import json

# Load the dataset
data = pd.read_csv('C:\\Users\\PMLS\\Desktop\\Sentiment_analysis\\Data\\cleaned_data.csv')

# Output directory for batches
output_dir = 'batches'
os.makedirs(output_dir, exist_ok=True)

# Set the batch size
batch_size = 100
num_batches = 10

# Check if the dataset size matches the required batches
total_rows = batch_size * num_batches
if len(data) < total_rows:
    print(f"Dataset contains only {len(data)} rows, fewer than the required {total_rows} rows!")
    exit()

# Split the dataset into 10 batches
for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size
    batch_data = data.iloc[start_idx:end_idx]

    # Save the batch to a JSON file
    batch_file = os.path.join(output_dir, f'batch_{batch_num + 1}.json')
    batch_data.to_json(batch_file, orient='records', indent=4)
    print(f"Batch {batch_num + 1} saved to {batch_file}")

print("10 batches created successfully.")

