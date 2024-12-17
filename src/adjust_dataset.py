import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = "./"
TRAIN_DIR = os.path.join(BASE_DIR, "audio/train")
TEST_DIR = os.path.join(BASE_DIR, "audio/test")
OUTPUT_DIR = "./data"  # New directory to hold train, val, and test

# Create output directories
for split in ["train", "validation", "test"]:
    for class_name in ["Sound_Guiatr", "Sound_Drum", "Sound_Violin", "Sound_Piano"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

# Load train.csv and test.csv
train_csv_path = os.path.join(BASE_DIR, "audio/train.csv")
test_csv_path = os.path.join(BASE_DIR, "audio/test.csv")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Split train into train (80%) and validation (20%)
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Class'])

# Function to copy files to their respective directories
def copy_files(dataframe, source_dir, dest_dir):
    for _, row in dataframe.iterrows():
        file_path = os.path.join(source_dir, row['FileName'])
        class_name = row['Class']
        target_folder = os.path.join(dest_dir, class_name)
        
        # Ensure class folder exists
        os.makedirs(target_folder, exist_ok=True)
        
        # Copy the file
        shutil.copy(file_path, os.path.join(target_folder, row['FileName']))

# Copy train files
copy_files(train_data, TRAIN_DIR, os.path.join(OUTPUT_DIR, "train"))

# Copy validation files
copy_files(val_data, TRAIN_DIR, os.path.join(OUTPUT_DIR, "validation"))

# Copy test files
copy_files(test_df, TEST_DIR, os.path.join(OUTPUT_DIR, "test"))

# Save new CSV files
train_data.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_data.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)