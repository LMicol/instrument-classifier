# This script is used to conver WAV data into pytorch tensor format
# I've used this dataset: https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset
# This script expects the data to be organized with the same folder structure

import os
import torch
import torchaudio
import torchaudio.transforms as transforms

folder_path = "..../musical-instruments-sound-dataset/versions/3/"
dest_path = "..../musical-instruments-sound-dataset/versions/3/tensors/"

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
    exit(-1)

# List all files in the folder
content = os.listdir(folder_path)

# Iterate over each file
for item in content:
    current_path = os.path.join(folder_path, item)
    # Get only 'train' and 'test' folders
    if os.path.isdir(current_path) and item in ['train', 'test']:
        print(f"Processing files in '{item}' folder:")
        files = os.listdir(current_path)
        for file in files:
            file_path = os.path.join(current_path,file)
            # Check if it's a wav file (not a subdirectory)
            if os.path.isfile(file_path) and file_path[-4:] == '.wav':          
                # Process spectrogram
                waveform, sample_rate = torchaudio.load(file_path)
                transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
                spectrogram = transform(waveform)
                file_name = os.path.join(dest_path,item,file)[:-4]
                torch.save(spectrogram, file_name+'.pt')
                print(f"Saved '{file_name}'")