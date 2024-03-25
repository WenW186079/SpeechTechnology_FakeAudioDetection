import os
import csv
import random

# Define the directory containing spoof data
spoof_directory = "/mount/resources/speech/corpora/MLAAD/fake/it/tts_models_it_mai_female_vits"

# Define the CSV file paths for spoof data
train_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/it_vits_train.csv'
dev_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/it_vits_dev.csv'
test_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/it_vits_test.csv'

# Initialize a list to store spoof files
spoof_files = []

# Iterate through files in the spoof directory
for filename in os.listdir(spoof_directory):
    # Create the full path to the file
    file_path = os.path.join(spoof_directory, filename)
    
    # Check if the file is a .wav file
    if filename.endswith(".wav"):
        # Append the file name to the list along with its label
        spoof_files.append({'file': filename, 'label': 'spoof'})

# Shuffle the files
random.shuffle(spoof_files)

# Select 1000 spoof files randomly
selected_spoof_files = random.sample(spoof_files, 1000)

# Determine the sizes of train, dev, and test sets
train_size = int(0.7 * len(selected_spoof_files))
dev_size = int(0.15 * len(selected_spoof_files))
test_size = len(selected_spoof_files) - train_size - dev_size

# Split the selected files into train, dev, and test sets
train_set = selected_spoof_files[:train_size]
dev_set = selected_spoof_files[train_size:train_size + dev_size]
test_set = selected_spoof_files[train_size + dev_size:]

# Combine the sets to form a single list
combined_set = train_set + dev_set + test_set

# Shuffle the combined set
random.shuffle(combined_set)

# Write the sets to CSV files
def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csv_file:
        # Define the CSV columns
        fieldnames = ['file', 'label']
        # Create a CSV writer
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        csv_writer.writeheader()
        
        # Write the data
        csv_writer.writerows(data)

# Write train set to train_spoof.csv
write_to_csv(train_csv_path, train_set)

# Write dev set to dev_spoof.csv
write_to_csv(dev_csv_path, dev_set)

# Write test set to test_spoof.csv
write_to_csv(test_csv_path, test_set)

print(f'Train CSV file created at: {train_csv_path}')
print(f'Dev CSV file created at: {dev_csv_path}')
print(f'Test CSV file created at: {test_csv_path}')
