import os
import csv
import random

# Define the directories and labels
directories_and_labels = {
    "/Users/wenwen/Desktop/SpeechTechnology/Project/corpora/m-ailabs-speech/ru_RU/female/hajdurova/chetvero_nischih/wavs": "bona-fide",
    "/Users/wenwen/Desktop/SpeechTechnology/Project/corpora/m-ailabs-speech/ru_RU/male/minaev/oblomov/wavs": "bona-fide"
}

# Define the CSV file paths
train_csv_path = '/Users/wenwen/Desktop/ru_real_train.csv'
dev_csv_path = '/Users/wenwen/Desktop/ru_real_dev.csv'
test_csv_path = '/Users/wenwen/Desktop/ru_real_test.csv'

# Initialize lists to store files
bona_fide_files = []

# Iterate through each directory and its corresponding label
for directory, label in directories_and_labels.items():
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate through files in the directory
        for filename in os.listdir(directory):
            # Create the full path to the file
            file_path = os.path.join(directory, filename)
            
            # Check if the file is a .wav file
            if filename.endswith(".wav"):
                # Append the file name to the list along with its label
                bona_fide_files.append({'file': filename, 'label': label})

# Shuffle the files
random.shuffle(bona_fide_files)

# Select 1000 bona-fide files randomly
selected_bona_fide_files = random.sample(bona_fide_files, 1000)

# Determine the sizes of train, dev, and test sets
train_size = int(0.7 * len(selected_bona_fide_files))
dev_size = int(0.15 * len(selected_bona_fide_files))
test_size = len(selected_bona_fide_files) - train_size - dev_size

# Split the selected files into train, dev, and test sets
train_set = selected_bona_fide_files[:train_size]
dev_set = selected_bona_fide_files[train_size:train_size + dev_size]
test_set = selected_bona_fide_files[train_size + dev_size:]

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

# Write train set to train.csv
write_to_csv(train_csv_path, train_set)

# Write dev set to dev.csv
write_to_csv(dev_csv_path, dev_set)

# Write test set to test.csv
write_to_csv(test_csv_path, test_set)

print(f'Train CSV file created at: {train_csv_path}')
print(f'Dev CSV file created at: {dev_csv_path}')
print(f'Test CSV file created at: {test_csv_path}')
