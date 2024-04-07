import os
import csv
import random

################ Need to adjust #############
spoof_directory = [
    "/mount/resources/speech/corpora/MLAAD/fake/uk/tts_models_uk_mai_vits",
    

]
################ Need to adjust #############
train_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_vits_train.csv'
dev_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_vits_dev.csv'
test_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_vits_test.csv'


spoof_files = []
# Iterate over each directory in spoof_directory
for directory in spoof_directory:
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith(".wav"):
            relative_path = file_path.replace("/mount/resources/speech/corpora/", "")
            spoof_files.append({'file': relative_path, 'label': 'spoof'})

random.shuffle(spoof_files)

selected_spoof_files = random.sample(spoof_files, 1000)

train_size = int(0.7 * len(selected_spoof_files))
dev_size = int(0.15 * len(selected_spoof_files))
test_size = len(selected_spoof_files) - train_size - dev_size

print("test_size:",test_size )

train_set = selected_spoof_files[:train_size]
dev_set = selected_spoof_files[train_size:train_size + dev_size]
test_set = selected_spoof_files[train_size + dev_size:]

def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csv_file:
        fieldnames = ['file', 'label']
   
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(data)

write_to_csv(train_csv_path, train_set)
write_to_csv(dev_csv_path, dev_set)
write_to_csv(test_csv_path, test_set)

print(f'Train CSV file created at: {train_csv_path}')
print(f'Dev CSV file created at: {dev_csv_path}')
print(f'Test CSV file created at: {test_csv_path}')
