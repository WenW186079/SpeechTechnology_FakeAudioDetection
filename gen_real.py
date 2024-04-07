import os
import csv
import random

#Here 'directories' and 'path'*3 need to be adjust


################ Need to adjust #############
directories = [
    "/mount/resources/speech/corpora/m-ailabs-speech/uk_UK/female",
    "/mount/resources/speech/corpora/m-ailabs-speech/uk_UK/male"
]

################ Need to adjust #############
train_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_real_train.csv'
dev_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_real_dev.csv'
test_csv_path = '/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/uk_real_test.csv'

label = "bona-fide"
female_files = []
male_files = []

for directory in directories:
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                
                if filename.endswith(".wav"):
                    relative_path = file_path.replace("/mount/resources/speech/corpora/", "")
                    
                    if directory.endswith("female"):
                        female_files.append({'file': relative_path, 'label': label})
                    elif directory.endswith("male"):
                        male_files.append({'file': relative_path, 'label': label})

random.shuffle(female_files)
random.shuffle(male_files)

selected_female_files = random.sample(female_files, 500)
selected_male_files = random.sample(male_files, 500)

selected_files = selected_female_files + selected_male_files

random.shuffle(selected_files)

train_size = int(0.7 * len(selected_files))
dev_size = int(0.15 * len(selected_files))
test_size = len(selected_files) - train_size - dev_size

train_set = selected_files[:train_size]
dev_set = selected_files[train_size:train_size + dev_size]
test_set = selected_files[train_size + dev_size:]

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
