import numpy as np
import pandas as pd
from pathlib import Path
import math
import random
import logging
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import os
import glob
from pathlib import Path


LOGGER = logging.getLogger(__name__)

SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
#FRAMES_NUMBER = 480_000  # <- originally 64_600
FRAMES_NUMBER = 64_600  


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

class SimpleAudioFakeDataset(Dataset):
    def __init__(
        self,
        transform=None,
        return_label: bool = True,
        return_meta: bool = False,
        oversample: bool = False,
        undersample: bool = False,
    ):
        self.transform = transform
        self.samples = pd.DataFrame()
        self.allowed_attacks = None
        self.seed = None
        self.return_label = return_label
        self.return_meta = return_meta
        self.oversample = oversample
        self.undersample = undersample
        self.read_samples()


    def df2tuples(self):
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append(
                (str(elem["path"]), elem["label"], elem["attack_type"])
            )

        self.samples = tuple_samples
        return self.samples
    
    def oversample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.get_group("bonafide"))
        spoof_length = len(samples.get_group("spoof"))

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.get_group("bonafide"))
        spoof_length = len(samples.get_group("spoof"))

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat(
                [samples.get_group("bonafide"), spoofs], ignore_index=True
            )

    def __getitem__(self, index) -> T_co:
        try:
            if isinstance(self.samples, pd.DataFrame):
                sample = self.samples.iloc[index]
                path = str(sample["path"])
                label = sample["label"]
                attack_type = sample["attack_type"]
                if type(attack_type) != str and math.isnan(attack_type):
                    attack_type = "N/A"
            else:
                path, label, attack_type = self.samples[index]
            
            # Check if the file has a ".wav" extension
            if not path.lower().endswith(".wav"):
                print(f"Skipping non-wav file at index {index}")
                return None

            waveform, sample_rate = torchaudio.load(path, normalize=APPLY_NORMALIZATION)
            real_sec_length = len(waveform[0]) / sample_rate

            # Apply trimming
            waveform, sample_rate = apply_trim(waveform, sample_rate)
            waveform, sample_rate = apply_preprocessing(waveform, sample_rate)

            # Return data only if the sample is valid
            if label is not None:
                return_data = [waveform, sample_rate]
                if self.return_label:
                    label = 1 if label == "bonafide" else 0
                    return_data.append(label)

                if self.return_meta:
                    return_data.append((attack_type, path, self.subset, real_sec_length))

                #print(f"Index: {index}, Path: {path}, Label: {label}, Attack Type: {attack_type}")
                #print(f"Waveform shape: {waveform.shape}, Sample Rate: {sample_rate}")

                return return_data
            else:
                print(f"Skipping invalid sample at index {index}")
                return None

        except Exception as e:
            print(f"Error processing file at index {index}: {e}")
            return None

    def __len__(self):
        return len(self.samples)



def apply_preprocessing(
    waveform,
    sample_rate,
    
):
    
    if sample_rate != SAMPLING_RATE and SAMPLING_RATE != -1:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    # Stereo to mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    # Trim too long utterances...
    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    # ... or pad too short ones.
    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def resample_file(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate



def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform


class MyDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        real_metafile_name,
        spoof_metafile_name,
        transform=None,
        seed=242,
        split_strategy="random",
        real_path = "/mount/resources/speech/corpora",
        spoof_path = "/mount/resources/speech/corpora",
        return_label=True,  
        return_meta=False,
    ):
        self.real_path = real_path
        self.real_metafile_name = real_metafile_name
        self.spoof_path = spoof_path
        self.spoof_metafile_name = spoof_metafile_name 
        super().__init__(
            transform=transform,
            return_label=return_label,
            return_meta=return_meta,
            )
        
        self.read_samples()
        self.seed = seed

    def read_samples(self):
        real_path = Path(self.real_path)
        spoof_path = Path(self.spoof_path)

        # Read real samples
        real_metafile_path = self.real_metafile_name
        real_samples = pd.read_csv(real_metafile_path)
        real_samples["path"] = real_samples["file"].apply(lambda n: real_path / n)
        real_samples["file"] = real_samples["file"].apply(lambda n: Path(n).stem)
        real_samples["label"] = real_samples["label"].map({"bona-fide": "bonafide", "spoof": "spoof"})
        real_samples["attack_type"] = real_samples["label"].map({"bonafide": "-", "spoof": "X"})
        real_samples.rename(columns={'file': 'sample_name'}, inplace=True)


        # Read spoof samples
        spoof_metafile_path = self.spoof_metafile_name
        spoof_samples = pd.read_csv(spoof_metafile_path)
        spoof_samples["path"] = spoof_samples["file"].apply(lambda n: str(spoof_path / n))
        spoof_samples["file"] = spoof_samples["file"].apply(lambda n: Path(n).stem)
        spoof_samples["label"] = spoof_samples["label"].map({"bona-fide": "bonafide", "spoof": "spoof"})
        spoof_samples["attack_type"] = spoof_samples["label"].map({"bonafide": "-", "spoof": "X"})
        spoof_samples.rename(columns={'file': 'sample_name'}, inplace=True)

        self.samples = pd.concat([real_samples, spoof_samples], ignore_index=True)
        self.samples = self.samples.sample(frac=1).reset_index(drop=True)
        '''
        print("Sample paths:")
        for path in self.samples['path']:
            print(path)
        '''

    def check_file_existence(self):
        missing_files = []
        for index, row in self.samples.iterrows():
            file_path = row["path"]
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        return missing_files

'''
if __name__ == "__main__":
    data_train = MyDataset(
        seed=242,
        real_path="/mount/resources/speech/corpora",
        real_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/en_real_train.csv",
        spoof_path= "/mount/resources/speech/corpora",
        spoof_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/en_gl_train.csv",

    )
    print("----------------------- Train Dataset --------------------")
    print("Number of samples:", len(data_train))
    print("First sample:", data_train[0])

    label_counts = data_train.samples['label'].value_counts()
    print("Number of real samples:", label_counts.get('bonafide', 0))
    print("Number of spoof samples:", label_counts.get('spoof', 0))

    print("Checking file existence...")
    missing_files = data_train.check_file_existence()
    if missing_files:
        print("Missing files:")
        for file_path in missing_files:
            print(file_path)
    else:
        print("All files exist.")

'''