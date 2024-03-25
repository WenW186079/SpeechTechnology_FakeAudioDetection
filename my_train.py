import numpy as np
import pandas as pd
from pathlib import Path
import logging
import math
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import argparse
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import yaml
import os

from data_loader import MyDataset
from src import specrnet, rawnet3
from src.trainer import GDTrainer
from src.commons import set_seed


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")

def train_nn(
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    config_save_path: str = "configs",
) -> Tuple[str, str]:
    logging.info("Loading data...")
    model_config = config["model"]
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    timestamp = time.time()
    checkpoint_path = ""

    ################################################################
    data_train = MyDataset(
        seed=242,
        real_path="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/real_dataset",
        real_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/de_real_train.csv",
        spoof_path= "/mount/resources/speech/corpora/MLAAD/fake/de/griffin_lim",
        spoof_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/de_gl_train.csv",

    )
    ################################################################
    data_dev = MyDataset(
        seed=242,
        real_path="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/real_dataset",
        real_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/de_real_dev.csv",
        spoof_path= "/mount/resources/speech/corpora/MLAAD/fake/de/griffin_lim",
        spoof_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/de_gl_dev.csv",
    )
   
    print("----------------------- get_datasets--------------------")
    print("Number of samples----data_train-----:", len(data_train))
    print("First sample:", data_train[0])
    print("Number of samples-------data_test-------:", len(data_dev))
    print("First sample:", data_dev[0])

    def get_model(model_name: str, config: Dict, device: str):
        if model_name == "rawnet3":
            return rawnet3.prepare_model()
        elif model_name == "specrnet":
            return specrnet.FrontendSpecRNet(
                device=device,
                **config,
            )
        
    current_model = get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )

    # If provided weights, apply corresponding ones (from an appropriate fold)
    model_path = config["checkpoint"]["path"]
    if model_path:
        current_model.load_state_dict(torch.load(model_path))
        logging.info(
            f"Finetuning '{model_name}' model, weights path: '{model_path}', on {len(data_train)} audio files."
        )
        if config["model"]["parameters"].get("freeze_encoder"):
            for param in current_model.whisper_model.parameters():
                param.requires_grad = False
    else:
        logging.info(f"Training '{model_name}' model on {len(data_train)} audio files.")
    current_model = current_model.to(device)

    ################################################################
    #use_scheduler = "rawnet3" in model_name.lower()
    use_scheduler = "specrnet" in model_name.lower()

    current_model = GDTrainer(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs=optimizer_config,
        use_scheduler=use_scheduler,
    ).train(
        dataset=data_train,
        model=current_model,
        test_dataset=data_dev,
    )

    if model_dir is not None:
        save_name = f"model__{model_name}__{timestamp}"
        save_model(
            model=current_model,
            model_dir=model_dir,
            name=save_name,
        )
        checkpoint_path = str(model_dir.resolve() / save_name / "ckpt.pth")

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"path": checkpoint_path}
        config_name = f"model__{model_name}__{timestamp}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        logging.info("Test config saved at location '{}'!".format(config_save_path))
    return config_save_path, checkpoint_path

def main():
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    ################################################################
    #config_file_path = "/mount/arbeitsdaten54/projekte/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/configs/training/rawnet3.yaml"
    config_file_path = "/mount/arbeitsdaten54/projekte/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/configs/training/specrnet.yaml"

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path("trained_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    train_nn(
        device=device,
        batch_size=5,
        epochs=3,
        model_dir=model_dir,
        config=config,
    )
    print("train-------------------------------done")

if __name__ == "__main__":
    main()
