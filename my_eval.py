import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from src import commons, specrnet, rawnet3
from src.eer import compute_eer
from data_loader import MyDataset, SimpleAudioFakeDataset

def evaluate_nn(
    model_paths: List[Path],
    model_config: Dict,
    device: str,
    batch_size: int = 8,
):
    logging.info("Loading data...")
    model_name, model_parameters = model_config["name"], model_config["parameters"]

    def get_model(model_name: str, config: Dict, device: str):
        if model_name == "rawnet3":
            return rawnet3.prepare_model()
        elif model_name == "specrnet":
            return specrnet.FrontendSpecRNet(
                device=device,
                **config,
            )

    # Load model architecture
    model = get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if len(model_paths):
        model.load_state_dict(torch.load(model_paths))
    
    model = model.to(device)

    ################################################################
    data_val = MyDataset(
        seed=242,
        real_path="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/real_dataset",
        real_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/en_real_test.csv",###
        spoof_path= "/mount/resources/speech/corpora/MLAAD/fake/en/tts_models_en_ljspeech_tacotron2-DCA",###
        spoof_metafile_name="/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/meta_path/en_tacotron2_test.csv",###
    )

    print("datasets_paths:",data_val)
    print("evaluate_nn--data_val---------------")
    print("Number of samples---------------:", len(data_val))
    print("First sample:", data_val[0])

    logging.info(
        f"Testing '{model_name}' model, weights path: '{model_paths}', on {len(data_val)} audio files."
    )
    test_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=3,
    )

    batches_number = len(data_val) // batch_size
    num_correct = 0.0
    num_total = 0.0

    y_pred = torch.Tensor([]).to(device)
    y = torch.Tensor([]).to(device)
    y_pred_label = torch.Tensor([]).to(device)

    for i, (batch_x, _, batch_y) in enumerate(test_loader):
        model.eval()
        if i % 10 == 0:
            print(f"Batch [{i}/{batches_number}]")

        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            num_total += batch_x.size(0)

            batch_pred = model(batch_x).squeeze(1)
            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + 0.5).int()

            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            y_pred = torch.concat([y_pred, batch_pred], dim=0)
            y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
            y = torch.concat([y, batch_y], dim=0)

    eval_accuracy = (num_correct / num_total) 

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y.cpu().numpy(), y_pred_label.cpu().numpy(), average="binary", beta=1.0
    )
    auc_score = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_pred.cpu().numpy())

    # Calculate EER
    ground_truth = y.cpu().numpy()
    predictions = y_pred.cpu().numpy()
    eer, threshold = compute_eer(ground_truth, predictions)
    print("Equal Error Rate (EER):", eer)
    print("Threshold at EER:", threshold)

    logging.info( f"EER: {eer:.4f}, ACC: {eval_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}, AUC: {auc_score:.4f}" )
    
def main():
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ######################################

    #config_path = "model__rawnet3__1711006815.150007.yaml"         #rawnet3 - en/gl
    #config_path = "model__rawnet3__1711125863.4510744.yaml"        #rawnet3 - de/gl
    #config_path = "model__rawnet3__1711006385.0831056.yaml"        #rawnet3 - en/vits
    #config_path = "model__rawnet3__1711005901.003168.yaml"         #rawnet3 - de/vits

    #config_path = "model__specrnet__1710998052.7580142.yaml"       #specrnet - en/griffin_lim
    #config_path = "model__specrnet__1711126954.0504417.yaml"       #specrnet - de/griffin_lim
    config_path = "model__specrnet__1711007175.876735.yaml"        #specrnet - en/vits
    #config_path = "model__specrnet__1711007372.530419.yaml"        #specrnet - de/vits
    
    full_path = os.path.join("/mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww/deepfake-whisper-features/configs/", config_path)
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    commons.set_seed(seed)

    evaluate_nn(
        model_paths=config["checkpoint"].get("path", []),
        model_config=config["model"],
        device=device,
    )

if __name__ == "__main__":
    main()
