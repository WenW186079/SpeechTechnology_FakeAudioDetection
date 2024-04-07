# Are current fake audio detection language-independent?

## Environment Setup

Install the required dependencies in your environment using:
`bash install.sh`

List of requirements: 

```
python=3.8  
pytorch==1.11.0  
torchaudio==0.11  
torchvision==0.12.0  
asteroid-filterbanks==0.4.0  
librosa==0.9.2  
pandas>=1.3.0  
numpy>=1.21.0  
scikit-learn>=0.24.0
``` 

Or directly use the set environment in IMS:    
```
cd /mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww
source speech/bin/activate
```

## Data
Spoof data source: [MLAAD Dataset](https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP#editor)  
Bona-fide data source: [Mailabs Speech Dataset](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)

### Data Selection Rule
- Random selection.  
- Balanced: Equal number of spoof and bona-fide samples (1000+1000). As MLAAD data is generated based on Mailabs, the duration of each label is similar.

### Selected Data
- Traditional generating spoof data:
  - Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
- Neural network-based generating spoof data:
  - VITS (En, De, Fr, It, Es, Pl, Uk)
  - VITS Neon (En, De)

Language codes: En-English, De-German, Fr-French, Ru-Russian, It-Italian, Es-Spanish, Pl-Polish, Uk-Ukrainian  
→ Germanic: English, German  
→ Romance: French, Italian, Spanish  
→ Slavic: Polish, Ukrainian, Russian  

Selected data are stored in meta CSV files: `/meta_path/`  
To ensure no overlap, real and spoof data are separated into three files: train, dev, test. For example:  
Real:`de_real_train.csv` `de_real_dev.csv` `de_real_test.csv`   
Spoof: `de_gl_train.csv` `de_gl_dev.csv` `de_gl_test.csv`  

## Training the Model
Two models are used:   
- RawNet3  
  - Input: raw audio  
  - No spectrogram-like features  
- SpecRNet  
  - Frontend algorithm: LFCC  

Adjust the `real_metafile_name` and `spoof_metafile_name` of training data and dev data in `my_train.py`.  
List of spoof path:  
- en_gl: /MLAAD/fake/en/griffin_lim
- en_vits: /MLAAD/fake/en/tts_models_en_ljspeech_vits
- de_gl: /MLAAD/fake/de/griffin_lim
- de_vits:/MLAAD/fake/de/tts_models_de_thorsten_vits
- it_gl:/MLAAD/fake/it/griffin_lim
- it_vits:/MLAAD/fake/it/tts_models_it_mai_female_vits + /MLAAD/fake/it/tts_models_it_mai_male_vits 
- ru_gl:/MLAAD/fake/ru/griffin_lim
- es_gl:/MLAAD/fake/es/griffin_lim
- es_vits:/MLAAD/fake/es/tts_models_es_css10_vits
- pl_gl:/MLAAD/fake/pl/griffin_lim
- pl_vits:/MLAAD/fake/pl/tts_models_pl_mai_female_vits ##Only have female samples
- fr_gl:/MLAAD/fake/fr/griffin_lim
- fr_vits:/MLAAD/fake/fr/tts_models_fr_css10_vits
- uk_gl:/MLAAD/fake/uk/griffin_lim
- uk_vits:/MLAAD/fake/uk/tts_models_uk_mai_vits

Adjust the model name:  
`"rawnet3"` or `"specrnet"`  

And corresponding model path:  
`/configs/training/rawnet3.yaml"` or  
`/configs/training/specrnet.yaml"`  

## Pretrained models
Pretrained models can be found in `/configs/`  
The list of models are stored in `models.json`, e.g.:  
```
{
  "models": [
    {
      "name": "rawnet3",
      "configs": [
        {
          "training data language": "en",
          "training data model": "griffin_lim",
          "file": "model__rawnet3__1711981315.3284469.yaml"
        },
        {
          "training data language": "de",
          "training data model": "griffin_lim",
          "file": "model__rawnet3__1711982401.4415708.yaml"
        },
    }
  ]
}
```
The trained models can be found in [trained_models](https://drive.google.com/drive/folders/1n7g5zXGX4D3aslLPvk4gS8rNANnP4jD-?usp=drive_link)

## Testing the Model
Adjust the `real_metafile_name` and `spoof_metafile_name` of test data and choose the trained model you want to use. Run `my_eval.py`.
    
## Performance
The test results of each test data with each model are stored in `test_results.csv`.

## Conclusion
- Based on the findings from the MLAAD dataset, it appears that the language factor does not significantly influence the detection of fake audio.
- When dealing with datasets of limited size, utilizing spectrogram like features for neural spoof generation tends to yield improved performance.



