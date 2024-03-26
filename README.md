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
  - Griffin Lim (En, De, Ru, It)  
- Neural network-based generating spoof data:  
  - VITS Neon (En, De)
  - VITS  (It)
  - Tacotron2 DCA  (En, De)

Note: Due to the limited amount of German data available for Tacotron2 DCA, it is included solely for comparison purposes. Additionally, since Italian data is not available for VITS Neon, VITS is utilized instead.

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

Adjust the training data path, dev data path, and metafile_name in `my_train.py`.  
List of spoof path:  
- en_gl: /MLAAD/fake/en/griffin_lim
- en_vits: /MLAAD/fake/en/tts_models_en_ljspeech_vits--neon
- en_tacotron2:/MLAAD/fake/en/tts_models_en_ljspeech_tacotron2-DCA
- de_gl: /MLAAD/fake/de/griffin_lim
- de_vits:/MLAAD/fake/de/tts_models_de_css10_vits-neon
- de_tacotron2:/MLAAD/fake/de/tts_models_de_thorsten_tacotron2-DCA
- it_gl:/MLAAD/fake/it/griffin_lim
- it_vits:/MLAAD/fake/it/tts_models_it_mai_female_vits
- ru_gl:/MLAAD/fake/ru/griffin_lim

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
      "name": "specrnet - en/griffin_lim",
      "config_file": "configs/model__specrnet__1710998052.7580142.yaml"
    },
    {
      "name": "specrnet - en/vits",
      "config_file": "configs/model__specrnet__1711007175.876735.yaml"
    }
  ]
}
```
  
## Testing the Model
Adjust the test data path, metafile_name, and choose the trained model you want to use. Run `my_eval.py`.

## Performance
The test results of each test data with each model are stored in `test_results.csv`.

## Conclusion
- The language factor impacts the effectiveness of fake audio detection, whether utilizing a raw waveform input model or employing a spectrogram features input model. Similar languages perform slightly better in detection outcomes.

- In general, for neural generating spoof, spectrogram-like features can lead better performance.



