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

## Dataset
Spoof data source: [MLAAD Dataset](https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP#editor)  
Bona-fide data source: [Mailabs Speech Dataset](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)

### Data Selection Rule
- Random selection.  
- Balanced: Equal number of spoof and bona-fide samples (1000+1000). As MLAAD data is generated based on Mailabs, the duration of each label is similar.

### Selected Dataset
- Traditional generating spoof dataset:
  - Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
- Neural network-based generating spoof dataset:
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


## Training Models
Two models are used:   
- RawNet3  
  - Input: raw audio  
  - No spectrogram-like features
  - The RawNet3 architecture is in a hybrid form of the ECAPA-TDNN and the RawNet2 with additional features including logarithm and normalisation.     
- SpecRNet  
  - Frontend algorithm: LFCC
  - A novel spectrogram– based model inspired by RawNet2 backbone.    

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
Pretrained models can be found [here](https://drive.google.com/drive/u/0/folders/1ysJuDmJSNiZ-ssyNeh_nDhT4eh_C-fLv). 
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


## Testing Models
Adjust the `real_metafile_name` and `spoof_metafile_name` of test data and choose the trained model you want to use. Run `my_eval.py`.
  
## Experiments

- Experiment 1: Investigating Language Impact on Raw Waveform Input Model
  - Round 1
    - Model: RawNet3  
    - Training data: en_Griffin Lim  
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)  
  - Round 2
    - Model: RawNet3
    - Training data: de_Griffin Lim
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
  - Round 3
    - Model: RawNet3
    - Training data: uk_Griffin Lim
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
  - Round 4
    - Model: RawNet3
    - Training data: en_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
  - Round 5
    - Model: RawNet3
    - Training data: de_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
  - Round 6
    - Model: RawNet3
    - Training data: uk_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
      
- Experiment 2: Investigating Language Impact on spectrogram features input model
  - Round 1
    - Model: SpecRNet  
    - Training data: en_Griffin Lim  
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)  
  - Round 2
    - Model: SpecRNet
    - Training data: de_Griffin Lim
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
  - Round 3
    - Model: SpecRNet
    - Training data: uk_Griffin Lim
    - Test data: Griffin Lim (En, De, Fr, Ru, It, Es, Pl, Uk)
  - Round 4
    - Model: SpecRNet
    - Training data: en_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
  - Round 5
    - Model: SpecRNet
    - Training data: de_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
  - Round 6
    - Model: SpecRNet
    - Training data: uk_VITS
    - Test data: VITS (En, De, Fr, It, Es, Pl, Uk)
    
## Performance
The test results of each test data with each model are stored in `test_results.csv`.   

## Results Analysis
- When using Griffin Lim to generate training and test fake audio, whether the results obtained by RawNet3 or SpecRNet detect model,
the trends in EER results remain consistent. Specifically, performances in German, Spanish, and Russian have notably declined, each to varying degrees.
- Linguistic homology doesn’t confer clear advantages. For instance, the performance of RawNet3(Train(de_gl)) in Test(en_gl) and Test(de_gl) is not as good as RawNet3(Train(uk_gl)). Additionally, SpecRNet(Train(uk_gl)) exhibits the poorest performance in Test(ru_gl).
- The score for each language family is calculated as the average of its constituent languages. For example, Germanic is computed as the average score of English and German. Although for RawNet3(Train(uk_gl)), Slavic performs the best among the three languages, SpecRNet(Train(uk_gl)) exhibits the opposite trend, with Slavic performing the worst among the three languages.
- For the fake audio generated by VITS, it is crucial to see the language when detecting it with RawNet3 detection model. Specifically, RawNet3 trained on German VITS data performs well only when detecting fake audio in the same language(here is German), while its performance is poor in other languages. Similarly, this trend holds for RawNet3 trained on English VITS data and on Ukrainian VITS data.
- For fake audio generated by VITS, language is not a significant factor when detected by SpecRNet model. For instance, SpecRNet trained on English VITS data performs well in tests on French, Italian, Polish, and Ukrainian VITS data without prior knowledge of the language.
- SpecRNet trained on German VITS data and SpecRNet trained on Ukrainian VITS data exhibit consistent performance trends across all test results. However, English, which belongs to the same language family of German, does not show consistency.
- When it comes to fake audio generated by VITS architecture, the SpecRNet detection model outperforms the RawNet3 detection model. Furthermore, it is speculated that when handling datasets with limited size, employing spectrogram-like features for neural spoof generation tends to result in improved performance.

## Conclusion
The influence of language varies across different fake audio generating architecture and detection models. According to the MLAAD dataset, for RawNet3 detection model training on fake audio generated by VITS, language plays a crucial role in detection. However, in other cases, language does not exhibit a significant effect.


