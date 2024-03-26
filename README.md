# Are current fake audio detection language-independent?

##Environment setting

Install required dependencies in your env using:
`bash install.sh`

List of requirements:
`python=3.8
pytorch==1.11.0
torchaudio==0.11
torchvision==0.12.0
asteroid-filterbanks==0.4.0
librosa==0.9.2
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0`

Or, dirtrctly use setted environment in IMS 
`cd /mount/arbeitsdaten/deepfake/SpeechTechnology2023/ww`
`source speech/bin/activate`

##Data
Spoof data source - MLAAD
Bona-fide data source - Mailabs

###Data Selection Rule
- Randomly pick.
- Balanced. Same number of spoof and bona-fide samples(1000+1000). As the MLAAD are generated based on Mailabs, the durance of each label are similar.

###Selected Data
Traditional generating spoof data:
- Griffin Lim (En, De, Ru, It)
Neural network-based generating spoof data:
- VITS Neon (En, De)
- VITS  (It)
- Tacotron2 DCA  (En, De)

Note: Due to the limited amount of German data available for Tacotron2 DCA, it is included solely for comparison purposes. Additionally, since Italian data is not available for VITS Neon, VITS is utilized instead.




# Train the model
Here two models are used: 
RawNet3
- Input: raw audio
- No spectrogram-like features
SpecRNet
- Frontend algorithm: LFCC


#Pretrained models
You can find pretrained models in /configs/
The list of models are stored in `models.json`, e.g.:
`{
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
    }`




