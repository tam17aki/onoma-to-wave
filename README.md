# Unofficial implementations of Onoma-to-Wave
This repository provides unofficial implementations of Onoma-to-Wave.

## Licence
MIT licence.

## Dependencies
We tested the implemention on Ubuntu 18.04. The verion of Python was `3.6.9`. The following modules are required:

- torch
- hydra
- progressbar2
- pandas
- soundfile
- librosa
- joblib
- numpy
- sklearn

## Datasets
You need to prepare the following two datasets.

   - [Real World Computing Partnership-Sound Scene Database (RWCP-SSD)](http://research.nii.ac.jp/src/en/RWCP-SSD.html)

   - [RWCP-SSD-Onomatopoeia](https://github.com/KeisukeImoto/RWCPSSD_Onomatopoeia)

## Configurations
- `unconditional/`: The models are NOT conditioned on sound event labels.
- `conditional/`:  The models are conditioned on sound event labels.


## Recipes
1. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

2. Run `preprocess.py`. It performs preprocessing steps.

3. Run `training.py`. It performs model training.

4. Run `inference.py`. It performs inference using trained model (i.e., generate audios from onomatopoeia).
    
## Extra: release of pretrained models
   
Pretrained models are also available on the repository for demonstration purposes. Download pretrained models from here: 
https://drive.google.com/drive/folders/1d2StFDdNhWVTBekSpZaSSkJB8Wn63G0G?usp=sharing

Scripts are also available to easily try audio generation.

- `demo.py` ... This is a script for environmental sound synthesis using trained models. Unlike `inference.py`, it can be easily synthesized based on onomatopoeia and acoustic events specified in the yaml file, and is somewhat simpler to implement since it does not use DataSet and DataLoader.

- `pretrained_uncond.pt` ... Trained model without sound event conditioning.

- `pretrained_cond.pt` ... Trained model with sound event conditioning.

First, edit `config.yaml` and provide the path to the trained model according to your environment. Next, provide the onomatopoeia (phoneme representation) you wish to synthesize and, if necessary, the acoustic events in the `config.yaml` file. We do not provide a demonstration that works on Google Colab like ESPnet.
  
## References

Yuki Okamoto, Keisuke Imoto, Shinnosuke Takamichi, Ryosuke Yamanishi, Takahiro Fukumori and Yoichi Yamashita, 
"Onoma-to-wave: Environmental Sound Synthesis from Onomatopoeic Words", 
APSIPA Transactions on Signal and Information Processing: Vol. 11: No. 1, e13. http://dx.doi.org/10.1561/116.00000049
```
@article{SIP-2021-0049,
  url = {http://dx.doi.org/10.1561/116.00000049},
  year = {2022},
  volume = {11},
  journal = {APSIPA Transactions on Signal and Information Processing},
  title = {Onoma-to-wave: Environmental Sound Synthesis from Onomatopoeic Words},
  doi = {10.1561/116.00000049},
  issn = {},
  number = {1},
  pages = {-},
  author = {Yuki Okamoto and Keisuke Imoto and Shinnosuke Takamichi and Ryosuke Yamanishi and Takahiro Fukumori and Yoichi Yamashita}
}
```
