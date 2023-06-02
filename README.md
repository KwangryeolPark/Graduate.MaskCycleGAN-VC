# MaskCycleGAN-VC
Unofficial **PyTorch** implementation of Kaneko et al.'s [**MaskCycleGAN-VC**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/maskcyclegan-vc/index.html) (2021) for non-parallel voice conversion.

MaskCycleGAN-VC is the state of the art method for non-parallel voice conversion using CycleGAN. It is trained using a novel auxiliary task of filling in frames (FIF) by applying a temporal mask to the input Mel-spectrogram. It demonstrates marked improvements over prior models such as CycleGAN-VC (2018), CycleGAN-VC2 (2019), and CycleGAN-VC3 (2020).

<p align="center">
<img src="imgs/MaskedCycleGAN-VC.png" width="500">
<br>
<b>Figure1: MaskCycleGAN-VC Training</b>
<br><br><br><br>
</p>

<p align="center">
<img src="imgs/generator.png" width="800">
<br>
<b>Figure2: MaskCycleGAN-VC Generator Architecture</b>
<br><br><br><br>
</p>

<p align="center">
<img src="imgs/discriminator.png" width="500">
<br>
<b>Figure3: MaskCycleGAN-VC PatchGAN Discriminator Architecture</b>
<br><br><br><br>
</p>

Paper: https://arxiv.org/pdf/2102.12841.pdf

Repository Contributors: [Claire Pajot](https://github.com/cmpajot), [Hikaru Hotta](https://github.com/HikaruHotta), [Sofian Zalouk](https://github.com/szalouk)

<br><br>
# 추가 사항
* 이 모델은 Voice conversion의 일종으로, source voice를 target voice로 변환해줍니다.
* 모든 음성 파일은 wav로 22050 sampling rate를 가집니다.
> Linux 환경에서 임의의 sampling rate로 된 wav 파일을 22050으로 바꾸는 코드는 다음과 같습니다.
```bash
#!/bin/bash

# 필요한 툴 설치: sox
sudo apt install sox
# 입력 폴더와 출력 폴더 경로 설정
input_folder="1"
output_folder="1_edit"

# 입력 폴더 내의 WAV 파일들에 대해 반복
for file in "$input_folder"/*.wav; do
    # 파일명 추출
    filename=$(basename "$file")
    # 새로운 파일 경로 설정
    output_file="$output_folder/${filename%.wav}.wav"
    # SoX를 사용하여 샘플링 속도 변경 (샘플링 속도: 22050)
    sox "$file" -r 22050 -c 1 -b 16 "$output_file" vol 0.5 dither -s 
done
```

> 파이썬 관련:   
> Python version: 3.8.16   
> > pip install -r requirements.txt   
> > This requirements file make you install pytorch==1.10.1, but you can may install more higher version if you need.   
> > The above behavior(version update) may occur error, but in pytorch==1.13.1+cu117, the error does not occure. 


> 실행 순서
> > 1. python 3.8로 가상환경 만들기
> > 2. pip install -r requirements.txt로 필요한 packages 설치하기.
> > > -> pytorch 버전은 CUDA version에 맞춰도 될듯.
> > 3. Dataset 설치하기 (보니까 Backend process로 설치되는듯?)
> > > 3.1. wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y   
> > > 3.2. wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip?sequence=3&isAllowed=y   
> > > 3.3. wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y   
> > > 3.4. mkdir vcc2018   
> > > 3.5. unzip vcc2018_database_training.zip?sequence=2 -d vcc2018/   
> > > 3.6. unzip vcc2018_database_evaluation.zip?sequence=3 -d vcc2018/   
> > > 3.7. unzip vcc2018_database_reference.zip?sequence=5 -d vcc2018/   
> > > 3.8. mv -v vcc2018/vcc2018_reference/* vcc2018/vcc2018_evaluation   
> > > 3.9. rm -rf vcc2018/vcc2018_reference   
> > > 3.10. vcc2018 아래에 vcc2018_training과 vcc2018_evaluation이 생긴 것을 확인.
> > 4. 준비된 source와 target wav 파일의 sampling rate를 22050으로 변환하기.
> > 5. source와 target wav 파일들을 vcc2018/vcc2018_training/SOURCE_DIR/여기와 vcc2018/vcc2018_training/TARGET_DIR/여기에 넣기.
> > > TARGET이 여러개인 경우, TARGET1_DIR, TARGET2_DIR 이렇게 해도 됨. (사실 디렉토리 이름은 상관 없음.)
> > 6. 데이터 전처리하기.
> > > 6.1. bash_scripts의 preprocessing.sh에서 DATA_DIRECTORY와 PREPROCESSED_DATA_DIRECTORY를 자신의 환경에 맞게 수정하기.
> > > > DATA_DIRECTORY는 3번 과정에서 만든 ~~~~/vcc2018_training이고, PREPRECESSED_DATA_DIRECTORY는 프로그램이 자동으로 생성할 precessed data의 위치로, 단순히 위치만 지정하면 됨. ex) ~/Graduate.MaskCycleGAN-VC/vcc_2018_preprocessed/vcc2018_training   
> > > > speaker_id는 5번 과정에서 언급한 SOURCE_DIR과 TARGET_DIR 또는 TARGET1_DIR, TARGET2_DIR을 공백으로 분리하여 넣으면 됨.   
> > 7. Training 시작.
> > > --name: ckpt가 저장될 이름.   
> > > --preprocessed_data_dir: 6.1.에서 지정한 PREPRECESSED_DATA_DIRECTORY   
> > > --speaker_A_id: 5.에서 언급한 SOURCE_DIR   
> > > --speaker_B_id: 5.에서 언급한 TARGET_DIR   
> > > --continue_train: checkpoint를 이용하여 training을 계속함.
> > > > 아마 다른 설정은 동일한 상태로 continue_train을 지정하면 --save_dir과 --name을 이용해서 프로그램이 자동으로 .ckpt 파일을 찾는듯.(혹은 .tar or .pth)
> > 8. Test 시작.
> > > 기본적인 건 training과 동일.   
> > > --ckpt_dir: checkpoint의 경로 ex) results/any_name/ckpts   
> > > --load_epoch: 가장 최근에 저장된 checkpoint epoch 횟수  
> > > > ckpts 폴더 아래의 .pth.tar 파일명의 앞 숫자들을 의미 ex) 00025_discriminator_A.pth.tar 파일에서는 00025가 load_epoch   
 

You should refer <a href="https://velog.io/@minjungh63/MaskCycleGAN-VC%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EC%9D%8C%EC%84%B1-%EC%98%A4%EB%94%94%EC%98%A4%EC%9D%98-%EB%AA%A9%EC%86%8C%EB%A6%AC-%EB%B0%94%EA%BE%B8%EA%B8%B0">MaskCycleGAN-VC를 이용하여 음성 오디오의 음색 바꾸기</a>

## Setup

Clone the repository.

```
git clone git@github.com:GANtastic3/MaskCycleGAN-VC.git
cd MaskCycleGAN-VC
```

Create the conda environment.
```
conda env create -f environment.yml
conda activate MaskCycleGAN-VC
```

## VCC2018 Dataset

The authors of the paper used the dataset from the Spoke task of [Voice Conversion Challenge 2018 (VCC2018)](https://datashare.ed.ac.uk/handle/10283/3061). This is a dataset of non-parallel utterances from 6 male and 6 female speakers. Each speaker utters approximately 80 sentences.

Download the dataset from the command line.
```
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip?sequence=3&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y
```

Unzip the dataset file.
```
mkdir vcc2018
apt-get install unzip
unzip vcc2018_database_training.zip?sequence=2 -d vcc2018/
unzip vcc2018_database_evaluation.zip?sequence=3 -d vcc2018/
unzip vcc2018_database_reference.zip?sequence=5 -d vcc2018/
mv -v vcc2018/vcc2018_reference/* vcc2018/vcc2018_evaluation
rm -rf vcc2018/vcc2018_reference
```

## Data Preprocessing

To expedite training, we preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. We convert waveforms to spectrograms using a [melgan vocoder](https://github.com/descriptinc/melgan-neurips) to ensure that you can decode voice converted spectrograms to waveform and listen to your samples during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```


## Training

Train MaskCycleGAN-VC to convert between `<speaker_A_id>` and `<speaker_B_id>`. You should start to get excellent results after only several hundred epochs.
```
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_<speaker_id_A>_<speaker_id_B> \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ \
    --speaker_A_id <speaker_A_id> \
    --speaker_B_id <speaker_B_id> \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --batch_size 1 \
    --lr 5e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```

To continue training from a previous checkpoint in the case that training is suspended, add the argument `--continue_train` while keeping all others the same. The model saver class will automatically load the most recently saved checkpoint and resume training.

Launch Tensorboard in a separate terminal window.
```
tensorboard --logdir results/logs
```

## Testing

Test your trained MaskCycleGAN-VC by converting between `<speaker_A_id>` and `<speaker_B_id>` on the evaluation dataset. Your converted .wav files are stored in `results/<name>/converted_audio`.

```
python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data1/cycleGAN_VC3/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts \
    --load_epoch 500 \
    --model_name generator_A2B \
```

Toggle between A->B and B->A conversion by setting `--model_name` as either `generator_A2B` or `generator_B2A`.

Select the epoch to load your model from by setting `--load_epoch`.

## Code Organization
```
├── README.md                       <- Top-level README.
├── environment.yml                 <- Conda environment
├── .gitignore
├── LICENSE
|
├── args
│   ├── base_arg_parser             <- arg parser
│   ├── train_arg_parser            <- arg parser for training (inherits base_arg_parser)
│   ├── cycleGAN_train_arg_parser   <- arg parser for training MaskCycleGAN-VC (inherits train_arg_parser)
│   ├── cycleGAN_test_arg_parser    <- arg parser for testing MaskCycleGAN-VC (inherits base_arg_parser)
│
├── bash_scripts
│   ├── mask_cyclegan_train.sh      <- sample script to train MaskCycleGAN-VC
│   ├── mask_cyclegan_test.sh       <- sample script to test MaskCycleGAN-VC
│
├── data_preprocessing
│   ├── preprocess_vcc2018.py       <- preprocess VCC2018 dataset
│
├── dataset
│   ├── vc_dataset.py               <- torch dataset class for MaskCycleGAN-VC
│
├── logger
│   ├── base_logger.sh              <- logging to Tensorboard
│   ├── train_logger.sh             <- logging to Tensorboard during training (inherits base_logger)
│
├── saver
│   ├── model_saver.py              <- saves and loads models
│
├── mask_cyclegan_vc
│   ├── model.py                    <- defines MaskCycleGAN-VC model architecture
│   ├── train.py                    <- training script for MaskCycleGAN-VC
│   ├── test.py                     <- training script for MaskCycleGAN-VC
│   ├── utils.py                    <- utility functions to train and test MaskCycleGAN-VC

```

## Acknowledgements

This repository was inspired by [jackaduma](https://github.com/jackaduma)'s implementation of [CycleGAN-VC2](https://github.com/jackaduma/CycleGAN-VC2).
