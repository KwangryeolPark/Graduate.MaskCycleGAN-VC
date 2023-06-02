#!/bin/bash
python -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_<source_speaker_id>_<target_speaker_id> \
    --save_dir results/outputs/ \
    --preprocessed_data_dir vcc2018_preprocessed/vvcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id <source_speaker_id> \
    --speaker_B_id <target_speaker_id> \
    --ckpt_dir /home/ubuntu/MaskCycleGAN-VC/results/mask_cyclegan_vc_MinjungF2_Minjung/ckpts \
    --load_epoch <가장 최근에 저장된 checkpoint epoch 횟수> \
    --model_name generator_A2B \