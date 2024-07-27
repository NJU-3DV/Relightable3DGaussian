#!/bin/bash

root_dir="datasets/neilfpp/data_dtu/DTU_scan"
list="24 37 40 55 63 65 69 83 97 105 106 110 114 118 122"
# list="24"

for i in $list
do
    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/dtu/${i}/3dgs \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        --lambda_mask_entropy 0.1 \
        --save_training_vis \
        --densify_grad_normal_threshold 999 \
        --lambda_depth_var 1e-2
    
    python eval_nvs.py --eval \
        -m output/dtu/${i}/3dgs \
        -c output/dtu/${i}/3dgs/chkpnt30000.pth

    python train.py --eval \
        -s ${root_dir}${i} \
        -m output/dtu/${i}/neilf \
        -c output/dtu/${i}/3dgs/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0 \
        --position_lr_final 0 \
        --normal_lr 0 \
        --sh_lr 0 \
        --opacity_lr 0 \
        --scaling_lr 0 \
        --rotation_lr 0 \
        --iterations 50000 \
        --lambda_base_color_smooth 1 \
        --lambda_roughness_smooth 0.5 \
        --lambda_light_smooth 1 \
        --lambda_light 0.01 \
        --light_init 3.0 \
        -t neilf --sample_num 32 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        --env_resolution 16 --env_lr 0.1 --roughness_lr 0.01

    python eval_nvs.py --eval \
        -m output/dtu/${i}/neilf \
        -c output/dtu/${i}/neilf/chkpnt50000.pth \
        -t neilf

done
