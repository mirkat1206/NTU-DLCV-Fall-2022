#!/bin/bash
pwd
echo

python3 hw4_1_write_config.py $1
echo
echo -------------------------------------
echo

cd DirectVoxGO
pwd 
echo ../$1

# python3 run.py --config ./configs/nerf/hw4.py --render_test --dump_images --json ../$1 --outdir ../$2
python3 run.py --config ./configs/nerf/hw4.py --render_only --render_test --dump_images --json ../$1 --outdir ../$2
# python3 run.py --config ./configs/nerf/hw4.py --render_test --eval_ssim --eval_lpips_vgg --dump_images --json ../$1 --outdir ../$2
# python3 run.py --config ./configs/nerf/hw4.py --render_only --render_test --eval_ssim --eval_lpips_vgg --dump_images
echo
echo -------------------------------------
echo