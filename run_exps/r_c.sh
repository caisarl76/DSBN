#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
 	--source-datasets RealWorld --target-datasets Clipart \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/r_c/stage1



CUDA_VISIBLE_DEVICES=7 python finetune_multi.py --model-name resnet50dsbn --exp-setting office-home \
	--source-datasets RealWorld --target-datasets Clipart \
       	--pseudo-target-loss default_ensemble --no-lambda \
	--teacher-model-path /mnt/dsbn_result/original_dsbn/r_c/stage1/best_resnet50dsbn+None+i0_RealWorld2Clipart.pth \
	--learning-rate 5e-5 --batch-size 40 --save-dir /mnt/dsbn_result/original_dsbn/r_c/stage2 --print-console



