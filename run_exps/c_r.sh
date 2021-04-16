#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
	--source-datasets Clipart --target-datasets RealWorld \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/c_r/stage1 


CUDA_VISIBLE_DEVICES=3 python finetune_multi.py --model-name resnet50dsbn --exp-setting office-home \
	--source-datasets Clipart --target-datasets RealWorld \ 
	--pseudo-target-loss default_ensemble --no-lambda \
	--teacher-model-path /mnt/dsbn_result/original_dsbn/c_r/stage1/best_resnet50dsbn+None+i0_Clipart2RealWorld.pth \
	--learning-rate 5e-5 --batch-size 40 --save-dir /mnt/dsbn_result/original_dsbn/c_r/stage2 --print-console


