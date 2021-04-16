#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
	--source-datasets Art --target-datasets RealWorld \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/a_r/stage1 


CUDA_VISIBLE_DEVICES=4 python finetune_multi.py --model-name resnet50dsbn --exp-setting office-home \
	--source-datasets Art --target-datasets RealWorld \ 
	--pseudo-target-loss default_ensemble --no-lambda \
	--teacher-model-path /mnt/dsbn_result/original_dsbn/a_r/stage1/best_resnet50dsbn+None+i0_Art2RealWorld.pth \
	--learning-rate 5e-5 --batch-size 40 --save-dir /mnt/dsbn_result/original_dsbn/a_r/stage2 --print-console


