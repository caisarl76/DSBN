#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
	--source-datasets RealWorld --target-datasets Product \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/r_p/stage1 

CUDA_VISIBLE_DEVICES=5 python finetune_multi.py --model-name resnet50dsbn --exp-setting office-home \
	--source-datasets RealWorld --target-datasets Product \
	--pseudo-target-loss default_ensemble --no-lambda \
	--teacher-model-path /mnt/dsbn_result/original_dsbn/r_p/stage1/best_resnet50dsbn+None+i0_RealWorld2Product.pth \
	--learning-rate 5e-5 --batch-size 40 --save-dir /mnt/dsbn_result/original_dsbn/r_p/stage2 --print-console

