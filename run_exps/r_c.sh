#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
	--source-datasets RealWorld --target-datasets Clipart \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/r_c/stage1 


