#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
	--source-datasets RealWorld --target-datasets Art \
	--batch-size 40 --print-console --save-dir /mnt/dsbn_result/original_dsbn/r_a/stage1 


