#!/bin/bash


save_dir=${4}
src=${3}
trg=${2}
gpu=${1}




function stage1(){
	CUDA_VISIBLE_DEVICES=${1} python trainval_multi.py --model-name resnet50dsbn --exp-setting office-home --sm-loss --adv-loss \
		--source-datasets ${2} --target-datasets ${3} \
		--batch-size 40 --print-console --save-dir ${4}/${2}_${3}/stage1
}

function stage2(){
	CUDA_VISIBLE_DEVICES=${1} python finetune_multi.py --model-name resnet50dsbn --exp-setting office-home \
		--source-datasets ${2} --target-datasets ${3} \
		--pseudo-target-loss default_ensemble --no-lambda \
		--teacher-model-path ${4}/${2}_${3}/stage1/best_resnet50dsbn+None+i0_${2}2${3}.pth \
		--learning-rate 5e-5 --batch-size 40 --save-dir ${4}/${2}_${3}/stage2 --print-console
}



stage1 gpu src trg save_dir
stage2 gpu src trg save_dir



