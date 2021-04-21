#!/bin/bash


gpu=${1}
src=${2}
trg=${3}
save_dir=${4}




function stage1(){
	echo gpu ${1} 
	echo src ${2}  trg ${3}
	echo save_dir ${4}

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

CUDA_VISIBLE_DEVICES=${1} python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ \
--save-root {4} \
        --save-dir pseudo/${2}_${3}/student --stage 2 --src-domain ${2} --trg-domain ${3}




stage1 ${1} ${2} ${3} ${4}

stage2 ${1} ${2} ${3} ${4}




