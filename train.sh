#!/bin/bash

python3 embedding.py --train --model-dir model2

python3 train.py --train --train-samp-ratio 1.0 --num-epoch 20 --train_data data/1_완료_SCS09_A_한국공연문화학회_2017_20210320_591.txt.bz2 \
 --test_data data/1_완료_SCS10_A_한국금융소비자학회_20210220_JKI.txt.bz2 --model_type kospacing --num-gpus 0 --num-cpus 5